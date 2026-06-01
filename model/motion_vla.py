import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, T5Config, T5ForConditionalGeneration

# ── 词表常量 ────────────────────────────────────────────────────────────────
T5_VOCAB_SIZE    = 32100
BASE_VOCAB_SIZE  = 4096
PHYS_VOCAB_SIZE  = 2048
TOTAL_VOCAB_SIZE = T5_VOCAB_SIZE + BASE_VOCAB_SIZE + PHYS_VOCAB_SIZE  # 38244

BASE_OFFSET = T5_VOCAB_SIZE                       # 32100
PHYS_OFFSET = T5_VOCAB_SIZE + BASE_VOCAB_SIZE     # 36196
BOS_ID      = 0
EOS_ID      = 1
SEP_ID      = 32099   # <extra_id_0>
PAD_ID      = 0


class VisualFeatureResampler(nn.Module):
    """Conv1d 下采样 + interpolate，将 Qwen 输出对齐到固定长度。"""
    def __init__(self, in_dim, out_dim, target_len=256):
        super().__init__()
        self.target_len = target_len
        self.proj_in    = nn.Linear(in_dim, out_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj_in(x)                          # (B, L, out_dim)
        x = x.transpose(1, 2)                        # (B, out_dim, L)
        x = self.conv_layers(x)
        orig_dtype = x.dtype
        x = F.interpolate(x.float(), size=self.target_len,
                          mode="linear", align_corners=False).to(orig_dtype)
        x = x.transpose(1, 2)                        # (B, target_len, out_dim)
        return self.norm(x)


class MotionMoELoRALayer(nn.Module):
    """
    MoE 残差层。8 个 Expert，每个为 Bottleneck LoRA（rank=32）。
    在 phys token 生成阶段，hidden state 已包含完整 base 上下文，
    MoE 在此路由到对应速度风格 expert，是其核心作用点。
    """
    def __init__(self, hidden_size, num_experts=8, lora_rank=32):
        super().__init__()
        self.num_experts = num_experts
        self.router  = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, lora_rank, bias=False),
                nn.SiLU(),
                nn.Linear(lora_rank, hidden_size, bias=False),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, L, H)
        router_logits   = self.router(x)                          # (B, L, E)
        routing_weights = F.softmax(router_logits, dim=-1)        # (B, L, E)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = routing_weights[:, :, i].unsqueeze(-1)            # (B, L, 1)
            out = out + expert(x) * w

        # load balancing auxiliary loss（挂在 module 上供 trainer 读取）
        self._router_logits = router_logits
        return out


class MotionVLA(nn.Module):
    """
    MotionVLA v2：
      Qwen3.5-VL → context_projector → T5 Decoder (扩容词表) → MoE → lm_head
    T5 Decoder 自回归生成单一序列：
      [BOS, base tokens (32100+), SEP, phys tokens (36196+), EOS]
    """
    def __init__(
        self,
        qwen_model_path,
        t5_model_path,
        num_experts=8,
        context_target_len=256,
    ):
        super().__init__()

        # ── 1. Qwen3.5-VL（冻结）──────────────────────────────────────────
        print(f"Loading Qwen3.5-VL from {qwen_model_path}...")
        self.vision_language_encoder = AutoModelForImageTextToText.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        for p in self.vision_language_encoder.parameters():
            p.requires_grad = False

        if hasattr(self.vision_language_encoder.config, "hidden_size"):
            qwen_hidden = self.vision_language_encoder.config.hidden_size
        elif hasattr(self.vision_language_encoder.config, "text_config"):
            qwen_hidden = self.vision_language_encoder.config.text_config.hidden_size
        else:
            qwen_hidden = 1024

        # ── 2. T5 Decoder（词表扩容，Full Fine-Tuning）────────────────────
        print(f"Loading T5 Decoder from {t5_model_path}...")
        t5_config    = T5Config.from_pretrained(t5_model_path)
        t5_hidden    = t5_config.d_model          # 768

        t5_full = T5ForConditionalGeneration.from_pretrained(t5_model_path)

        # 词表扩容：32100 → 38244
        t5_full.resize_token_embeddings(TOTAL_VOCAB_SIZE)
        print(f"T5 词表扩容: {T5_VOCAB_SIZE} → {TOTAL_VOCAB_SIZE}")

        self.action_decoder = t5_full.decoder     # 含已扩容的 embed_tokens
        self.lm_head        = t5_full.lm_head     # Linear(768, 38244)

        # lm_head 权重与 embed_tokens 解绑（扩容后已经独立）
        self.lm_head.weight = nn.Parameter(
            self.lm_head.weight.clone().detach()
        )

        # T5 在 forward 中会做 sequence_output *= model_dim**-0.5 再送入 lm_head，
        # 我们直接用 decoder 绕过了这一步，需在 lm_head 前手动施加相同缩放。
        self.lm_head_scale = t5_config.d_model ** -0.5   # ≈ 0.03608 for d_model=768

        # 新增 action token 的 lm_head 行和 embed_tokens 行用小值重新初始化。
        # resize_token_embeddings 用多元正态初始化（std≈4.6），经过 lm_head_scale 后
        # logit std 仍远大于 1，softmax 极度集中导致初始 loss 虚高。
        # 用 std=0.02（标准小值初始化）可使初始 logit std≈0.02，接近均匀分布。
        with torch.no_grad():
            nn.init.normal_(self.lm_head.weight[T5_VOCAB_SIZE:], mean=0.0, std=0.02)
            nn.init.normal_(
                self.action_decoder.embed_tokens.weight[T5_VOCAB_SIZE:],
                mean=0.0, std=0.02
            )

        # ── 3. context_projector（Qwen → T5 空间）────────────────────────
        self.context_projector = VisualFeatureResampler(
            in_dim=qwen_hidden,
            out_dim=t5_hidden,
            target_len=context_target_len,
        ).to(torch.bfloat16)

        # ── 4. MoE Layer ──────────────────────────────────────────────────
        self.moe_layer = MotionMoELoRALayer(
            hidden_size=t5_hidden,
            num_experts=num_experts,
        ).to(torch.bfloat16)

        self.pad_token_id = PAD_ID

        # 训练时 logit mask：只允许有效 action token 参与 softmax
        # 原始 T5 词表中 32098 个 token 永远不是 target，让它们竞争 softmax 会虚增 loss
        # 有效 token：EOS(1), SEP(32099), base[32100:36196], phys[36196:38244]
        train_mask = torch.full((TOTAL_VOCAB_SIZE,), float("-inf"))
        train_mask[EOS_ID]                                    = 0.0
        train_mask[SEP_ID]                                    = 0.0
        train_mask[BASE_OFFSET: BASE_OFFSET + BASE_VOCAB_SIZE] = 0.0
        train_mask[PHYS_OFFSET: PHYS_OFFSET + PHYS_VOCAB_SIZE] = 0.0
        self.register_buffer("train_logit_mask", train_mask)

    # ── 特征提取 ──────────────────────────────────────────────────────────
    def extract_qwen_context(self, input_ids, attention_mask,
                             pixel_values, image_grid_thw):
        with torch.no_grad():
            out = self.vision_language_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            return out.hidden_states[-2]   # 倒数第二层

    # ── 前向传播 ──────────────────────────────────────────────────────────
    def forward(
        self,
        qwen_input_ids,
        qwen_attention_mask,
        pixel_values,
        image_grid_thw,
        input_ids,     # decoder 输入  (B, L-1)
        target_ids,    # 预测目标      (B, L-1)
        attn_mask,     # decoder mask  (B, L-1)
    ):
        # 1. Qwen 上下文
        ctx = self.extract_qwen_context(
            qwen_input_ids, qwen_attention_mask,
            pixel_values, image_grid_thw,
        )                                                     # (B, L_q, H_q)

        # 2. 投影到 T5 空间（ctx 用完立即释放，节省显存）
        enc_hs = self.context_projector(
            ctx.to(self.context_projector.proj_in.weight.dtype)
        ).float()                                             # (B, 256, 768)
        del ctx

        # 3. T5 Decoder（embed_tokens 内部查表）
        dec_out = self.action_decoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=torch.ones(
                enc_hs.shape[:2], dtype=torch.long, device=enc_hs.device
            ),
            return_dict=True,
        )
        h = dec_out.last_hidden_state                        # (B, L-1, 768)

        # 4. MoE 残差
        h = h + self.moe_layer(h.to(torch.bfloat16)).to(h.dtype)

        # 5. lm_head → logits（T5 原生 forward 中 sequence_output *= model_dim**-0.5）
        if self.lm_head.weight.dtype != torch.float32:
            self.lm_head.to(torch.float32)
        logits = self.lm_head(h.float() * self.lm_head_scale)   # (B, L-1, 38244)

        # 6. Loss（训练时 mask 掉无效 token，只在 6146 个有效 token 上竞争）
        masked_logits = logits + self.train_logit_mask
        loss = F.cross_entropy(
            masked_logits.reshape(-1, TOTAL_VOCAB_SIZE),
            target_ids.reshape(-1),
            ignore_index=PAD_ID,
        )

        return {"loss": loss, "logits": logits}

    # ── 推理生成 ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_motion(
        self,
        qwen_input_ids,
        qwen_attention_mask,
        pixel_values,
        image_grid_thw,
        max_base_len=700,
        max_phys_len=150,
    ):
        """
        自回归生成单一序列。
        推理时用 logit masking 强制：
          - SEP 出现前：只在 base token 范围 [32100, 36195] 采样
          - SEP 出现后：只在 phys token 范围 [36196, 38243] 采样
        """
        device     = qwen_input_ids.device
        batch_size = qwen_input_ids.shape[0]

        ctx    = self.extract_qwen_context(
            qwen_input_ids, qwen_attention_mask,
            pixel_values, image_grid_thw)
        enc_hs = self.context_projector(
            ctx.to(self.context_projector.proj_in.weight.dtype)
        ).float()
        enc_mask = torch.ones(enc_hs.shape[:2], dtype=torch.long, device=device)

        generated = torch.full((batch_size, 1), BOS_ID,
                               dtype=torch.long, device=device)
        in_phys   = [False] * batch_size
        done      = [False] * batch_size

        # 预先构建 mask（-inf 屏蔽不允许的 token）
        base_mask_vec = torch.full((TOTAL_VOCAB_SIZE,), float("-inf"), device=device)
        base_mask_vec[BASE_OFFSET: BASE_OFFSET + BASE_VOCAB_SIZE] = 0.0
        base_mask_vec[SEP_ID] = 0.0   # 允许输出 SEP 来结束 base 段

        phys_mask_vec = torch.full((TOTAL_VOCAB_SIZE,), float("-inf"), device=device)
        phys_mask_vec[PHYS_OFFSET: PHYS_OFFSET + PHYS_VOCAB_SIZE] = 0.0
        phys_mask_vec[EOS_ID] = 0.0   # 允许输出 EOS 来结束 phys 段

        for _ in range(max_base_len + max_phys_len):
            if all(done):
                break

            dec_out = self.action_decoder(
                input_ids=generated,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                return_dict=True,
            )
            h      = dec_out.last_hidden_state[:, -1:, :]   # (B, 1, 768)
            h      = h + self.moe_layer(h.to(torch.bfloat16)).to(h.dtype)
            logits = self.lm_head(h.float() * self.lm_head_scale).squeeze(1)  # (B, 38244)

            # logit masking
            for b in range(batch_size):
                if done[b]:
                    continue
                if in_phys[b]:
                    logits[b] = logits[b] + phys_mask_vec
                else:
                    logits[b] = logits[b] + base_mask_vec

            next_tok = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # 更新状态
            for b in range(batch_size):
                t = next_tok[b, 0].item()
                if t == SEP_ID:
                    in_phys[b] = True
                elif t == EOS_ID:
                    done[b] = True

            generated = torch.cat([generated, next_tok], dim=1)

        return generated   # (B, seq_len)  含 BOS/SEP/EOS
