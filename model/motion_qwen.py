"""
MotionQwen: Qwen3.5-VL + LoRA + optional MoE
Server version — CUDA optimized, no hardcoded paths.

Vocab design (DS-FAST v2):
  Qwen original:   [0,      248320)
  Base tokens:     [248320, 252416)   4096 tokens
  Phys tokens:     [252416, 256512)   4096 tokens  (v2: was 2048)
  Special:         256512=BOS, 256513=SEP, 256514=EOS
  Total:           256515
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText
from peft import get_peft_model, LoraConfig, TaskType

# ── Vocab constants ──────────────────────────────────────────────
QWEN_ORIG_VOCAB  = 248320
BASE_VOCAB_SIZE  = 4096
PHYS_VOCAB_SIZE  = 4096   # v2
BASE_OFFSET      = QWEN_ORIG_VOCAB
PHYS_OFFSET      = QWEN_ORIG_VOCAB + BASE_VOCAB_SIZE   # 252416
MOTION_BOS_ID    = QWEN_ORIG_VOCAB + BASE_VOCAB_SIZE + PHYS_VOCAB_SIZE      # 256512
MOTION_SEP_ID    = MOTION_BOS_ID + 1                                         # 256513
MOTION_EOS_ID    = MOTION_BOS_ID + 2                                         # 256514
TOTAL_VOCAB      = MOTION_BOS_ID + 3                                         # 256515


class MotionMoELayer(nn.Module):
    """
    Mixture-of-Experts residual layer.
    Inserted between Qwen's last hidden state and lm_head.
    Each expert is a low-rank bottleneck (LoRA-style).
    """
    def __init__(self, hidden_size: int, num_experts: int = 8, lora_rank: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.router  = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, lora_rank, bias=False),
                nn.SiLU(),
                nn.Linear(lora_rank, hidden_size, bias=False),
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        router_logits   = self.router(x)                       # (B, L, E)
        routing_weights = F.softmax(router_logits, dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = routing_weights[..., i].unsqueeze(-1)          # (B, L, 1)
            out = out + expert(x) * w
        self._router_logits = router_logits
        return out


class MotionQwen(nn.Module):
    """
    Qwen3.5-VL + LoRA + optional MoE for motion token generation.

    Key design:
    - motion_embed / motion_lm_head_w: only 6147 new rows stored as nn.Parameter
      (avoids gradients on the full 248K-row embedding, saving ~1 GB)
    - Forward hook injects trainable motion embeddings into Qwen's embed_tokens
    - Logit mask at training time restricts prediction to motion-token vocab
    """

    def __init__(
        self,
        qwen_model_path: str,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        use_moe: bool = True,
        num_experts: int = 8,
        device_map: str = "auto",
    ):
        super().__init__()
        self.use_moe = use_moe

        # ── 1. Load Qwen3.5-VL ──────────────────────────────────
        print(f"Loading Qwen3.5-VL from {qwen_model_path} ...")
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            qwen_model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        # ── 2. Vocabulary expansion: 248320 → 254467 ────────────
        self.qwen.resize_token_embeddings(TOTAL_VOCAB)
        print(f"Vocab expanded: {QWEN_ORIG_VOCAB} → {TOTAL_VOCAB}")

        with torch.no_grad():
            emb = self.qwen.get_input_embeddings()
            nn.init.normal_(emb.weight[QWEN_ORIG_VOCAB:], mean=0.0, std=0.02)
            lm_head = self._get_lm_head()
            if lm_head is not None:
                nn.init.normal_(lm_head.weight[QWEN_ORIG_VOCAB:], mean=0.0, std=0.02)

        # ── 3. LoRA on attention + FFN projections ───────────────
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()

        # ── 4. Optional MoE layer ────────────────────────────────
        qwen_hidden = (self.qwen.config.text_config.hidden_size
                       if hasattr(self.qwen.config, "text_config")
                       else self.qwen.config.hidden_size)
        if use_moe:
            self.moe_layer = MotionMoELayer(
                hidden_size=qwen_hidden,
                num_experts=num_experts,
            ).to(torch.bfloat16)
            print(f"MoE enabled: hidden={qwen_hidden}, experts={num_experts}")

        # ── 5. Trainable motion embeddings (memory-efficient) ────
        # PEFT freezes all base model params (including new embed rows).
        # Storing only 6147 new rows as nn.Parameter (~12 MB vs ~1 GB for full embed).
        _N = TOTAL_VOCAB - QWEN_ORIG_VOCAB  # 6147

        with torch.no_grad():
            _emb_w = self.qwen.get_input_embeddings().weight
            _lm_h  = self._get_lm_head()
            _lm_w  = _lm_h.weight if _lm_h is not None else None
            e_init = _emb_w[QWEN_ORIG_VOCAB:].float().clone()
            l_init = _lm_w[QWEN_ORIG_VOCAB:].float().clone() if _lm_w is not None else None

        self.motion_embed     = nn.Parameter(e_init)           # (6147, H) float32
        self.motion_lm_head_w = nn.Parameter(l_init) if l_init is not None else None
        print(f"Motion embedding params: {_N}×{qwen_hidden} = {_N*qwen_hidden/1e6:.1f}M")

        # Forward hook: replace motion token embeddings at runtime
        def _motion_embed_hook(module, inp, out):
            ids = inp[0]
            motion_mask = ids >= QWEN_ORIG_VOCAB
            if not motion_mask.any():
                return out
            idx         = (ids - QWEN_ORIG_VOCAB).clamp(min=0)
            motion_embs = self.motion_embed[idx].to(out.dtype)
            return torch.where(motion_mask.unsqueeze(-1).expand_as(out), motion_embs, out)

        self.qwen.get_input_embeddings().register_forward_hook(_motion_embed_hook)

        # ── 6. Logit mask: restrict predictions to motion tokens ─
        train_mask = torch.full((TOTAL_VOCAB,), float("-inf"))
        train_mask[MOTION_BOS_ID] = 0.0
        train_mask[MOTION_SEP_ID] = 0.0
        train_mask[MOTION_EOS_ID] = 0.0
        train_mask[BASE_OFFSET: BASE_OFFSET + BASE_VOCAB_SIZE] = 0.0
        train_mask[PHYS_OFFSET: PHYS_OFFSET + PHYS_VOCAB_SIZE] = 0.0
        self.register_buffer("train_logit_mask", train_mask)

    def _get_lm_head(self):
        for m in [self.qwen,
                  getattr(self.qwen, "base_model", None),
                  getattr(getattr(self.qwen, "base_model", None), "model", None)]:
            if m is not None and hasattr(m, "lm_head"):
                return m.lm_head
        return None

    def forward(self, input_ids, attention_mask, labels,
                pixel_values=None, image_grid_thw=None, mm_token_type_ids=None):
        if self.use_moe:
            out = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            h = out.hidden_states[-1].to(torch.bfloat16)   # (B, L, H)
            h = h + self.moe_layer(h)

            lm_head = self._get_lm_head()
            logits  = lm_head(h)                     # (B, L, V) bfloat16
            if self.motion_lm_head_w is not None:
                logits = logits.clone()
                logits[..., QWEN_ORIG_VOCAB:] = h @ self.motion_lm_head_w.to(h.dtype).T

            masked = logits + self.train_logit_mask.to(h.dtype)
            loss = F.cross_entropy(
                masked.float().reshape(-1, TOTAL_VOCAB),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            out = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                return_dict=True,
            )
            loss = out.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate_motion(self, input_ids, attention_mask,
                        pixel_values=None, image_grid_thw=None,
                        max_new_tokens=800, temperature=1.0):
        device = input_ids.device
        B = input_ids.shape[0]

        base_mask = torch.full((TOTAL_VOCAB,), float("-inf"), device=device)
        base_mask[BASE_OFFSET: BASE_OFFSET + BASE_VOCAB_SIZE] = 0.0
        base_mask[MOTION_SEP_ID] = 0.0

        phys_mask = torch.full((TOTAL_VOCAB,), float("-inf"), device=device)
        phys_mask[PHYS_OFFSET: PHYS_OFFSET + PHYS_VOCAB_SIZE] = 0.0
        phys_mask[MOTION_EOS_ID] = 0.0

        bos      = torch.full((B, 1), MOTION_BOS_ID, dtype=torch.long, device=device)
        init_ids = torch.cat([input_ids, bos], dim=1)
        init_attn = torch.cat([attention_mask,
                               torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)

        out = self.qwen(
            input_ids=init_ids,
            attention_mask=init_attn,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=self.use_moe,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values

        if self.use_moe:
            h = out.hidden_states[-1][:, -1:, :].float()
            h = h + self.moe_layer(h.to(torch.bfloat16)).float()
            logits = self._get_lm_head()(h).squeeze(1)
            if self.motion_lm_head_w is not None:
                logits[..., QWEN_ORIG_VOCAB:] = h.squeeze(1) @ self.motion_lm_head_w.to(h.dtype).T
        else:
            logits = out.logits[:, -1, :]

        generated = [[] for _ in range(B)]
        in_phys   = [False] * B
        done      = [False] * B

        for b in range(B):
            logits[b] += base_mask
        if temperature != 1.0:
            logits /= temperature
        next_tok = logits.argmax(dim=-1, keepdim=True)

        for b in range(B):
            t = next_tok[b, 0].item()
            generated[b].append(t)
            if t == MOTION_SEP_ID: in_phys[b] = True
            if t == MOTION_EOS_ID: done[b]    = True

        cur_len = init_ids.shape[1]
        for step in range(max_new_tokens - 1):
            if all(done):
                break
            cur_len += 1
            cur_attn = torch.ones(B, cur_len, dtype=torch.long, device=device)

            out = self.qwen(
                input_ids=next_tok,
                attention_mask=cur_attn,
                output_hidden_states=self.use_moe,
                use_cache=True,
                past_key_values=past_kv,
                return_dict=True,
            )
            past_kv = out.past_key_values

            if self.use_moe:
                h = out.hidden_states[-1][:, -1:, :].float()
                h = h + self.moe_layer(h.to(torch.bfloat16)).float()
                logits = self._get_lm_head()(h).squeeze(1)
                if self.motion_lm_head_w is not None:
                    logits[..., QWEN_ORIG_VOCAB:] = h.squeeze(1) @ self.motion_lm_head_w.to(h.dtype).T
            else:
                logits = out.logits[:, -1, :]

            for b in range(B):
                if not done[b]:
                    logits[b] += (phys_mask if in_phys[b] else base_mask)
            if temperature != 1.0:
                logits /= temperature
            next_tok = logits.argmax(dim=-1, keepdim=True)

            for b in range(B):
                t = next_tok[b, 0].item()
                generated[b].append(t)
                if t == MOTION_SEP_ID: in_phys[b] = True
                if t == MOTION_EOS_ID: done[b]    = True

        full = []
        for b in range(B):
            full.append(torch.tensor(
                init_ids[b].tolist() + generated[b], dtype=torch.long, device=device))
        max_len = max(t.shape[0] for t in full)
        result  = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, t in enumerate(full):
            result[b, :t.shape[0]] = t
        return result
