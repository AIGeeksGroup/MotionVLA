"""
DSFT: Dual-Stream Frequency-domain Tokenizer
============================================

Core design:
  1. Split 276-dim ViMoGen motion (or 263-dim HumanML3D motion) into two
     streams along the feature dimension:
       Base (201-dim, ViMoGen): body_pose_6d[0:126] + joints[126:192] +
                                root_orient_6d[258:264] + root_trans[270:273]
       Phys ( 75-dim, ViMoGen): joints_vel[192:258] + root_vel_6d[264:270] +
                                root_trans_vel[273:276]

  2. For each stream, apply DCT along the time axis and keep the first K rows:
       [T, D] --DCT(axis=0)--> [T, D] --[:K, :]--> [K, D]
       K << T preserves the dominant frequency components.

  3. Flatten [K, D] into K*D integers (after scale + integerize), then encode
     them with a per-stream BPE tokenizer.

  4. Decoding inverts each step: BPE^-1 → [K, D] → zero-pad to [T, D] → IDCT.

The two streams have different spectral profiles, so they are tokenized with
independent codebooks (default base_vocab=4096, phys_vocab=2048).
"""

import json, os
from typing import Optional
import numpy as np
from scipy.fft import dct, idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


# ── ViMoGen 276-dim feature-dimension partition ────────────
BASE_SLICES = [(0, 126), (126, 192), (258, 264), (270, 273)]
PHYS_SLICES = [(192, 258), (264, 270), (273, 276)]
BASE_DIM    = 201
PHYS_DIM    = 75


def split_276(motion: np.ndarray):
    """[T, 276] → (base [T,201], phys [T,75])"""
    base = np.concatenate([motion[:, s:e] for s, e in BASE_SLICES], axis=1)
    phys = np.concatenate([motion[:, s:e] for s, e in PHYS_SLICES], axis=1)
    assert base.shape[1] == BASE_DIM
    assert phys.shape[1] == PHYS_DIM
    return base, phys


class SingleStreamFASTTokenizer:
    """
    单流 FAST Tokenizer：
      encode: [T, D] → token list
      decode: token list + T → [T, D]
    """

    def __init__(
        self,
        bpe: PreTrainedTokenizerFast,
        scale: float,
        min_token: int,
        K: int,
        action_dim: int,
    ):
        self.bpe       = bpe
        self.scale     = scale
        self.min_token = min_token
        self.K         = K
        self.action_dim = action_dim

    # ── encode ─────────────────────────────────────────────
    def encode(self, motion: np.ndarray) -> list[int]:
        """
        motion: [T, D]
        返回：token id 列表
        """
        T = motion.shape[0]
        # 1. DCT 沿时间轴
        freq = dct(motion, axis=0, norm="ortho")   # [T, D]
        # 2. 截取前 K 行
        K_eff = min(self.K, T)
        freq_k = freq[:K_eff, :]                   # [K_eff, D]
        # 3. 量化
        vals = np.around(freq_k.flatten() * self.scale).astype(int)
        # 4. 偏移到非负（BPE 需要可打印字符）
        vals_shifted = vals - self.min_token        # 全部 >= 0
        # 5. 转字符串 → BPE
        token_str = "".join(map(chr, vals_shifted))
        ids = self.bpe(token_str)["input_ids"]
        return ids

    # ── decode ─────────────────────────────────────────────
    def decode(self, token_ids: list[int], T: int) -> np.ndarray:
        """
        token_ids: BPE token 列表
        T: 原始帧数（用于 IDCT zero-padding）
        返回：重建动作 [T, D]
        """
        K_eff = min(self.K, T)
        expected_len = K_eff * self.action_dim

        decoded_str = self.bpe.decode(token_ids)
        vals_shifted = np.array(list(map(ord, decoded_str)))

        # 修剪或补零到正确长度
        if len(vals_shifted) >= expected_len:
            vals_shifted = vals_shifted[:expected_len]
        else:
            vals_shifted = np.pad(
                vals_shifted, (0, expected_len - len(vals_shifted)))

        vals = (vals_shifted + self.min_token).reshape(K_eff, self.action_dim)
        freq_k = vals / self.scale                  # [K_eff, D]

        # zero-pad 到 [T, D]，再 IDCT
        freq_full = np.zeros((T, self.action_dim), dtype=np.float32)
        freq_full[:K_eff, :] = freq_k
        motion = idct(freq_full, axis=0, norm="ortho")
        return motion.astype(np.float32)

    # ── 保存 / 加载 ─────────────────────────────────────────
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.bpe.save_pretrained(directory)
        cfg = {
            "scale":      self.scale,
            "min_token":  self.min_token,
            "K":          self.K,
            "action_dim": self.action_dim,
        }
        with open(os.path.join(directory, "fast_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "SingleStreamFASTTokenizer":
        bpe = PreTrainedTokenizerFast.from_pretrained(directory)
        with open(os.path.join(directory, "fast_config.json")) as f:
            cfg = json.load(f)
        return cls(bpe=bpe, **cfg)

    # ── 训练（类方法）──────────────────────────────────────
    @classmethod
    def fit(
        cls,
        data: list[np.ndarray],   # list of [T, D]
        K: int,
        scale: float,
        vocab_size: int,
        action_dim: int,
    ) -> "SingleStreamFASTTokenizer":
        """
        从一批 [T, D] 数据训练 BPE tokenizer。
        """
        # 1. 提取所有序列的前K行DCT
        all_chars = []
        for motion in data:
            T = motion.shape[0]
            if T < 2:
                continue
            K_eff = min(K, T)
            freq   = dct(motion, axis=0, norm="ortho")
            freq_k = freq[:K_eff, :]
            vals   = np.around(freq_k.flatten() * scale).astype(int)
            all_chars.extend(vals.tolist())

        arr = np.array(all_chars)
        min_token = int(arr.min())
        max_token = int(arr.max())
        token_range = max_token - min_token

        print(f"  DCT系数整数范围: [{min_token}, {max_token}] = {token_range}")
        assert vocab_size > token_range, (
            f"vocab_size={vocab_size} 必须 > token_range={token_range}，"
            f"请增大 vocab_size 或减小 scale"
        )

        # 2. 构造字符串迭代器
        def _iter():
            for motion in data:
                T = motion.shape[0]
                if T < 2:
                    continue
                K_eff = min(K, T)
                freq   = dct(motion, axis=0, norm="ortho")
                freq_k = freq[:K_eff, :]
                vals   = np.around(freq_k.flatten() * scale).astype(int)
                vals_s = (vals - min_token).astype(int)
                yield "".join(map(chr, vals_s))

        # 3. 训练 BPE
        alphabet = [chr(i) for i in range(token_range + 1)]
        bpe_tok  = ByteLevelBPETokenizer()
        trainer  = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=512,
        )
        bpe_tok._tokenizer.train_from_iterator(_iter(), trainer=trainer)

        fast = PreTrainedTokenizerFast(
            tokenizer_object=bpe_tok,
            clean_up_tokenization_spaces=False,
        )
        return cls(
            bpe=fast,
            scale=scale,
            min_token=min_token,
            K=K,
            action_dim=action_dim,
        )


class DSFTTokenizer:
    """
    DSFT (Dual-Stream Frequency-domain Tokenizer) top-level interface.
    Manages a Base and a Phys SingleStreamFASTTokenizer.
    """

    def __init__(
        self,
        base_tok: SingleStreamFASTTokenizer,
        phys_tok: SingleStreamFASTTokenizer,
    ):
        self.base_tok = base_tok
        self.phys_tok = phys_tok

    def encode(self, motion: np.ndarray) -> dict:
        """
        motion: [T, 276]
        returns: {"base_tokens": list[int], "phys_tokens": list[int], "T": int}
        """
        import torch
        if isinstance(motion, torch.Tensor):
            motion = motion.numpy()
        base, phys = split_276(motion)
        T = motion.shape[0]
        return {
            "base_tokens": self.base_tok.encode(base),
            "phys_tokens": self.phys_tok.encode(phys),
            "T": T,
        }

    def decode(self, base_tokens: list[int], phys_tokens: list[int], T: int):
        """
        returns: (base_recon [T,201], phys_recon [T,75])
        """
        base_recon = self.base_tok.decode(base_tokens, T)
        phys_recon = self.phys_tok.decode(phys_tokens, T)
        return base_recon, phys_recon

    def save(self, directory: str):
        self.base_tok.save(os.path.join(directory, "base"))
        self.phys_tok.save(os.path.join(directory, "phys"))
        print(f"DSFT tokenizer saved to: {directory}")

    @classmethod
    def load(cls, directory: str) -> "DSFTTokenizer":
        base_tok = SingleStreamFASTTokenizer.load(os.path.join(directory, "base"))
        phys_tok = SingleStreamFASTTokenizer.load(os.path.join(directory, "phys"))
        return cls(base_tok, phys_tok)

    @classmethod
    def fit(
        cls,
        data: list[np.ndarray],        # list of [T, 276]
        K_base: int   = 5,
        K_phys: int   = 25,
        scale: float  = 10.0,
        base_vocab: int = 4096,
        phys_vocab: int = 2048,
    ) -> "DSFTTokenizer":
        """
        Train a dual-stream tokenizer from raw 276-dim motion data.
        """
        base_data, phys_data = [], []
        for motion in data:
            b, p = split_276(motion)
            base_data.append(b)
            phys_data.append(p)

        print(f"=== Training Base tokenizer (K={K_base}, D=201, vocab={base_vocab}) ===")
        base_tok = SingleStreamFASTTokenizer.fit(
            base_data, K=K_base, scale=scale,
            vocab_size=base_vocab, action_dim=BASE_DIM,
        )

        print(f"\n=== Training Phys tokenizer (K={K_phys}, D=75, vocab={phys_vocab}) ===")
        phys_tok = SingleStreamFASTTokenizer.fit(
            phys_data, K=K_phys, scale=scale,
            vocab_size=phys_vocab, action_dim=PHYS_DIM,
        )

        return cls(base_tok, phys_tok)


# Backwards-compatible alias for any external code still importing the old name.
DSFASTTokenizer = DSFTTokenizer
