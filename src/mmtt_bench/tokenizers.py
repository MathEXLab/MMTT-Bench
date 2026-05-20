# benchmark/tokenizers.py
"""
Text and time-series tokenizers for the benchmark.

Text tokenizers wrap embedding models and expose a single interface:
    .encode(texts: list[str]) -> np.ndarray  shape (N, embed_dim)

Available text tokenizers are registered by name in TEXT_TOKENIZERS.
Available TS tokenizers are registered by name in TS_TOKENIZERS.
"""

import numpy as np


class SentenceTransformerTokenizer:
    """
    Wraps sentence_transformers.SentenceTransformer.encode().
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None   # lazy-load on first encode call

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        return np.array(self._model.encode(texts))


class HuggingFaceTokenizer:
    """
    CLS-token embeddings via HuggingFace AutoModel.

    Tokenizes with padding=True and truncation=True, then returns the
    CLS token representation: last_hidden_state[:, 0, :].
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._model     = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model     = AutoModel.from_pretrained(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        self._load()
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            out = self._model(**encoded)
        return out.last_hidden_state[:, 0, :].numpy()


class LLMTokenizer:
    """
    Mean-pool embeddings from a decoder-only LLM (e.g. GPT-2, Llama-7b/8b).

    Decoder-only models have no CLS token, so this mean-pools the last hidden
    state over non-padding token positions. The pad token is set to EOS when
    the model has no dedicated pad token (standard for GPT-2, Llama, etc.).

    Loads lazily on first encode() call. Uses CUDA if available.
    Texts are encoded in mini-batches to avoid OOM on large models.

    Usage:
        tok = LLMTokenizer('gpt2')
        tok = LLMTokenizer('meta-llama/Llama-2-7b-hf', device='cuda', batch_size=4)
        embeddings = tok.encode(texts)  # (N, hidden_dim)
    """
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 8):
        self.model_name = model_name
        self.device     = device   # auto-detected on first _load() if None
        self.batch_size = batch_size
        self._tokenizer = None
        self._model     = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # GPT-2, Llama, and most causal models ship without a pad token.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(self.model_name)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = self._model.to(self.device)
        self._model.eval()

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model(**encoded)
        hidden = out.last_hidden_state                             # (B, L, D)
        mask   = encoded['attention_mask'].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)    # (B, D)
        return pooled.cpu().numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        batches = [
            texts[i: i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.concatenate([self._encode_batch(b) for b in batches], axis=0)


# ---------------------------------------------------------------------------
# Registry — maps embedding_type strings to tokenizer classes.
#
# Usage:
#   tok = TEXT_TOKENIZERS['sentence_transformer'](
#             'sentence-transformers/all-MiniLM-L6-v2')
#   embeddings = tok.encode(texts)   # (N, embed_dim)
# ---------------------------------------------------------------------------

TEXT_TOKENIZERS = {
    'sentence_transformer': SentenceTransformerTokenizer,
    'huggingface':          HuggingFaceTokenizer,
    'llm':                  LLMTokenizer,
}

def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Left-pad arr with zeros along the time axis to reach target_len.

    Handles both 1-D (L,) and 2-D (L, C) arrays so TS tokenizers work
    identically for univariate and multivariate signals.
    """
    arr     = np.asarray(arr)
    pad_len = target_len - arr.shape[0]
    if pad_len <= 0:
        return arr
    if arr.ndim == 1:
        return np.pad(arr, (pad_len, 0), mode='constant', constant_values=0)
    # 2-D: pad along time axis
    return np.vstack([np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype), arr])

#TS tokenizers
# ============================================================================

class PatchTSTokenizer:
    """
    Learned patch projection with d_model=1 (channel-independent PatchTST).

    Univariate input  (L,)    → output (n_patches,)
    Multivariate input (L, C) → each channel encoded independently,
                                 outputs concatenated → (n_patches * C,)

    This is the channel-independent (CI) variant of PatchTST: the same
    Linear(patch_len, 1) projection is applied to every channel.

    Call .build() once before a preprocessing loop so weights are
    constructed with the same seed for every point.

    Usage:
        tok = PatchTSTokenizer(patch_len=8, stride=4, lookback_steps=16)
        tok.build()
        vec = tok.encode(y_lookback)   # (n_patches,) or (n_patches * C,)
    """
    def __init__(self, patch_len=8, stride=4, lookback_steps=16, seed=42):
        self.patch_len      = patch_len
        self.stride         = stride
        self.lookback_steps = lookback_steps
        self.seed           = seed
        self._proj          = None

    def build(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self._proj = nn.Linear(self.patch_len, 1)
        return self

    def _encode_1d(self, y: np.ndarray) -> np.ndarray:
        """Encode a single 1-D channel to (n_patches,)."""
        import torch
        if self._proj is None:
            self.build()
        if len(y) < self.lookback_steps:
            y = _pad(y, self.lookback_steps)
        ts      = torch.tensor(y.reshape(1, -1, 1), dtype=torch.float32)
        patches = ts.permute(0, 2, 1).unfold(2, self.patch_len, self.stride)
        B_, C_, N, P = patches.shape
        t   = self._proj(patches.contiguous().view(B_ * C_, N, P))
        out = t.view(B_, C_, N, 1).mean(1)       # (1, n_patches, 1)
        return out.squeeze().detach().numpy()     # (n_patches,)

    def encode(self, y) -> np.ndarray:
        """
        Encode lookback window.
        y : (L,) or (L, C) ndarray
        Returns (n_patches,) for 1-D or (n_patches * C,) for multivariate.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return self._encode_1d(y)
        # (L, C) — channel-independent: encode each column, concatenate
        return np.concatenate([self._encode_1d(y[:, c]) for c in range(y.shape[1])])

    @property
    def n_patches(self):
        return max(1, (self.lookback_steps - self.patch_len) // self.stride + 1)

    def out_dim(self, n_channels: int = 1) -> int:
        return self.n_patches * n_channels


class PatchMeanTSTokenizer:
    """
    Patch means — no learned weights.

    Univariate (L,)   → (n_patches,)
    Multivariate (L,C) → (n_patches * C,)
    """
    def __init__(self, patch_len=8, stride=4, lookback_steps=16):
        self.patch_len      = patch_len
        self.stride         = stride
        self.lookback_steps = lookback_steps

    def build(self):
        return self

    def _encode_1d(self, y: np.ndarray) -> np.ndarray:
        if len(y) < self.lookback_steps:
            y = _pad(y, self.lookback_steps)
        n = max(1, (self.lookback_steps - self.patch_len) // self.stride + 1)
        patches = np.array([
            y[i * self.stride: i * self.stride + self.patch_len]
            for i in range(n)
        ])
        return patches.mean(axis=1)

    def encode(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return self._encode_1d(y)
        return np.concatenate([self._encode_1d(y[:, c]) for c in range(y.shape[1])])

    @property
    def n_patches(self):
        return max(1, (self.lookback_steps - self.patch_len) // self.stride + 1)

    def out_dim(self, n_channels: int = 1) -> int:
        return self.n_patches * n_channels


class IdentityTSTokenizer:
    """
    Raw lookback values — no patching.

    Univariate (L,)   → (lookback_steps,)
    Multivariate (L,C) → (lookback_steps * C,)  — channels concatenated
    """
    def __init__(self, lookback_steps=16, **_):
        self.lookback_steps = lookback_steps

    def build(self):
        return self

    def _encode_1d(self, y: np.ndarray) -> np.ndarray:
        if len(y) < self.lookback_steps:
            y = _pad(y, self.lookback_steps)
        return y[-self.lookback_steps:]

    def encode(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return self._encode_1d(y)
        return np.concatenate([self._encode_1d(y[:, c]) for c in range(y.shape[1])])

    def out_dim(self, n_channels: int = 1) -> int:
        return self.lookback_steps * n_channels


# Registry
TS_TOKENIZERS = {
    'patch':      PatchTSTokenizer,       # learned projection, d_model=1
    'patch_mean': PatchMeanTSTokenizer,   # patch means, no weights
    'identity':   IdentityTSTokenizer,    # raw lookback values
}
