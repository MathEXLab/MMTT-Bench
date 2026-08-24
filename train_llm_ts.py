#!/usr/bin/env python3
"""
train_llm_ts.py — LLM-based time series forecasting on mmtt-bench / TimeMMD.

Models
------
  TimeLLM  (KimMeen/Time-LLM, ICLR 2024)
    TS patches reprogrammed into LLM token space via multi-head cross-attention
    with learned prototypes, then processed by a frozen GPT2 backbone.
    Trains: patch embedding · reprogramming layer · output head.

  GPT4MTS  (Flora-jia-jfr/GPT4MTS, 2023)
    TS patches linearly projected to LLM embedding space; enriched prompt
    includes domain context, per-sample TS statistics, and text annotation.
    Trains: patch projection · output head.

  LLMTime  (Gruver et al., NeurIPS 2023  —  ngruver/llmtime)
    Zero-shot.  Scales history with quantile normalisation (alpha=0.95,
    beta=0.3), serialises to space-separated base-10 digit tokens, and asks
    a causal LM to continue the digit stream.  Forecast = median of n_samples
    completions, inverse-scaled back to the original range.
    No training — pure in-context digit continuation.

  PromptCast  (Xue & Salim 2023  —  adapted from ngruver/llmtime/models/promptcast.py)
    Zero-shot.  Same digit serialisation as LLMTime but wraps the series in a
    natural-language question template and uses identity scaling.  Domain
    context and optional annotation are prepended as a free-text preamble.
    No training — pure causal completion.

  DirectPrompt  (CiK-style direct prompting; protocol of arXiv:2506.21611)
    Zero-shot.  Context + plain-decimal history in natural language; an
    instruction-tuned LLM is asked to return the forecast as comma-separated
    values between <forecast></forecast> tags.  Completions that do not parse
    to pred_len values are rejected; forecast = median of parsed completions.
    Use with an instruct model (e.g. Qwen2.5-Instruct) — a base LM like GPT2
    falls back to primed plain-text completion but is a strawman here.

Text conditions mirror train_transformer_cfa.py:
  domain_only  — domain prompt only, no per-sample annotation (baseline)
  correct      — domain prompt + correct annotation
  incorrect    — domain prompt + incorrect annotation
  irrelevant   — domain prompt + irrelevant annotation

Domain prompts come from Appendix C of:
  "When Does Multimodality Lead to Better Time Series Forecasting?" (TimeMMD).

Usage
-----
    # LLM-backbone (trained)
    python train_llm_ts.py --config timemmd_llm.yaml \\
        --dataset Agriculture --model TimeLLM --pred-len 6

    # Zero-shot
    python train_llm_ts.py --config timemmd_llm.yaml \\
        --dataset Agriculture --model LLMTime --pred-len 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Reversible Instance Normalisation  (Kim et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalise(self, x: torch.Tensor):
        """x: [B, L, C] → (x_norm, mean [B,1,C], std [B,1,C])"""
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std, mean, std

    def denormalise(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Patching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n_patches(seq_len: int, patch_len: int, stride: int) -> int:
    return (seq_len - patch_len) // stride + 1


def _make_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """x: [B, seq_len, 1] → patches [B, n_patches, patch_len]"""
    return x[:, :, 0].unfold(1, patch_len, stride)


# ─────────────────────────────────────────────────────────────────────────────
# Reprogramming layer  (TimeLLM)
# ─────────────────────────────────────────────────────────────────────────────

class ReprogrammingLayer(nn.Module):
    """
    Multi-head cross-attention that maps TS patch embeddings (d_model) into
    LLM space (d_llm) via n_tokens learned prototype vectors.

    Q = patch embeddings  [B, N, d_model]
    K = V = prototypes    [n_tokens, d_llm]  (shared across batch)
    output                [B, N, d_llm]
    """
    def __init__(self, d_model: int, n_heads: int, d_llm: int, n_tokens: int):
        super().__init__()
        assert d_llm % n_heads == 0, f"d_llm={d_llm} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head  = d_llm // n_heads
        self.scale   = self.d_head ** -0.5

        # Learned prototypes in LLM embedding space
        self.prototypes = nn.Parameter(torch.randn(n_tokens, d_llm) * 0.02)
        self.W_Q   = nn.Linear(d_model, d_llm, bias=False)
        self.W_out = nn.Linear(d_llm,   d_llm, bias=False)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        """pe: [B, N, d_model] → [B, N, d_llm]"""
        B, N, _ = pe.shape
        Q = self.W_Q(pe)                                          # [B, N, d_llm]
        K = V = self.prototypes.unsqueeze(0).expand(B, -1, -1)   # [B, T, d_llm]

        def _split(t: torch.Tensor) -> torch.Tensor:
            s = t.shape[1]
            return t.reshape(B, s, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = _split(Q), _split(K), _split(V)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, N, -1)     # [B, N, d_llm]
        return self.W_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# TimeLLM
# ─────────────────────────────────────────────────────────────────────────────

class TimeLLM(nn.Module):
    """
    TimeLLM (ICLR 2024, KimMeen/Time-LLM).

    Data flow  (enc_in=1 for mmtt-bench):
      x [B, seq_len, 1]
        → RevIN normalise
        → patches → Linear(patch_len, d_model) → ReprogrammingLayer → [B, N, d_llm]
        → cat([prompt_embs [B, L_p, d_llm], ts_emb [B, N, d_llm]])
        → frozen GPT2 (gradients flow through for reprogramming updates)
        → take last N hidden states → mean-pool
        → Linear(d_llm, pred_len) → [B, pred_len]
        → RevIN denormalise → [B, pred_len, 1]

    Only patch_embed, reprogramming, output_proj are trained.
    """
    def __init__(
        self,
        seq_len:   int,
        pred_len:  int,
        d_model:   int,
        patch_len: int,
        stride:    int,
        n_heads:   int,
        n_tokens:  int,
        d_llm:     int,
        dropout:   float,
        llm_model: nn.Module,
        tokenizer,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.d_llm     = d_llm
        self.llm       = llm_model
        self.tokenizer = tokenizer
        self.N         = _n_patches(seq_len, patch_len, stride)

        self.patch_embed   = nn.Linear(patch_len, d_model)
        self.reprogramming = ReprogrammingLayer(d_model, n_heads, d_llm, n_tokens)
        self.drop          = nn.Dropout(dropout)
        self.output_proj   = nn.Linear(d_llm, pred_len)
        self.revin         = RevIN()

        for p in self.llm.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, prompt_texts: list[str]) -> torch.Tensor:
        """x: [B, seq_len, 1] → [B, pred_len, 1]"""
        B      = x.size(0)
        device = x.device

        x_norm, mean, std = self.revin.normalise(x)          # [B, seq_len, 1]

        patches = _make_patches(x_norm, self.patch_len, self.stride)  # [B, N, patch_len]
        pe      = self.drop(self.patch_embed(patches))                  # [B, N, d_model]
        ts_emb  = self.reprogramming(pe)                                # [B, N, d_llm]

        # Prompt embeddings — computed without gradient (frozen embed table).
        # Gradients still flow through the LLM forward to reach ts_emb.
        with torch.no_grad():
            enc        = self.tokenizer(prompt_texts, return_tensors='pt',
                                        padding=True, truncation=True, max_length=512)
            ids        = enc.input_ids.to(device)
            prompt_mask = enc.attention_mask.to(device)          # [B, L_p]
            prompt_emb  = self.llm.get_input_embeddings()(ids).float()  # [B, L_p, d_llm]

        n_prompt = prompt_emb.size(1)
        combined = torch.cat([prompt_emb, ts_emb], dim=1)       # [B, L_p+N, d_llm]
        attn_mask = torch.cat([
            prompt_mask,
            torch.ones(B, self.N, dtype=torch.long, device=device),
        ], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        hidden = self.llm(
            inputs_embeds  = combined.to(dtype=llm_dtype),
            attention_mask = attn_mask,
        ).last_hidden_state.float()                              # [B, L_p+N, d_llm]

        ts_out = hidden[:, n_prompt:, :]                         # [B, N, d_llm]
        out    = self.output_proj(ts_out.mean(dim=1))            # [B, pred_len]
        out    = self.revin.denormalise(
            out.unsqueeze(-1), mean[:, :1, :], std[:, :1, :]
        )                                                         # [B, pred_len, 1]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# GPT4MTS
# ─────────────────────────────────────────────────────────────────────────────

class GPT4MTS(nn.Module):
    """
    GPT4MTS (Flora-jia-jfr/GPT4MTS).

    Data flow  (enc_in=1 for mmtt-bench):
      x [B, seq_len, 1]
        → RevIN normalise
        → patches → Linear(patch_len, d_llm) → [B, N, d_llm]
        → cat([prompt_embs, ts_emb])
        → frozen GPT2
        → take last N hidden states → mean-pool
        → Linear(d_llm, pred_len) → RevIN denormalise → [B, pred_len, 1]

    Prompt = domain description + per-sample TS statistics + sample text annotation.
    Only patch_proj and output_proj are trained.
    """
    def __init__(
        self,
        seq_len:   int,
        pred_len:  int,
        patch_len: int,
        stride:    int,
        d_llm:     int,
        dropout:   float,
        llm_model: nn.Module,
        tokenizer,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.d_llm     = d_llm
        self.llm       = llm_model
        self.tokenizer = tokenizer
        self.N         = _n_patches(seq_len, patch_len, stride)

        self.patch_proj  = nn.Linear(patch_len, d_llm)
        self.drop        = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_llm, pred_len)
        self.revin       = RevIN()

        for p in self.llm.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _ts_stats(x: torch.Tensor) -> list[str]:
        """Per-sample TS statistics as natural-language string (after RevIN)."""
        strs = []
        for i in range(x.size(0)):
            xi    = x[i, :, 0].cpu().float().numpy()
            trend = 'upward' if float(xi[-1]) > float(xi[0]) else 'downward'
            strs.append(
                f"Input statistics: min={xi.min():.3f}, max={xi.max():.3f}, "
                f"mean={xi.mean():.3f}, std={xi.std():.3f}, trend={trend}."
            )
        return strs

    def forward(
        self,
        x:            torch.Tensor,    # [B, seq_len, 1]
        domain_texts: list[str],
        sample_texts: list[str],
    ) -> torch.Tensor:                 # [B, pred_len, 1]
        B      = x.size(0)
        device = x.device

        x_norm, mean, std = self.revin.normalise(x)

        stat_strs   = self._ts_stats(x_norm)
        prompt_strs = [
            '\n'.join(filter(None, [d, s, f"Context: {t.strip()}" if t.strip() else '']))
            for d, s, t in zip(domain_texts, stat_strs, sample_texts)
        ]

        patches = _make_patches(x_norm, self.patch_len, self.stride)
        ts_emb  = self.drop(self.patch_proj(patches))        # [B, N, d_llm]

        with torch.no_grad():
            enc        = self.tokenizer(prompt_strs, return_tensors='pt',
                                        padding=True, truncation=True, max_length=512)
            ids        = enc.input_ids.to(device)
            prompt_mask = enc.attention_mask.to(device)
            prompt_emb  = self.llm.get_input_embeddings()(ids).float()

        n_prompt = prompt_emb.size(1)
        combined = torch.cat([prompt_emb, ts_emb], dim=1)
        attn_mask = torch.cat([
            prompt_mask,
            torch.ones(B, self.N, dtype=torch.long, device=device),
        ], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        hidden = self.llm(
            inputs_embeds  = combined.to(dtype=llm_dtype),
            attention_mask = attn_mask,
        ).last_hidden_state.float()

        ts_out = hidden[:, n_prompt:, :]                      # [B, N, d_llm]
        out    = self.output_proj(ts_out.mean(dim=1))         # [B, pred_len]
        out    = self.revin.denormalise(
            out.unsqueeze(-1), mean[:, :1, :], std[:, :1, :]
        )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot model registry
# ─────────────────────────────────────────────────────────────────────────────

_ZERO_SHOT_MODELS: frozenset[str] = frozenset({'LLMTime', 'PromptCast', 'DirectPrompt'})


# ─────────────────────────────────────────────────────────────────────────────
# Digit serialisation  (adapted from ngruver/llmtime, data/serialize.py)
#
# Format: base-10 digits, space between each digit (bit_sep=' '),
#         space-comma between time steps (time_sep=' ,'), trailing separator.
# Negative values prefixed with ' -'; positive values have no prefix.
# Leading zeros stripped (keep ≥1 digit).
#
# Example (prec=3, max_val=10 → max_bit_pos=1 → 4 digits per value):
#   3.52  →  " 3 5 2 0"
#  -0.5   →  " - 5 0 0"
#   [3.52, -0.5] → " 3 5 2 0 , - 5 0 0 ,"
# ─────────────────────────────────────────────────────────────────────────────

def _max_bit_pos(max_val: float) -> int:
    return max(1, math.ceil(math.log10(max(max_val, 1.0001))))


def _serialize_arr(arr: np.ndarray, prec: int = 3, max_val: float = 10.0) -> str:
    mbp = _max_bit_pos(max_val)
    parts: list[str] = []
    for v in arr:
        sign = 1 if v >= 0 else -1
        v    = abs(float(v))
        digits: list[int] = []
        for i in range(mbp):
            d = int(v / 10 ** (mbp - i - 1))
            digits.append(d % 10)
            v -= d * 10 ** (mbp - i - 1)
        for i in range(prec):
            d = int(v / 10 ** (-i - 1))
            digits.append(d % 10)
            v -= d * 10 ** (-i - 1)
        # strip leading zeros, keep ≥1 digit
        first_nz = next((k for k, d in enumerate(digits) if d != 0), len(digits) - 1)
        digits   = digits[first_nz:]
        digit_str = ''.join(' ' + str(d) for d in digits)
        parts.append((' -' + digit_str) if sign < 0 else digit_str)
    return ' ,'.join(parts) + ' ,'


def _deserialize_str(
    text: str, prec: int = 3, max_val: float = 10.0, steps: int = None
) -> list[float] | None:
    """Return a list of floats (length ≤ steps), or None if the first step fails."""
    mbp   = _max_bit_pos(max_val)
    total = mbp + prec
    tokens = [t for t in text.split(' ,') if t.strip()]
    if steps is not None:
        tokens = tokens[:steps]
    values: list[float] = []
    for tok in tokens:
        tok  = tok.strip()
        sign = -1 if tok.startswith('-') else 1
        tok  = tok.lstrip('-').strip()
        raw  = [c for c in tok.split(' ') if c.isdigit()]
        if not raw:
            if not values:
                return None
            break
        digits = [int(c) for c in raw]
        # left-pad / right-truncate to total digits
        if len(digits) < total:
            digits = [0] * (total - len(digits)) + digits
        else:
            digits = digits[-total:]
        val  = sum(d * 10 ** (mbp - 1 - i) for i, d in enumerate(digits))
        val += 0.5 * 10 ** (-prec)   # half-bin correction (matches llmtime)
        values.append(sign * val)
    return values if values else None


# ─────────────────────────────────────────────────────────────────────────────
# Quantile scaler  (ngruver/llmtime, models/utils.py  get_scaler)
# ─────────────────────────────────────────────────────────────────────────────

def _get_scaler(history: np.ndarray, alpha: float = 0.95, beta: float = 0.3):
    """
    Shift–scale so alpha-quantile of (shifted) history maps to 1.
    Returns (transform, inv_transform) callables.
    """
    h    = history[~np.isnan(history)].astype(float)
    lo   = h.min() - beta * (h.max() - h.min())
    q    = float(np.quantile(h - lo, alpha))
    if q == 0.0:
        q = 1.0
    transform     = lambda x: (x - lo) / q      # noqa: E731
    inv_transform = lambda x: x * q  + lo       # noqa: E731
    return transform, inv_transform


# ─────────────────────────────────────────────────────────────────────────────
# LLMTime  (Gruver et al., NeurIPS 2023)
# ─────────────────────────────────────────────────────────────────────────────

class LLMTime:
    """
    Zero-shot forecaster: history → quantile-scaled digit string → LLM
    continuation → median of n_samples → inverse-scaled forecast.

    Ignores domain_texts / sample_texts; pure digit autoregression.
    Needs a causal LM (AutoModelForCausalLM) and matching tokenizer.
    """

    def __init__(
        self,
        llm_model,
        tokenizer,
        pred_len:  int,
        n_samples: int   = 10,
        prec:      int   = 3,
        temp:      float = 0.7,
        alpha:     float = 0.95,
        beta:      float = 0.3,
        max_val:   float = 10.0,
        record_transcripts: bool = False,
        gen_batch_size: int | None = None,
    ):
        self.llm       = llm_model
        self.tokenizer = tokenizer
        self.pred_len  = pred_len
        self.n_samples = n_samples
        self.prec      = prec
        self.temp      = temp
        self.alpha     = alpha
        self.beta      = beta
        self.max_val   = max_val
        self.record_transcripts = record_transcripts
        self.transcripts: list[dict] = []
        # Max series generated in parallel per generate() call (each expands to
        # gen_batch_size * n_samples sequences via num_return_sequences). None
        # = try the whole batch at once, backing off automatically on OOM.
        self.gen_batch_size = gen_batch_size

    def predict(
        self,
        x:            torch.Tensor,   # [B, seq_len, 1]
        domain_texts: list[str],      # unused — pure digit continuation
        sample_texts: list[str],      # unused
    ) -> torch.Tensor:                # [B, pred_len, 1]
        B, device = x.size(0), x.device
        arrs: list[np.ndarray] = []
        inv_transforms = []
        input_strs: list[str] = []
        for i in range(B):
            arr              = x[i, :, 0].cpu().float().numpy()
            transform, inv_t = _get_scaler(arr, self.alpha, self.beta)
            arrs.append(arr)
            inv_transforms.append(inv_t)
            input_strs.append(_serialize_arr(transform(arr), self.prec, self.max_val))

        # One generate() call for the whole batch: B series × n_samples
        # completions each, instead of B*n_samples sequential single-item calls.
        raw_preds, completions = self._sample_batch(input_strs, device)

        preds: list[list[float]] = []
        for i in range(B):
            inv_t  = inv_transforms[i]
            scaled = [
                ((s := [inv_t(v) for v in vals]) + [s[-1]] * self.pred_len)[:self.pred_len]
                for vals in raw_preds[i]
            ]
            vals_i = (np.median(np.array(scaled), axis=0).tolist()
                      if scaled else [0.0] * self.pred_len)
            preds.append(vals_i)
            if self.record_transcripts:
                self.transcripts.append({
                    'history':     arrs[i].tolist(),
                    'prompt':      input_strs[i],
                    'completions': completions[i],
                    'forecast':    vals_i,
                })
        out = torch.tensor(preds, dtype=torch.float32, device=device)
        return out.unsqueeze(-1)   # [B, pred_len, 1]

    def _sample_batch(
        self, input_strs: list[str], device
    ) -> tuple[list[list[list[float]]], list[list[str]]]:
        """
        Generate n_samples completions for every series in the batch, chunked
        so at most `chunk * n_samples` sequences are ever in flight at once.
        Starts at gen_batch_size (or the whole batch if unset) and halves on
        CUDA OOM, down to a single series per generate() call, so this can't
        OOM worse than the pre-batching implementation.

        Returns (raw_preds, completions), both length B:
          raw_preds[i]    — list of parsed digit-space value lists (one per
                             accepted completion for series i; unscaled).
          completions[i]  — list of n_samples raw decoded strings for series i.
        """
        B = len(input_strs)
        raw_preds:   list = [None] * B
        completions: list = [None] * B
        chunk = min(self.gen_batch_size or B, B)

        start = 0
        while start < B:
            end = min(start + chunk, B)
            try:
                r_chunk, c_chunk = self._generate_chunk(input_strs[start:end], device)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                if chunk == 1:
                    raise RuntimeError(
                        'LLMTime: CUDA OOM even at gen_batch_size=1 (single series, '
                        f'n_samples={self.n_samples}). Try lowering n_samples or using '
                        'a smaller generative_llm model.'
                    ) from exc
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f'    [LLMTime] CUDA OOM — retrying with gen_batch_size={chunk}')
                continue
            for j, idx in enumerate(range(start, end)):
                raw_preds[idx], completions[idx] = r_chunk[j], c_chunk[j]
            start = end

        return raw_preds, completions

    def _generate_chunk(
        self, input_strs: list[str], device
    ) -> tuple[list[list[list[float]]], list[list[str]]]:
        """One generate() call for a chunk of series (batch dim = len(input_strs),
        num_return_sequences = n_samples), instead of per-series sequential calls."""
        B = len(input_strs)
        # Each serialised step uses at most (prec+2) digit tokens × 2 GPT2 tokens each.
        # Reserve that budget before encoding so input + output never exceeds model max.
        model_max = getattr(self.llm.config, 'max_position_embeddings', 1024)
        max_new   = int((self.prec + 3) * 2 * self.pred_len * 1.5) + 10
        max_input = max(model_max - max_new, 64)

        # Left padding: generate() then appends new tokens at the same column
        # for every row regardless of each prompt's real (unpadded) length.
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        try:
            enc = self.tokenizer(
                input_strs, return_tensors='pt', truncation=True,
                max_length=max_input, padding=True,
            ).to(device)
        finally:
            self.tokenizer.padding_side = orig_padding_side

        input_len = enc['input_ids'].shape[1]
        # Recalculate max_new tightly now that we know the actual (padded) input length
        max_new = max(model_max - input_len, max_new)

        with torch.no_grad():
            out_ids = self.llm.generate(
                **enc,
                max_new_tokens       = max_new,
                do_sample            = True,
                temperature          = self.temp,
                num_return_sequences = self.n_samples,
                pad_token_id         = self.tokenizer.eos_token_id,
            )
        # generate() expands the batch via repeat_interleave, so each series'
        # n_samples completions are contiguous: rows [i*n_samples:(i+1)*n_samples].

        raw_preds: list[list[list[float]]] = []
        completions: list[list[str]] = []
        for i in range(B):
            block = out_ids[i * self.n_samples : (i + 1) * self.n_samples]
            texts = [
                self.tokenizer.decode(row[input_len:], skip_special_tokens=True)
                for row in block
            ]
            completions.append(texts)
            parsed = [
                vals for text in texts
                if (vals := _deserialize_str(text, self.prec, self.max_val, self.pred_len))
                is not None
            ]
            raw_preds.append(parsed)
        return raw_preds, completions


# ─────────────────────────────────────────────────────────────────────────────
# PromptCast  (Xue & Salim 2023 — adapted from ngruver/llmtime/models/promptcast.py)
# ─────────────────────────────────────────────────────────────────────────────

class PromptCast:
    """
    Zero-shot forecaster: natural-language preamble + digit history →
    LLM causal completion → digit forecast.

    Same digit serialisation as LLMTime; identity scaling (no normalisation).
    domain_texts / sample_texts are prepended as a free-text preamble so that
    all four text conditions (domain_only / correct / incorrect / irrelevant)
    are supported through the standard pipeline.

    Needs a causal LM (AutoModelForCausalLM) and matching tokenizer.
    """

    def __init__(
        self,
        llm_model,
        tokenizer,
        pred_len:  int,
        n_samples: int   = 10,
        prec:      int   = 3,
        temp:      float = 0.8,
        max_val:   float = 10.0,
        record_transcripts: bool = False,
    ):
        self.llm       = llm_model
        self.tokenizer = tokenizer
        self.pred_len  = pred_len
        self.n_samples = n_samples
        self.prec      = prec
        self.temp      = temp
        self.max_val   = max_val
        self.record_transcripts = record_transcripts
        self.transcripts: list[dict] = []

    def predict(
        self,
        x:            torch.Tensor,   # [B, seq_len, 1]
        domain_texts: list[str],
        sample_texts: list[str],
    ) -> torch.Tensor:                # [B, pred_len, 1]
        B, device = x.size(0), x.device
        preds: list[list[float]] = []
        for i in range(B):
            arr     = x[i, :, 0].cpu().float().numpy()
            seq_len = len(arr)
            # Identity scaling: no normalisation (matches original promptcast.py)
            body    = _serialize_arr(arr, self.prec, self.max_val).rstrip()
            if body.endswith(','):
                body = body[:-1].rstrip()

            # Free-text preamble (domain context + optional annotation)
            ctx_parts: list[str] = []
            if domain_texts[i].strip():
                ctx_parts.append(domain_texts[i].strip())
            if sample_texts[i].strip():
                ctx_parts.append(f'Note: {sample_texts[i].strip()}')
            preamble = '\n\n'.join(ctx_parts)
            header   = preamble + '\n\n' if preamble else ''

            # PromptCast completion template (from promptcast.py)
            prompt = (
                f'{header}'
                f'The values for the past {seq_len} time steps are {body}.'
                f' What will the values for the next {self.pred_len} time steps be?'
                f' The values for the next {self.pred_len} time steps will be'
            )
            vals, completions = self._sample_median(prompt, device)
            preds.append(vals)
            if self.record_transcripts:
                self.transcripts.append({
                    'history':     arr.tolist(),
                    'prompt':      prompt,
                    'completions': completions,
                    'forecast':    vals,
                })
        out = torch.tensor(preds, dtype=torch.float32, device=device)
        return out.unsqueeze(-1)

    def _sample_median(self, prompt: str, device) -> tuple[list[float], list[str]]:
        model_max = getattr(self.llm.config, 'max_position_embeddings', 1024)
        max_new   = int((self.prec + 3) * 2 * self.pred_len * 1.5) + 10
        max_input = max(model_max - max_new, 64)

        enc = self.tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=max_input,
        ).to(device)
        max_new = max(model_max - enc['input_ids'].shape[1], max_new)

        all_preds: list[list[float]] = []
        completions: list[str] = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                out_ids = self.llm.generate(
                    **enc,
                    max_new_tokens = max_new,
                    do_sample      = True,
                    temperature    = self.temp,
                    pad_token_id   = self.tokenizer.eos_token_id,
                )
            new_text = self.tokenizer.decode(
                out_ids[0, enc['input_ids'].shape[1]:], skip_special_tokens=True
            )
            completions.append(new_text)
            # Prepend a time_sep so _deserialize_str treats the first token as
            # the start of a new time step (matching the LLMTime digit format).
            vals = _deserialize_str(' ,' + new_text, self.prec, self.max_val, self.pred_len)
            if vals is not None:
                all_preds.append((vals + [vals[-1]] * self.pred_len)[:self.pred_len])

        if not all_preds:
            return [0.0] * self.pred_len, completions
        return np.median(np.array(all_preds), axis=0).tolist(), completions


# ─────────────────────────────────────────────────────────────────────────────
# DirectPrompt  (CiK-style direct prompting — protocol of arXiv:2506.21611)
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


class DirectPrompt:
    """
    Zero-shot forecaster following the prompting-based protocol of
    "When Does Multimodality Lead to Better Time Series Forecasting?"
    (arXiv:2506.21611), whose prompts follow CiK (Williams et al. 2025):
    free-text context + plain-decimal history, forecast requested between
    <forecast></forecast> tags.

    No digit serialisation and no rescaling — values are used as-is (the
    dataset already standardises them) and parsed back with a float regex.
    A completion is rejected unless it yields >= pred_len finite values;
    forecast = element-wise median over the accepted completions.  If every
    completion is rejected, falls back to persistence (repeat last value)
    and counts it in `stats`.

    Uses the tokenizer's chat template when available (instruct models);
    base LMs get a plain-text prompt primed with the opening <forecast> tag.
    """

    def __init__(
        self,
        llm_model,
        tokenizer,
        pred_len:  int,
        n_samples: int   = 10,
        temp:      float = 0.7,
        decimals:  int   = 3,
        record_transcripts: bool = False,
    ):
        self.llm       = llm_model
        self.tokenizer = tokenizer
        self.pred_len  = pred_len
        self.n_samples = n_samples
        self.temp      = temp
        self.decimals  = decimals
        self.stats     = {'completions': 0, 'rejected': 0, 'fallback': 0}
        self.record_transcripts = record_transcripts
        self.transcripts: list[dict] = []

    def predict(
        self,
        x:            torch.Tensor,   # [B, seq_len, 1]
        domain_texts: list[str],
        sample_texts: list[str],
    ) -> torch.Tensor:                # [B, pred_len, 1]
        B, device = x.size(0), x.device
        preds: list[list[float]] = []
        for i in range(B):
            arr    = x[i, :, 0].cpu().float().numpy()
            prompt = self._build_prompt(arr, domain_texts[i], sample_texts[i])
            vals, completions = self._sample_median(prompt, fallback=float(arr[-1]))
            preds.append(vals)
            if self.record_transcripts:
                self.transcripts.append({
                    'history':     arr.tolist(),
                    'prompt':      prompt,
                    'completions': completions,
                    'forecast':    vals,
                })
        out = torch.tensor(preds, dtype=torch.float32, device=device)
        return out.unsqueeze(-1)

    def _build_prompt(self, arr: np.ndarray, domain: str, note: str) -> str:
        ctx_parts = [p for p in (domain.strip(),) if p]
        if note.strip():
            ctx_parts.append(f'Note: {note.strip()}')
        context = ('\n\n'.join(ctx_parts) + '\n\n') if ctx_parts else ''
        hist    = ', '.join(f'{v:.{self.decimals}f}' for v in arr)
        return (
            f'{context}'
            f'Here is a historical time series covering the last {len(arr)} '
            f'time steps:\n'
            f'<history>\n{hist}\n</history>\n\n'
            f'Predict the next {self.pred_len} values of this time series, '
            f'taking the context above into account.\n'
            f'Respond with exactly {self.pred_len} comma-separated numeric '
            f'values between <forecast> and </forecast> tags, .'
            f'and your reasoning for the prediction between <reasoning> '
            f'and </reasoning> tags, and nothing else.'
        )

    def _encode(self, prompt: str, device) -> torch.Tensor:
        if getattr(self.tokenizer, 'chat_template', None):
            ids = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                add_generation_prompt=True, return_tensors='pt',
            )
            if not torch.is_tensor(ids):   # newer transformers: BatchEncoding
                ids = ids['input_ids']
        else:
            # Base LM: prime the completion with the opening tag
            ids = self.tokenizer(
                prompt + '\n<forecast>\n', return_tensors='pt'
            ).input_ids
        # Keep the END of the prompt (recent history + instructions) if the
        # model's context is too small for it.
        model_max = getattr(self.llm.config, 'max_position_embeddings', 4096)
        max_input = model_max - self._max_new()
        if ids.shape[1] > max_input:
            ids = ids[:, -max_input:]
        return ids.to(device)

    def _max_new(self) -> int:
        # ~6 tokens per formatted value plus tags/whitespace slack
        return self.pred_len * (self.decimals + 7) + 320

    def _parse_forecast(self, text: str) -> list[float] | None:
        m = re.search(r'<forecast>(.*?)</forecast>', text, re.S)
        if m:
            region = m.group(1)
        else:
            # Opening tag may sit in the prompt (base-LM priming) or the
            # closing tag may be missing.
            region = text.split('</forecast>')[0]
            if '<forecast>' in region:
                region = region.split('<forecast>', 1)[1]
        vals = [float(v) for v in _FLOAT_RE.findall(region)]
        vals = vals[:self.pred_len]
        if len(vals) < self.pred_len:
            return None
        if any(not math.isfinite(v) or abs(v) > 1e6 for v in vals):
            return None
        return vals

    def _sample_median(self, prompt: str, fallback: float) -> tuple[list[float], list[str]]:
        device = next(self.llm.parameters()).device
        ids    = self._encode(prompt, device)
        with torch.no_grad():
            out_ids = self.llm.generate(
                ids,
                attention_mask       = torch.ones_like(ids),
                max_new_tokens       = self._max_new(),
                do_sample            = True,
                temperature          = self.temp,
                num_return_sequences = self.n_samples,
                pad_token_id         = self.tokenizer.eos_token_id,
            )
        accepted: list[list[float]] = []
        completions: list[str] = []
        for k in range(out_ids.shape[0]):
            new_text = self.tokenizer.decode(
                out_ids[k, ids.shape[1]:], skip_special_tokens=True
            )
            completions.append(new_text)
            vals = self._parse_forecast(new_text)
            self.stats['completions'] += 1
            if vals is None:
                self.stats['rejected'] += 1
            else:
                accepted.append(vals)
        if not accepted:
            self.stats['fallback'] += 1
            return [fallback] * self.pred_len, completions
        return np.median(np.array(accepted), axis=0).tolist(), completions


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (mirrors MmttDataset in train_transformer_cfa.py)
# ─────────────────────────────────────────────────────────────────────────────

class MmttDataset(Dataset):
    """
    Annotation-point dataset for mmtt-bench JSON splits.
    Each sample is one annotation point: (lookback window, future horizon, text).
    Interface mirrors MM-TSFlib Dataset_Custom.
    """
    def __init__(
        self,
        json_path:  str,
        seq_len:    int,
        label_len:  int,
        pred_len:   int,
        categories: list[str] | None = None,
        scale:      bool = True,
        scaler      = None,
    ):
        with open(json_path) as f:
            raw = json.load(f)

        dense_y = np.array(raw['metadata']['y'], dtype=np.float64)
        dense_t = [round(t, 4) for t in raw['metadata']['t']]
        n_dense = len(dense_y)

        lookbacks, futures, texts = [], [], []

        for pt in raw['points']:
            if categories and pt.get('category') not in categories:
                continue
            t_i = round(pt['t'], 4)
            if t_i not in dense_t:
                continue
            idx = dense_t.index(t_i)

            # Lookback ends at the annotation point inclusive (matches data.py)
            start = max(0, idx + 1 - seq_len)
            lb    = dense_y[start:idx + 1]
            if len(lb) < seq_len:
                lb = np.concatenate([np.full(seq_len - len(lb), lb[0] if len(lb) else 0.0), lb])
            lb = lb[-seq_len:]

            future_end = idx + 1 + pred_len
            if future_end > n_dense:
                continue
            fut = dense_y[idx + 1 : future_end]

            text = ''
            if pt.get('samples'):
                text = pt['samples'][0].get('text', '')

            lookbacks.append(lb)
            futures.append(fut)
            texts.append(text)

        if not lookbacks:
            raise ValueError(f'No samples in {json_path} (categories={categories})')

        lb_arr  = np.array(lookbacks, dtype=np.float32)
        fut_arr = np.array(futures,   dtype=np.float32)

        if scale:
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(lb_arr.reshape(-1, 1))
            lb_arr  = scaler.transform(lb_arr.reshape(-1, 1)).reshape(lb_arr.shape)
            fut_arr = scaler.transform(fut_arr.reshape(-1, 1)).reshape(fut_arr.shape)
        self.scaler = scaler

        self.data_x   = lb_arr[:, :, np.newaxis]                       # [N, seq_len, 1]
        label_part    = lb_arr[:, -label_len:, np.newaxis]
        future_part   = fut_arr[:, :,           np.newaxis]
        self.data_y   = np.concatenate([label_part, future_part], axis=1)

        self.texts    = texts
        self.seq_len  = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        N = len(texts)
        self.x_mark = np.zeros((N, seq_len,             4), dtype=np.float32)
        self.y_mark = np.zeros((N, label_len + pred_len, 4), dtype=np.float32)

    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return np.array([self.texts[int(i)] for i in indices], dtype=object)

    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return np.zeros((len(indices), self.pred_len, 1), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (
            self.data_x[index],
            self.data_y[index],
            self.x_mark[index],
            self.y_mark[index],
            index,
        )


def _texts(raw) -> list[str]:
    arr = np.asarray(raw)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return [str(x) for x in arr]


def _get_loader(
    json_path:  str,
    seq_len:    int,
    label_len:  int,
    pred_len:   int,
    batch_size: int,
    categories: list[str] | None,
    scale:      bool,
    scaler      = None,
    shuffle:    bool = False,
) -> tuple[MmttDataset, DataLoader]:
    ds = MmttDataset(
        json_path  = json_path,
        seq_len    = seq_len,
        label_len  = label_len,
        pred_len   = pred_len,
        categories = categories,
        scale      = scale,
        scaler     = scaler,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return ds, loader


# ─────────────────────────────────────────────────────────────────────────────
# LLM loading
# ─────────────────────────────────────────────────────────────────────────────

def load_llm(llm_model: str, device: torch.device):
    """Load and freeze a GPT2 or LLAMA3 backbone. Returns (tokenizer, model, d_llm)."""
    from transformers import (
        GPT2Config, GPT2Model, GPT2Tokenizer,
        LlamaConfig, LlamaModel, AutoTokenizer,
    )

    if llm_model == 'GPT2':
        cfg = GPT2Config.from_pretrained('openai-community/gpt2')
        try:
            model = GPT2Model.from_pretrained('openai-community/gpt2', config=cfg,
                                              local_files_only=True)
            tok   = GPT2Tokenizer.from_pretrained('openai-community/gpt2',
                                                   local_files_only=True)
        except Exception:
            model = GPT2Model.from_pretrained('openai-community/gpt2', config=cfg)
            tok   = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        d_llm = 768

    elif llm_model == 'LLAMA3':
        path  = 'meta-llama/Meta-Llama-3-8B-Instruct'
        cfg   = LlamaConfig.from_pretrained(path)
        model = LlamaModel.from_pretrained(path, config=cfg)
        tok   = AutoTokenizer.from_pretrained(path)
        d_llm = 4096

    else:
        raise ValueError(f'Unsupported llm_model: {llm_model}')

    if tok.pad_token is None:
        if tok.eos_token:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tok))

    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device).eval()

    return tok, model, d_llm


# ─────────────────────────────────────────────────────────────────────────────
# Generative LLM loading (for LLMTime / PromptCast)
# ─────────────────────────────────────────────────────────────────────────────

def load_generative_llm(model_name: str, device: torch.device):
    """
    Load AutoModelForCausalLM for text generation (LLMTime / PromptCast).
    Returns (tokenizer, model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

    if model_name == 'GPT2':
        try:
            tok   = GPT2Tokenizer.from_pretrained('openai-community/gpt2', local_files_only=True)
            model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', local_files_only=True)
        except Exception:
            tok   = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
            model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
        model = model.to(device)
    else:
        tok   = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map='auto' if device.type == 'cuda' else None,
        )
        if device.type != 'cuda':
            model = model.to(device)

    if tok.pad_token is None:
        if tok.eos_token:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tok))

    model.eval()
    return tok, model


def _build_zero_shot_model(
    model_name: str, tokenizer, llm_model, pred_len: int, cfg: dict
):
    """Instantiate LLMTime or PromptCast from config. Built once per pred_len."""
    record_transcripts = cfg.get('save_transcripts', False)
    if model_name == 'LLMTime':
        kw = cfg.get('llmtime', {})
        return LLMTime(
            llm_model  = llm_model,
            tokenizer  = tokenizer,
            pred_len   = pred_len,
            n_samples  = kw.get('n_samples', 10),
            prec       = kw.get('prec',      3),
            temp       = kw.get('temp',      0.7),
            alpha      = kw.get('alpha',     0.95),
            beta       = kw.get('beta',      0.3),
            max_val    = kw.get('max_val',   10.0),
            record_transcripts = record_transcripts,
            gen_batch_size = kw.get('gen_batch_size', None),
        )
    if model_name == 'PromptCast':
        kw = cfg.get('promptcast', {})
        return PromptCast(
            llm_model  = llm_model,
            tokenizer  = tokenizer,
            pred_len   = pred_len,
            n_samples  = kw.get('n_samples', 10),
            prec       = kw.get('prec',      3),
            temp       = kw.get('temp',      0.8),
            max_val    = kw.get('max_val',   10.0),
            record_transcripts = record_transcripts,
        )
    if model_name == 'DirectPrompt':
        kw = cfg.get('directprompt', {})
        return DirectPrompt(
            llm_model  = llm_model,
            tokenizer  = tokenizer,
            pred_len   = pred_len,
            n_samples  = kw.get('n_samples', 10),
            temp       = kw.get('temp',      0.7),
            decimals   = kw.get('decimals',  3),
            record_transcripts = record_transcripts,
        )
    raise ValueError(f'Unknown zero-shot model: {model_name}')


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    seq_len:    int,
    pred_len:   int,
    patch_len:  int,
    stride:     int,
    d_model:    int,
    n_heads:    int,
    n_tokens:   int,
    d_llm:      int,
    dropout:    float,
    llm_model:  nn.Module,
    tokenizer,
    device:     torch.device,
) -> nn.Module:
    if model_name == 'TimeLLM':
        return TimeLLM(
            seq_len=seq_len, pred_len=pred_len, d_model=d_model,
            patch_len=patch_len, stride=stride, n_heads=n_heads,
            n_tokens=n_tokens, d_llm=d_llm, dropout=dropout,
            llm_model=llm_model, tokenizer=tokenizer,
        ).float().to(device)
    elif model_name == 'GPT4MTS':
        return GPT4MTS(
            seq_len=seq_len, pred_len=pred_len, patch_len=patch_len,
            stride=stride, d_llm=d_llm, dropout=dropout,
            llm_model=llm_model, tokenizer=tokenizer,
        ).float().to(device)
    else:
        raise ValueError(f'Unknown model: {model_name}. Choose TimeLLM or GPT4MTS.')


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass (unified interface for both models)
# ─────────────────────────────────────────────────────────────────────────────

def _forward(
    model,
    model_name:   str,
    batch_x:      torch.Tensor,
    domain_texts: list[str],
    sample_texts: list[str],
) -> torch.Tensor:
    if model_name in _ZERO_SHOT_MODELS:
        return model.predict(batch_x, domain_texts, sample_texts)
    if model_name == 'TimeLLM':
        prompts = [
            f"{d}\n\n{s}".rstrip() if s.strip() else d
            for d, s in zip(domain_texts, sample_texts)
        ]
        return model(batch_x, prompts)
    # GPT4MTS
    return model(batch_x, domain_texts, sample_texts)


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_setting(
    model_name:   str,
    pred_len:     int,
    patch_len:    int,
    stride:       int,
    d_model:      int,
    n_heads:      int,
    n_tokens:     int,
    d_llm:        int,
    dropout:      float,
    llm_model:    nn.Module,
    tokenizer,
    train_ds:     MmttDataset,
    train_loader: DataLoader,
    val_ds:       MmttDataset,
    val_loader:   DataLoader,
    domain_prompt: str,
    use_sample_text: bool,
    lr:           float,
    epochs:       int,
    patience:     int,
    device:       torch.device,
    prebuilt_model = None,
) -> dict:
    # Zero-shot models skip training entirely
    if prebuilt_model is not None:
        return _compute_metrics(
            prebuilt_model, model_name, val_ds, val_loader,
            domain_prompt, use_sample_text, pred_len, device,
        )

    seq_len = train_ds.seq_len
    model   = build_model(
        model_name=model_name, seq_len=seq_len, pred_len=pred_len,
        patch_len=patch_len, stride=stride, d_model=d_model,
        n_heads=n_heads, n_tokens=n_tokens, d_llm=d_llm,
        dropout=dropout, llm_model=llm_model, tokenizer=tokenizer,
        device=device,
    )

    # Only train non-LLM parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f'    trainable params: {n_trainable:,}')

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, _, _, index in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            raw_texts    = _texts(train_ds.get_text(index))
            sample_texts = raw_texts if use_sample_text else [''] * len(raw_texts)
            domain_texts = [domain_prompt] * len(raw_texts)

            outputs = _forward(model, model_name, batch_x, domain_texts, sample_texts)
            outputs = outputs[:, -pred_len:, :]
            targets = batch_y[:, -pred_len:, :]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

        val_loss = _eval_loss(
            model, model_name, val_ds, val_loader, criterion,
            domain_prompt, use_sample_text, pred_len, device,
        )
        print(f'    epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}')

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'    early stop at epoch {epoch}')
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return _compute_metrics(
        model, model_name, val_ds, val_loader,
        domain_prompt, use_sample_text, pred_len, device,
    )


def _set_eval(model) -> None:
    if hasattr(model, 'eval') and callable(model.eval):
        model.eval()

def _set_train(model) -> None:
    if hasattr(model, 'train') and callable(model.train):
        model.train()


@torch.no_grad()
def _eval_loss(
    model, model_name, ds, loader, criterion,
    domain_prompt, use_sample_text, pred_len, device,
) -> float:
    _set_eval(model)
    total, n = 0.0, 0
    for batch_x, batch_y, _, _, index in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        raw_texts    = _texts(ds.get_text(index))
        sample_texts = raw_texts if use_sample_text else [''] * len(raw_texts)
        domain_texts = [domain_prompt] * len(raw_texts)

        out  = _forward(model, model_name, batch_x, domain_texts, sample_texts)
        out  = out[:, -pred_len:, :]
        true = batch_y[:, -pred_len:, :].to(device)
        total += criterion(out, true).item() * len(batch_x)
        n     += len(batch_x)
    _set_train(model)
    return total / n


@torch.no_grad()
def _compute_metrics(
    model, model_name, ds, loader,
    domain_prompt, use_sample_text, pred_len, device,
) -> dict:
    _set_eval(model)
    preds, trues = [], []
    for batch_x, batch_y, _, _, index in loader:
        batch_x = batch_x.float().to(device)

        raw_texts    = _texts(ds.get_text(index))
        sample_texts = raw_texts if use_sample_text else [''] * len(raw_texts)
        domain_texts = [domain_prompt] * len(raw_texts)

        out = _forward(model, model_name, batch_x, domain_texts, sample_texts)
        preds.append(out[:, -pred_len:, :].cpu().numpy())
        trues.append(batch_y[:, -pred_len:, :].numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    mse   = float(mean_squared_error(trues, preds))
    mae   = float(mean_absolute_error(trues, preds))
    _set_train(model)
    metrics = {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mae}
    stats = getattr(model, 'stats', None)
    if stats and stats.get('completions'):
        print(f"    parse stats: {stats['completions']} completions, "
              f"{stats['rejected']} rejected, "
              f"{stats['fallback']} persistence fallbacks")
        metrics['parse_rejected_frac'] = stats['rejected'] / stats['completions']
        metrics['fallback_forecasts']  = stats['fallback']
        model.stats = {k: 0 for k in stats}
    return metrics


def _dump_transcripts(
    model, ds, ds_name: str, pred_len: int, model_name: str, cond: str,
    root: Path, enabled: bool,
) -> None:
    """
    Write out per-sample (history, prompt, raw completions, forecast,
    ground truth) records collected by a zero-shot model's last
    _compute_metrics call, then clear them for the next text condition.

    Relies on val loaders being shuffle=False, so `model.transcripts`
    (appended in dataset order by `predict`) lines up 1:1 with `ds.data_y`.
    """
    if not enabled or model is None or not getattr(model, 'record_transcripts', False):
        return
    transcripts = model.transcripts
    model.transcripts = []
    if not transcripts:
        return
    future = ds.data_y[:, -pred_len:, 0]
    for rec, gt in zip(transcripts, future):
        rec['ground_truth'] = gt.tolist()
    out_path = root / ds_name / f'pred{pred_len}_{model_name}_{cond}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(transcripts, f, indent=2)
    print(f'    transcripts → {out_path}  ({len(transcripts)} samples)')


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='TimeLLM / GPT4MTS on mmtt-bench / TimeMMD'
    )
    p.add_argument('--config',   required=True,  help='Path to timemmd_llm.yaml')
    p.add_argument('--dataset',  default=None,   help='Dataset name; omit to run all')
    p.add_argument('--model',    default=None,   help='TimeLLM or GPT4MTS; omit for all')
    p.add_argument('--pred-len', type=int, default=None, dest='pred_len',
                   help='Single prediction horizon; omit to run all for the freq')
    p.add_argument('--text-cond', default=None, dest='text_cond',
                   help='domain_only | correct | incorrect | irrelevant; omit for all')
    p.add_argument('--output',  '-o', default=None,
                   help='Directory to write per-(dataset, pred_len) JSON results')
    p.add_argument('--save-transcripts', action='store_true', dest='save_transcripts',
                   help='Save full prompts/completions for zero-shot models '
                        '(LLMTime/PromptCast/DirectPrompt) under <output>/transcripts/. '
                        'Can also be set via `save_transcripts: true` in the config.')
    p.add_argument('--device',  default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[2021],
                   help='Random seeds; results are averaged across seeds')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _set_seed(args.seeds[0])

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['save_transcripts'] = args.save_transcripts or cfg.get('save_transcripts', False)
    transcripts_root = Path(args.output or '.') / 'transcripts'

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    # ── resolve config ────────────────────────────────────────────────────
    llm_model_name = cfg.get('llm_model', 'GPT2')
    d_model    = cfg.get('d_model',  128)
    n_heads    = cfg.get('n_heads',  8)
    n_tokens   = cfg.get('n_tokens', 100)
    dropout    = cfg.get('dropout',  0.1)
    epochs     = cfg.get('train_epochs', 10)
    patience   = cfg.get('patience', 5)
    batch_size = cfg.get('batch_size', 16)
    lr         = cfg.get('learning_rate', 1e-4)
    scale_data = cfg.get('scale_data', True)

    models_to_run = ([args.model] if args.model
                     else cfg.get('models', ['TimeLLM', 'GPT4MTS']))

    all_text_conds = ['domain_only', 'correct', 'incorrect', 'irrelevant']
    text_conds = ([args.text_cond] if args.text_cond else all_text_conds)

    domain_prompts = cfg.get('domain_prompts', {})
    freq_settings  = cfg['freq_settings']

    all_datasets = cfg['datasets']
    if args.dataset:
        all_datasets = [d for d in all_datasets if d['name'] == args.dataset]
        if not all_datasets:
            raise ValueError(f"Dataset '{args.dataset}' not in config.")

    # ── load backbone LLM (for TimeLLM / GPT4MTS) ───────────────────────
    backbone_needed = any(m not in _ZERO_SHOT_MODELS for m in models_to_run)
    if backbone_needed:
        print(f'Loading backbone LLM: {llm_model_name} ...')
        tokenizer, llm_obj, d_llm = load_llm(llm_model_name, device)
        print(f'  d_llm={d_llm}')
    else:
        tokenizer = llm_obj = None
        d_llm = 768   # placeholder; unused for zero-shot-only runs

    # ── load generative LLM (for LLMTime / PromptCast) ──────────────────
    gen_tok = gen_llm = None
    if any(m in _ZERO_SHOT_MODELS for m in models_to_run):
        gen_cfg  = cfg.get('generative_llm', {})
        gen_name = gen_cfg.get('model_name', llm_model_name)
        print(f'Loading generative LLM for zero-shot: {gen_name} ...')
        gen_tok, gen_llm = load_generative_llm(gen_name, device)
        print(f'  generative LLM ready')

    all_results: dict = {}

    for ds_info in all_datasets:
        ds_name    = ds_info['name']
        freq       = ds_info['freq']
        fs         = freq_settings[freq]
        seq_len    = fs['seq_len']
        label_len  = fs['label_len']
        patch_len  = fs['patch_len']
        stride     = fs['stride']
        pred_lens  = [args.pred_len] if args.pred_len else fs['pred_lengths']
        domain_prompt = domain_prompts.get(ds_name, '')

        print(f'\n{"═"*70}')
        print(f'Dataset: {ds_name}  freq={freq}  seq={seq_len}  '
              f'patch={patch_len}/{stride}')

        for pred_len in pred_lens:
            print(f'\n  pred_len={pred_len}')

            # ── load data splits (domain_only uses all data; others filter by cat)
            try:
                all_train_ds, all_train_loader = _get_loader(
                    ds_info['train_json'], seq_len, label_len, pred_len,
                    batch_size, categories=None,
                    scale=scale_data, shuffle=True,
                )
                all_val_ds, all_val_loader = _get_loader(
                    ds_info['val_json'], seq_len, label_len, pred_len,
                    batch_size, categories=None,
                    scale=scale_data, scaler=all_train_ds.scaler,
                )
                cat_loaders: dict[str, tuple] = {}
                for cat in ['correct', 'incorrect', 'irrelevant']:
                    if cat not in text_conds:
                        continue
                    try:
                        tr_ds, tr_loader = _get_loader(
                            ds_info['train_json'], seq_len, label_len, pred_len,
                            batch_size, categories=[cat],
                            scale=scale_data, shuffle=True,
                        )
                        va_ds, va_loader = _get_loader(
                            ds_info['val_json'], seq_len, label_len, pred_len,
                            batch_size, categories=[cat],
                            scale=scale_data, scaler=tr_ds.scaler,
                        )
                        cat_loaders[cat] = (tr_ds, tr_loader, va_ds, va_loader)
                        print(f'    {cat}: train={len(tr_ds)}  val={len(va_ds)}')
                    except ValueError as e:
                        print(f'    [skip] {cat}: {e}')
            except Exception as e:
                print(f'  [skip] data loading: {e}')
                continue

            print(f'  all:   train={len(all_train_ds)}  val={len(all_val_ds)}')

            def _run_seeds(
                tr_ds, tr_loader, va_ds, va_loader,
                use_text: bool, model_name: str,
                prebuilt_model=None,
            ) -> dict | None:
                # Zero-shot models: no seed averaging — they do their own sampling
                if prebuilt_model is not None:
                    _set_seed(args.seeds[0])
                    try:
                        m = train_one_setting(
                            model_name=model_name, pred_len=pred_len,
                            patch_len=patch_len, stride=stride,
                            d_model=d_model, n_heads=n_heads,
                            n_tokens=n_tokens, d_llm=d_llm,
                            dropout=dropout, llm_model=llm_obj,
                            tokenizer=tokenizer,
                            train_ds=tr_ds, train_loader=tr_loader,
                            val_ds=va_ds, val_loader=va_loader,
                            domain_prompt=domain_prompt,
                            use_sample_text=use_text,
                            lr=lr, epochs=epochs, patience=patience,
                            device=device, prebuilt_model=prebuilt_model,
                        )
                        print(f'    zero-shot  MSE={m["mse"]:.4f}  MAE={m["mae"]:.4f}')
                        return m
                    except Exception as exc:
                        import traceback
                        print(f'    ERROR: {exc}'); traceback.print_exc()
                        return None

                seed_metrics = []
                for seed in args.seeds:
                    _set_seed(seed)
                    try:
                        m = train_one_setting(
                            model_name=model_name,
                            pred_len=pred_len,
                            patch_len=patch_len,
                            stride=stride,
                            d_model=d_model,
                            n_heads=n_heads,
                            n_tokens=n_tokens,
                            d_llm=d_llm,
                            dropout=dropout,
                            llm_model=llm_obj,
                            tokenizer=tokenizer,
                            train_ds=tr_ds,
                            train_loader=tr_loader,
                            val_ds=va_ds,
                            val_loader=va_loader,
                            domain_prompt=domain_prompt,
                            use_sample_text=use_text,
                            lr=lr,
                            epochs=epochs,
                            patience=patience,
                            device=device,
                        )
                        seed_metrics.append(m)
                        print(f'    seed={seed}  MSE={m["mse"]:.4f}  MAE={m["mae"]:.4f}')
                    except Exception as exc:
                        import traceback
                        print(f'    seed={seed} ERROR: {exc}')
                        traceback.print_exc()
                if not seed_metrics:
                    return None
                avg = {k: float(np.mean([m[k] for m in seed_metrics]))
                       for k in seed_metrics[0]}
                if len(seed_metrics) > 1:
                    avg.update({
                        f'{k}_std': float(np.std([m[k] for m in seed_metrics], ddof=1))
                        for k in seed_metrics[0]
                    })
                    avg['n_seeds'] = len(seed_metrics)
                return avg

            for model_name in models_to_run:
                print(f'\n  ── model: {model_name} ──')

                # Build zero-shot model once per (pred_len, model_name)
                prebuilt = (
                    _build_zero_shot_model(model_name, gen_tok, gen_llm, pred_len, cfg)
                    if model_name in _ZERO_SHOT_MODELS and gen_llm is not None
                    else None
                )

                # domain_only: train/evaluate on all data, no sample text
                if 'domain_only' in text_conds:
                    print(f'  ── {model_name}  domain_only (all data, no annotation) ──')
                    metrics = _run_seeds(all_train_ds, all_train_loader,
                                        all_val_ds, all_val_loader,
                                        use_text=False, model_name=model_name,
                                        prebuilt_model=prebuilt)
                    if metrics:
                        key = f'{ds_name}|pred{pred_len}|{model_name}|domain_only'
                        all_results[key] = metrics
                        print(f'  domain_only  MSE={metrics["mse"]:.4f}  '
                              f'MAE={metrics["mae"]:.4f}')
                    _dump_transcripts(prebuilt, all_val_ds, ds_name, pred_len,
                                      model_name, 'domain_only',
                                      transcripts_root, cfg['save_transcripts'])

                # per-category with sample text
                for cat in ['correct', 'incorrect', 'irrelevant']:
                    if cat not in text_conds or cat not in cat_loaders:
                        continue
                    tr_ds, tr_loader, va_ds, va_loader = cat_loaders[cat]
                    print(f'\n  ── {model_name}  text_cond={cat} ──')
                    metrics = _run_seeds(tr_ds, tr_loader, va_ds, va_loader,
                                        use_text=True, model_name=model_name,
                                        prebuilt_model=prebuilt)
                    if metrics:
                        key = f'{ds_name}|pred{pred_len}|{model_name}|{cat}'
                        all_results[key] = metrics
                        print(f'  {cat}  MSE={metrics["mse"]:.4f}  '
                              f'MAE={metrics["mae"]:.4f}')
                    _dump_transcripts(prebuilt, va_ds, ds_name, pred_len,
                                      model_name, cat,
                                      transcripts_root, cfg['save_transcripts'])

    # ── JSON output ───────────────────────────────────────────────────────
    if args.output:
        seen: set[tuple[str, int]] = set()
        for key in all_results:
            parts = key.split('|')
            seen.add((parts[0], int(parts[1].replace('pred', ''))))

        out_root = Path(args.output)
        for ds_name, pred_len in sorted(seen):
            prefix = f'{ds_name}|pred{pred_len}|'
            results_block: dict = {}
            for key, metrics in all_results.items():
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix):].split('|')
                model_name, cond = parts[0], parts[1]
                results_block.setdefault(cond, {})[model_name] = metrics

            payload = {
                'config':   {
                    'llm_model': llm_model_name,
                    'd_model':   d_model,
                    'n_tokens':  n_tokens,
                },
                'dataset':    ds_name,
                'pred_len':   pred_len,
                'models':     models_to_run,
                'categories': ['domain_only', 'correct', 'incorrect', 'irrelevant'],
                'results':    results_block,
            }
            out_path = out_root / ds_name / f'pred{pred_len}.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f'  Saved → {out_path}')

    # ── summary ───────────────────────────────────────────────────────────
    print('\n\nSummary')
    print(f'{"Key":<70}  {"MSE":>8}  {"MAE":>8}')
    print('─' * 90)
    for k, v in all_results.items():
        print(f'{k:<70}  {v["mse"]:8.4f}  {v["mae"]:8.4f}')
    print('\nDone.')


if __name__ == '__main__':
    main()
