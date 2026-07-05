#!/usr/bin/env python3
"""
train_transformer_cfa.py — Reproduce CFA paper experiments on TimeMMD data.

Closely mirrors Lee et al. (2026) "Rethinking Multimodal Fusion for Time Series
Forecasting: Text Modalities Need Constrained Fusion" (arXiv:2603.22372) using:

  - CFA backbone models  (cfa/models/)  which accept text_context
  - MM-TSFlib Dataset_Custom data loader (sliding windows, StandardScaler)
  - BERT/GPT2 text encoding: embedding lookup → MLP(d_llm→d_llm/8→pred_len) → avg-pool
  - All CFA injection modes: unimodal, first/middle/last-additive, first/middle/last-concat,
    film, cfa, gating, orthogonal

Key invariant from CFA run.py (line 174): text_emb = pred_len.

Usage
-----
    python train_transformer_cfa.py --config timemmd_cfa.yaml \\
        --dataset Agriculture \\
        --model PatchTST \\
        --injection-mode cfa \\
        --pred-len 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

# Path setup
_HERE   = Path(__file__).resolve().parent          # mmtt-bench/
_CFA    = _HERE.parent / 'cfa'                     # cfa/
_MMTSF  = _HERE.parent / 'MM-TSFlib'              # MM-TSFlib/

# Insert MM-TSFlib first so data_provider resolves there,
# then CFA last so it sits at position 0 and its models/ (with text_context)
# shadow MM-TSFlib's models/ for backbone imports.
sys.path.insert(0, str(_MMTSF))
sys.path.insert(0, str(_CFA))


# MLP  (identical to CFA exp_long_term_forecasting_text_integrated.py)
class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], dropout_rate: float = 0.3):
        super().__init__()
        self.layers  = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


# TEXT ENCODER

def load_text_encoder(llm_model: str, llm_layers: int, device: torch.device):
    """
    Load LLM tokenizer and model.  Parameters are frozen; only the embedding
    lookup (get_input_embeddings) is used unless use_fullmodel=True.

    Matches CFA's __init__ in Exp_Long_Term_Forecast.
    """
    from transformers import (
        BertConfig, BertModel, BertTokenizer,
        GPT2Config, GPT2Model, GPT2Tokenizer,
        AutoTokenizer, LlamaConfig, LlamaModel,
    )

    if llm_model == 'BERT':
        cfg = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        cfg.num_hidden_layers = llm_layers
        try:
            model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased', config=cfg, local_files_only=True)
            tok   = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased', local_files_only=True)
        except Exception:
            model = BertModel.from_pretrained('google-bert/bert-base-uncased', config=cfg)
            tok   = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
        d_llm = 768

    elif llm_model == 'GPT2':
        cfg = GPT2Config.from_pretrained('openai-community/gpt2')
        cfg.num_hidden_layers = llm_layers
        try:
            model = GPT2Model.from_pretrained(
                'openai-community/gpt2', config=cfg, local_files_only=True)
            tok   = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2', local_files_only=True)
        except Exception:
            model = GPT2Model.from_pretrained('openai-community/gpt2', config=cfg)
            tok   = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        d_llm = 768

    elif llm_model == 'LLAMA3':
        path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        cfg  = LlamaConfig.from_pretrained(path)
        cfg.num_hidden_layers = llm_layers
        model = LlamaModel.from_pretrained(path, config=cfg)
        tok   = AutoTokenizer.from_pretrained(path)
        d_llm = 4096

    else:
        raise ValueError(f'Unsupported llm_model: {llm_model}')

    if tok.eos_token:
        tok.pad_token = tok.eos_token
    else:
        tok.add_special_tokens({'pad_token': '[PAD]'})

    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device).eval()

    return tok, model, d_llm


def _texts(raw) -> list[str]:
    """
    Convert get_text() output to a flat list of strings.

    Dataset_Custom stores self.text from df_raw[[col]].values where df_raw is
    rebuilt by concatenating existing columns + text_name.  When text_name is
    already present in the general cols (e.g. Final_Search_4 in TimeMMD CSVs),
    the column appears twice, giving shape (N, 2) instead of (N, 1).
    We always take the first occurrence.
    """
    import numpy as np
    arr = np.asarray(raw)
    if arr.ndim > 1:
        arr = arr[:, 0]   # first occurrence of the duplicated column
    return [str(x) for x in arr]


def encode_text(
    texts: list[str],
    tokenizer,
    llm_model: nn.Module,
    mlp: MLP,
    pool_type: str,
    use_fullmodel: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a batch of text strings to [B, text_emb] following CFA:

        tokens → get_input_embeddings() → [B, L, d_llm]
        → (optional full LLM forward)
        → MLP(d_llm → d_llm/8 → text_emb)  [per token]
        → avg/max pool over tokens           [B, text_emb]
    """
    prompt = [
        # f"<|start_prompt|Make predictions about the future based on "
        # f"the following information: {t}<|<end_prompt>|>"
        # for t in texts
        f"{t}"
        for t in texts
    ]
    token_ids = tokenizer(
        prompt, return_tensors='pt', padding=True, truncation=True, max_length=1024
    ).input_ids.to(device)

    with torch.no_grad():
        emb = llm_model.get_input_embeddings()(token_ids)   # [B, L, d_llm]
        if use_fullmodel:
            emb = llm_model(inputs_embeds=emb).last_hidden_state

    emb = mlp(emb)   # [B, L, text_emb]

    if pool_type == 'avg':
        return emb.mean(dim=1)   # [B, text_emb]
    elif pool_type == 'max':
        return emb.max(dim=1)[0]
    else:
        return emb.mean(dim=1)


# MODEL CONFIG
# Per-model overrides that match CFA paper Table E.1 defaults
_MODEL_OVERRIDES: dict[str, dict] = {
    'FEDformer': {'n_heads': 8},   # hardcoded in FourierBlock weight shapes
}

def make_cfa_config(
    seq_len:          int,
    pred_len:         int,
    enc_in:           int,
    label_len:        int,
    injection_mode:   str,
    cfa_reduction:    int = 8,
    d_model:          int = 512,
    n_heads:          int = 8,
    e_layers:         int = 2,
    d_layers:         int = 1,
    d_ff:             int = 2048,
    dropout:          float = 0.1,
    factor:           int = 1,
    moving_avg:       int = 25,
) -> SimpleNamespace:
    """Build config namespace expected by CFA backbone models."""
    # moving_avg must be odd and ≤ seq_len
    mv = min(moving_avg, seq_len)
    if mv % 2 == 0:
        mv = max(1, mv - 1)

    return SimpleNamespace(
        task_name        = 'long_term_forecast',
        seq_len          = seq_len,
        label_len        = label_len,
        pred_len         = pred_len,
        enc_in           = enc_in,
        dec_in           = enc_in,
        c_out            = enc_in,
        d_model          = d_model,
        n_heads          = n_heads,
        e_layers         = e_layers,
        d_layers         = d_layers,
        d_ff             = d_ff,
        dropout          = dropout,
        factor           = factor,
        moving_avg       = mv,
        embed            = 'timeF',
        freq             = 'h',
        activation       = 'gelu',
        output_attention = False,
        distil           = True,
        num_class        = 1,
        # Nonstationary_Transformer projector layers
        p_hidden_dims    = [128, 128],
        p_hidden_layers  = 2,
        # CFA-specific
        text_injection_mode = injection_mode,
        text_emb            = pred_len,   # CFA invariant: text_emb = pred_len
        cfa_reduction       = cfa_reduction,
    )


# DATA LOADING  (MM-TSFlib Dataset_Custom)
def get_dataset_and_loader(
    root_path:  str,
    data_path:  str,
    flag:       str,          # 'train' | 'val' | 'test'
    seq_len:    int,
    label_len:  int,
    pred_len:   int,
    text_len:   int,
    batch_size: int,
    freq:       str = 'h',
    shuffle:    bool = None,
) -> tuple:
    """Wrap MM-TSFlib Dataset_Custom with a fake args namespace."""
    from data_provider.data_loader import Dataset_Custom

    fake_args = SimpleNamespace(
        use_closedllm = 0,
        text_len      = text_len,
    )

    ds = Dataset_Custom(
        args      = fake_args,
        root_path = root_path,
        flag      = flag,
        size      = [seq_len, label_len, pred_len],
        features  = 'S',
        data_path = data_path,
        target    = 'OT',
        scale     = True,
        timeenc   = 0,
        freq      = freq,
    )

    should_shuffle = (flag == 'train') if shuffle is None else shuffle
    loader = DataLoader(ds, batch_size=batch_size, shuffle=should_shuffle, drop_last=False)
    return ds, loader


# DATA LOADING  (mmtt-bench annotation-point datasets)
class MmttDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping mmtt-bench JSON annotation-point splits.

    Unlike MM-TSFlib's Dataset_Custom (sliding windows), each sample is one
    annotation point — the ts lookback ending at that point, the future horizon
    starting one step later, and the raw text annotation.  This preserves the
    independence of samples needed for MI estimation.

    The interface mirrors Dataset_Custom so the CFA training loop (including
    get_text / get_prior_y calls) works without modification.

    Parameters
    ----------
    json_path : path to mmtt-bench split JSON (e.g. agriculture_train.json)
    seq_len   : lookback steps
    label_len : decoder context steps (last seq_len steps reused as label)
    pred_len  : forecast horizon
    categories: list of categories to include, e.g. ['correct'].  None = all.
    scaler    : pre-fitted StandardScaler to use (pass train scaler for val).
                If None and flag='train', a new scaler is fitted here.
    """

    def __init__(
        self,
        json_path:  str,
        seq_len:    int,
        label_len:  int,
        pred_len:   int,
        categories: list[str] | None = None,
        scale:      bool = True,
        scaler=None,
    ):
        import json
        from sklearn.preprocessing import StandardScaler

        with open(json_path) as f:
            raw = json.load(f)

        dense_y  = np.array(raw['metadata']['y'], dtype=np.float64)
        dense_t  = [round(t, 4) for t in raw['metadata']['t']]
        n_dense  = len(dense_y)

        lookbacks, futures, texts = [], [], []

        for pt in raw['points']:
            if categories and pt.get('category') not in categories:
                continue
            t_i = round(pt['t'], 4)
            if t_i not in dense_t:
                continue
            idx = dense_t.index(t_i)

            # Lookback: seq_len steps ending just before the annotation point
            start    = max(0, idx - seq_len)
            lb       = dense_y[start:idx]
            if len(lb) < seq_len:                    # left-pad with first value
                lb = np.concatenate([np.full(seq_len - len(lb), lb[0] if len(lb) else 0.0), lb])
            lb = lb[-seq_len:]

            # Future: pred_len steps after the annotation point
            future_end = idx + 1 + pred_len
            if future_end > n_dense:
                continue
            fut = dense_y[idx + 1 : future_end]     # [pred_len]

            # Text: take the first sample surface form
            text = ''
            if pt.get('samples'):
                text = pt['samples'][0].get('text', '')

            lookbacks.append(lb)
            futures.append(fut)
            texts.append(text)

        if not lookbacks:
            raise ValueError(f'No samples found in {json_path} '
                             f'(categories={categories})')

        lb_arr  = np.array(lookbacks, dtype=np.float32)   # [N, seq_len]
        fut_arr = np.array(futures,   dtype=np.float32)   # [N, pred_len]

        # Normalise using a StandardScaler fitted on the lookback values.
        # Set scale=False for synthetic data already in [-1,1] range so that
        # results are directly comparable to train_transformer.py raw-scale output.
        if scale:
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(lb_arr.reshape(-1, 1))
            lb_arr  = scaler.transform(lb_arr.reshape(-1, 1)).reshape(lb_arr.shape)
            fut_arr = scaler.transform(fut_arr.reshape(-1, 1)).reshape(fut_arr.shape)
        self.scaler = scaler  # None when scale=False

        # data_x: [N, seq_len, 1]
        self.data_x = lb_arr[:, :, np.newaxis]

        # data_y: [N, label_len + pred_len, 1]  (label = last label_len of lookback)
        label_part  = lb_arr[:, -label_len:, np.newaxis]         # [N, label_len, 1]
        future_part = fut_arr[:, :, np.newaxis]                   # [N, pred_len, 1]
        self.data_y = np.concatenate([label_part, future_part], axis=1)

        self.texts    = texts
        self.seq_len  = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        N = len(texts)
        # Zero time marks (no real timestamps in mmtt-bench)
        self.x_mark = np.zeros((N, seq_len, 4),               dtype=np.float32)
        self.y_mark = np.zeros((N, label_len + pred_len, 4),  dtype=np.float32)

    # Dataset_Custom-compatible helpers
    def get_text(self, indices):
        """Return raw text strings for a batch of sample indices."""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return np.array([self.texts[int(i)] for i in indices], dtype=object)

    def get_prior_y(self, indices):
        """Return zeros — mmtt-bench has no prior_history_avg column."""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return np.zeros((len(indices), self.pred_len, 1), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (
            self.data_x[index],      # [seq_len, 1]
            self.data_y[index],      # [label_len + pred_len, 1]
            self.x_mark[index],      # [seq_len, 4]
            self.y_mark[index],      # [label_len + pred_len, 4]
            index,
        )


def get_mmtt_dataset_and_loader(
    train_json: str,
    val_json:   str,
    flag:       str,
    seq_len:    int,
    label_len:  int,
    pred_len:   int,
    batch_size: int,
    categories: list[str] | None = None,
    scale:      bool = True,
    train_scaler=None,
) -> tuple:
    """
    Build a MmttDataset and DataLoader for the requested split.

    Always builds train first (to get scaler) then val.  Call with
    flag='train' to get (train_ds, train_loader, fitted_scaler), then
    call again with flag='val' passing the scaler.
    """
    json_path = train_json if flag == 'train' else val_json
    ds = MmttDataset(
        json_path  = json_path,
        seq_len    = seq_len,
        label_len  = label_len,
        pred_len   = pred_len,
        categories = categories,
        scale      = scale,
        scaler     = train_scaler,
    )
    shuffle = (flag == 'train')
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return ds, loader


# TRAIN / EVALUATE
def train_one_setting(
    model_name:      str,
    cfg:             SimpleNamespace,
    train_ds,        train_loader,
    val_ds,          val_loader,
    tokenizer,
    llm_model:       nn.Module,
    mlp:             MLP,
    pool_type:       str,
    use_fullmodel:   bool,
    lr_model:        float,
    lr_mlp:          float,
    epochs:          int,
    patience:        int,
    device:          torch.device,
) -> tuple[dict, dict]:
    """
    Build a CFA backbone, train it with text_context, return best val + test metrics.
    Returns (val_metrics, preds_dict) where preds_dict has arrays for external eval.
    """
    import importlib
    mod      = importlib.import_module(f'models.{model_name}')
    backbone = mod.Model(cfg).float().to(device)

    opt_model = torch.optim.Adam(backbone.parameters(), lr=lr_model)
    opt_mlp   = torch.optim.Adam(mlp.parameters(),      lr=lr_mlp)
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    best_state     = None
    mlp_best_state = None
    no_improve     = 0

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────────
        backbone.train(); mlp.train()
        for batch_x, batch_y, batch_x_mark, batch_y_mark, index in train_loader:
            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            batch_text   = _texts(train_ds.get_text(index))

            text_ctx = encode_text(
                batch_text, tokenizer, llm_model, mlp,
                pool_type, use_fullmodel, device,
            )   # [B, pred_len]

            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :])
            dec_inp = torch.cat(
                [batch_y[:, :cfg.label_len, :], dec_inp], dim=1
            ).float().to(device)

            opt_model.zero_grad(); opt_mlp.zero_grad()

            if cfg.text_injection_mode == 'unimodal':
                outputs = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                   text_context=text_ctx)

            outputs = outputs[:, -cfg.pred_len:, 0:]
            loss    = criterion(outputs, batch_y[:, -cfg.pred_len:, 0:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            opt_model.step(); opt_mlp.step()

        # validate 
        val_loss = _evaluate_loss(
            backbone, mlp, val_ds, val_loader, criterion,
            cfg, tokenizer, llm_model, pool_type, use_fullmodel, device,
        )
        print(f'    epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
            mlp_best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'    early stop (patience={patience})')
                break

    # restore best
    backbone.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    mlp.load_state_dict({k: v.to(device) for k, v in mlp_best_state.items()})

    val_metrics = _compute_metrics(
        backbone, mlp, val_ds, val_loader,
        cfg, tokenizer, llm_model, pool_type, use_fullmodel, device,
    )
    return val_metrics


@torch.no_grad()
def _evaluate_loss(backbone, mlp, ds, loader, criterion, cfg,
                   tokenizer, llm_model, pool_type, use_fullmodel, device):
    backbone.eval(); mlp.eval()
    total, n = 0.0, 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark, index in loader:
        batch_x      = batch_x.float().to(device)
        batch_y      = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        batch_text = _texts(ds.get_text(index))

        text_ctx = encode_text(batch_text, tokenizer, llm_model, mlp,
                               pool_type, use_fullmodel, device)

        dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1).to(device)

        if cfg.text_injection_mode == 'unimodal':
            out = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            out = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                           text_context=text_ctx)

        out   = out[:, -cfg.pred_len:, 0:]
        true  = batch_y[:, -cfg.pred_len:, 0:].to(device)
        total += criterion(out, true).item() * len(batch_x)
        n     += len(batch_x)

    backbone.train(); mlp.train()
    return total / n


@torch.no_grad()
def _compute_metrics(backbone, mlp, ds, loader, cfg,
                     tokenizer, llm_model, pool_type, use_fullmodel, device):
    backbone.eval(); mlp.eval()
    preds, trues = [], []
    for batch_x, batch_y, batch_x_mark, batch_y_mark, index in loader:
        batch_x      = batch_x.float().to(device)
        batch_y      = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        batch_text = _texts(ds.get_text(index))
        text_ctx = encode_text(batch_text, tokenizer, llm_model, mlp,
                               pool_type, use_fullmodel, device)

        dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1).to(device)

        if cfg.text_injection_mode == 'unimodal':
            out = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            out = backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                           text_context=text_ctx)

        preds.append(out[:, -cfg.pred_len:, 0:].cpu().numpy())
        trues.append(batch_y[:, -cfg.pred_len:, 0:].cpu().numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    mse   = float(mean_squared_error(trues, preds))
    mae   = float(mean_absolute_error(trues, preds))
    backbone.train(); mlp.train()
    return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mae}


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Reproduce CFA paper results on TimeMMD data'
    )
    p.add_argument('--config',          required=True,
                   help='Path to timemmd_cfa.yaml')
    p.add_argument('--dataset',         default=None,
                   help='Dataset name (must match an entry in YAML). '
                        'If omitted, runs all datasets.')
    p.add_argument('--model',           default=None,
                   help='Model name (e.g. PatchTST). '
                        'If omitted, runs all models from YAML.')
    p.add_argument('--injection-mode',  default=None, dest='injection_mode',
                   help='CFA injection mode. If omitted, runs all from YAML.')
    p.add_argument('--pred-len',        type=int, default=None, dest='pred_len',
                   help='Single prediction horizon. If omitted, runs all for the freq.')
    p.add_argument('--llm-model',       default=None, dest='llm_model',
                   help='Override LLM model from YAML (BERT|GPT2|LLAMA3).')
    p.add_argument('--lr-multiplier',   type=float, default=None,
                   dest='lr_multiplier',
                   help='Multiply both learning rates by this factor.')
    p.add_argument('--output',  '-o',   default=None,
                   help='Path to write JSON results.')
    p.add_argument('--device',          default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[2021],
                   help='One or more random seeds.  When multiple seeds are given '
                        'each (model, mode) configuration is trained once per seed '
                        'and the metrics are averaged, matching the CFA paper\'s '
                        'practice of averaging across multiple runs. '
                        'Example: --seeds 2021 42 123')
    return p.parse_args()


# MAIN
def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main() -> None:
    args = parse_args()
    _set_seed(args.seeds[0])   # initial seed; reset per-run below

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    # ── resolve run parameters ─────────────────────────────────────────────
    llm_model_name = args.llm_model or cfg_yaml.get('llm_model', 'BERT')
    text_len       = cfg_yaml.get('text_len', 4)
    pool_type      = cfg_yaml.get('pool_type', 'avg')
    use_fullmodel  = bool(cfg_yaml.get('use_fullmodel', False))
    llm_layers     = cfg_yaml.get('llm_layers', 6)
    cfa_reduction  = cfg_yaml.get('cfa_reduction', 8)
    d_model        = cfg_yaml.get('d_model', 512)
    n_heads        = cfg_yaml.get('n_heads', 8)
    e_layers       = cfg_yaml.get('e_layers', 2)
    d_layers       = cfg_yaml.get('d_layers', 1)
    d_ff           = cfg_yaml.get('d_ff', 2048)
    dropout        = cfg_yaml.get('dropout', 0.1)
    train_epochs   = cfg_yaml.get('train_epochs', 10)
    patience       = cfg_yaml.get('patience', 5)
    batch_size     = cfg_yaml.get('batch_size', 32)
    lr_model_base  = cfg_yaml.get('learning_rate', 1e-4)
    lr_mlp_base    = cfg_yaml.get('lr_mlp', 1e-2)
    lr_mult        = args.lr_multiplier or cfg_yaml.get('lr_multiplier', 1.0)
    lr_model       = lr_model_base * lr_mult
    lr_mlp         = lr_mlp_base   * lr_mult

    models_to_run  = ([args.model]          if args.model
                      else cfg_yaml.get('models', ['PatchTST']))
    modes_to_run   = ([args.injection_mode] if args.injection_mode
                      else cfg_yaml.get('injection_modes', ['unimodal', 'cfa']))
    freq_settings  = cfg_yaml['freq_settings']

    # filter datasets
    all_datasets   = cfg_yaml['datasets']
    if args.dataset:
        all_datasets = [d for d in all_datasets if d['name'] == args.dataset]
        if not all_datasets:
            raise ValueError(f"Dataset '{args.dataset}' not found in YAML.")

    # ── load LLM once ─────────────────────────────────────────────────────
    print(f'Loading text encoder: {llm_model_name} ({llm_layers} layers) ...')
    tokenizer, llm_model_obj, d_llm = load_text_encoder(
        llm_model_name, llm_layers, device
    )
    print(f'  d_llm={d_llm}')

    all_results: dict = {}  # {compound_key: metrics}

    data_format   = cfg_yaml.get('data_format', 'mmtsf')  # 'mmtsf' | 'mmtt'
    BASELINE_MODE = 'unimodal'
    non_baseline_modes = [m for m in modes_to_run if m != BASELINE_MODE]

    # Text conditions vary only in mmtt mode (CSV mode has a single text column).
    text_conditions = (['correct', 'incorrect', 'irrelevant']
                       if data_format == 'mmtt' else ['correct'])

    # ── outer loops: dataset → pred_len → text_condition → model → mode ───
    for ds_info in all_datasets:
        ds_name   = ds_info['name']
        freq      = ds_info['freq']
        fsettings = freq_settings[freq]
        seq_len   = fsettings['seq_len']
        label_len = fsettings['label_len']
        pred_lens = [args.pred_len] if args.pred_len else fsettings['pred_lengths']

        print(f'\n{"═"*70}')
        print(f'Dataset: {ds_name}  freq={freq}  seq={seq_len}  format={data_format}')

        for pred_len in pred_lens:
            print(f'\n  pred_len={pred_len}')
            unimodal_done: set[str] = set()  # backbone names whose unimodal is cached

            for text_cond in text_conditions:
                print(f'\n  ── text condition: {text_cond} ──')

                # ── build data loaders for this text condition ─────────────
                try:
                    if data_format == 'mmtt':
                        train_json = ds_info['train_json']
                        val_json   = ds_info['val_json']
                        scale_data = ds_info.get('scale', cfg_yaml.get('scale_data', True))
                        train_ds, train_loader = get_mmtt_dataset_and_loader(
                            train_json, val_json, 'train',
                            seq_len, label_len, pred_len, batch_size,
                            [text_cond], scale=scale_data)
                        val_ds, val_loader = get_mmtt_dataset_and_loader(
                            train_json, val_json, 'val',
                            seq_len, label_len, pred_len, batch_size,
                            [text_cond], scale=scale_data,
                            train_scaler=train_ds.scaler)
                    else:
                        root_path = ds_info['root_path']
                        data_path = ds_info['data_path']
                        train_ds, train_loader = get_dataset_and_loader(
                            root_path, data_path, 'train', seq_len, label_len,
                            pred_len, text_len, batch_size, freq='h')
                        val_ds, val_loader = get_dataset_and_loader(
                            root_path, data_path, 'val', seq_len, label_len,
                            pred_len, text_len, batch_size, freq='h')
                except Exception as e:
                    print(f'  [skip] data loading failed: {e}')
                    continue

                enc_in = train_ds.data_x.shape[-1]
                print(f'  train={len(train_ds)}  val={len(val_ds)}  enc_in={enc_in}')

                for model_name in models_to_run:
                    eff_overrides = dict(_MODEL_OVERRIDES.get(model_name, {}))
                    run_n_heads   = eff_overrides.get('n_heads', n_heads)

                    def _run_seeds(injection_mode: str) -> dict | None:
                        seed_metrics = []
                        cfg = make_cfa_config(
                            seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
                            label_len=label_len, injection_mode=injection_mode,
                            cfa_reduction=cfa_reduction, d_model=d_model,
                            n_heads=run_n_heads, e_layers=e_layers,
                            d_layers=d_layers, d_ff=d_ff, dropout=dropout,
                        )
                        for seed in args.seeds:
                            _set_seed(seed)
                            mlp_run = MLP([d_llm, d_llm // 8, pred_len],
                                          dropout_rate=0.3).to(device)
                            try:
                                m = train_one_setting(
                                    model_name=model_name, cfg=cfg,
                                    train_ds=train_ds, train_loader=train_loader,
                                    val_ds=val_ds, val_loader=val_loader,
                                    tokenizer=tokenizer, llm_model=llm_model_obj,
                                    mlp=mlp_run, pool_type=pool_type,
                                    use_fullmodel=use_fullmodel,
                                    lr_model=lr_model, lr_mlp=lr_mlp,
                                    epochs=train_epochs, patience=patience,
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

                    # ── unimodal baseline: trained once per backbone ───────
                    # Text is ignored by unimodal; train on the first text_cond pass.
                    if BASELINE_MODE in modes_to_run and model_name not in unimodal_done:
                        unimodal_done.add(model_name)
                        print(f'\n  ── {model_name}  mode={BASELINE_MODE} (baseline, once) ──')
                        metrics = _run_seeds(BASELINE_MODE)
                        if metrics:
                            uni_key = f'{ds_name}|pred{pred_len}|{model_name}|{BASELINE_MODE}'
                            all_results[uni_key] = metrics
                            print(f'  baseline avg  MSE={metrics["mse"]:.4f}  '
                                  f'MAE={metrics["mae"]:.4f}')

                    # ── non-baseline modes for this text condition ─────────
                    for injection_mode in non_baseline_modes:
                        print(f'\n  ── {model_name}  mode={injection_mode}  text={text_cond} ──')
                        metrics = _run_seeds(injection_mode)
                        if metrics:
                            key = (f'{ds_name}|pred{pred_len}|{model_name}'
                                   f'|{injection_mode}|{text_cond}')
                            all_results[key] = metrics
                            print(f'  avg  MSE={metrics["mse"]:.4f}±'
                                  f'{metrics.get("mse_std", 0):.4f}  '
                                  f'MAE={metrics["mae"]:.4f}±'
                                  f'{metrics.get("mae_std", 0):.4f}')

    # ── JSON output ────────────────────────────────────────────────────────
    # One file per (dataset, pred_len):  <output_dir>/<dataset>/pred<N>.json
    #
    # plot.py --models compatible:
    #   categories  = text conditions: correct / incorrect / irrelevant
    #   models      = compound "backbone|injection_mode" strings
    #   no_text     = unimodal baseline (text ignored), same for all compounds
    #
    # Use backbone_models and fusion_methods keys to filter by either axis.
    if args.output:
        run_config = {
            'llm_model':     llm_model_name,
            'text_len':      text_len,
            'pool_type':     pool_type,
            'use_fullmodel': use_fullmodel,
            'cfa_reduction': cfa_reduction,
            'd_model':       d_model,
            'lr_multiplier': lr_mult,
            'data_format':   data_format,
        }

        seen: set[tuple[str, int]] = set()
        for key in all_results:
            parts = key.split('|')
            if len(parts) >= 4:
                seen.add((parts[0], int(parts[1].replace('pred', ''))))

        out_root = Path(args.output)
        for ds_name, pred_len in sorted(seen):
            prefix = f'{ds_name}|pred{pred_len}|'
            backbones: set[str] = set()
            nb_modes:  set[str] = set()
            for key in all_results:
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix):].split('|')
                backbones.add(parts[0])
                if parts[1] != BASELINE_MODE:
                    nb_modes.add(parts[1])

            sorted_backbones = sorted(backbones)
            sorted_nb_modes  = sorted(nb_modes)
            compound_models  = [f'{b}|{m}'
                                 for b in sorted_backbones
                                 for m in sorted_nb_modes]

            results: dict = {'no_text': {}}
            for compound in compound_models:
                backbone, mode = compound.rsplit('|', 1)
                uni_metrics = all_results.get(f'{prefix}{backbone}|{BASELINE_MODE}', {})
                results['no_text'][compound] = {
                    'overall':      uni_metrics,
                    'per_category': {cat: uni_metrics for cat in text_conditions},
                }
                for cat in text_conditions:
                    cat_metrics = all_results.get(
                        f'{prefix}{backbone}|{mode}|{cat}', {})
                    results.setdefault(cat, {})[compound] = {
                        'overall':      cat_metrics,
                        'per_category': {cat: cat_metrics},
                    }

            payload = {
                'config':          run_config,
                'dataset':         ds_name,
                'pred_len':        pred_len,
                'models':          compound_models,
                'backbone_models': sorted_backbones,
                'fusion_methods':  sorted_nb_modes,
                'categories':      list(text_conditions),
                'results':         results,
            }

            out_path = out_root / ds_name / f'pred{pred_len}.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f'  Saved → {out_path}')

    # ── console summary ────────────────────────────────────────────────────
    print('\n\nSummary')
    print(f'{"Key":<70}  {"MSE":>8}  {"MAE":>8}')
    print('─' * 90)
    for k, v in all_results.items():
        print(f'{k:<70}  {v["mse"]:8.4f}  {v["mae"]:8.4f}')
    print('\nDone.')


if __name__ == '__main__':
    main()
