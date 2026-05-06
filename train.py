#!/usr/bin/env python3
"""
train.py — Train sklearn regressors on joint (ts + text) embeddings.

Builds category_data exactly as Experiment does, then:
  1. Pools annotation points (matching metrics.py)
  2. Applies PCA to text embeddings (same options as Experiment)
  3. Concatenates ts + text_pca as the joint input embedding
  4. Trains Ridge and MLP on joint embeddings → predict y_future
  5. Evaluates regression metrics set
  6. Optionally writes a JSON results file for use with compare.py

Usage
-----
    python train.py --train-path benchmark_splits/train.json \\
                    --val-path benchmark_splits/val.json \\
                    --output results/final/sine_model.json

    # From a final_config YAML (shares params with run.py):
    python train.py --config final_configs/sine.yaml \\
                    --output results/final/sine_model.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    'ridge': lambda: Ridge(alpha=1.0),
    'mlp':   lambda: MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
    ),
    'svr':   lambda: SVR(kernel='rbf', C=10.0, epsilon=0.01),
    'gbr':   lambda: GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    ),
    'rf':    lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    ),
    'knn':   lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _pool_and_embed(category_data: dict, pca_dim: int, pca_text: PCA = None):
    """
    Pool annotation points, PCA-reduce text, build joint embedding.

    Mirrors the pooling + PCA steps in MutualInformationMetric.compute().

    Returns
    -------
    pooled   : {cat: {'ts', 'x_pca', 'joint', 'y'}}
    pca_text : fitted PCA (for reuse on held-out splits)
    """
    from src.mmtt_bench.metrics import _pool_by_annotation_point

    pooled: dict[str, dict] = {}
    for cat, vals in category_data.items():
        x_p, ts_p, y_p = _pool_by_annotation_point(
            np.array(vals['x']),
            np.array(vals['ts']),
            np.array(vals['y']),
            vals['n_samples'][0],
        )
        pooled[cat] = {'ts': ts_p, 'x_raw': x_p, 'y': y_p}

    if pca_text is None:
        all_x = np.vstack([v['x_raw'] for v in pooled.values()])
        pca_text = PCA(random_state=42).fit(all_x)

    for cat, vals in pooled.items():
        x_pca = pca_text.transform(vals['x_raw'])[:, :pca_dim]
        vals['x_pca']  = x_pca
        vals['joint']  = np.hstack([vals['ts'], x_pca])

    return pooled, pca_text


def _flatten(pooled: dict) -> dict:
    """Concatenate all categories into flat arrays with per-row category labels."""
    rows = []
    for cat, vals in pooled.items():
        rows.append({
            'X_joint': vals['joint'],
            'X_ts':    vals['ts'],
            'X_text':  vals['x_pca'],
            'y':       vals['y'],
            'cats':    [cat] * len(vals['joint']),
        })
    return {
        'X_joint': np.vstack([r['X_joint'] for r in rows]),
        'X_ts':    np.vstack([r['X_ts']    for r in rows]),
        'X_text':  np.vstack([r['X_text']  for r in rows]),
        'y':       np.vstack([r['y']        for r in rows]),
        'cats':    np.array([c for r in rows for c in r['cats']]),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        'mse':  mse,
        'rmse': float(np.sqrt(mse)),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'r2':   float(r2_score(y_true, y_pred)),
    }



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train Ridge/MLP on joint ts+text embeddings'
    )
    # ── Config shortcut (shares YAML format with run.py) ──────────────────
    p.add_argument('--config', default=None,
                   help='Path to a run.py-style YAML config. Sets data paths and '
                        'tokenizer params; CLI flags override individual fields.')
    # ── Data ─────────────────────────────────────────────────────────────
    p.add_argument('--train-path', default=None,
                   help='Path to training split JSON (overrides config)')
    p.add_argument('--val-path', default=None,
                   help='Path to validation split JSON (derived from train-path if omitted)')
    # ── Text embedding ────────────────────────────────────────────────────
    p.add_argument('--embedding-type', default=None,
                   choices=['sentence_transformer', 'huggingface'],
                   help='Text tokenizer type')
    p.add_argument('--model-name', default=None,
                   help='HuggingFace model name for text embedding')
    # ── TS tokenizer ──────────────────────────────────────────────────────
    p.add_argument('--ts-tokenizer', default=None,
                   choices=['patch', 'patch_mean', 'identity'],
                   help='Time-series tokenizer')
    p.add_argument('--lookback-steps', type=int, default=None)
    p.add_argument('--horizon-steps',  type=int, default=None)
    p.add_argument('--patch-len',      type=int, default=None)
    p.add_argument('--stride',         type=int, default=None)
    # ── Embedding ─────────────────────────────────────────────────────────
    p.add_argument('--pca-dim', type=int, default=None,
                   help='PCA dims to keep from text embedding before joint concat')
    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument('--models', nargs='+', default=None,
                   choices=list(MODELS.keys()),
                   help='Which models to train')
    p.add_argument('--text-categories', nargs='*', default=None,
                   help='Text categories to compare (default: all in data). '
                        'Each run alongside no_text.')
    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument('--output', '-o', default=None,
                   help='Path to write JSON results (for use with compare.py)')

    args = p.parse_args()

    # Load YAML config and apply as defaults for any unset CLI flags.
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # data_path in YAML → train_path; derive val_path by sibling convention
        if args.train_path is None and 'data_path' in cfg:
            args.train_path = cfg['data_path']
        if args.val_path is None and 'data_path' in cfg:
            train_p = cfg['data_path']
            args.val_path = str(train_p).replace('train.json', 'val.json')
        ck = cfg.get('compute_kwargs', {})
        _yaml_defaults = {
            'embedding_type': cfg.get('embedding_type', 'sentence_transformer'),
            'model_name':     cfg.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            'ts_tokenizer':   cfg.get('ts_tokenizer', 'patch_mean'),
            'lookback_steps': cfg.get('lookback_steps', 4),
            'horizon_steps':  cfg.get('horizon_steps', 1),
            'patch_len':      cfg.get('patch_len', 2),
            'stride':         cfg.get('stride', 2),
            'pca_dim':        ck.get('pca_dim', cfg.get('pca_dim', 8)),
        }
        for key, val in _yaml_defaults.items():
            cli_key = key.replace('_', '-')
            if getattr(args, key) is None:
                setattr(args, key, val)

    # Apply remaining hard defaults for anything still None.
    _hard_defaults = {
        'embedding_type': 'sentence_transformer',
        'model_name':     'sentence-transformers/all-MiniLM-L6-v2',
        'ts_tokenizer':   'patch_mean',
        'lookback_steps': 4,
        'horizon_steps':  1,
        'patch_len':      2,
        'stride':         2,
        'pca_dim':        8,
        'models':         ['ridge', 'mlp', 'svr', 'gbr', 'rf', 'knn'],
    }
    for key, val in _hard_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.train_path is None:
        p.error('--train-path (or --config with data_path) is required')
    if args.val_path is None:
        args.val_path = args.train_path.replace('train.json', 'val.json')

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── 1. Build category_data (same pipeline as Experiment) ──────────────
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    from src.mmtt_bench.data      import build_category_data, load_split
    from src.mmtt_bench.tokenizers import TEXT_TOKENIZERS, TS_TOKENIZERS

    def _build(path):
        raw = load_split(path)
        return build_category_data(
            raw,
            text_tokenizer = text_tok,
            ts_tokenizer   = ts_tok,
            horizon_steps  = args.horizon_steps,
            lookback_steps = args.lookback_steps,
            patch_len      = args.patch_len,
        )

    text_tok = TEXT_TOKENIZERS[args.embedding_type](args.model_name)
    ts_tok   = TS_TOKENIZERS[args.ts_tokenizer](
        patch_len      = args.patch_len,
        stride         = args.stride,
        lookback_steps = args.lookback_steps,
    ).build()

    print('Building train category data ...')
    train_cat = _build(args.train_path)
    print('Building val category data ...')
    val_cat   = _build(args.val_path)

    # ── 2. Pool + build joint embedding ───────────────────────────────────
    # PCA is fitted on train text embeddings and reused for val.
    print(f'Pooling and embedding (pca_dim={args.pca_dim}) ...')
    pooled_train, pca_text = _pool_and_embed(train_cat, args.pca_dim)
    pooled_val,   _        = _pool_and_embed(val_cat,   args.pca_dim, pca_text=pca_text)

    categories = list(pooled_train.keys())
    print(f'Categories: {categories}')
    for cat, vals in pooled_train.items():
        print(f'  {cat}: train N={len(vals["y"])}  '
              f'ts_dim={vals["ts"].shape[1]}  '
              f'joint_dim={vals["joint"].shape[1]}')

    train = _flatten(pooled_train)
    val   = _flatten(pooled_val)
    ts_dim = train['X_ts'].shape[1]

    text_cats    = args.text_categories if args.text_categories is not None else categories
    input_configs = [None] + text_cats   # None → no_text

    # Shared config fields logged to every run
    base_config = {
        'train_path':     args.train_path,
        'val_path':       args.val_path,
        'embedding_type': args.embedding_type,
        'model_name':     args.model_name,
        'ts_tokenizer':   args.ts_tokenizer,
        'lookback_steps': args.lookback_steps,
        'horizon_steps':  args.horizon_steps,
        'patch_len':      args.patch_len,
        'stride':         args.stride,
        'pca_dim':        args.pca_dim,
        'all_categories': categories,
    }

    # Accumulates results for CSV + JSON export.
    # Structure: {config_name: {model_key: {'overall': metrics, 'per_category': {cat: metrics}}}}
    all_results: dict[str, dict[str, dict]] = {}

    # ── Loop over input configurations ────────────────────────────────────
    for input_config in input_configs:
        config_name = input_config or 'no_text'

        # Select rows and features for this configuration
        if input_config is None:
            # ts-only baseline: use all data, drop text columns
            X_tr_raw     = train['X_ts']
            X_val_raw    = val['X_ts']
            y_tr         = train['y'][:, 0]
            y_val        = val['y'][:, 0]
            val_cats_arr = val['cats']
        else:
            # ts + text: filter to rows annotated with this text category
            tr_mask      = train['cats'] == input_config
            vl_mask      = val['cats']   == input_config
            X_tr_raw     = train['X_joint'][tr_mask]
            X_val_raw    = val['X_joint'][vl_mask]
            y_tr         = train['y'][tr_mask, 0]
            y_val        = val['y'][vl_mask,   0]
            val_cats_arr = val['cats'][vl_mask]

        if len(X_tr_raw) == 0 or len(X_val_raw) == 0:
            print(f'\n[skip] {config_name}: no data in train or val split')
            continue

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr_raw)
        X_val_s = scaler.transform(X_val_raw)

        print(f'\n══ {config_name}: train={X_tr_s.shape}  val={X_val_s.shape} ══')

        config_results: dict[str, dict] = {}

        # ── Train and evaluate each model ─────────────────────────────────
        for model_key in args.models:
            model  = MODELS[model_key]()
            print(f'  {model_key} ...', end=' ', flush=True)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)

            # get feature importance for tree-based models
            if model_key in ['rf', 'gbr'] and input_config is not None:
                # After fitting GBR on X_joint
                importances = model.feature_importances_
                ts_importance = importances[:ts_dim].sum()
                text_importance = importances[ts_dim:].sum()
                print(f"\n    feature importance — TS: {ts_importance:.3f}  Text: {text_importance:.3f}")

            overall = _regression_metrics(y_val, y_pred)
            print(f'R²={overall["r2"]:.3f}  RMSE={overall["rmse"]:.4f}')

            # ── Per-category breakdown ─────────────────────────────────────
            per_cat: dict[str, dict] = {}
            eval_cats = categories if input_config is None else [input_config]
            for cat in eval_cats:
                mask  = val_cats_arr == cat
                n_cat = int(mask.sum())
                if n_cat < 2:
                    continue
                cat_reg = _regression_metrics(y_val[mask], y_pred[mask])
                per_cat[cat] = cat_reg
                print(f'    {cat:<22s} N={n_cat:<4d} '
                      f'R²={cat_reg["r2"]:.3f}  RMSE={cat_reg["rmse"]:.4f}')

            config_results[model_key] = {'overall': overall, 'per_category': per_cat}

        all_results[config_name] = config_results

    # ── Write summary CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(args.train_path), 'results.csv')
    reg_metrics = ['mse', 'rmse', 'mae', 'r2']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'config'] + reg_metrics)
        for config_name, model_results in all_results.items():
            for mk in args.models:
                overall = model_results.get(mk, {}).get('overall', {})
                row = [mk, config_name] + [overall.get(m, '') for m in reg_metrics]
                writer.writerow(row)
    print(f'\nResults saved to {csv_path}')

    # ── Write JSON output (for compare.py) ────────────────────────────────
    if args.output:
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            'config':     base_config,
            'models':     args.models,
            'categories': categories,
            'results':    all_results,
        }
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'JSON results saved to {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
