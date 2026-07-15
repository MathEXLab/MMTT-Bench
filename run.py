"""
Run a benchmark experiment or sweep and save results to JSON.

Usage
-----
    # Single dataset
    python run.py config.yaml --output results/my_run.json

    # Multi-dataset (datasets: list in config) — writes one JSON per dataset
    python run.py config.yaml --output results/final/

Config format (see example_config.yaml for full reference):

    data_path: benchmark_splits_zero/train.json
    model_name: sentence-transformers/all-MiniLM-L6-v2
    ts_tokenizer: identity
    lookback_steps: 1
    horizon_steps: 1
    patch_len: 1
    stride: 1
    compute_kwargs:
      pca_dim: 4

    # One or more metrics — category_data is built once and shared.
    metrics:
      mutual_information:
        n_neighbours: 3
        n_bootstrap: 20
      pid:
        n_clusters_ts: 4
        n_clusters_text: 3
        n_bins_y: 3

    # Optional sweep — varies one parameter across all metrics.
    sweep:
      param: pca_dim          # top-level field or key in compute_kwargs
      values: [1, 2, 4, 8, 16, 32]

    # Optional multi-dataset list — overrides data_path; output must be a directory.
    # datasets:
    #   - benchmark_splits/train.json
    #   - benchmark_splits_zero30/train.json
    #   - benchmark_splits_rossler/train.json
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

# Cap OpenBLAS threads before any numpy/scipy import to prevent segfaults on
# machines with >128 cores (OpenBLAS's compiled maximum).
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "64")

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _serialise_results(results: dict) -> dict:
    """Drop large non-serialisable arrays (transformed_text)."""
    out = {}
    for k, v in results.items():
        if k == "transformed_text":
            continue
        if isinstance(v, dict):
            out[k] = {cat: (val.tolist() if isinstance(val, np.ndarray) else val)
                      for cat, val in v.items()}
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DATA_FIELDS = {
    "data_path", "embedding_type", "model_name", "device", "llm_batch_size",
    "ts_tokenizer", "horizon_steps", "lookback_steps", "patch_len", "stride",
}

_DEFAULTS = {
    "embedding_type": "sentence_transformer",
    "model_name":     "sentence-transformers/all-MiniLM-L6-v2",
    "ts_tokenizer":   "identity",
    "horizon_steps":  1,
    "lookback_steps": 1,
    "patch_len":      1,
    "stride":         1,
    "device":         None,
    "llm_batch_size": 8,
}


def _parse_metrics(cfg: dict) -> dict:
    """
    Return {metric_name: metric_kwargs} from the config.
    Accepts the new `metrics` dict block.
    """
    if "metrics" in cfg:
        return {name: (kwargs or {}) for name, kwargs in cfg["metrics"].items()}
    raise KeyError("Config must contain a 'metrics' block.")


def _build_category_data(cfg: dict):
    """Build category_data from config — expensive (loads model)."""
    from src.mmtt_bench.data import (
        build_category_data, build_category_data_dummy_embedding,
        load_split, shuffle_text_annotations,
    )
    from src.mmtt_bench.tokenizers import TEXT_TOKENIZERS, TS_TOKENIZERS

    raw            = load_split(cfg["data_path"])
    embedding_type = cfg.get("embedding_type", "sentence_transformer")
    ts_tok = TS_TOKENIZERS[cfg.get("ts_tokenizer", "identity")](
        patch_len      = cfg.get("patch_len",      _DEFAULTS["patch_len"]),
        stride         = cfg.get("stride",         _DEFAULTS["stride"]),
        lookback_steps = cfg.get("lookback_steps", _DEFAULTS["lookback_steps"]),
    ).build()

    if embedding_type == "dummy":
        # Oracle 1-D embedding derived from the true y_future value instead
        # of a real text tokenizer — see build_category_data_dummy_embedding.
        category_data = build_category_data_dummy_embedding(
            raw,
            ts_tokenizer   = ts_tok,
            horizon_steps  = cfg.get("horizon_steps",  _DEFAULTS["horizon_steps"]),
            lookback_steps = cfg.get("lookback_steps", _DEFAULTS["lookback_steps"]),
            patch_len      = cfg.get("patch_len",      _DEFAULTS["patch_len"]),
            seed           = cfg.get("dummy_seed", 0),
        )
    else:
        model_name = cfg.get("model_name", _DEFAULTS["model_name"])
        if embedding_type == "llm":
            text_tok = TEXT_TOKENIZERS[embedding_type](
                model_name,
                device     = cfg.get("device",         _DEFAULTS["device"]),
                batch_size = cfg.get("llm_batch_size", _DEFAULTS["llm_batch_size"]),
            )
        else:
            text_tok = TEXT_TOKENIZERS[embedding_type](model_name)

        category_data = build_category_data(
            raw,
            text_tokenizer = text_tok,
            ts_tokenizer   = ts_tok,
            horizon_steps  = cfg.get("horizon_steps",  _DEFAULTS["horizon_steps"]),
            lookback_steps = cfg.get("lookback_steps", _DEFAULTS["lookback_steps"]),
            patch_len      = cfg.get("patch_len",      _DEFAULTS["patch_len"]),
        )

    if cfg.get("shuffle", False):
        print("  [shuffle] Shuffling text annotations to random time points ...")
        rng = np.random.default_rng(cfg.get("shuffle_seed", 0))
        category_data = shuffle_text_annotations(category_data, rng)

    return category_data


def _run_metrics(category_data, metrics_cfg: dict, compute_kwargs: dict) -> dict:
    """Run all metrics on the same category_data. Returns {metric: results}."""
    from src.mmtt_bench.metrics import METRICS

    results_by_metric = {}
    for metric_name, metric_kwargs in metrics_cfg.items():
        print(f"  [{metric_name}] computing ...")
        metric_obj = METRICS[metric_name](**metric_kwargs)
        rng = np.random.default_rng(42)
        results_by_metric[metric_name] = _serialise_results(
            metric_obj.compute(category_data, rng=rng, **compute_kwargs)
        )
    return results_by_metric


# ---------------------------------------------------------------------------
# Sweep parameter logic (mirrors sweep.py but works on raw cfg dicts)
# ---------------------------------------------------------------------------

def _apply_sweep_value(cfg: dict, param: str, value) -> dict:
    """Return a copy of cfg with the sweep param set to value."""
    cfg = deepcopy(cfg)
    if param in cfg:
        cfg[param] = value
    elif param in cfg.get("compute_kwargs", {}):
        cfg["compute_kwargs"][param] = value
    else:
        # Search inside each metric's kwargs block
        found = False
        for metric_kwargs in cfg.get("metrics", {}).values():
            if metric_kwargs and param in metric_kwargs:
                metric_kwargs[param] = value
                found = True
        if not found:
            raise ValueError(
                f"Sweep param {param!r} not found as a top-level key, "
                "inside compute_kwargs, or inside any metric's kwargs."
            )
    return cfg


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_single(cfg: dict) -> dict:
    metrics_cfg    = _parse_metrics(cfg)
    compute_kwargs = cfg.get("compute_kwargs", {})

    print("Building category data ...")
    category_data = _build_category_data(cfg)

    return {
        "type":               "single",
        "config":             {k: v for k, v in cfg.items()
                               if k not in ("metrics", "sweep")},
        "metrics":            list(metrics_cfg.keys()),
        "results_by_metric":  _run_metrics(category_data, metrics_cfg,
                                           compute_kwargs),
    }


def run_sweep(cfg: dict) -> dict:
    sweep_cfg = cfg.pop("sweep")
    param     = sweep_cfg["param"]
    values    = sweep_cfg["values"]

    records = []
    for val in values:
        print(f"\n── {param}={val!r} ──")
        point_cfg      = _apply_sweep_value(cfg, param, val)
        compute_kwargs = point_cfg.get("compute_kwargs", {})
        metrics_cfg    = _parse_metrics(point_cfg)

        # Only rebuild category_data when the sweep touches a data field.
        if val == values[0] or param in _DATA_FIELDS:
            print("  Building category data ...")
            category_data = _build_category_data(point_cfg)

        records.append({
            "sweep_value":       val,
            "results_by_metric": _run_metrics(category_data, metrics_cfg,
                                              compute_kwargs),
        })

    return {
        "type":         "sweep",
        "config":       {k: v for k, v in cfg.items() if k != "metrics"},
        "metrics":      list(metrics_cfg.keys()),
        "sweep_param":  param,
        "sweep_values": values,
        "runs":         records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_one(cfg: dict, output_path: Path) -> None:
    """Run a single config (sweep or single) and write JSON to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = run_sweep(cfg) if "sweep" in cfg else run_single(cfg)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, cls=_Encoder)
    print(f"\nSaved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a benchmark experiment.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON path, or directory when config has datasets: list")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Multi-dataset mode: datasets: list overrides data_path, output must be a dir.
    if "datasets" in cfg:
        datasets = cfg.pop("datasets")
        out_dir  = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset_path in datasets:
            stem      = Path(dataset_path).parent.name  # e.g. benchmark_splits_zero30
            out_path  = out_dir / f"{stem}.json"
            run_cfg   = {**cfg, "data_path": dataset_path}
            print(f"\n{'═' * 60}")
            print(f"Dataset: {dataset_path}")
            print(f"{'═' * 60}")
            _run_one(run_cfg, out_path)
    else:
        out = Path(args.output)
        if out.is_dir():
            stem = Path(cfg.get("data_path", "run")).stem
            out  = out / f"{stem}_metrics.json"
            print(f"Output is a directory — writing to {out}")
        _run_one(cfg, out)


if __name__ == "__main__":
    main()
