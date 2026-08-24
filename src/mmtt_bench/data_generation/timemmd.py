#!/usr/bin/env python3
"""
timemmd_cfa.py

Convert MM-TSFlib TimeMMD CSVs to mmtt-bench annotation-point JSON splits
following CFA paper conventions.

Source
------
The eight TimeMMD datasets from MM-TSFlib, hardcoded below (DATASETS).
Pass --mm-tsf-root to set the MM-TSFlib data directory
(default: /root/emma-code/MM-TSFlib/data).

Text column
-----------
Final_Search_4  (default, matching CFA text_len=4).  The "Available facts are
as follows:" boilerplate and date prefixes are stripped by clean_search_text.

Split
-----
70 / 10 / 20  (train / val / test), matching Dataset_Custom exactly:
  num_train = int(N * 0.7)
  num_test  = int(N * 0.2)
  num_vali  = N - num_train - num_test

Frequency settings  (from run_timemmd.sh get_forecast_settings)
----------------------------------------------------------------
  daily:   seq_len=96,  label_len=48,  pred_lengths=[48, 96, 192, 336]
  weekly:  seq_len=36,  label_len=18,  pred_lengths=[12, 24,  36,  48]
  monthly: seq_len=8,   label_len=4,   pred_lengths=[ 6,  8,  10,  12]

These are hardcoded in FREQ_SETTINGS and stored in each JSON's metadata.

CFA windowing vs mmtt-bench annotation points
---------------------------------------------
CFA (Dataset_Custom) generates *sliding windows*.  For each split, it extends
the start boundary back by seq_len rows so the first window has a full
lookback:
  border1s = [0,  num_train - seq_len,  N - num_test - seq_len]
  border2s = [num_train,  num_train + num_vali,  N]

Number of CFA sliding windows per split:
  train : num_train - seq_len - pred_len + 1
  val   : num_vali  - pred_len + 1
  test  : num_test  - pred_len + 1

This script instead produces ONE annotation point per split timestep:
  train : num_train   points  (= int(N * 0.7))
  val   : num_vali    points
  test  : num_test    points  (= int(N * 0.2))

Annotation points are seq_len/pred_len agnostic.  The full N-row dense signal
is stored in metadata.t / metadata.y so that build_category_data() (run.py)
and MmttDataset (train_transformer_cfa.py) can extract any lookback length at
runtime, crossing split boundaries exactly as Dataset_Custom does.

Annotation points within pred_len steps of the end of the full signal are
silently dropped at runtime when no future window exists.

Categories
----------
  correct    — text at that timestep  (natural pairing)
  incorrect  — text shuffled within the same dataset/split via constrained
               derangement (no text returns to its original position)
  irrelevant — text sampled uniformly from the same-split rows of a different
               domain CSV

Usage
-----
    python src/mmtt_bench/data_generation/timemmd_cfa.py \\
        --output-dir datasets/timemmd_cfa_splits

    # Non-default MM-TSFlib root:
    python src/mmtt_bench/data_generation/timemmd_cfa.py \\
        --mm-tsf-root /path/to/MM-TSFlib/data \\
        --output-dir  datasets/timemmd_cfa_splits

Output
------
  <output-dir>/
    Agriculture/
      train.json  val.json  test.json
    ...

  Each JSON metadata block:
  {
    "split": "train", "text_col": "Final_Search_4",
    "freq": "monthly", "seq_len": 8, "label_len": 4,
    "pred_lengths": [6, 8, 10, 12],
    "n_points": 347, "blank_points": 0.05,
    "t": [...all N unix-second timestamps...],
    "y": [...all N OT values...]
  }
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded dataset catalogue  (mirrors timemmd_cfa.yaml datasets block)
# ---------------------------------------------------------------------------

DATASETS = [
    # name          subdir            csv                                      freq
    ("Agriculture", "Algriculture",   "US_RetailBroilerComposite_Month.csv",  "monthly"),
    ("Climate",     "Climate",        "US_precipitation_month.csv",            "monthly"),
    ("Economy",     "Economy",        "US_TradeBalance_Month.csv",             "monthly"),
    ("Energy",      "Energy",         "US_GasolinePrice_Week.csv",             "weekly"),
    ("PublicHealth","Public_Health",  "US_FLURATIO_Week.csv",                  "weekly"),
    ("Security",    "Security",       "US_FEMAGrant_Month.csv",                "monthly"),
    ("SocialGood",  "SocialGood",     "Unadj_UnemploymentRate_ALL_processed.csv", "monthly"),
    ("Traffic",     "Traffic",        "US_VMT_Month.csv",                      "monthly"),
]

DEFAULT_MM_TSF_ROOT = Path("/root/emma-code/MM-TSFlib/data")


# ---------------------------------------------------------------------------
# Frequency settings  (from run_timemmd.sh get_forecast_settings)
# ---------------------------------------------------------------------------

FREQ_SETTINGS: dict[str, dict] = {
    'daily':   {'seq_len': 96, 'label_len': 48, 'pred_lengths': [48, 96, 192, 336]},
    'weekly':  {'seq_len': 36, 'label_len': 18, 'pred_lengths': [12, 24,  36,  48]},
    'monthly': {'seq_len':  8, 'label_len':  4, 'pred_lengths': [ 6,  8,  10,  12]},
}


# ---------------------------------------------------------------------------
# Text cleaning  (same as timemmd.py)
# ---------------------------------------------------------------------------

def clean_search_text(text: str) -> str:
    """Strip boilerplate header and date prefixes from Final_Search_N text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^Available facts are as follows:\s*', '', text)
    text = text.rstrip(';').strip()
    facts = re.split(r'\d{4}-\d{2}-\d{2}:\s*', text)
    facts = [f.strip().rstrip('.').strip() for f in facts
             if f.strip() and f.strip() not in ('NA', 'NA.')]
    return ' '.join(facts)


# ---------------------------------------------------------------------------
# Constrained derangement  (same as build_tats_splits.py)
# ---------------------------------------------------------------------------

def constrained_derangement(texts: np.ndarray, rng: np.random.Generator,
                             max_attempts: int = 1000) -> np.ndarray:
    """
    Return permutation perm s.t. texts[perm[i]] != texts[i] for all i.

    Uses rejection sampling, then sort-and-shift, then cyclic shift.
    """
    N   = len(texts)
    arr = np.array(texts)

    for _ in range(max_attempts):
        perm = rng.permutation(N)
        if np.all(arr[perm] != arr):
            return perm

    _, counts = np.unique(arr, return_counts=True)
    max_count = int(counts.max())
    if max_count > N - max_count:
        warnings.warn(
            f"Constrained derangement impossible "
            f"({max_count}/{N} positions share the same text); "
            "using cyclic shift."
        )
        return np.roll(np.arange(N), N // 2 + 1)

    sort_order = np.argsort(arr, kind="stable")
    ranks = np.empty(N, dtype=int)
    ranks[sort_order] = np.arange(N)
    for shift in range(max_count, N):
        candidate = sort_order[(ranks + shift) % N]
        if np.all(arr[candidate] != arr):
            return candidate

    warnings.warn("Sort-and-shift fallback exhausted; using cyclic shift.")
    return np.roll(np.arange(N), N // 2 + 1)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic):  return obj.item()
        return super().default(obj)


# ---------------------------------------------------------------------------
# 70 / 10 / 20 split — mirrors Dataset_Custom exactly
# ---------------------------------------------------------------------------

def _split_borders(n: int):
    """
    Return (border1s, border2s) for train / val / test.

    No seq_len overlap — the full N-row dense signal in metadata handles
    cross-split lookbacks at runtime, just as Dataset_Custom's extended
    val/test borders do.
    """
    num_train = int(n * 0.7)
    num_test  = int(n * 0.2)
    num_vali  = n - num_train - num_test
    border1s  = [0,         num_train,             num_train + num_vali]
    border2s  = [num_train, num_train + num_vali,  n                   ]
    return border1s, border2s


# ---------------------------------------------------------------------------
# Load one MM-TSFlib CSV
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path, text_col: str, clean: bool) -> dict:
    """
    Return {
        't':     list[int]    unix-second timestamps (all rows, sorted)
        'y':     list[float]  OT values
        'texts': list[str]    cleaned text strings (empty string if missing)
    }
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in ['date', 'OT', text_col] if c not in df.columns]
    if missing:
        avail = [c for c in df.columns if 'Search' in c or 'Output' in c]
        raise ValueError(f"Missing columns {missing}. Available text cols: {avail}")

    df['_t'] = pd.to_datetime(df['date']).astype('int64') // 10**9
    df = df.sort_values('_t').reset_index(drop=True)

    texts = (
        [clean_search_text(v) for v in df[text_col]]
        if clean
        else [str(v) if pd.notna(v) else "" for v in df[text_col]]
    )

    return {
        't':     df['_t'].tolist(),
        'y':     [float(v) for v in df['OT']],
        'texts': texts,
    }


# ---------------------------------------------------------------------------
# Build and write splits for one dataset
# ---------------------------------------------------------------------------

def write_splits(
    name:            str,
    data:            dict,
    freq:            str,
    out_dir:         Path,
    text_col:        str,
    irr_pools:       dict,   # {split_name: list[str]} from OTHER datasets
    rng:             np.random.Generator,
    drop_empty_text: bool,
    min_points:      int,
) -> dict:
    """Write train/val/test JSONs; return per-split point counts for summary."""
    t_all     = data['t']
    y_all     = data['y']
    texts_all = data['texts']
    n         = len(t_all)
    fs        = FREQ_SETTINGS[freq]

    border1s, border2s = _split_borders(n)
    split_names = ['train', 'val', 'test']
    counts: dict[str, int] = {}

    for s_idx, split_name in enumerate(split_names):
        b1, b2 = border1s[s_idx], border2s[s_idx]
        idxs   = list(range(b1, b2))

        if drop_empty_text:
            idxs = [i for i in idxs if texts_all[i]]

        if len(idxs) < min_points:
            counts[split_name] = 0
            continue

        split_texts = np.array([texts_all[i] for i in idxs], dtype=object)
        blank_count = int((split_texts == "").sum())
        blank_ratio = blank_count / len(idxs)

        # ── incorrect: constrained derangement within this split ─────────────
        perm = constrained_derangement(split_texts, rng)
        incorrect_texts = split_texts[perm]
        collisions = int(np.sum(incorrect_texts == split_texts))
        if collisions:
            warnings.warn(
                f"[{name}/{split_name}] {collisions}/{len(idxs)} collision(s) "
                "in incorrect condition (unavoidable given label distribution)."
            )

        # ── irrelevant: uniform sample from other datasets' same split ────────
        pool = irr_pools.get(split_name, [])
        if not pool:
            warnings.warn(
                f"[{name}/{split_name}] irrelevant pool is empty "
                "(only one dataset?); falling back to own texts."
            )
            pool = split_texts.tolist()
        pool_arr         = np.array(pool, dtype=object)
        irrelevant_texts = pool_arr[rng.integers(0, len(pool_arr), size=len(idxs))]

        # ── assemble points ───────────────────────────────────────────────────
        points = []
        for local_i, global_i in enumerate(idxs):
            t_i = t_all[global_i]
            y_i = y_all[global_i]
            points.append({"t": t_i, "y": y_i, "category": "correct",
                           "samples": [{"text": str(split_texts[local_i])}]})
            points.append({"t": t_i, "y": y_i, "category": "incorrect",
                           "samples": [{"text": str(incorrect_texts[local_i])}]})
            points.append({"t": t_i, "y": y_i, "category": "irrelevant",
                           "samples": [{"text": str(irrelevant_texts[local_i])}]})

        payload = {
            "metadata": {
                "split":        split_name,
                "text_col":     text_col,
                "freq":         freq,
                "seq_len":      fs['seq_len'],
                "label_len":    fs['label_len'],
                "pred_lengths": fs['pred_lengths'],
                "n_points":     len(idxs),
                "blank_points": blank_ratio,
                "t": t_all,
                "y": y_all,
            },
            "points": points,
        }

        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, cls=_NumpyEncoder)

        counts[split_name] = len(idxs)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build mmtt-bench CFA-style TimeMMD annotation-point splits "
                    "(correct / incorrect / irrelevant, 70/10/20 split).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mm-tsf-root",     default=str(DEFAULT_MM_TSF_ROOT), metavar="DIR",
                    help=f"MM-TSFlib data root (default: {DEFAULT_MM_TSF_ROOT})")
    ap.add_argument("--output-dir",      required=True,
                    help="Directory to write per-dataset split JSONs")
    ap.add_argument("--text-len",        type=int, default=4, choices=[2, 4, 6],
                    help="Which Final_Search_N column to use (default: 4, CFA default)")
    ap.add_argument("--no-clean-text",   action="store_true",
                    help="Skip clean_search_text() boilerplate stripping")
    ap.add_argument("--drop-empty-text", action="store_true",
                    help="Omit rows with empty text entirely (default: keep as \"\")")
    ap.add_argument("--min-points",      type=int, default=50,
                    help="Skip splits with fewer usable rows than this (default: 50)")
    ap.add_argument("--seed",            type=int, default=42,
                    help="RNG seed for derangement and irrelevant sampling")

    args     = ap.parse_args()
    rng      = np.random.default_rng(args.seed)
    text_col = f"Final_Search_{args.text_len}"
    clean    = not args.no_clean_text
    out_root = Path(args.output_dir)
    root     = Path(args.mm_tsf_root)

    print(f"Source       : {root}")
    print(f"Text column  : {text_col}  (clean_search_text={'on' if clean else 'off'})")
    print(f"Split        : 70 / 10 / 20  (train / val / test)")
    print(f"Categories   : correct, incorrect, irrelevant")
    print(f"Output       : {out_root}")
    print()
    print("Frequency settings (hardcoded from run_timemmd.sh):")
    for freq, fs in FREQ_SETTINGS.items():
        print(f"  {freq:<8}  seq_len={fs['seq_len']:>3}  label_len={fs['label_len']:>2}"
              f"  pred_lengths={fs['pred_lengths']}")
    print()

    # ── Windowing explanation ─────────────────────────────────────────────────
    print("CFA windowing vs mmtt-bench annotation points")
    print("─" * 60)
    print("CFA (Dataset_Custom) slides a seq_len window one step at a time,")
    print("extending val/test start back by seq_len rows for full lookback context.")
    print("  CFA windows: train = num_train - seq_len - pred_len + 1")
    print("               val   = num_vali  - pred_len + 1")
    print("               test  = num_test  - pred_len + 1")
    print()
    print("This script produces ONE annotation point per split timestep:")
    print("  Annotation pts: train = int(N*0.7),  val = remainder,  test = int(N*0.2)")
    print("The full N-row signal is stored in metadata so build_category_data()")
    print("and MmttDataset recover any lookback at runtime, including cross-split")
    print("overlap — identical to Dataset_Custom's extended val/test borders.")
    print()

    # ── Load all datasets ─────────────────────────────────────────────────────
    loaded: dict[str, dict] = {}
    freqs:  dict[str, str]  = {}
    for name, subdir, csv, freq in DATASETS:
        csv_path = root / subdir / csv
        try:
            loaded[name] = load_csv(csv_path, text_col, clean)
            freqs[name]  = freq
        except Exception as exc:
            print(f"SKIPPED {name}: {exc}")

    if not loaded:
        raise SystemExit("All datasets failed to load.")

    # ── Pre-build irrelevant text pools ───────────────────────────────────────
    split_texts_per_ds: dict[str, dict[str, list[str]]] = {}
    for name, data in loaded.items():
        b1s, b2s = _split_borders(len(data['t']))
        split_texts_per_ds[name] = {
            sn: [data['texts'][i]
                 for i in range(b1s[si], b2s[si])
                 if data['texts'][i]]
            for si, sn in enumerate(['train', 'val', 'test'])
        }

    irr_pools: dict[str, dict[str, list[str]]] = {
        name: {
            sn: [t
                 for other, pools in split_texts_per_ds.items()
                 if other != name
                 for t in pools.get(sn, [])]
            for sn in ['train', 'val', 'test']
        }
        for name in loaded
    }

    # ── Write per-dataset split JSONs ─────────────────────────────────────────
    summary: dict[str, dict] = {}
    for name, data in loaded.items():
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = write_splits(
            name            = name,
            data            = data,
            freq            = freqs[name],
            out_dir         = out_dir,
            text_col        = text_col,
            irr_pools       = irr_pools[name],
            rng             = rng,
            drop_empty_text = args.drop_empty_text,
            min_points      = args.min_points,
        )
        summary[name] = {"n": len(data['t']), "freq": freqs[name], **counts}

    # ── Summary table ─────────────────────────────────────────────────────────
    # Show annotation point counts alongside the CFA sliding-window counts for
    # the shortest and longest pred_len in each frequency band.
    print()
    print("Annotation points per dataset  (3 categories × n_points in each JSON)")
    print(f"{'Dataset':<13} {'freq':<8} {'N rows':>7}  "
          f"{'ann train':>9}  {'ann val':>7}  {'ann test':>8}")
    print("─" * 60)
    for name, s in summary.items():
        freq = s['freq']
        fs   = FREQ_SETTINGS[freq]
        n    = s['n']
        tr   = s.get('train', 0)
        vl   = s.get('val',   0)
        te   = s.get('test',  0)
        print(f"  {name:<11} {freq:<8} {n:>7}  "
              f"  {str(tr or 'skip'):>7}    {str(vl or 'skip'):>5}    {str(te or 'skip'):>6}")

    print()
    print("Equivalent CFA sliding windows  (pred-len dependent)")
    print(f"{'Dataset':<13} {'seq':>4}  "
          f"{'train [min–max pred]':>22}  {'val':>16}  {'test':>16}")
    print("─" * 76)
    for name, s in summary.items():
        freq = s['freq']
        fs   = FREQ_SETTINGS[freq]
        n    = s['n']
        seq  = fs['seq_len']
        pmin, pmax = fs['pred_lengths'][0], fs['pred_lengths'][-1]
        b1s, b2s  = _split_borders(n)
        tr_rows = b2s[0] - b1s[0]
        vl_rows = b2s[1] - b1s[1]
        te_rows = b2s[2] - b1s[2]
        # CFA windows: train = tr_rows - seq - pred + 1; val/test = rows - pred + 1
        tr_hi = max(0, tr_rows - seq - pmin + 1)
        tr_lo = max(0, tr_rows - seq - pmax + 1)
        vl_hi = max(0, vl_rows - pmin + 1)
        vl_lo = max(0, vl_rows - pmax + 1)
        te_hi = max(0, te_rows - pmin + 1)
        te_lo = max(0, te_rows - pmax + 1)
        print(f"  {name:<11} {seq:>4}  "
              f"  {f'{tr_lo}–{tr_hi}':>20}  {f'{vl_lo}–{vl_hi}':>14}  {f'{te_lo}–{te_hi}':>14}")
    print(f"  (window range = shortest–longest pred_len per freq:"
          f" monthly {FREQ_SETTINGS['monthly']['pred_lengths'][0]}–"
          f"{FREQ_SETTINGS['monthly']['pred_lengths'][-1]},"
          f" weekly {FREQ_SETTINGS['weekly']['pred_lengths'][0]}–"
          f"{FREQ_SETTINGS['weekly']['pred_lengths'][-1]})")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
