#!/usr/bin/env python3
"""
fintexts.py

Convert the local FinTexTS dataset (/root/emma-code/FinTexTS, 100 tickers,
one HF `datasets`-format arrow dir per ticker) to mmtt-bench annotation-
point JSON splits — one independent dataset per ticker, matching the
per-patient design used for TTC's medical data ("treat each stock as a
completely new dataset, average at the end").

Source
------
  <TICKER>/{train,validation,test}/  — HF `Dataset.load_from_disk()` dirs.
    Columns: date, ticker, open, high, low, close, plus 21 text columns
    across 5 groups: macro_category1-5, sector_category1-5,
    targetCompany_category1-3, relatedCompany_category1-3, filing_* x5
    (financialStatement/governanceRisks/overviewProduct/
    recentEventCatalyst/strategyMarketOps).

    Every ticker shares an identical 1304-row daily trading calendar
    (2019-01-01..2023-12-29, verified — no gaps beyond weekends), already
    split on disk train=784 (<2022-01-01) / validation=260 (2022) /
    test=260 (>=2023-01-01), matching fintexts.ipynb's own chronological
    cutoff. Used as-is — already split, no re-cutting.

    Most text fields are populated almost every day (macro/sector/filing
    missing ~0-4%); targetCompany_category1-3 are missing 27-61% of days
    (company-specific news doesn't happen daily). No day has all 21
    fields missing. y = close, matching the single-target convention used
    elsewhere in this codebase (open/high/low are available but unused).

Text
----
Matches the FinTexTS paper's actual architecture (arXiv:2603.02702, Sec 4.1):
"summarize the news content for each category using SBERT, and then apply
mean pooling across all categories and levels to obtain a final text
representation" — i.e. embed each of the 21 fields SEPARATELY, then mean-pool
the resulting embedding vectors. NOT concatenate-then-embed-once (an earlier
version of this script did that; it needed a long-context model to avoid
truncating the ~3000-token concatenation, and plausibly diluted/mixed the
per-field signal into one blob).

Each point's "samples" list holds all 21 fields, one text per sample, missing
fields as "" (not omitted) — every point needs the SAME sample count, since
_pool_by_annotation_point (metrics.py) reads n_samples from the first point in
a category and reshapes the whole category's embeddings by it; a variable
count per point would silently misalign the reshape. This makes the existing
mmtt-bench pipeline do the paper's per-field-embed-then-mean-pool for free —
_attach_text_embeddings embeds every sample independently and
_pool_by_annotation_point/​_build_category_data average them per point, in
both run.py's MI pipeline (data.py) and train_transformer.py's MmttDataset
(patched to do the same, since it previously only read samples[0]).

Per-stock z-score normalisation
----------------------------------
Close prices vary by orders of magnitude across tickers (e.g. NVDA
~17.7+-12.5 vs TSLA ~170.8+-108.3 in this window) — pooling raw-scale MSE
across 100 stocks would be dominated by whichever ticker has the largest
price level, not whichever has the best fit. y is z-scored per ticker
using ONLY that ticker's train-split mean/std (applied unchanged to val
and test — never fit normalisation on val/test) before being written to
the JSON, so every downstream consumer (run.py's MI estimators, train.py's
MSE/MAE/R^2, already computed in train.py's _regression_metrics) operates
on comparable units with no code changes needed elsewhere. mu/sigma are
recorded in each split's metadata for de-normalising if ever needed.

Categories
----------
  correct    — the day's own 21-field bundle
  incorrect  — a different day's whole 21-field bundle (constrained
               derangement within that ticker's own split, at the bundle
               level — not per-field — so a shuffled point still reads as
               one coherent day's news, just the wrong day)
  irrelevant — a random OTHER ticker's whole 21-field bundle from the same
               split (excludes the ticker's own bundles)

Usage
-----
    python src/mmtt_bench/data_generation/fintexts.py \\
        --output-dir datasets/fintexts_splits

Output
------
  <output-dir>/fintexts_1/<TICKER>/{train,val,test}.json   (x100)

Note: with 100 tickers x ~784 train rows x 3 categories, a full run.py
pass over every ticker embeds ~235k texts for train alone (plus val/test)
— expect this to take a while; run.py's `datasets:` multi-dataset mode is
the natural way to batch all 100 in one invocation.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("/root/emma-code/FinTexTS")
SPLIT_DIRS  = {"train": "train", "val": "validation", "test": "test"}
TARGET_COL  = "close"
TEXT_PREFIXES = ("macro_category", "sector_category", "targetCompany_category",
                 "relatedCompany_category", "filing_")

MIN_TRAIN_POINTS = 400  # KSG-viability floor; per-ticker train N=784 clears this


# ---------------------------------------------------------------------------
# Constrained derangement (same as timemmd.py / ttc.py)
# ---------------------------------------------------------------------------

def constrained_derangement(texts: np.ndarray, rng: np.random.Generator,
                             max_attempts: int = 1000) -> np.ndarray:
    """Return permutation perm s.t. texts[perm[i]] != texts[i] for all i."""
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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _list_tickers() -> list[str]:
    return sorted(p.name for p in DATA_ROOT.iterdir()
                 if p.is_dir() and (p / "train").exists())


def _load_ticker(ticker: str) -> dict[str, pd.DataFrame]:
    from datasets import Dataset
    out = {}
    for split, dirname in SPLIT_DIRS.items():
        df = Dataset.load_from_disk(str(DATA_ROOT / ticker / dirname)).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        out[split] = df.sort_values("date").reset_index(drop=True)
    return out


def _field_texts_df(df: pd.DataFrame, text_cols: list[str]) -> np.ndarray:
    """(n_rows, len(text_cols)) array of per-field text, "" where missing.

    One column per field rather than one concatenated string per row — each
    field becomes its own sample, embedded independently and mean-pooled
    downstream (see module docstring). Working off raw numpy arrays per
    column (not df.iterrows(), which reconstructs a full Series per row) for
    speed across a 27-column frame x ~130k rows.
    """
    n = len(df)
    out = np.empty((n, len(text_cols)), dtype=object)
    for j, c in enumerate(text_cols):
        col = df[c].to_numpy()
        out[:, j] = [str(v) if pd.notna(v) else "" for v in col]
    return out


def _bundle_keys(field_texts: np.ndarray) -> np.ndarray:
    """One opaque string per row, for constrained_derangement's content-
    equality check — cheap way to reuse it unmodified on whole bundles."""
    return np.array(["\x1f".join(row) for row in field_texts])


# ---------------------------------------------------------------------------
# Point construction
# ---------------------------------------------------------------------------

def _write_split(ds_dir: Path, split: str, t: list[int], y: list[float],
                 points: list[dict], mu: float, sigma: float) -> None:
    metadata = {"split": split, "n_points": len(points) // 3, "blank_points": 0.0,
               "t": t, "y": y, "zscore_mu": mu, "zscore_sigma": sigma}
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "points": points}, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved {len(points)} points -> {out_path}")


def _bundle_to_samples(row: np.ndarray) -> list[dict]:
    return [{"text": txt} for txt in row]


def _build_points(t: list[int], y: list[float], field_texts: np.ndarray,
                  other_bundles: list[np.ndarray], rng: np.random.Generator) -> list[dict]:
    points = [{"t": ti, "y": yi, "category": "correct", "samples": _bundle_to_samples(row)}
              for ti, yi, row in zip(t, y, field_texts)]

    # Derange whole bundles (not per-field) so an "incorrect" point still
    # reads as one coherent day's news — just the wrong day.
    perm = constrained_derangement(_bundle_keys(field_texts), rng)
    points += [{"t": ti, "y": yi, "category": "incorrect", "samples": _bundle_to_samples(row)}
              for ti, yi, row in zip(t, y, field_texts[perm])]

    # Drawing a fresh rng.choice(other_bundles) per point would re-convert
    # the whole pool to an array each call (same trap as the old per-text
    # version) — draw all indices at once instead.
    irr_idx = rng.integers(0, len(other_bundles), size=len(t))
    points += [{"t": ti, "y": yi, "category": "irrelevant",
               "samples": _bundle_to_samples(other_bundles[idx])}
              for ti, yi, idx in zip(t, y, irr_idx)]

    return points


def _process_ticker(ticker: str, dfs: dict[str, pd.DataFrame], field_texts: dict[str, np.ndarray],
                    other_bundle_pool: dict[str, list[np.ndarray]], out_dir: Path,
                    rng: np.random.Generator) -> None:
    mu    = float(dfs["train"][TARGET_COL].mean())
    sigma = float(dfs["train"][TARGET_COL].std()) + 1e-8

    ds_dir = out_dir / "fintexts_1" / ticker
    for split, df in dfs.items():
        if split == "train" and len(df) < MIN_TRAIN_POINTS:
            print(f"  Skipping {ticker}/{split} — only {len(df)} points "
                 f"(minimum {MIN_TRAIN_POINTS})")
            continue
        t      = [int(d.value // 10**9) for d in df["date"]]
        y      = ((df[TARGET_COL].astype(float) - mu) / sigma).tolist()
        bundle = field_texts[split]
        points = _build_points(t, y, bundle, other_bundle_pool[split], rng)
        _write_split(ds_dir, split, t, y, points, mu, sigma)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert the local FinTexTS dataset to mmtt-bench "
                    "annotation-point JSON splits, one independent dataset per ticker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", default="datasets/fintexts_splits",
                    help="Directory to write fintexts_1/<ticker>/{train,val,test}.json")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for derangement and irrelevant sampling (default: 42)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    tickers = _list_tickers()
    print(f"Found {len(tickers)} tickers")

    print("Loading all tickers ...")
    all_dfs = {t: _load_ticker(t) for t in tickers}
    text_cols = [c for c in all_dfs[tickers[0]]["train"].columns
                if c.startswith(TEXT_PREFIXES)]
    print(f"{len(text_cols)} text fields per day, embedded separately and "
         f"mean-pooled downstream: {text_cols}")

    print("Building per-field text arrays + cross-ticker bundle pools ...")
    field_texts = {
        t: {split: _field_texts_df(df, text_cols) for split, df in dfs.items()}
        for t, dfs in all_dfs.items()
    }
    # list of per-row bundles (each a (21,) array), not a flat pool of
    # individual field texts — irrelevant sampling draws whole bundles.
    bundle_pool_by_split = {
        split: {t: list(field_texts[t][split]) for t in tickers}
        for split in SPLIT_DIRS
    }

    for i, ticker in enumerate(tickers):
        print(f"\n== {ticker} ({i + 1}/{len(tickers)}) ==")
        other_bundle_pool = {
            split: [row for other, bundles in bundle_pool_by_split[split].items()
                   if other != ticker for row in bundles]
            for split in SPLIT_DIRS
        }
        _process_ticker(ticker, all_dfs[ticker], field_texts[ticker], other_bundle_pool, out_dir, rng)


if __name__ == "__main__":
    main()