#!/usr/bin/env python3
"""
ttc.py

Convert the local Multimodal_Forecasting CSV source
(/root/emma-code/Multimodal_Forecasting/data) to mmtt-bench
annotation-point JSON splits, matching that repo's own split methodology
(utils/dataset.py: get_embedding_loader / get_embedding_loader_from_list)
so results stay comparable to anything already run against it.

Source
------
  climate_2014_2023_final.csv  — one row per date, columns:
      date, temp, precip, humidity, windspeed, text
    A single clean daily series, 2014-01-01..2023-12-01, verified no
    gaps/duplicate dates/empty text. y = temp; the other numeric columns
    are available in the CSV but unused here, matching the single-target
    convention used elsewhere in this codebase and matching
    config/baseline/climate.yaml's numerical_columns: [temp].

  medical/patient_<id>.csv  x 73  — one row per (patient, date), columns:
      date, Respiratory_Rate, Heart_Rate, SaO2, FiO2, text
    Each file is one patient's own short (41-167 day, median 110) daily
    stay with its own de-identified date offset. y = Heart_Rate, matching
    config/baseline/medical.yaml's numerical_columns: [Heart_Rate].

Splitting — matches utils/dataset.py exactly
-----------------------------------------------
Both configs use train_split=0.8, valid_split=0.9 (an 80/10/10 cut,
train_n=int(n*0.8), valid_n=int(n*0.9), chronological):

  climate — get_embedding_loader: one dataframe, cut once.
  medical — get_embedding_loader_from_list: df_list = sorted(glob(dir/*)),
            and EACH patient's own dataframe is cut 80/10/10
            independently, then all patients' train slices (resp.
            valid/test) are pooled via ConcatDataset. Point-level windows
            (input_window/output_window) are computed per patient
            *before* pooling, so a window never crosses a patient
            boundary — patients are never mixed at the raw-signal level,
            only after each has been independently windowed.

mmtt-bench's own windowing (build_category_data's lookback/horizon slicing
of metadata['t']/['y'] by array position, driven by lookback_steps /
horizon_steps chosen later in the run.py/train.py config) works the same
way here: each patient gets its OWN split JSON files
(<output-dir>/medical_1/<patient_id>/{train,val,test}.json), so windowing
never has the chance to cross a patient boundary — no margin-trimming or
synthetic timestamps needed (an earlier version of this script pooled all
patients into one dense array per split and worked around the resulting
cross-patient window risk that way; this is simpler and matches the
source repo directly). run.py's existing multi-dataset mode consumes a
`datasets:` list of paths and writes one output JSON per dataset (named
from the parent directory), so per-patient results can be computed with a
single run.py invocation; averaging across patients is left to the
caller ("treat each patient as a completely new dataset and average at
the end").

Per-patient N is small (median ~88 train / ~11 val / ~11 test rows) —
well under KSG's ~400-point reliability floor for a single dataset, so
per-patient MI estimates will be individually noisy. That's expected: the
point of computing 73 independent estimates and averaging is to get a
usable aggregate despite that, the same way get_embedding_loader_from_list
pools many short per-patient *training* sets rather than expecting any
one of them to be sufficient alone.

Categories
----------
  correct    — the date's own text
  incorrect  — constrained derangement (medical: within each patient's own
               rows, since patients are now processed independently;
               climate: within the split)
  irrelevant — sampled from the other dataset's pooled text for the same
               split (a medical patient's irrelevant text is sampled from
               climate's same split; climate's irrelevant text is sampled
               from the pooled text of all medical patients' same split)

Usage
-----
    python src/mmtt_bench/data_generation/ttc.py \\
        --output-dir datasets/ttc_splits

Output
------
  <output-dir>/
    climate_1/                    train.json  val.json  test.json
    medical_1/<patient_id>/       train.json  val.json  test.json   (x73)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Source paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("/root/emma-code/Multimodal_Forecasting/data")
CLIMATE_CSV = DATA_ROOT / "climate_2014_2023_final.csv"
MEDICAL_DIR = DATA_ROOT / "medical"

TRAIN_SPLIT = 0.8   # matches Multimodal_Forecasting's train_split
VALID_SPLIT = 0.9   # matches Multimodal_Forecasting's valid_split

MIN_TRAIN_POINTS         = 400   # KSG-viability floor for climate (one big dataset)
MIN_PATIENT_TRAIN_POINTS = 20    # sanity floor only — per-patient N is expected to be small


# ---------------------------------------------------------------------------
# Constrained derangement (same as timemmd.py / build_tats_splits.py)
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
# 80/10/10 helper (matches utils/dataset.py exactly)
# ---------------------------------------------------------------------------

def _split_sizes(n: int) -> tuple[int, int, int]:
    train_n = int(n * TRAIN_SPLIT)
    valid_n = int(n * VALID_SPLIT)
    return train_n, valid_n - train_n, n - valid_n  # (n_train, n_val, n_test)


def _write_split(ds_dir: Path, split: str,
                 t: list[int], y: list[float], points: list[dict]) -> None:
    metadata = {"split": split, "n_points": len(points) // 3, "blank_points": 0.0,
               "t": t, "y": y}
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "points": points}, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved {len(points)} points -> {out_path}")


def _build_points(t: list[int], y: list[float], texts: np.ndarray,
                  other_texts: list[str], rng: np.random.Generator) -> list[dict]:
    points = [{"t": ti, "y": yi, "category": "correct", "samples": [{"text": txt}]}
              for ti, yi, txt in zip(t, y, texts)]

    perm = constrained_derangement(texts, rng)
    points += [{"t": ti, "y": yi, "category": "incorrect", "samples": [{"text": txt}]}
              for ti, yi, txt in zip(t, y, texts[perm])]

    points += [{"t": ti, "y": yi, "category": "irrelevant",
               "samples": [{"text": str(rng.choice(other_texts))}]}
              for ti, yi in zip(t, y)]

    return points


# ---------------------------------------------------------------------------
# Climate: single clean daily series, chronological 80/10/10
# ---------------------------------------------------------------------------

def _load_climate() -> pd.DataFrame:
    df = pd.read_csv(CLIMATE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert not df["date"].duplicated().any(), "climate CSV has duplicate dates"
    assert not (df["text"].isna() | (df["text"] == "")).any(), "climate CSV has empty text"
    return df


def _climate_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n_train, n_val, n_test = _split_sizes(len(df))
    return {
        "train": df.iloc[:n_train],
        "val":   df.iloc[n_train:n_train + n_val],
        "test":  df.iloc[n_train + n_val:],
    }


def _process_climate(out_dir: Path, medical_text_pool: dict[str, list[str]],
                     rng: np.random.Generator) -> dict[str, list[str]]:
    print("\n== climate ==")
    df = _load_climate()
    splits = _climate_splits(df)
    text_pool = {}
    ds_dir = out_dir / "climate_1"

    for split, sdf in splits.items():
        text_pool[split] = sdf["text"].tolist()
        if split == "train" and len(sdf) < MIN_TRAIN_POINTS:
            print(f"  Skipping train — only {len(sdf)} points (minimum {MIN_TRAIN_POINTS})")
            continue
        t     = [int(d.value // 10**9) for d in sdf["date"]]
        y     = sdf["temp"].astype(float).tolist()
        texts = sdf["text"].to_numpy()
        points = _build_points(t, y, texts, medical_text_pool[split], rng)
        _write_split(ds_dir, split, t, y, points)

    return text_pool


# ---------------------------------------------------------------------------
# Medical: 73 independent per-patient series, each its own 80/10/10 cut
# ---------------------------------------------------------------------------

def _load_medical_patients() -> dict[str, pd.DataFrame]:
    patients = {}
    for f in sorted(MEDICAL_DIR.glob("patient_*.csv")):
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        patients[f.stem] = df
    return patients


def _process_medical(out_dir: Path, climate_text_pool: dict[str, list[str]],
                     rng: np.random.Generator) -> dict[str, list[str]]:
    print("\n== medical ==")
    patients = _load_medical_patients()
    print(f"  {len(patients)} patients")
    text_pool = {"train": [], "val": [], "test": []}
    skipped = {"train": 0, "val": 0, "test": 0}

    for pid, df in patients.items():
        n_train, n_val, n_test = _split_sizes(len(df))
        splits = {
            "train": df.iloc[:n_train],
            "val":   df.iloc[n_train:n_train + n_val],
            "test":  df.iloc[n_train + n_val:],
        }
        ds_dir = out_dir / "medical_1" / pid

        for split, sdf in splits.items():
            text_pool[split].extend(sdf["text"].tolist())
            floor = MIN_PATIENT_TRAIN_POINTS if split == "train" else 1
            if len(sdf) < floor:
                skipped[split] += 1
                continue
            t     = [int(d.value // 10**9) for d in sdf["date"]]
            y     = sdf["Heart_Rate"].astype(float).tolist()
            texts = sdf["text"].to_numpy()
            points = _build_points(t, y, texts, climate_text_pool[split], rng)
            _write_split(ds_dir, split, t, y, points)

    for split, n in skipped.items():
        if n:
            print(f"  {split}: skipped {n}/{len(patients)} patients (too few rows)")

    return text_pool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert the local Multimodal_Forecasting CSV source to "
                    "mmtt-bench annotation-point JSON splits, using that repo's "
                    "own 80/10/10 split methodology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", default="datasets/ttc_splits",
                    help="Directory to write climate_1/ and medical_1/<patient>/ splits")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for derangement and irrelevant sampling (default: 42)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    # Two passes: text pools need each other's per-split texts for irrelevant
    # sampling, so build both raw split tables before writing any points.
    climate_df    = _load_climate()
    climate_split_tables = _climate_splits(climate_df)
    climate_pool  = {s: sdf["text"].tolist() for s, sdf in climate_split_tables.items()}

    patients = _load_medical_patients()
    medical_pool = {"train": [], "val": [], "test": []}
    for df in patients.values():
        n_train, n_val, n_test = _split_sizes(len(df))
        medical_pool["train"].extend(df["text"].iloc[:n_train].tolist())
        medical_pool["val"].extend(df["text"].iloc[n_train:n_train + n_val].tolist())
        medical_pool["test"].extend(df["text"].iloc[n_train + n_val:].tolist())

    _process_climate(out_dir, medical_pool, rng)
    _process_medical(out_dir, climate_pool, rng)


if __name__ == "__main__":
    main()
