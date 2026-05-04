import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EMPTY_STRING_FILL = "Sorry, no available information"

T_COL   = 'Date'
Y_COL   = 'OT'
TEXT_COL = 'Final_Search_2'


def clean_search_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^Available facts are as follows:\s*', '', text)
    text = text.rstrip(';').strip()
    facts = re.split(r'\d{4}-\d{2}-\d{2}:\s*', text)
    facts = [f.strip().rstrip('.').strip() for f in facts
             if f.strip() and f.strip() not in ('NA', 'NA.')]
    return ' '.join(facts)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic):  return obj.item()
        return super().default(obj)


def _create_irrelevant_point(row, path: str, text_dict: dict,
                              rng: np.random.Generator) -> dict:
    other_paths = [p for p in text_dict if p != path]
    irr_text = rng.choice(text_dict[rng.choice(other_paths)])
    if irr_text == "":
        irr_text = EMPTY_STRING_FILL
    return {
        "t":        int(row[T_COL]),
        "y":        float(row[Y_COL]),
        "category": "irrelevant",
        "samples":  [{"text": irr_text}],
    }


def process_dataset(path: str, text_dict: dict, rng: np.random.Generator,
                    out_dir: Path) -> None:
    df = pd.read_csv(path)
    t_col = T_COL if T_COL in df.columns else T_COL.lower()

    df[t_col]   = pd.to_datetime(df[t_col]).astype('int64') // 10**9
    df[TEXT_COL] = df[TEXT_COL].apply(clean_search_text)
    df = df[[t_col, Y_COL, TEXT_COL]].sort_values(by=t_col)

    # Rename to canonical column names so downstream code is uniform
    df = df.rename(columns={t_col: T_COL})

    split_idx = int(len(df) * 0.85)
    if split_idx < 400:
        print(f"Skipping {path} — train split has only {split_idx} rows (minimum 400)")
        return

    stem = Path(path).stem
    splits = [("train", df[:split_idx]), ("val", df[split_idx:])]

    for split_name, split_df in splits:
        points = [
            {
                "t":        int(row[T_COL]),
                "y":        float(row[Y_COL]),
                "category": "correct",
                "samples":  [{"text": row[TEXT_COL] if row[TEXT_COL] != "" else EMPTY_STRING_FILL}],
            }
            for _, row in split_df.iterrows()
        ]
        points += [
            _create_irrelevant_point(row, path, text_dict, rng)
            for _, row in split_df.iterrows()
        ]

        metadata = {
            "split":        split_name,
            "n_points":     len(split_df),
            "blank_points": (split_df[TEXT_COL] == "").sum() / len(split_df),
            "t":            df[T_COL].tolist(),
            "y":            df[Y_COL].tolist(),
        }

        out_path = out_dir / f"{stem}_{split_name}.json"
        with open(out_path, 'w') as f:
            json.dump({'metadata': metadata, 'points': points},
                      f, indent=2, cls=_NumpyEncoder)
        print(f"Saved {len(points)} points → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TimeMMD CSV files to benchmark split format."
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing TimeMMD CSV files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write <name>_train.json and <name>_val.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for irrelevant text sampling (default: 42)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(glob.glob(str(input_dir / "*.csv")))
    if not data_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # Pre-load all text columns so irrelevant sampling can draw from other datasets
    text_dict: dict[str, list] = {}
    for path in data_paths:
        df = pd.read_csv(path)
        text_dict[path] = df[TEXT_COL].apply(clean_search_text).tolist()

    rng = np.random.default_rng(seed=args.seed)

    for path in data_paths:
        print(f"\nProcessing {path}")
        process_dataset(path, text_dict, rng, output_dir)
