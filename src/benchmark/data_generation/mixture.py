"""
mixture.py
----------
Creates a single combined benchmark split directory that augments an existing
dataset with mixture categories for MI analysis.

The output keeps the original correct / incorrect / irrelevant points for every
annotation time-point, then adds one extra point per (contam_type, level)
combination.  Each extra point is a copy of either the correct or contaminating-
category point (depending on the random draw), with its `category` field
renamed to e.g. `mixture_25_incorrect`.

Because build_category_data() extracts categories dynamically from the data,
all mixture categories are picked up automatically alongside the originals.

Usage
-----
    python mixture.py --source benchmark_splits_sinehalf

Output
------
    benchmark_splits_sinehalf_mixtures/
        train.json   — original points + 6 mixture categories per annotation time
        val.json
        test.json
"""

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path


CONTAMINATION_LEVELS = [25, 50, 75]
CONTAMINATION_TYPES  = ['incorrect', 'irrelevant']
SPLITS               = ['train', 'val', 'test']


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  saved {len(data['points']):>5} points → {path}")


def _build_combined_split(
    points:  list[dict],
    metadata: dict,
    levels:  list[int],
    types:   list[str],
    seed:    int = 0,
) -> dict:
    """
    Return a split dict containing the original points plus one extra point
    per annotation time per (contam_type, level) combination.

    Extra points are copies of either the correct or contaminating-category
    point, with `category` set to `mixture_{n}_{contam_type}`.
    """
    by_time: dict[float, dict[str, dict]] = defaultdict(dict)
    for pt in points:
        by_time[pt['t']][pt['category']] = pt

    times = sorted(by_time.keys())

    # Pre-compute which times are contaminated for each (type, level) combo.
    # Use a deterministic seed derived from (type, level) so different combos
    # don't share the same random selection.
    contam_sets: dict[tuple, set] = {}
    for contam_type in types:
        for n_pct in levels:
            n_contam = math.ceil(len(times) * n_pct / 100)
            rng = random.Random(seed + hash((contam_type, n_pct)) % (2**32))
            contam_sets[(contam_type, n_pct)] = set(rng.sample(times, n_contam))

    # Start with all original points
    combined = list(points)

    # Add one mixture point per annotation time per combo
    for contam_type in types:
        for n_pct in levels:
            cat_name = f'mixture_{n_pct}_{contam_type}'
            contam_times = contam_sets[(contam_type, n_pct)]
            n_contam_actual = 0
            for t in times:
                group = by_time[t]
                src_cat = contam_type if t in contam_times else 'correct'
                if src_cat in group:
                    pt = copy.deepcopy(group[src_cat])
                    pt['category'] = cat_name
                    combined.append(pt)
                    if src_cat == contam_type:
                        n_contam_actual += 1

            pct_actual = 100 * n_contam_actual / len(times)
            print(f"    {cat_name:<28}  {n_contam_actual}/{len(times)} "
                  f"contaminated ({pct_actual:.0f}%)")

    combined.sort(key=lambda p: (p['t'], p['category']))

    from collections import Counter
    cats = Counter(p['category'] for p in combined)
    print(f"    categories: {dict(cats)}")

    return {'metadata': metadata, 'points': combined}


def make_combined(
    source_dir:  str,
    output_dir:  str | None = None,
    levels:      list[int]  = CONTAMINATION_LEVELS,
    types:       list[str]  = CONTAMINATION_TYPES,
    seed:        int        = 0,
) -> None:
    src = Path(source_dir)
    out = Path(output_dir) if output_dir else Path(f"{src.name}_mixtures")

    print(f"Output → {out}/")
    for split in SPLITS:
        split_path = src / f"{split}.json"
        if not split_path.exists():
            print(f"  skipping {split} (not found)")
            continue
        print(f"\n  {split}:")
        raw = _load(split_path)
        combined = _build_combined_split(
            raw['points'],
            raw['metadata'],
            levels=levels,
            types=types,
            seed=seed,
        )
        _save(combined, out / f"{split}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment a benchmark split with mixture categories.')
    parser.add_argument('--source',     required=True,
                        help='Source benchmark split directory')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <source>_mixtures)')
    parser.add_argument('--levels',  nargs='+', type=int, default=CONTAMINATION_LEVELS,
                        help='Contamination percentages (default: 25 50 75)')
    parser.add_argument('--types',   nargs='+', default=CONTAMINATION_TYPES,
                        help='Contamination categories (default: incorrect irrelevant)')
    parser.add_argument('--seed',    type=int, default=0)
    args = parser.parse_args()

    make_combined(
        source_dir=args.source,
        output_dir=args.output_dir,
        levels=args.levels,
        types=args.types,
        seed=args.seed,
    )
