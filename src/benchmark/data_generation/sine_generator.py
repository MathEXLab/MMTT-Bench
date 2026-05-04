"""
generator.py
-----------------------
Generates a synthetic benchmark consisting of time-series data points annotated with natural language descriptions of their local behavior.
The time-series signal can be specified in the config, currently supports sine, square, sawtooth, and noisy sine waves.

Generation is based on a hybrid model:

1. Text templates define the semantic contract (ground truth anchor)
2. LLM providers paraphrase for surface diversity
3. Output is saved as JSON

Adding a new LLM provider:
Subclass BaseLLMProvider and implement the `complete(prompt) -> str` method,
then register it in the PROVIDERS dict at the bottom of this file.

"""

import argparse
import json
import numpy as np
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal


#from models import BaseLLMProvider, AnthropicProvider, OpenAIProvider, GeminiProvider, HuggingFaceProvider
from sine_templates import build_data_point, AnnotationPoint
from dotenv import load_dotenv
load_dotenv()   # loads .env from the project directory into os.environ


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


# Signal library
# Add new signal types here. Each function takes a time array `t` plus optional
# keyword arguments (passed through from the config's signal section).

def _sine(t: np.ndarray, **_) -> np.ndarray:
    return np.sin(t)

def _noisy_sine(t: np.ndarray, noise_std: float = 0.1, seed: Optional[int] = None,
                **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sin(t) + rng.normal(0, noise_std, len(t))

def _make_mask(
    t:             np.ndarray,
    transition_mask: float = 0.3,
    min_constant_len:   float = np.pi/2,
    max_constant_len:  float = 2*np.pi,
    seed:          Optional[int] = 42,
) -> np.ndarray:
    """
    Build the boolean mask used by _sine_with_constants.

    Exposed separately so callers can obtain the exact mask that was applied
    (e.g. for transition detection) without recomputing y.  Uses the same
    rng sequence as _sine_with_constants, so the mask is identical for equal args.
    """
    rng    = np.random.default_rng(seed)
    n      = len(t)
    target = min(int(transition_mask * n), n)
    mask   = np.zeros(n, dtype=bool)
    while mask.sum() < target:
        start_t = rng.uniform(float(t[0]), float(t[-1]))
        seg_len = rng.uniform(float(min_constant_len), float(max_constant_len))
        s = int(np.searchsorted(t, start_t))
        e = min(int(np.searchsorted(t, start_t + seg_len)), n)
        mask[s:e] = True
    return mask


def _sine_with_constants(
    t:             np.ndarray,
    transition_mask: float = 0.3,
    min_constant_len:  float = np.pi/2,
    max_constant_len:  float = 2*np.pi,
    seed:          Optional[int] = None,
    constant: Optional[float] = -0.5,
    **_,
) -> np.ndarray:
    """
    Sine wave with randomly placed zero (flatline) segments.

    Parameters
    ----------
    transition_mask : target fraction of t-values to set to zero (0..1)
    max_constant_len  : maximum length of each zero segment in t units;
                    each segment's length is drawn uniformly from [0, max_constant_len]
    seed          : random seed for reproducibility
    """
    y = np.sin(t).copy()
    y[_make_mask(t, transition_mask, min_constant_len,max_constant_len, seed)] = constant
    return y


SIGNAL_FUNCTIONS: dict[str, callable] = {
    "sine":            _sine,
    "noisy_sine":      _noisy_sine,
    "sine_with_constants": _sine_with_constants,
}

# Theoretical derivative of the underlying clean signal — used for annotation
# direction/magnitude so that noisy variants still describe the true pattern.
SIGNAL_DERIVATIVES: dict[str, callable] = {
    "sine":            lambda t, **_: np.cos(t),
    "noisy_sine":      lambda t, **_: np.cos(t),   # derivative of sin, not the noise
    "sine_with_constants": lambda t, **_: np.cos(t),   # underlying sine's derivative
}


def _find_transition_annotations(constant_mask: np.ndarray, indices: np.ndarray):
    """
    Scan annotation positions for zero-segment transitions.

    For each consecutive annotation pair (j, j+1), detects whether the
    signal transitions from non-zero to zero (before_drop) or from zero
    to non-zero (before_recovery) somewhere in the interval [indices[j],
    indices[j+1]).

    Parameters
    ----------
    constant_mask : boolean array (same length as the signal) — True where the
                signal was explicitly set to zero by _make_mask.  Pass
                the mask directly rather than inferring zeros from y values,
                which would also catch natural sine zeros at t=0, 2π, …

    Returns
    -------
    before_drop     : list of positions into `indices` — the annotation just
                      before the signal drops to zero
    before_recovery : list of positions into `indices` — the annotation just
                      before the signal recovers from zero
    """
    is_constant = constant_mask

    # Find all non-zero→zero and zero→non-zero transitions in the raw signal
    diff          = np.diff(is_constant.astype(np.int8))
    drop_locs     = np.where(diff > 0)[0]   # transition at (loc, loc+1)
    recovery_locs = np.where(diff < 0)[0]

    def _annot_before(loc):
        """Index into `indices` of the last annotation point ≤ loc."""
        j = int(np.searchsorted(indices, loc, side='right')) - 1
        return j if 0 <= j < len(indices) - 1 else None

    seen_drop     = set()
    seen_recovery = set()
    before_drop, before_recovery = [], []

    for loc in drop_locs:
        j = _annot_before(loc)
        if j is not None and j not in seen_drop:
            before_drop.append(j)
            seen_drop.add(j)

    for loc in recovery_locs:
        j = _annot_before(loc)
        if j is not None and j not in seen_recovery:
            before_recovery.append(j)
            seen_recovery.add(j)

    for drop, recovery in zip(before_drop, before_recovery):
        for j in np.arange(drop, recovery, 1):
            before_drop.append(j)

    return before_drop, before_recovery


def save_json(annotations: dict[AnnotationPoint], path: str,
              signal_type: str = "sine", n_samples: int=1) -> None:
    """Serialize annotations to a JSON file."""

    def _serialise(a: AnnotationPoint) -> dict:
        d = asdict(a)
        # asdict handles nested dataclasses automatically
        return d

    with open(path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "signal_type":     signal_type,
                    "n_annotations":   len(annotations),
                    "n_samples": n_samples,
                },
                "annotations": [_serialise(a) for a in annotations],
            },
            f,
            indent=2,
        )
    print(f"Saved {len(annotations)} annotations → {path}")


def determine_phase(t, period=2 * np.pi):
    """
    Determine phase from t position in the cycle.

    Uses t rather than y/dy so noisy signals always get an annotation.
    Quarter-cycle boundaries have 1/8-cycle tolerance for floating-point
    rounding in np.arange.
    """
    phase_frac = (t % period) / period   # 0..1
    if phase_frac < 0.125 or phase_frac >= 0.875:
        return "ascending_zero"
    elif phase_frac < 0.375:
        return "peak"
    elif phase_frac < 0.625:
        return "descending_zero"
    else:
        return "trough"
    

def assign_split(
    t: float,
    period: float = 2 * np.pi,
    train_cycles: tuple[int, int] = (1, 5),   # inclusive cycle numbers
    val_cycles:   tuple[int, int] = (6, 6),
    test_cycles:  tuple[int, int] = (7, 8),
) -> Literal["train", "val", "test"] | None:
    """
    Assign a time point to a split based on which cycle it falls in.
    Cycle numbering starts at 1.
    Returns None if t falls outside all defined ranges.

    Parameters
    ----------
    t            : time value
    period       : length of one full cycle (default 2π)
    train_cycles : (first, last) cycle numbers assigned to train
    val_cycles   : (first, last) cycle numbers assigned to val
    test_cycles  : (first, last) cycle numbers assigned to test
    """
    cycle_number = int(t // period) + 1   # 1-indexed

    if train_cycles[0] <= cycle_number <= train_cycles[1]:
        return "train"
    elif val_cycles[0] <= cycle_number <= val_cycles[1]:
        return "val"
    elif test_cycles[0] <= cycle_number <= test_cycles[1]:
        return "test"
    return None


def split_dataset(
    annotations: list[AnnotationPoint],
    period: float = 2 * np.pi,
    train_cycles: tuple[int, int] = (1, 5),
    val_cycles:   tuple[int, int] = (6, 6),
    test_cycles:  tuple[int, int] = (7, 8),
) -> dict[str, list[dict]]:
    """
    Split a list of AnnotationPoint objects into train/val/test
    by cycle number, preserving temporal order within each split.

    Parameters
    ----------
    annotations   : list of AnnotationPoint dataclass instances
    period        : cycle length (default 2π for a sine wave)
    train_cycles  : (first, last) inclusive cycle numbers for training
    val_cycles    : (first, last) inclusive cycle numbers for validation
    test_cycles   : (first, last) inclusive cycle numbers for test

    Returns
    -------
    dict with keys "train", "val", "test" — each a temporally ordered
    list of dicts (serialisation-ready)
    """
    splits = {"train": [], "val": [], "test": []}

    for point in annotations:
        split = assign_split(point.t, period, train_cycles, val_cycles, test_cycles)
        if split is not None:
            splits[split].append(asdict(point))

    for key in splits:
        splits[key].sort(key=lambda p: p["t"])

    return splits


def check_split_balance(splits: dict[str, list[dict]]) -> None:
    """
    Print a balance report: counts per split broken down by
    category, tense, and phase. Use this to verify no cell is empty.
    """
    from collections import Counter

    for split_name, points in splits.items():
        print(f"\n--- {split_name} ({len(points)} points) ---")

        cat_counts   = Counter(p["category"] for p in points)
        tense_counts = Counter(p["tense"]    for p in points)
        phase_counts = Counter(p["phase"]    for p in points)

        print("  Categories:")
        for k, v in sorted(cat_counts.items()):
            print(f"    {k:<25} {v}")
        print("  Tenses:")
        for k, v in sorted(tense_counts.items()):
            print(f"    {k:<25} {v}")
        print("  Phases:")
        for k, v in sorted(phase_counts.items()):
            print(f"    {k:<25} {v}")


def save_splits(
    splits: dict[str, list[dict]],
    output_dir: str = ".",
    t: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    period: float = 2 * np.pi,
    n_samples: int=1,
    train_cycles: tuple[int, int] = (1, 5),
    val_cycles:   tuple[int, int] = (6, 6),
    test_cycles:  tuple[int, int] = (7, 8),
) -> None:
    """Save each split to a separate JSON file.

    If `t` and `y` are provided, each file also includes the full time and
    signal arrays masked to that split's cycle range as ``metadata.t`` and
    ``metadata.y``.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    cycle_ranges = {
        "train": train_cycles,
        "val":   val_cycles,
        "test":  test_cycles,
    }

    for split_name, points in splits.items():
        metadata: dict = {"split": split_name, "n_points": len(points),
                          "n_samples": n_samples}

        if t is not None and y is not None:
            first, last = cycle_ranges[split_name]
            t_min = (first - 1) * period
            t_max = last * period
            mask = (t >= t_min) & (t < t_max)
            metadata["t"] = t[mask]
            metadata["y"] = y[mask]

        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "metadata": metadata,
                    "points":   points,
                },
                f,
                indent=2,
                cls=_NumpyEncoder,
            )
        print(f"Saved {len(points)} points → {path}")


if __name__ == "__main__":
    # import yaml
    parser = argparse.ArgumentParser(description="Generate annotated benchmark dataset.")
    parser.add_argument("--output-dir", default="benchmark_splits",
                        help="Path to output JSON file (default: benchmark_annotations.json)")
    # parser = argparse.ArgumentParser(description="Generate annotated benchmark dataset.")
    # parser.add_argument("--config", default="config.yaml",
    #                     help="Path to YAML config file (default: config.yaml)")
    args = parser.parse_args()

    # with open(args.config) as f:
    #     cfg = yaml.safe_load(f)


    # --- Time series ---
    # scfg        = cfg.get("signal", {})
    # signal_type = scfg.get("type", "sine")
    # signal_fn   = SIGNAL_FUNCTIONS[signal_type]
    # signal_kwargs = {k: v for k, v in scfg.items()
    #              if k not in ("type", "t_start", "t_end", "n_points")}

    seed = 42
    signal_type    = "sine_with_constants"
    constant       = -0.5
    transition_mask  = 0.3    # fraction of t-values set to zero
    signal_fn      = SIGNAL_FUNCTIONS[signal_type]
    signal_period  = 2 * np.pi
    n_signals      = 512
    t_start        = 0
    t_end          = (n_signals * signal_period) + (signal_period/4) # go past by one annotation step

    train_cycles = (1, int(0.625 * n_signals))
    val_cycles   = (int(0.625 * n_signals) + 1, int(0.8125 * n_signals))
    test_cycles  = (int(0.8125 * n_signals) + 1, n_signals)

    annotation_step       = signal_period / 4
    min_constant_len   = annotation_step
    max_constant_len   = signal_period - annotation_step # maximum length of each zero segment
    annotations_per_point = 1

    t  = np.arange(t_start, t_end, annotation_step)
    y  = signal_fn(t, transition_mask=transition_mask, min_constant_len=min_constant_len, max_constant_len=max_constant_len, seed=seed, constant=constant)
    dy = SIGNAL_DERIVATIVES[signal_type](t)

    indices = np.arange(0, len(t), dtype=int)

    # Pre-compute transition positions so the main loop can override phase
    drop_positions     = set()
    recovery_positions = set()
    if signal_type == 'sine_with_constants':
        constant_mask = _make_mask(t, transition_mask=transition_mask,
                                    max_constant_len=max_constant_len, seed=seed)
        before_drop, before_recovery = _find_transition_annotations(constant_mask, indices)
        drop_positions     = {indices[j] for j in before_drop}
        recovery_positions = {indices[j] for j in before_recovery}
        print(f"Transition positions: {len(drop_positions)} drop, "
              f"{len(recovery_positions)} recovery")

    annotations = []
    template_seed = 0
    for i in indices:
        if i in drop_positions:
            phase = 'drop_to_constant'
        elif i in recovery_positions:
            phase = 'recover_from_constant'
        else:
            phase = determine_phase(t[i], period=signal_period)

        data_point = build_data_point(
            t        = t[i],
            y        = y[i],
            dy       = dy[i],
            phase    = phase,
            n_samples= annotations_per_point,
            seed     = None,
        )
        annotations.extend(data_point.values())
        seed+=1

    print(f"Total annotations: {len(annotations)}")
    # --- Providers ---
    #providers = build_providers(cfg.get("providers", [{"name": "mock"}]))

    # --- Generate ---
    # acfg   = cfg.get("annotations", {})
    # domain = acfg.get("domain", "signal")

    splits = split_dataset(annotations, period=signal_period,
        train_cycles= train_cycles,val_cycles= val_cycles,test_cycles= test_cycles
        )
    #check_split_balance(splits)
    save_splits(splits, output_dir=args.output_dir, t=t, y=y, period=signal_period,
                n_samples = annotations_per_point,
                train_cycles=train_cycles, val_cycles=val_cycles, test_cycles=test_cycles)

    # --- Export ---
    # save_json(annotations, args.output,
    #          signal_type=signal_type)