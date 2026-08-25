"""
coupling.py
-----------
Creates benchmark split directories that augment an existing dataset with
continuous coupling-strength categories for MI analysis, answering the
reviewer ask that text-signal coupling is "never varied continuously".

Three modes, each a different axis from full coupling to no coupling:

rho (corpus axis, default)
    Coupling strength rho = fraction of annotations describing the true
    future.  For each rho level every annotation time gets one extra point,
    copied from 'correct' with probability rho, otherwise from the
    contaminating category ('incorrect' by default; add 'irrelevant' for
    the dilution axis).  rho = 100 reproduces the correct corpus, rho = 0
    the fully inverted one.  Mechanically this is mixture.py on a fine grid
    over the full range - each annotation is still wholly correct or wholly
    wrong.  Category: coupling_{rho}_{contam_type}.

noise (content axis - per-annotation continuous channel)
    Each annotation's text is re-rendered from a noisy observation of the
    local gradient: the annotation describes the direction/magnitude of
    dy + N(0, sigma), using the same templates as the original corpus.
    Every annotation passes through a continuous noisy channel - direction
    flips when the noise flips the effective sign, so per-point reliability
    varies smoothly with sigma (and with |dy|: peaks/troughs degrade before
    zero crossings).  At transition points, where correct text uses the
    drop/recovery templates, the annotation is swapped for the time-matched
    incorrect text with the empirical flip probability of the directional
    points at that sigma.  Category: coupling_noise_{sigma}.

jitter (temporal axis - per-annotation continuous channel)
    Each annotation gets the *correct* text of a nearby time point,
    offset by round(N(0, sigma)) annotation steps.  Coupling decays
    continuously with sigma while the text stays lexically identical to
    the correct corpus - the same temporally-misaligned-text probe
    reviewers asked for on real data.  Category: coupling_jitter_{sigma}.

Every generated point records its origin in `source_category`
('correct' / 'incorrect', or 'jittered' for displaced jitter text), so the
dummy-oracle embedding path (run.py embedding_type: dummy) resolves the
oracle value from realized correctness.  The oracle is undefined for
displaced jitter text, so dummy runs on jitter datasets fail loudly rather
than report misleading values.  `analytic_dummy_mi()` gives the exact
ground-truth MI of the rho-mode oracle channel for overlaying on estimated
dose-response curves.

Because build_category_data() extracts categories dynamically from the data,
all coupling categories are picked up automatically alongside the originals
by run.py and train.py / train_transformer.py.

Usage
-----
    python coupling.py --source datasets/benchmark_splits
    python coupling.py --source datasets/benchmark_splits --mode noise
    python coupling.py --source datasets/benchmark_splits --mode jitter

Output
------
    benchmark_splits_coupling[_noise|_jitter]/
        train.json   - original points + one coupling category per level
        val.json
        test.json
"""

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


COUPLING_LEVELS      = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CONTAMINATION_TYPES  = ['incorrect']
NOISE_SIGMAS         = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
JITTER_SIGMAS        = [0.5, 1, 2, 4, 8, 16]
SPLITS               = ['train', 'val', 'test']
TRANSITION_PHASES    = {'drop_to_constant', 'recover_from_constant'}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  saved {len(data['points']):>5} points → {path}")


def _combo_rng(seed: int, *key) -> random.Random:
    """Independent deterministic rng per (mode, level) combination."""
    return random.Random(seed + hash(key) % (2**32))


def analytic_dummy_mi(rho: float, p_y) -> float:
    """
    Exact MI (nats) of the rho-mode dummy-oracle channel.

    The channel: X equals the true Y value with probability rho, otherwise a
    uniform draw from the remaining K-1 valid values.  `p_y` is the marginal
    distribution of Y over its K discrete values (e.g. the empirical
    frequencies of y_future in a split).  Note zero MI falls at rho = 1/K,
    not rho = 0: a fully inverted corpus still excludes the true value.

    Use at analysis time to overlay the ground-truth curve on estimated MI:

        truth = [analytic_dummy_mi(rho / 100, p_y) for rho in COUPLING_LEVELS]
    """
    p_y = np.asarray(p_y, dtype=float)
    p_y = p_y / p_y.sum()
    k   = len(p_y)
    # p(x|y): rho on the diagonal, (1-rho)/(k-1) off it
    cond = np.full((k, k), (1 - rho) / (k - 1))
    np.fill_diagonal(cond, rho)
    p_x = p_y @ cond
    mi  = 0.0
    for yi in range(k):
        for xi in range(k):
            if cond[yi, xi] > 0 and p_x[xi] > 0:
                mi += p_y[yi] * cond[yi, xi] * math.log(cond[yi, xi] / p_x[xi])
    return mi


def _group_by_time(points: list[dict]) -> tuple[dict, list]:
    by_time: dict[float, dict[str, dict]] = defaultdict(dict)
    for pt in points:
        by_time[pt['t']][pt['category']] = pt
    return by_time, sorted(by_time.keys())


def _finish_split(combined: list[dict], metadata: dict, coupling_meta: dict) -> dict:
    combined.sort(key=lambda p: (p['t'], p['category']))
    cats = Counter(p['category'] for p in combined)
    print(f"    categories: {dict(cats)}")
    metadata = dict(metadata)
    metadata['coupling'] = coupling_meta
    return {'metadata': metadata, 'points': combined}


# ---------------------------------------------------------------------------
# rho mode - corpus-composition sweep (mixture.py on a fine, full-range grid)
# ---------------------------------------------------------------------------

def _build_rho_split(points, metadata, levels, types, seed):
    by_time, times = _group_by_time(points)
    combined = list(points)
    realized = {}

    for contam_type in types:
        for rho in levels:
            cat_name = f'coupling_{rho}_{contam_type}'
            # Exact-count sampling (as in mixture.py) so realized coupling
            # matches rho deterministically.
            n_decoupled = math.ceil(len(times) * (100 - rho) / 100)
            rng = _combo_rng(seed, contam_type, rho)
            decoupled = set(rng.sample(times, n_decoupled))
            n_correct = 0
            for t in times:
                src_cat = contam_type if t in decoupled else 'correct'
                if src_cat in by_time[t]:
                    pt = copy.deepcopy(by_time[t][src_cat])
                    pt['category'] = cat_name
                    pt['source_category'] = src_cat
                    combined.append(pt)
                    n_correct += (src_cat == 'correct')

            realized[cat_name] = 100 * n_correct / len(times)
            print(f"    {cat_name:<26}  {n_correct}/{len(times)} correct "
                  f"(rho = {realized[cat_name]:.0f}%)")

    return _finish_split(combined, metadata, {
        'mode': 'rho', 'levels': levels, 'types': types,
        'seed': seed, 'realized_rho': realized,
    })


# ---------------------------------------------------------------------------
# noise mode - per-annotation continuous channel on annotation content
# ---------------------------------------------------------------------------

def _build_noise_split(points, metadata, sigmas, seed):
    # Script-style import, matching sine_generator.py (the script's own
    # directory is on sys.path when run directly).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sine_templates import (_build_directional_text, _true_direction,
                                _true_magnitude)

    by_time, times = _group_by_time(points)
    combined = list(points)
    realized = {}

    for sigma in sigmas:
        cat_name = f'coupling_noise_{sigma}'
        rng = _combo_rng(seed, 'noise', sigma)

        # Pass 1: directional points - re-render text from a noisy gradient.
        new_pts, deferred, n_flipped = [], [], 0
        for t in times:
            ptc = by_time[t].get('correct')
            if ptc is None:
                continue
            if ptc.get('phase') in TRANSITION_PHASES:
                deferred.append(t)
                continue

            dy_noisy  = ptc['dy'] + rng.gauss(0, sigma)
            direction = _true_direction(ptc['y'], dy_noisy)
            magnitude = _true_magnitude(dy_noisy)
            s0   = ptc['samples'][0]
            text, dir_w, mag_w = _build_directional_text(
                s0['template_idx'], s0.get('tense', ptc.get('tense', 'future')),
                direction, magnitude, rng)

            flipped = direction != _true_direction(ptc['y'], ptc['dy'])
            n_flipped += flipped
            pt = copy.deepcopy(ptc)
            pt['category']            = cat_name
            pt['source_category']     = 'incorrect' if flipped else 'correct'
            pt['described_direction'] = direction
            pt['described_magnitude'] = magnitude
            pt['samples'] = [dict(s0, text=text, direction_word=dir_w,
                                  magnitude_word=mag_w)]
            new_pts.append(pt)

        # Pass 2: transition points have no gradient to perturb - swap to the
        # time-matched incorrect text at the directional flip rate so the
        # whole corpus degrades at one consistent sigma.
        flip_prob = n_flipped / max(len(new_pts), 1)
        for t in deferred:
            group = by_time[t]
            src_cat = ('incorrect'
                       if rng.random() < flip_prob and 'incorrect' in group
                       else 'correct')
            pt = copy.deepcopy(group[src_cat])
            pt['category'] = cat_name
            pt['source_category'] = src_cat
            new_pts.append(pt)

        combined.extend(new_pts)
        n_correct = sum(p['source_category'] == 'correct' for p in new_pts)
        realized[cat_name] = 100 * n_correct / len(new_pts)
        print(f"    {cat_name:<26}  {n_correct}/{len(new_pts)} correct "
              f"(directional flip p = {flip_prob:.2f})")

    return _finish_split(combined, metadata, {
        'mode': 'noise', 'sigmas': sigmas, 'seed': seed,
        'realized_pct_correct': realized,
    })


# ---------------------------------------------------------------------------
# jitter mode - per-annotation continuous channel on temporal alignment
# ---------------------------------------------------------------------------

def _build_jitter_split(points, metadata, sigmas, seed):
    by_time, times = _group_by_time(points)
    combined = list(points)
    realized = {}

    for sigma in sigmas:
        cat_name = f'coupling_jitter_{sigma}'
        rng = _combo_rng(seed, 'jitter', sigma)
        n_aligned, abs_offsets, n_total = 0, [], 0
        for i, t in enumerate(times):
            ptc = by_time[t].get('correct')
            if ptc is None:
                continue
            delta = int(round(rng.gauss(0, sigma)))
            j     = min(max(i + delta, 0), len(times) - 1)
            src   = by_time[times[j]].get('correct')
            if src is None:
                continue
            pt = copy.deepcopy(ptc)
            pt['category'] = cat_name
            # Displaced text has no dummy-oracle analogue: 'jittered' makes
            # the dummy path raise instead of silently returning the oracle
            # value for the wrong time point.
            pt['source_category'] = 'correct' if j == i else 'jittered'
            pt['samples'] = copy.deepcopy(src['samples'])
            pt['jitter_offset'] = j - i
            combined.append(pt)
            n_total   += 1
            n_aligned += (j == i)
            abs_offsets.append(abs(j - i))

        realized[cat_name] = {
            'pct_aligned':     100 * n_aligned / n_total,
            'mean_abs_offset': float(np.mean(abs_offsets)),
        }
        print(f"    {cat_name:<26}  {n_aligned}/{n_total} aligned "
              f"(mean |offset| = {np.mean(abs_offsets):.2f} steps)")

    return _finish_split(combined, metadata, {
        'mode': 'jitter', 'sigmas': sigmas, 'seed': seed,
        'realized': realized,
    })


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_coupled(
    source_dir:  str,
    mode:        str        = 'rho',
    output_dir:  str | None = None,
    levels:      list[int]  = COUPLING_LEVELS,
    types:       list[str]  = CONTAMINATION_TYPES,
    sigmas:      list[float] | None = None,
    seed:        int        = 0,
) -> None:
    src    = Path(source_dir)
    suffix = '_coupling' if mode == 'rho' else f'_coupling_{mode}'
    out    = Path(output_dir) if output_dir else src.parent / f"{src.name}{suffix}"
    if sigmas is None:
        sigmas = NOISE_SIGMAS if mode == 'noise' else JITTER_SIGMAS

    print(f"Output → {out}/")
    for split in SPLITS:
        split_path = src / f"{split}.json"
        if not split_path.exists():
            print(f"  skipping {split} (not found)")
            continue
        print(f"\n  {split}:")
        raw = _load(split_path)
        if mode == 'rho':
            combined = _build_rho_split(raw['points'], raw['metadata'],
                                        levels=levels, types=types, seed=seed)
        elif mode == 'noise':
            combined = _build_noise_split(raw['points'], raw['metadata'],
                                          sigmas=sigmas, seed=seed)
        elif mode == 'jitter':
            combined = _build_jitter_split(raw['points'], raw['metadata'],
                                           sigmas=sigmas, seed=seed)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        _save(combined, out / f"{split}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment a benchmark split with coupling-strength categories.')
    parser.add_argument('--source',     required=True,
                        help='Source benchmark split directory')
    parser.add_argument('--mode',       default='rho',
                        choices=['rho', 'noise', 'jitter'],
                        help='Coupling axis: rho = corpus composition, '
                             'noise = per-annotation gradient noise, '
                             'jitter = per-annotation temporal offset')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <source>_coupling[_mode])')
    parser.add_argument('--levels',  nargs='+', type=int, default=COUPLING_LEVELS,
                        help='rho mode: coupling strengths as %% of correct '
                             'annotations (default: 0 10 ... 100)')
    parser.add_argument('--types',   nargs='+', default=CONTAMINATION_TYPES,
                        help="rho mode: contaminating categories (default: "
                             "incorrect; add irrelevant for the dilution axis)")
    parser.add_argument('--sigmas',  nargs='+', type=float, default=None,
                        help='noise/jitter modes: channel widths '
                             f'(defaults: {NOISE_SIGMAS} / {JITTER_SIGMAS})')
    parser.add_argument('--seed',    type=int, default=0)
    args = parser.parse_args()

    make_coupled(
        source_dir=args.source,
        mode=args.mode,
        output_dir=args.output_dir,
        levels=args.levels,
        types=args.types,
        sigmas=args.sigmas,
        seed=args.seed,
    )
