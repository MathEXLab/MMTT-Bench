# benchmark/data.py
"""
Data loading and preprocessing for the sine-wave benchmark.

Handles loading raw JSON splits, attaching lookback/future windows to each
annotation point, building per-category arrays, and embedding text via a
registered tokenizer.  The public entry point is ``build_category_data``.

TS tokenization (_pad, _PatchProjector, _make_projector,
_tokenize_lookback) has moved to benchmark/tokenizers.py.
This module imports _pad and TS_TOKENIZERS from there.
"""

import json
import warnings
import numpy as np
from pathlib import Path

from .tokenizers import _pad, TS_TOKENIZERS


# ---------------------------------------------------------------------------
# Step 1: attach y_lookback, y_future, x_text to each data point
# ---------------------------------------------------------------------------

def _stack_3d_signal(metadata: dict) -> np.ndarray:
    """
    Return the dense signal as a numpy array.

    """
    y = np.asarray(metadata['y'])
    if 'y_comp' in metadata and 'z_comp' in metadata:
        return np.column_stack([
            y,
            np.asarray(metadata['y_comp']),
            np.asarray(metadata['z_comp']),
        ])
    return y


def _attach_lookback_and_future(
    data,
    horizon_steps,
    lookback_steps,
    ts_tokenizer,
):
    """
    Attaches to each point:
        y_lookback : tokenised lookback  — (n_patches,) or (n_patches * C,)
        y_future   : future window       — (H,) for 1-D, (H, C) for 3-D
        x_text     : list of raw text strings

    Automatically detects 3-D Rössler data from metadata keys
    ('y_comp', 'z_comp') and stacks all channels before encoding.
    The ts_tokenizer handles both 1-D and multivariate (L, C) inputs via
    its channel-independent encode().
    """
    t_dense      = data['metadata']['t']
    dense_signal = _stack_3d_signal(data['metadata'])   # (L,) or (L, C)
    is_3d        = dense_signal.ndim == 2
    t_rounded    = [round(t, 4) for t in t_dense]
    n_dense      = dense_signal.shape[0]

    for i, point in enumerate(data['points']):
        t_i = round(point['t'], 4)
        if t_i not in t_rounded:
            continue
        idx = t_rounded.index(t_i)

        # ── Lookback ────────────────────────────────────────────────────
        # Ends at the annotation point inclusive: y(t) is an observation
        # available at forecast time (standard sliding-window convention),
        # and the text is generated from y(t)/dy(t) — excluding it would
        # give text privileged access to the current signal state.
        start    = max(0, idx + 1 - lookback_steps)
        y_window = dense_signal[start:idx + 1]      # (lb,) or (lb, C)
        if y_window.shape[0] < lookback_steps:
            y_window = _pad(y_window, lookback_steps)   # _pad handles both dims
        y_lookback_tok = ts_tokenizer.encode(y_window)  # (n_patches,) or (n_patches*C,)

        # ── Future ──────────────────────────────────────────────────────
        # When pre_lead_steps is set we want the single snapshot at exactly
        # t + PRE_LEAD (the event the text describes), not the full window
        # leading up to it — the full window would be thousands of dimensions
        # and make CCA/MI estimation degenerate.
        future_end = idx + 1 + horizon_steps
        if future_end > n_dense:
            continue
        if horizon_steps > 1 and data['metadata'].get('pre_lead_steps', 0) > 0:
            y_future = dense_signal[future_end - 1:future_end]  # single endpoint
        else:
            y_future = dense_signal[idx + 1:future_end]         # (H,) or (H, C)

        # ── Text ────────────────────────────────────────────────────────
        text_list = [s['text'] for s in point['samples']]

        data['points'][i]['y_lookback'] = y_lookback_tok
        data['points'][i]['y_future']   = y_future
        data['points'][i]['x_text']     = text_list
        if is_3d:
            data['points'][i]['signal_3d'] = True   # flag for downstream use

    return data


# ---------------------------------------------------------------------------
# Step 2: build category_data skeleton (ts and y, no embeddings yet)
# One row per annotation point.
# ---------------------------------------------------------------------------

def _build_category_data(data, categories):
    """
    One row per annotation point -- 'x' filled by _attach_text_embeddings.
    """
    category_data = {cat: {'ts': [], 'y': [], 'x': [], 'n_samples':[]}
                     for cat in categories}

    for pt in data['points']:
        if 'x_text' not in pt:
            continue
        cat = pt.get('category')
        if cat not in category_data:
            continue
        if 'y_lookback' in pt and 'y_future' in pt:
            ts_row = np.array(pt['y_lookback']).ravel()
            y_row  = pt['y_future']
            n_samples = len(pt['x_text'])
            # Repeat ts and y once per text sample so all arrays align
            for _ in range(n_samples):
                category_data[cat]['ts'].append(ts_row)
                category_data[cat]['y'].append(y_row)
                category_data[cat]['n_samples'].append(n_samples)


    return category_data


# ---------------------------------------------------------------------------
# Step 3: embed text and attach to category_data
# ---------------------------------------------------------------------------

def _attach_text_embeddings(data, category_data, text_tokenizer):
    """
    Single batch embedding call; each text sample gets its own embedding row.

    Iterates over annotation points, collects all text strings into one list,
    calls text_tokenizer.encode() once, then slices the result back into
    per-category arrays.  The resulting 'x' array has one row per text sample
    (matching the expanded ts/y arrays from _build_category_data).
    """
    all_texts       = []
    cat_text_slices = {cat: [] for cat in category_data}

    for pt in data['points']:
        if 'x_text' not in pt:
            continue
        cat = pt.get('category')
        if cat not in category_data:
            continue
        start = len(all_texts)
        all_texts.extend(pt['x_text'])
        cat_text_slices[cat].append((start, len(all_texts)))

    all_embeddings = text_tokenizer.encode(all_texts)

    for cat, slices in cat_text_slices.items():
        per_point = [all_embeddings[s:e] for s, e in slices]

        category_data[cat]['x'] = (
            np.vstack(per_point)
            if per_point else np.empty((0,))
        )

    return category_data


# ---------------------------------------------------------------------------
# Step 3 (alt): dummy oracle embedding, for testing metrics without a real
# text tokenizer.
# ---------------------------------------------------------------------------

_DUMMY_VALID_VALUES = (-1.0, -0.5, 0.0, 1.0)


def _attach_dummy_embeddings(data, category_data, rng,
                              valid_values=_DUMMY_VALID_VALUES):
    """
    Replace text embeddings with a 1-D oracle value derived from the
    ground-truth sine value at the annotated point, instead of encoding
    ``x_text``. Each row becomes a single-number vector:

        correct    -> [true y_future value]
        incorrect  -> [random value from valid_values, excluding the true one]
        irrelevant -> [0.0]

    Lets metric tests target a known signal (e.g. unique_text should be
    near-maximal for 'correct'/'incorrect' data and near-zero for
    'irrelevant') without depending on a real text_tokenizer.
    """
    cat_rows = {cat: [] for cat in category_data}

    for pt in data['points']:
        if 'x_text' not in pt or 'y_future' not in pt:
            continue
        cat = pt.get('category')
        if cat not in category_data:
            continue

        correct_value = float(np.asarray(pt['y_future']).ravel()[0])
        n_samples = len(pt['x_text'])

        # Derived categories (e.g. coupling_70_incorrect from coupling.py)
        # record which base category the point was copied from.
        src_cat = pt.get('source_category', cat)

        for _ in range(n_samples):
            if src_cat == 'irrelevant':
                value = 0.0
            elif src_cat == 'correct':
                value = correct_value
            elif src_cat == 'incorrect':
                choices = [v for v in valid_values
                           if not np.isclose(v, correct_value)]
                value = rng.choice(choices)
            else:
                raise ValueError(
                    f"Unknown category '{cat}' for dummy embedding; "
                    f"expected 'correct', 'incorrect', or 'irrelevant'."
                )
            cat_rows[cat].append([value])

    for cat, rows in cat_rows.items():
        category_data[cat]['x'] = np.array(rows) if rows else np.empty((0, 1))

    return category_data


# ---------------------------------------------------------------------------
# Step 4: finalise -- convert lists to arrays, validate alignment
# ---------------------------------------------------------------------------

def _finalise(category_data):
    for cat, vals in category_data.items():
        ts = np.array(vals['ts'])
        y  = np.array(vals['y'])
        x  = np.array(vals['x'])

        if not (len(ts) == len(y) == len(x)):
            raise ValueError(
                f"Row count mismatch for '{cat}': "
                f"ts={len(ts)}, y={len(y)}, x={len(x)}."
            )

        category_data[cat]['ts'] = ts
        category_data[cat]['y']  = y.reshape(len(y), -1)
        category_data[cat]['x']  = x

    return category_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shuffle_text_annotations(category_data, rng):
    """
    Shuffle text embeddings (x) to random time points within each category.

    Sanity check: metrics measuring real text-timeseries correlation should
    score lower on shuffled data than on the original.
    """
    for vals in category_data.values():
        idx = rng.permutation(len(vals['x']))
        vals['x'] = vals['x'][idx]
    return category_data


def load_split(path):
    """Load a benchmark split JSON file."""
    with open(path) as f:
        return json.load(f)


def build_category_data(
    raw,
    text_tokenizer,
    ts_tokenizer,
    horizon_steps  = 1,
    lookback_steps = 16,
    patch_len      = 8,
):
    """
    Full pipeline from raw JSON to category_data.

    If raw['metadata']['pre_lead_steps'] is present and > 0, it overrides
    horizon_steps so that y_future ends at the time the text describes
    (i.e. the forecast target is the annotated event, not 1 step ahead).

    Parameters
    ----------
    raw              : dict from load_split()
    text_tokenizer   : object with .encode(list[str]) -> np.ndarray
    horizon_steps    : dense steps ahead for Y_future — overridden by
                       metadata['pre_lead_steps'] when that key is present
    lookback_steps   : dense steps in lookback window
    patch_len        : patch length for ts tokenisation

    Returns
    -------
    Univariate:
        {category: {'ts': (N, n_patches), 'y': (N, H), 'x': (N, embed_dim)}}
    3-D Rössler:
        {category: {'ts': (N, n_patches*3), 'y': (N, H*3), 'x': (N, embed_dim)}}
        y columns are [x_0…x_{H-1}, y_0…y_{H-1}, z_0…z_{H-1}] (H values per
        channel, channels concatenated by np.reshape(N, -1) in _finalise)
    """

    # Use pre_lead_steps from metadata if present — ensures the forecast
    # target matches the time the text annotations describe.
    pre_lead_steps = raw.get('metadata', {}).get('pre_lead_steps', 0)
    if pre_lead_steps > 0:
        horizon_steps = pre_lead_steps

    # Shallow copy so caller's dict is not mutated
    data = dict(raw)
    data['points'] = [dict(pt) for pt in raw['points']]

    data          = _attach_lookback_and_future(
                        data, horizon_steps, lookback_steps, ts_tokenizer)
    categories = list(dict.fromkeys(x['category'] for x in data['points']))
    category_data = _build_category_data(data, categories)
    category_data = _attach_text_embeddings(data, category_data, text_tokenizer)
    category_data = _finalise(category_data)

    for cat, vals in category_data.items():
        N = len(vals['y'])
        d_conservative = int(np.sqrt(N / 2))
        if d_conservative < 3:
            warnings.warn(
                f"Category '{cat}' has only N={N} samples. "
                f"KSG reliable up to d={d_conservative} dims.",
                UserWarning
            )

    return category_data


def build_category_data_dummy_embedding(
    raw,
    ts_tokenizer,
    horizon_steps  = 1,
    lookback_steps = 16,
    patch_len      = 8,
    seed           = 0,
):
    """
    Same pipeline as build_category_data, but 'x' is a synthetic oracle
    embedding (see _attach_dummy_embeddings) instead of a real text
    tokenizer's output. embed_dim is always 1, so no PCA/dim-reduction
    step is needed on the resulting category_data before feeding it to
    the metrics.

    Parameters
    ----------
    raw          : dict from load_split()
    ts_tokenizer : time-series tokenizer, same as build_category_data
    seed         : seeds the RNG used to pick 'incorrect' values

    Returns
    -------
    {category: {'ts': (N, n_patches), 'y': (N, H), 'x': (N, 1)}}
    """
    rng = np.random.default_rng(seed)

    pre_lead_steps = raw.get('metadata', {}).get('pre_lead_steps', 0)
    if pre_lead_steps > 0:
        horizon_steps = pre_lead_steps

    data = dict(raw)
    data['points'] = [dict(pt) for pt in raw['points']]

    data          = _attach_lookback_and_future(
                        data, horizon_steps, lookback_steps, ts_tokenizer)
    categories = list(dict.fromkeys(x['category'] for x in data['points']))
    category_data = _build_category_data(data, categories)
    category_data = _attach_dummy_embeddings(data, category_data, rng)
    category_data = _finalise(category_data)

    return category_data
