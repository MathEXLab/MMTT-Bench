"""
plot.py — Plot benchmark results from run.py and train.py.

Default usage (requires both --metrics and --models):
-----------------------------------------------------
    python plot.py \\
        --metrics results/sine_metrics.json \\
        --models  results/sine_model.json \\
        --output-dir results/plots/

Produces all plots in one pass:
    - Bar charts of the conditional text contribution per annotation category
      (one panel per metric family: KSG MI, PID, V-information, CCA, …)
    - Scatter plots: info metric vs performance delta, one panel per metric
    - Correlation trend lines per model with shared y-axis
    - Delta bar charts grouped by model and category

Multiple datasets side-by-side:
    python plot.py \\
        --metrics results/sine_metrics.json results/rossler_metrics.json \\
        --models  results/sine_model.json   results/rossler_model.json \\
        --labels  sine rossler

Sweep visualisation (run.py sweep output only):
    python plot.py results/sine_sweep.json --output-dir plots/
    python plot.py results/r1.json results/r2.json --overlay --labels a b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from src.mmtt_bench.viz import (
    METRIC_LABELS,
    METRIC_DISPLAY,
    CONDITIONAL_KEY,
    PERF_METRIC_LABEL,
    MODEL_MARKERS,
    MODEL_LINESTYLES,
    plot_correlation_trends,
    plot_embedding_sensitivity,
    plot_ordering_grid,
    plot_annotation_example,
    plot_realworld_mi_grid,
    plot_model_mixture,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_metrics(path: str) -> dict:
    """Load a run.py single-result JSON → results_by_metric."""
    data = _load(path)
    if data.get("type") != "single":
        raise ValueError(
            f"{path} is a sweep result (type={data['type']!r}). "
            "Compare mode expects a single-run result."
        )
    return data["results_by_metric"]


def load_model_results(path: str) -> dict:
    """Load a train.py JSON output."""
    return _load(path)


def filter_model_data(model_data: dict,
                      fix_model: str | None,
                      fix_mode:  str | None) -> dict:
    """
    Subset compound 'backbone|injection_mode' model keys by one fixed axis,
    then strip that axis so only the varying dimension remains as the label.

    --fix-model PatchTST  →  keep 'PatchTST|*' keys, rename to injection mode
    --fix-mode  cfa       →  keep '*|cfa'     keys, rename to backbone name

    If neither flag is set the data is returned unchanged.
    """
    if not fix_model and not fix_mode:
        return model_data

    import copy
    data = copy.deepcopy(model_data)

    def _keep(key: str) -> bool:
        if fix_model:
            return key == fix_model or key.startswith(f'{fix_model}|')
        return key == fix_mode or key.endswith(f'|{fix_mode}')

    def _rename(key: str) -> str:
        if '|' not in key:
            return key
        if fix_model:
            return key.split('|', 1)[1]   # 'PatchTST|cfa' → 'cfa'
        return key.rsplit('|', 1)[0]       # 'PatchTST|cfa' → 'PatchTST'

    data['models'] = [_rename(m) for m in data['models'] if _keep(m)]

    new_results: dict = {}
    for cat, cat_results in data['results'].items():
        new_results[cat] = {
            _rename(k): v for k, v in cat_results.items() if _keep(k)
        }
    data['results'] = new_results
    return data


# ---------------------------------------------------------------------------
# Delta + table computation
# ---------------------------------------------------------------------------

def compute_deltas(model_data: dict) -> dict[str, dict[str, dict[str, float]]]:
    """
    For each (model, category) compute delta = text_model − no_text_baseline,
    plus raw metrics and a 'no_text' baseline row.

    Returns
    -------
    {model_name: {'delta_r2': {cat: float}, 'r2': {cat: float}, ...}}
    """
    results    = model_data["results"]
    no_text    = results.get("no_text", {})
    categories = model_data.get("categories", list(
        next(iter(no_text.values()), {}).get("per_category", {}).keys()
    ))

    deltas: dict[str, dict[str, dict[str, float]]] = {}

    for model_key in model_data.get("models", list(no_text.keys())):
        baseline_per_cat = no_text.get(model_key, {}).get("per_category", {})
        no_text_overall  = no_text.get(model_key, {}).get("overall", {})
        model_deltas: dict[str, dict[str, float]] = {
            "delta_r2":   {},
            "delta_rmse": {},
            "delta_mae":  {},
            "delta_mse":  {},
            "r2":         {},
            "rmse":       {},
            "mae":        {},
            "mse":        {},
        }

        # no_text row: zero delta, raw metrics from baseline overall result
        if no_text_overall:
            for raw_key in ("r2", "rmse", "mae", "mse"):
                model_deltas[raw_key]["no_text"] = no_text_overall.get(raw_key, np.nan)
            for delta_key in ("delta_r2", "delta_rmse", "delta_mae", "delta_mse"):
                model_deltas[delta_key]["no_text"] = 0.0

        for cat in categories:
            baseline    = baseline_per_cat.get(cat)
            text_result = results.get(cat, {}).get(model_key, {}).get("overall")
            if baseline is None or text_result is None:
                continue
            nan = float("nan")
            r2_t = text_result.get("r2")
            r2_b = baseline.get("r2")
            model_deltas["delta_r2"][cat]   = (r2_t - r2_b) if r2_t is not None and r2_b is not None else nan
            model_deltas["delta_rmse"][cat] = text_result.get("rmse", nan) - baseline.get("rmse", nan)
            model_deltas["delta_mae"][cat]  = text_result.get("mae",  nan) - baseline.get("mae",  nan)
            model_deltas["delta_mse"][cat]  = text_result.get("mse",  nan) - baseline.get("mse",  nan)
            model_deltas["r2"][cat]         = r2_t if r2_t is not None else nan
            model_deltas["rmse"][cat]       = text_result.get("rmse", nan)
            model_deltas["mae"][cat]        = text_result.get("mae",  nan)
            model_deltas["mse"][cat]        = text_result.get("mse",  nan)

        deltas[model_key] = model_deltas

    return deltas


def build_eval_table(
    metrics_by_family: dict,
    model_delta:       dict[str, dict[str, dict[str, float]]],
    categories:        list[str],
) -> dict[str, pd.DataFrame]:
    """
    Build one DataFrame per model with info metrics + delta and raw performance.

    Rows: no_text + each text category.
    Columns: info metrics, delta_r2/rmse/mae/mse, r2/rmse/mae/mse.
    """
    tables:   dict[str, pd.DataFrame] = {}
    all_cats = ["no_text"] + list(categories)

    for model_name, deltas in model_delta.items():
        if model_name == 'knn':
            continue
        # delta_r2/r2 are empty for CFA output (no r2 metric); only require
        # at least one of the non-r2 metrics to have data before including.
        if not any(deltas.get(k) for k in ("delta_mse", "delta_mae", "delta_rmse",
                                            "mse", "mae", "rmse")):
            continue

        row_data: dict[str, list] = {"category": all_cats}

        # Info metrics — no_text row gets NaN (no text, no information to measure)
        for family, results in metrics_by_family.items():
            ckey = CONDITIONAL_KEY.get(family, "conditional")
            if ckey not in results:
                continue
            col_vals = []
            std_vals = []
            for cat in all_cats:
                if cat == "no_text":
                    col_vals.append(np.nan)
                    std_vals.append(np.nan)
                    continue
                v = results[ckey].get(cat, np.nan)
                if isinstance(v, (list, np.ndarray)) and len(v):
                    col_vals.append(float(np.mean(v)))
                    # Prefer pre-computed _std (e.g. KSG subsampling extrapolation)
                    # over np.std of the list when available.
                    precomputed_std = results.get(f"{ckey}_std", {}).get(cat)
                    if precomputed_std is not None and np.isfinite(precomputed_std):
                        std_vals.append(float(precomputed_std))
                    else:
                        std_vals.append(float(np.std(v)))
                else:
                    col_vals.append(float(v) if not isinstance(v, (list, np.ndarray)) else np.nan)
                    std_vals.append(np.nan)
            row_data[family] = col_vals
            row_data[f'{family}_std'] = std_vals

        for delta_key in ("delta_r2", "delta_rmse", "delta_mae", "delta_mse"):
            row_data[delta_key] = [deltas[delta_key].get(cat, np.nan) for cat in all_cats]

        for raw_key in ("r2", "rmse", "mae", "mse"):
            row_data[raw_key] = [deltas[raw_key].get(cat, np.nan) for cat in all_cats]

        tables[model_name] = pd.DataFrame(row_data).set_index("category")

    return tables

def print_correlations(
    tables:       dict[str, pd.DataFrame],
    info_metrics: list[str],
    perf_metrics: list[str] | None = None,
    label:        str = "",
) -> None:
    if perf_metrics is None:
        perf_metrics = ["delta_r2", "delta_rmse"]

    print(f"\n{'═' * 70}")
    if label:
        print(f"  {label}")
    print(f"{'═' * 70}")

    for model_name, df in tables.items():
        print(f"\n  MODEL: {model_name}")
        print(f"  {'─' * 60}")
        for info_m in info_metrics:
            if info_m not in df.columns:
                continue
            for perf_m in perf_metrics:
                if perf_m not in df.columns:
                    continue
                mask   = df[info_m].notna() & df[perf_m].notna()
                x_vals = df.loc[mask, info_m].values
                y_vals = df.loc[mask, perf_m].values
                if len(x_vals) < 3:
                    continue
                rho, p1 = spearmanr(x_vals, y_vals)
                tau, p2 = kendalltau(x_vals, y_vals)
                print(f"  {info_m:<25s} vs {perf_m:<14s}"
                      f"  Spearman ρ={rho:+.2f} (p={p1:.3f})"
                      f"  Kendall τ={tau:+.2f} (p={p2:.3f})")
                # tab = df[info_m]
                # std = df[f'{info_m}_std']
                # n = df['n']

                # # for each cat
                # p = [two_sided_z_test(
                #     tab.loc[cat], std.loc[cat], n.loc[cat], tab.loc["no_text"], std.loc["no_text"], n.loc["no_text"]
                #     ) for cat in categories] + [0]

                # mi_diff = [tab.loc[cat] - tab.loc["no_text"] for cat in categories]
                # mi_std = [std.loc[cat] - std.loc["no_text"] for cat in categories]

                # model_diff = [tables[model]['delta_mse'].loc[cat] for cat in categories]


    print(f"{'═' * 70}")


# ---------------------------------------------------------------------------
# Info-metrics mode helpers
# ---------------------------------------------------------------------------

def _results_by_metric(data: dict) -> dict[str, dict]:
    if "results_by_metric" in data:
        return data["results_by_metric"]
    metric = data.get("config", {}).get("metric", "metric")
    return {metric: data["results"]}


def _runs_by_metric(data: dict) -> dict[str, list]:
    runs = data["runs"]
    if "results_by_metric" in runs[0]:
        metrics = list(runs[0]["results_by_metric"].keys())
        return {
            m: [{"sweep_value": r["sweep_value"], "results": r["results_by_metric"][m]}
                for r in runs]
            for m in metrics
        }
    metric = data.get("config", {}).get("metric", "metric")
    return {metric: runs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure 5 — loading the real-world results
#
# Drawing lives in viz.plot_realworld_mi_grid(); everything here is data
# assembly, matching the split used by the rest of this module.
# ---------------------------------------------------------------------------

CATEGORIES = ["correct", "incorrect", "irrelevant"]
ESTIMATORS = ["mutual_information", "mine", "infonce", "cca", "v_information", "pid"]
BASELINE = "no_text"
SERIES = [BASELINE] + CATEGORIES

FINTEXTS_TICKERS = ["AMD", "BA", "COST", "DIS", "GOOGL", "INTC", "NFLX", "NVDA", "T", "TSLA"]
TIMEMMD = [("Agriculture", 6), ("Climate", 6), ("Energy", 12),
           ("PublicHealth", 12), ("SocialGood", 6), ("Traffic", 6)]
DATASET_LABEL = {"PublicHealth": "Public\nHealth", "SocialGood": "Social\nGood",
                 "FinTexTS": "FinTexTS\n(10 tickers)"}


# ── Loading ──────────────────────────────────────────────────────────────────
def _first(pattern: str) -> str | None:
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def _stat(md: dict, field: str, cat: str) -> tuple[float, float]:
    """Mean and std of a metric field for one category, tolerating both the
    mean/std layout and a raw list of bootstrap draws."""
    if f"{field}_mean" in md:
        return float(md[f"{field}_mean"][cat]), float(md[f"{field}_std"][cat])
    v = md[field][cat]
    v = v if isinstance(v, list) else [v]
    return float(np.mean(v)), float(np.std(v))


def _pid_text_info(md: dict, cat: str) -> tuple[float, float]:
    """PID identity: I(ts,text;Y) - I(ts;Y) = U_text + S exactly, so PID reports
    the net text contribution directly. Component covariances are unavailable,
    so sigma sums the two variances."""
    u, us = _stat(md, "unique_text", cat)
    s, ss = _stat(md, "synergy", cat)
    return u + s, float(np.hypot(us, ss))


def mi_value(md: dict, est: str, cat: str, quantity: str) -> tuple[float, float]:
    """Raw estimate for one (estimator, category), before any contrast is taken.

    joint        I(X_ts, X_text; Y) — the quantity both derived measures come
                 from: net gain is joint minus the ts-only value, alignment is
                 joint|correct minus joint|incorrect.
    text         I(X_text; Y), the text-only marginal.
    conditional  the estimator's own paired-resample conditional field.

    PID stores components rather than a joint field; they sum to the joint by
    construction (I(ts,text;Y) = R + U_ts + U_text + S), so it is reassembled.
    """
    if est == "pid":
        if quantity == "conditional":
            return _pid_text_info(md, cat)
        if quantity == "text":
            return _stat(md, "unique_text", cat)
        parts = [_stat(md, k, cat) for k in ("redundancy", "unique_ts", "unique_text", "synergy")]
        return float(sum(m for m, _ in parts)), float(np.sqrt(sum(s ** 2 for _, s in parts)))

    pre = "v_" if est == "v_information" else ""
    return _stat(md, f"{pre}{quantity}", cat)


def ts_only(md: dict, est: str) -> float:
    """Time-series-only reference. Identical across categories by construction
    (same series, no text), so averaged over them for numerical stability."""
    try:
        if est == "pid":
            vals = [_stat(md, "redundancy", c)[0] + _stat(md, "unique_ts", c)[0]
                    for c in CATEGORIES]
        else:
            pre = "v_" if est == "v_information" else ""
            vals = [_stat(md, f"{pre}ts", c)[0] for c in CATEGORIES]
        return float(np.mean(vals))
    except (KeyError, TypeError):
        return np.nan


def net_contribution(md: dict, est: str, cat: str) -> tuple[float, float]:
    """Net information text adds beyond the series, with its standard error.

    Read from the estimator's own paired-resample conditional field (joint - ts
    within the SAME draw), never as joint_mean - ts_mean. The joint and marginal
    estimates share resamples and are strongly correlated, so dividing the
    difference of means by the joint's own standard deviation ignores that
    correlation, overstates sigma and understates the contribution — by a factor
    of ~3 for CCA on these datasets.
    """
    if est == "pid":
        return _pid_text_info(md, cat)
    pre = "v_" if est == "v_information" else ""
    try:
        return _stat(md, f"{pre}conditional", cat)
    except (KeyError, TypeError):
        return np.nan, np.nan


def pct_mse_change(model_path: str) -> dict[str, np.ndarray]:
    """% change in MSE vs the no-text baseline, one value per configuration.
    Negative means adding text lowered error.

    Only configurations present in *every* category are kept, so the returned
    arrays are aligned index-by-index and the paired test below compares like
    with like. Building each category independently would silently offset the
    arrays whenever one category is missing a run.
    """
    d = json.load(open(model_path))
    res = d["results"]
    usable = [m for m in d["models"]
              if all(m in res.get(c, {}) for c in CATEGORIES + ["no_text"])
              and res["no_text"][m]["overall"]["mse"]]
    return {c: np.array([(res[c][m]["overall"]["mse"] - res["no_text"][m]["overall"]["mse"])
                         / res["no_text"][m]["overall"]["mse"] * 100 for m in usable], float)
            for c in CATEGORIES}


def _mi_block(metrics, quantity: str, pooled: bool = False) -> dict:
    """{estimator: {'vals': {cat: (mean, std)}, 'ts': float}} for one dataset."""
    out: dict[str, dict] = {}
    for est in ESTIMATORS:
        srcs = [m for m in (metrics if pooled else [metrics]) if est in m]
        if not srcs:
            continue
        vals = {}
        for cat in CATEGORIES:
            try:
                got = [mi_value(m[est], est, cat, quantity) for m in srcs]
            except (KeyError, TypeError):
                continue
            vals[cat] = (float(np.mean([g[0] for g in got])),
                         float(np.mean([g[1] for g in got])))
        net = {}
        for cat in CATEGORIES:
            got = [net_contribution(m[est], est, cat) for m in srcs]
            got = [g for g in got if np.isfinite(g[0]) and np.isfinite(g[1])]
            if got:
                net[cat] = (float(np.mean([g[0] for g in got])),
                            float(np.mean([g[1] for g in got])))
        if vals:
            out[est] = {"vals": vals, "net": net,
                        "ts": float(np.nanmean([ts_only(m[est], est) for m in srcs]))}
    return out


def load_all(quantity: str) -> list[dict]:
    rows = []

    for name, pl in TIMEMMD:
        mi_path = f"results/timemmd_cfa/{name}.json"
        mdl = _first(f"results/timemmd_all/transformer/{name}_pl{pl}.json/**/*.json")
        if not (os.path.exists(mi_path) and mdl):
            print(f"  [skip] {name}")
            continue
        metrics = json.load(open(mi_path))["results_by_metric"]
        rows.append(dict(name=name, mse=pct_mse_change(mdl),
                         mi=_mi_block(metrics, quantity)))

    # FinTexTS: pool configurations across tickers, average MI across tickers.
    per_cat: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    tick_metrics = []
    for tk in FINTEXTS_TICKERS:
        mdl = _first(f"results/fintexts/transformer_perfield/{tk}/**/*.json")
        if mdl:
            for c, v in pct_mse_change(mdl).items():
                per_cat[c].extend(v.tolist())
        mi_path = f"results/fintexts_perfield/{tk}.json"
        if os.path.exists(mi_path):
            tick_metrics.append(json.load(open(mi_path))["results_by_metric"])
    if tick_metrics and any(len(v) for v in per_cat.values()):
        rows.append(dict(name="FinTexTS",
                         mse={c: np.asarray(v, float) for c, v in per_cat.items()},
                         mi=_mi_block(tick_metrics, quantity, pooled=True)))
    return rows


def preference_p(mse: dict[str, np.ndarray]) -> float:
    """Wilcoxon signed-rank p for best vs second-best category, paired by
    configuration. Answers: is there any corpus preference to detect at all?"""
    usable = [c for c in CATEGORIES if len(mse[c])]
    if len(usable) < 2:
        return np.nan
    order = sorted(usable, key=lambda c: np.median(mse[c]))
    try:
        return float(wilcoxon(mse[order[0]], mse[order[1]])[1])
    except ValueError:
        return np.nan


def ts_stat(md: dict, est: str) -> tuple[float, float]:
    """Series-only estimate and its bootstrap sigma. Identical across text
    categories by construction (same series, no text), so averaged over them.
    PID has no ts field; R + U_ts is the series-only share by definition."""
    try:
        if est == "pid":
            pairs = [(_stat(md, "redundancy", c), _stat(md, "unique_ts", c)) for c in CATEGORIES]
            m = float(np.mean([a[0] + b[0] for a, b in pairs]))
            s = float(np.mean([np.hypot(a[1], b[1]) for a, b in pairs]))
            return m, s
        pre = "v_" if est == "v_information" else ""
        vals = [_stat(md, f"{pre}ts", c) for c in CATEGORIES]
        return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
    except (KeyError, TypeError):
        return np.nan, np.nan


def collect(rows: list[dict], quantity: str = "joint") -> dict:
    """{estimator: {dataset_index: {series: (mean, std, differs_from_baseline)}}}

    quantity="joint"        marker is I(X_ts, X_text; Y), baseline is the
                            estimator's series-only value; the conditional is
                            then the gap between them.
    quantity="conditional"  marker is I(X_text; Y | X_ts) read from the paired
                            resample field, baseline is exactly zero because a
                            corpus that adds nothing has no conditional
                            information by definition.
    """
    out: dict[str, dict[int, dict[str, tuple]]] = {e: {} for e in ESTIMATORS}
    for idx, r in enumerate(rows):
        srcs = r["_metrics"] if r.get("_pooled") else [r["_metrics"]]
        for est in ESTIMATORS:
            have = [m for m in srcs if est in m]
            if not have:
                continue
            entry: dict[str, tuple] = {}
            if quantity == "conditional":
                entry[BASELINE] = (0.0, 0.0, False)
            else:
                bt = [ts_stat(m[est], est) for m in have]
                entry[BASELINE] = (float(np.mean([b[0] for b in bt])),
                                   float(np.mean([b[1] for b in bt])), False)
            for cat in CATEGORIES:
                # Significance always uses the paired conditional, never joint
                # minus ts as separate means: the two share resamples and are
                # correlated, so an unpaired sigma would be too wide.
                net = [net_contribution(m[est], est, cat) for m in have]
                net = [n for n in net if np.isfinite(n[0]) and np.isfinite(n[1])]
                if not net:
                    continue
                # KSG returns an exactly-zero conditional with exactly zero
                # variance on some datasets: the joint estimate is bit-identical
                # to the series-only one in every resample. That is a reading,
                # not a gap, so it is plotted — and with no spread it can never
                # differ from the baseline.
                sd = float(np.mean([n[1] for n in net]))
                differs = sd > 0 and abs(np.mean([n[0] for n in net])) > 1.96 * sd
                if quantity == "conditional":
                    got = net
                else:
                    try:
                        got = [mi_value(m[est], est, cat, "joint") for m in have]
                    except (KeyError, TypeError):
                        continue
                entry[cat] = (float(np.mean([g[0] for g in got])),
                              float(np.mean([g[1] for g in got])), differs)
            if len(entry) > 1:
                out[est][idx] = entry
    return out


def load_realworld_results(quantity: str = "joint"):
    """Everything Figure 5 needs: per-dataset MSE distributions and MI estimates.

    Returns (rows, data) for viz.plot_realworld_mi_grid(). Datasets with no
    results on disk are skipped, so a partial run still plots.
    """
    rows = load_all(quantity)
    by_name = {}
    for name, _ in TIMEMMD:
        p = f"results/timemmd_cfa/{name}.json"
        if os.path.exists(p):
            by_name[name] = (json.load(open(p))["results_by_metric"], False)
    tick = [json.load(open(f"results/fintexts_perfield/{t}.json"))["results_by_metric"]
            for t in FINTEXTS_TICKERS
            if os.path.exists(f"results/fintexts_perfield/{t}.json")]
    if tick:
        by_name["FinTexTS"] = (tick, True)
    rows = [r for r in rows if r["name"] in by_name]
    for r in rows:
        r["_metrics"], r["_pooled"] = by_name[r["name"]]
    return rows, collect(rows, quantity)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results (info metrics or model performance).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Info-metrics mode
    parser.add_argument("results", nargs="*",
                        help="Path(s) to run.py JSON(s)")
    parser.add_argument("--figure1", action="store_true",
                        help="build Figure 1 (signal with example annotations) and exit")
    parser.add_argument("--fig1-data", default="datasets/benchmark_splits/train.json",
                        help="split JSON Figure 1 draws from")
    parser.add_argument("--figure2", nargs="*", metavar="SWEEP_JSON",
                        help="build Figure 2 (ordering grid + encoder sensitivity). "
                             "Pass the sweep JSONs, or none to look in results/mmtt_bench/")
    parser.add_argument("--figure5", action="store_true",
                        help="build Figure 5 (real-world MI grid) from results/ and exit")
    parser.add_argument("--fig5-quantity", default="joint", choices=["joint", "conditional"],
                        help="joint: I(ts,text;Y) against the series-only rule. "
                             "conditional: I(text;Y|ts) against a rule at zero")
    parser.add_argument("--fig5-no-mse", action="store_true",
                        help="drop the downstream-MSE column from Figure 5")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save plots (default: show interactively)")

    # Compare mode
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Path(s) to run.py single-result JSON(s)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Path(s) to train.py JSON(s), matching --metrics order")
    parser.add_argument("--model-filter", nargs="*", default=None,
                        help="Only show these model keys (e.g. ridge mlp)")
    parser.add_argument("--fix-model", default=None, metavar="BACKBONE",
                        help="Fix one backbone and vary injection modes "
                             "(e.g. --fix-model PatchTST). Strips backbone prefix "
                             "from compound 'backbone|mode' keys.")
    parser.add_argument("--fix-mode", default=None, metavar="MODE",
                        help="Fix one injection mode and vary backbones "
                             "(e.g. --fix-mode cfa). Strips mode suffix "
                             "from compound 'backbone|mode' keys.")
    parser.add_argument("--perf-metric", default="delta_mse",
                        choices=["delta_r2", "delta_rmse", "delta_mae", "delta_mse",
                                 "r2", "rmse", "mae", "mse"],
                        help="Performance metric to plot (default: delta_r2)")

    args = parser.parse_args()


    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _out(name: str) -> str | None:
        return str(out_dir / name) if out_dir else None

    if args.figure1:
        raw = json.load(open(args.fig1_data))
        plot_annotation_example(raw, save_path=_out("figure1_annotations.png"))
        return

    if args.figure2 is not None:
        paths = args.figure2 or sorted(glob.glob("results/mmtt_bench/sweep_*.json"))
        if not paths:
            parser.error("no sweep JSONs given and none found in results/mmtt_bench/ — "
                         "run reproduce/01_mmtt_bench.sh first")
        # The published figure shows PCA dim and lookback; the encoder and patch
        # sweeps are available as extra panels but were cut for width.
        CHOSEN = {"pca_dim": 16, "lookback_steps": 2, "patch_len": 1}
        NAMES = {"pca_dim": "PCA dim", "lookback_steps": "Lookback",
                 "model_name": "Encoder", "patch_len": "Patch/stride"}
        panels, encoder = [], None
        for path in paths:
            d = json.load(open(path))
            param = d["sweep_param"]
            if param == "model_name":
                encoder = d
                continue
            key = param[0] if isinstance(param, list) else param
            vals = d["sweep_values"]
            labels = ["/".join(str(x) for x in v) if isinstance(v, list) else str(v)
                      for v in vals]
            chosen = next((i for i, v in enumerate(vals)
                           if (v[0] if isinstance(v, list) else v) == CHOSEN.get(key)), None)
            panels.append({"name": NAMES.get(key, key), "runs": d["runs"],
                           "labels": labels, "chosen": chosen})
        if panels:
            plot_ordering_grid(panels, save_path=_out("figure2_ordering_grid.png"))
        if encoder is not None:
            labels = [str(v).split("/")[-1] for v in encoder["sweep_values"]]
            plot_embedding_sensitivity(encoder["runs"], labels,
                                       save_path=_out("figure2_encoder_sensitivity.png"))
        return

    if args.figure5:
        rows, data = load_realworld_results(args.fig5_quantity)
        if not rows:
            parser.error("no real-world results found under results/ — "
                         "run reproduce/02_timemmd.sh and 03_fintexts.sh first")
        print(f"Figure 5 from {len(rows)} datasets: {', '.join(r['name'] for r in rows)}")
        plot_realworld_mi_grid(rows, data, quantity=args.fig5_quantity,
                               show_mse=not args.fig5_no_mse,
                               save_path=_out("figure5_realworld_mi.png"))
        return

    # ── Default mode: --metrics + --models ───────────────────────────────────
    if args.metrics or args.models:
        if not args.metrics or not args.models:
            parser.error("--metrics and --models must be provided together")
        if len(args.metrics) != len(args.models):
            parser.error("--metrics and --models must have the same number of paths")

        labels = [Path(p).stem for p in args.metrics]

        for label, metrics_path, models_path in zip(labels, args.metrics, args.models):
            print(f"\n{'━' * 70}")
            print(f"  Dataset: {label}")
            print(f"  Metrics: {metrics_path}")
            print(f"  Models:  {models_path}")
            print(f"{'━' * 70}")

            metrics_data = load_metrics(metrics_path)
            model_data   = load_model_results(models_path)
            model_data   = filter_model_data(model_data, args.fix_model, args.fix_mode)

            all_deltas = compute_deltas(model_data)
            if args.model_filter:
                all_deltas = {k: v for k, v in all_deltas.items() if k in args.model_filter}

            categories   = model_data.get("categories", [])
            tables       = build_eval_table(metrics_data, all_deltas, categories)
            info_metrics = [m for m in CONDITIONAL_KEY if m in metrics_data]


            print_correlations(tables, info_metrics, label=label)

            raw_key = args.perf_metric[6:] if args.perf_metric.startswith("delta_") else args.perf_metric
            best_model_scores = {
                model: model_data["results"]["no_text"][model]["overall"].get(raw_key)
                for model in model_data["models"]
                if model in model_data["results"].get("no_text", {})
            }

            plot_correlation_trends(
                tables, info_metrics,
                scores_dict=best_model_scores,
                perf_metric=args.perf_metric,
                title=f"{label} — correlation trend ({args.perf_metric})",
                save_path=_out(f"{label}_correlation_trends_{args.perf_metric}.png"),
            )

            if "mixture" in Path(metrics_path).name or "mixture" in Path(models_path).name:
                plot_model_mixture(
                    tables, info_metrics, categories,
                    perf_metric=args.perf_metric,
                    title=f"{label} — mixture degradation ({args.perf_metric})",
                    save_path=_out(f"{label}_mixture_{args.perf_metric}.png"),
                )
        return

    # ── Sweep / overlay mode (positional results args) ────────────────────────
    if not args.results:
        parser.error("provide --metrics and --models to plot results, "
                     "or pass sweep JSON file(s) for sweep visualisation")

    datasets = [_load(p) for p in args.results]


    data = datasets[0]
    stem = Path(args.results[0]).stem

    if data["type"] == "single":
        parser.error("single-run JSONs no longer have a plot mode; the per-metric "
                     "tables they fed are produced by reproduce/make_artifacts.py")

    elif data["type"] == "sweep":
        # Figure 2, right panel: conditional MI per text encoder. The ordering
        # grid (left panel) needs several sweeps at once, so it is built by
        # reproduce/make_artifacts.py rather than from a single file here.
        labels = [str(v) for v in data["sweep_values"]]
        plot_embedding_sensitivity(
            data["runs"], labels,
            save_path=_out(f"{stem}_sensitivity.png"),
        )
if __name__ == "__main__":
    main()
