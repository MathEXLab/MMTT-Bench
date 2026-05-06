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
import json
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
    plot_overlay,
    print_single_table,
    plot_single_bar,
    plot_sweep_lines,
    plot_sweep_all_metrics,
    plot_conditional_comparison,
    plot_conditional_sweep,
    print_sweep_table,
    plot_info_vs_delta,
    plot_correlation_trends,
    plot_delta_bars,
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
            model_deltas["delta_r2"][cat]   = text_result["r2"]   - baseline["r2"]
            model_deltas["delta_rmse"][cat] = text_result["rmse"] - baseline["rmse"]
            model_deltas["delta_mae"][cat]  = text_result["mae"]  - baseline["mae"]
            model_deltas["delta_mse"][cat]  = text_result["mse"]  - baseline["mse"]
            model_deltas["r2"][cat]         = text_result["r2"]
            model_deltas["rmse"][cat]       = text_result["rmse"]
            model_deltas["mae"][cat]        = text_result["mae"]
            model_deltas["mse"][cat]        = text_result["mse"]

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
        if not all(deltas.values()):
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results (info metrics or model performance).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Info-metrics mode
    parser.add_argument("results", nargs="*",
                        help="Path(s) to run.py JSON(s)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save plots (default: show interactively)")
    parser.add_argument("--submetric", default=None,
                        help="MI submetric to highlight in sweep line plot")
    parser.add_argument("--overlay", action="store_true",
                        help="Overlay multiple sweep files (requires ≥2 files)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Display labels for overlay or compare mode")

    # Compare mode
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Path(s) to run.py single-result JSON(s)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Path(s) to train.py JSON(s), matching --metrics order")
    parser.add_argument("--model-filter", nargs="*", default=None,
                        help="Only show these model keys (e.g. ridge mlp)")
    parser.add_argument("--perf-metric", default="delta_r2",
                        choices=["delta_r2", "delta_rmse", "delta_mae", "delta_mse",
                                 "r2", "rmse", "mae", "mse"],
                        help="Performance metric to plot (default: delta_r2)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _out(name: str) -> str | None:
        return str(out_dir / name) if out_dir else None

    # ── Default mode: --metrics + --models ───────────────────────────────────
    if args.metrics or args.models:
        if not args.metrics or not args.models:
            parser.error("--metrics and --models must be provided together")
        if len(args.metrics) != len(args.models):
            parser.error("--metrics and --models must have the same number of paths")

        labels = args.labels or [Path(p).stem for p in args.metrics]

        for label, metrics_path, models_path in zip(labels, args.metrics, args.models):
            print(f"\n{'━' * 70}")
            print(f"  Dataset: {label}")
            print(f"  Metrics: {metrics_path}")
            print(f"  Models:  {models_path}")
            print(f"{'━' * 70}")

            metrics_data = load_metrics(metrics_path)
            model_data   = load_model_results(models_path)

            all_deltas = compute_deltas(model_data)
            if args.model_filter:
                all_deltas = {k: v for k, v in all_deltas.items() if k in args.model_filter}

            categories   = model_data.get("categories", [])
            tables       = build_eval_table(metrics_data, all_deltas, categories)
            info_metrics = [m for m in CONDITIONAL_KEY if m in metrics_data]

            # Bar chart per metric family
            for metric_name, results in metrics_data.items():
                print_single_table(results, label=METRIC_LABELS.get(metric_name, metric_name))
                plot_single_bar(
                    results,
                    title=METRIC_DISPLAY.get(metric_name, metric_name),
                    save_path=_out(f"{label}_{metric_name}.png"),
                )

            print_correlations(tables, info_metrics, label=label)

            raw_key = args.perf_metric[6:] if args.perf_metric.startswith("delta_") else args.perf_metric
            best_model_scores = {
                model: model_data["results"]["no_text"][model]["overall"][raw_key]
                for model in model_data["models"]
            }

            plot_info_vs_delta(
                tables, info_metrics,
                scores_dict=best_model_scores,
                perf_metric=args.perf_metric,
                save_path=_out(f"{label}_scatter_{args.perf_metric}.png"),
            )
            plot_correlation_trends(
                tables, info_metrics,
                scores_dict=best_model_scores,
                perf_metric=args.perf_metric,
                title=f"{label} — correlation trend ({args.perf_metric})",
                save_path=_out(f"{label}_correlation_trends_{args.perf_metric}.png"),
            )
            plot_delta_bars(
                tables,
                perf_metric=args.perf_metric,
                title=f"{label} — {args.perf_metric} by category",
                save_path=_out(f"{label}_delta_bars_{args.perf_metric}.png"),
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

    if args.overlay:
        if len(datasets) < 2:
            parser.error("--overlay requires at least 2 result files")
        labels    = args.labels or [Path(p).stem for p in args.results]
        submetric = args.submetric or "conditional"
        plot_overlay(datasets, labels, metric=submetric, save_path=_out("overlay.png"))
        return

    data = datasets[0]
    stem = Path(args.results[0]).stem

    if data["type"] == "single":
        results_by_metric = _results_by_metric(data)
        for metric_name, results in results_by_metric.items():
            print_single_table(results, label=METRIC_LABELS.get(metric_name, metric_name))
            plot_single_bar(results, title=METRIC_DISPLAY.get(metric_name, metric_name),
                            save_path=_out(f"{stem}_{metric_name}.png"))
        plot_conditional_comparison(
            results_by_metric, title=stem,
            save_path=_out(f"{stem}_conditional.png"),
        )

    elif data["type"] == "sweep":
        runs_by_metric = _runs_by_metric(data)
        for metric_name, runs in runs_by_metric.items():
            sweep_data = {
                "sweep_param":  data["sweep_param"],
                "sweep_values": data["sweep_values"],
                "runs":         runs,
            }
            print_sweep_table(sweep_data)
            plot_sweep_all_metrics(sweep_data,
                                   title=METRIC_DISPLAY.get(metric_name, metric_name),
                                   save_path=_out(f"{stem}_{metric_name}_sweep.png"))
            if args.submetric:
                plot_sweep_lines(sweep_data, metric=args.submetric,
                                 save_path=_out(f"{stem}_{metric_name}_{args.submetric}.png"))
        plot_conditional_sweep(
            runs_by_metric,
            sweep_param=data["sweep_param"],
            sweep_values=data["sweep_values"],
            title=stem,
            save_path=_out(f"{stem}_conditional_sweep.png"),
        )


if __name__ == "__main__":
    main()
