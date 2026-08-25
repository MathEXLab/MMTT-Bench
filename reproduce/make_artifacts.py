#!/usr/bin/env python
"""
make_artifacts.py — regenerate every table and figure in the paper from saved
results, without rerunning any experiment.

Reads only what 01/02/03_*.sh wrote under results/ and emits LaTeX tables to
paper_artifacts/tables/ and figures to paper_artifacts/figures/. Nothing here
trains a model or estimates mutual information, so it runs in seconds and can be
re-run freely while a figure is being tuned.

    python reproduce/make_artifacts.py                  # everything available
    python reproduce/make_artifacts.py --which mmtt_bench
    python reproduce/make_artifacts.py --list           # what maps to what

Missing inputs are reported and skipped rather than raising, so a partial run
(say, MMTT-Bench only) still produces its share of the artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_TABLES = ROOT / "paper_artifacts" / "tables"
OUT_FIGURES = ROOT / "paper_artifacts" / "figures"

CATEGORIES = ["correct", "incorrect", "irrelevant"]
ESTIMATORS = [
    ("mutual_information", "KSG"), ("mine", "MINE"), ("infonce", "InfoNCE"),
    ("cca", "CCA"), ("deep_cca", "Deep CCA"), ("v_information", "V-information"),
    ("pid", r"PID (U$_{\text{text}}$)"),
]

# artifact -> (produced by, reads)
MANIFEST = [
    ("Table 4  encoder sweep",        "01", "mmtt_bench/sweep_embedding.json"),
    ("Table 5  PCA sweep",            "01", "mmtt_bench/sweep_pca.json"),
    ("Table 6  lookback sweep",       "01", "mmtt_bench/sweep_lookback.json"),
    ("Table 7  patch sweep",          "01", "mmtt_bench/sweep_patch.json"),
    ("Table 8  conditional MI",       "01", "mmtt_bench/metrics.json"),
    ("Table 9  downstream MSE",       "01", "mmtt_bench/sklearn.json + transformers/"),
    ("Table 10 shuffle control",      "01", "mmtt_bench/{metrics,shuffle}.json"),
    ("Table 11 observation noise",    "01", "mmtt_bench/noise.json"),
    ("Table 12 temporal jitter",      "01", "mmtt_bench/coupling_jitter.json + _sklearn"),
    ("Table 13 dataset features",     "-",  "datasets/ (computed directly)"),
    ("Table 17 alignment/conditional", "02/03", "timemmd/mi/ + fintexts/mi/"),
    ("Table 18 real-world MSE",       "02/03", "timemmd/transformer/ + fintexts/transformer/"),
    ("Figure 2 parameter sweeps",     "01", "mmtt_bench/sweep_*.json"),
    ("Figure 3 MI vs MSE",            "01", "mmtt_bench/metrics.json + models"),
    ("Figure 4 mixture corpora",      "01", "mmtt_bench/mixture*.json"),
    ("Figure 5 real-world MI",        "02/03", "timemmd/mi/ + fintexts/mi/"),
    ("Figure 6 embeddings",           "01", "datasets/benchmark_splits/"),
]


# ── helpers ─────────────────────────────────────────────────────────────────
def load(rel: str):
    p = RESULTS / rel
    return json.load(open(p)) if p.exists() else None


def conditional(md: dict, est: str, cat: str) -> float:
    """Conditional MI for one estimator/category, in that estimator's own terms.

    PID decomposes rather than conditioning, so its comparable quantity is the
    unique-information atom; V-information reports a ridge Delta-R^2 and stores
    it under a v_ prefix.
    """
    if est == "pid":
        return float(np.mean(md["unique_text"][cat]))
    pre = "v_" if est == "v_information" else ""
    return float(md[f"{pre}conditional_mean"][cat])


def conditional_std(md: dict, est: str, cat: str) -> float:
    if est == "pid":
        return float(np.std(md["unique_text"][cat]))
    pre = "v_" if est == "v_information" else ""
    return float(md[f"{pre}conditional_std"][cat])


def latex_table(rows, header, caption, label, align=None, note=None) -> str:
    align = align or ("l" + "c" * (len(header) - 1))
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", "\\small", f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}",
           "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\\\[2pt]\\footnotesize {note}")
    out += ["\\end{table}", ""]
    return "\n".join(out)


def write(name: str, body: str) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / name).write_text(body)
    print(f"  wrote tables/{name}")


def skip(name: str, why: str) -> None:
    print(f"  [skip] {name} — {why}")


# ── tables ──────────────────────────────────────────────────────────────────
def table_8() -> None:
    d = load("mmtt_bench/metrics.json")
    if not d:
        return skip("Table 8", "results/mmtt_bench/metrics.json missing (run 01)")
    md = d["results_by_metric"]
    rows = [[name] + [f"${conditional(md[e], e, c):.3f} \\pm {conditional_std(md[e], e, c):.3f}$"
                      for c in CATEGORIES]
            for e, name in ESTIMATORS if e in md]
    write("table08_conditional_mi.tex", latex_table(
        rows, ["Estimator", "Correct", "Incorrect", "Irrelevant"],
        r"Conditional mutual information $I(X_{\text{text}}; Y \mid X_{\text{ts}})$ on "
        r"MMTT-Bench, in nats, mean $\pm$ standard deviation over 20 bootstrap resamples.",
        "tab:conditional-mi"))


def table_sweep(rel: str, param_label: str, num: str, caption: str) -> None:
    d = load(rel)
    if not d:
        return skip(f"Table {num}", f"results/{rel} missing (run 01)")
    rows = []
    for rec in d["records"]:
        md, val = rec["results_by_metric"], rec["sweep_value"]
        val = ", ".join(str(v) for v in val) if isinstance(val, list) else val
        for e, name in ESTIMATORS:
            if e in md:
                rows.append([name, str(val)] + [f"{conditional(md[e], e, c):.3f}" for c in CATEGORIES])
    # group by estimator, blanking repeats so the table reads as blocks
    rows.sort(key=lambda r: [n for _, n in ESTIMATORS].index(r[0]))
    seen = set()
    for r in rows:
        if r[0] in seen:
            r[0] = ""
        else:
            seen.add(r[0])
    write(f"table{num}_{rel.split('/')[-1].replace('.json','')}.tex", latex_table(
        rows, ["Estimator", param_label, "Correct", "Incorrect", "Irrelevant"],
        caption, f"tab:sweep-{num}"))


def table_10() -> None:
    pre, post = load("mmtt_bench/metrics.json"), load("mmtt_bench/shuffle.json")
    if not (pre and post):
        return skip("Table 10", "metrics.json or shuffle.json missing (run 01)")
    a, b = pre["results_by_metric"], post["results_by_metric"]
    rows = []
    for c in CATEGORIES:
        row = [c.capitalize()]
        for e in ("mine", "infonce"):
            if e in a and e in b:
                row += [f"${conditional(a[e], e, c):.2f}\\pm{conditional_std(a[e], e, c):.2f}$",
                        f"${conditional(b[e], e, c):.2f}\\pm{conditional_std(b[e], e, c):.2f}$"]
        rows.append(row)
    write("table10_shuffle.tex", latex_table(
        rows, ["Category", "MINE pre", "MINE post", "InfoNCE pre", "InfoNCE post"],
        "Conditional MI before and after shuffling annotations across time points. "
        "A drop confirms the signal lives in the (text, $Y_{\\text{future}}$) pairing "
        "rather than in the embedding distribution.", "tab:shuffle"))


def table_12() -> None:
    mi, mdl = load("mmtt_bench/coupling_jitter.json"), load("mmtt_bench/coupling_jitter_sklearn.json")
    if not mi:
        return skip("Table 12", "results/mmtt_bench/coupling_jitter.json missing (run 01)")
    md = mi["results_by_metric"]
    keys = [("correct", "0 (correct)")] + sorted(
        [(k, k.rsplit("_", 1)[1]) for k in md["cca"]["conditional_mean"] if k.startswith("coupling_jitter_")],
        key=lambda t: float(t[1]))
    ests = [(e, n) for e, n in ESTIMATORS if e in md and e != "deep_cca"]
    rows = []
    for key, label in keys:
        row = [label] + [f"{conditional(md[e], e, key):.3f}".replace("-", "$-$") for e, _ in ests]
        if mdl:
            row.append(f"{np.mean([mdl['results'][key][m]['overall']['mse'] for m in mdl['models']]):.3f}")
        rows.append(row)
    header = [r"Jitter $\sigma$"] + [n for _, n in ests] + (["mean MSE"] if mdl else [])
    write("table12_jitter.tex", latex_table(
        rows, header,
        "MI estimates under temporal jitter, where each annotation is replaced by the correct "
        "text of a nearby timepoint, $\\sigma$ being the standard deviation of the displacement "
        "in annotation steps.", "tab:jitter"))


def table_13() -> None:
    """Dataset characteristics, computed from the splits rather than from results."""
    def series(path):
        d = json.load(open(path))
        pts = {p["t"]: p["y"] for p in d["points"] if p.get("category") == "correct"}
        t = np.array(sorted(pts))
        return t, np.array([pts[k] for k in t], float), d.get("metadata", {})

    def acf(v, lag):
        if lag >= len(v) - 2:
            return np.nan
        a, b = v[:-lag] - v[:-lag].mean(), v[lag:] - v[lag:].mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / den) if den else np.nan

    def stats(y, period):
        n = len(y); x = np.arange(n, dtype=float)
        fit = np.vstack([x, np.ones(n)]).T @ np.linalg.lstsq(
            np.vstack([x, np.ones(n)]).T, y, rcond=None)[0]
        sst = ((y - y.mean()) ** 2).sum()
        h = n // 2
        return dict(
            trend=float(1 - ((y - fit) ** 2).sum() / sst) if sst else np.nan,
            lag1=acf(y, 1),
            # differenced first, so persistence is not mistaken for seasonality
            seas=acf(np.diff(y), period),
            vr=float(np.var(y[h:]) / np.var(y[:h])) if np.var(y[:h]) else np.nan,
            ms=float(abs(y[h:].mean() - y[:h].mean()) / np.std(y)) if np.std(y) else np.nan)

    specs = [("MMTT-Bench", ROOT / "datasets/benchmark_splits", "synthetic", 24)]
    for name, per, freq in [("Agriculture", 12, "monthly"), ("Climate", 12, "monthly"),
                            ("Energy", 52, "weekly"), ("PublicHealth", 52, "weekly"),
                            ("SocialGood", 12, "monthly"), ("Traffic", 12, "monthly")]:
        specs.append((name, ROOT / f"datasets/timemmd_cfa_splits/{name}", freq, per))

    rows = []
    for name, base, freq, period in specs:
        tr = base / "train.json"
        if not tr.exists():
            continue
        _, y, _ = series(tr)
        counts = [sum(1 for q in json.load(open(base / f"{s}.json"))["points"]
                      if q.get("category") == "correct") if (base / f"{s}.json").exists() else 0
                  for s in ("train", "val", "test")]
        st = stats(y, period)
        rows.append([name, freq, *[str(c) for c in counts],
                     f"{st['trend']:.2f}", f"{st['lag1']:.2f}", f"{st['seas']:.2f}",
                     f"{st['vr']:.2f}", f"{st['ms']:.2f}"])
    if not rows:
        return skip("Table 13", "no dataset splits found")
    write("table13_dataset_features.tex", latex_table(
        rows, ["Dataset", "Frequency", "Train", "Val", "Test", "Trend $R^2$",
               "Lag-1 ACF", "Seasonal ACF", "Var. ratio", "Mean shift"],
        "MMTT-Bench and real-world dataset feature comparison. Counts are annotated points per "
        "text category. Seasonal ACF is computed on the first-differenced series at the natural "
        "period so that persistence is not mistaken for seasonality; variance ratio and mean "
        "shift compare the two halves of the series and together indicate departure from "
        "stationarity.", "tab:dataset-features", align="ll" + "r" * 8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all",
                    choices=["all", "mmtt_bench", "timemmd", "fintexts"])
    ap.add_argument("--list", action="store_true", help="show the artifact manifest and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'artifact':34s} {'script':7s} reads")
        for a, s, r in MANIFEST:
            print(f"{a:34s} {s:7s} results/{r}")
        return

    print(f"Reading results from {RESULTS}\n")
    if args.which in ("all", "mmtt_bench"):
        print("MMTT-Bench:")
        table_8()
        table_sweep("mmtt_bench/sweep_embedding.json", "Encoder", "04",
                    "MI estimator results for different text embedding models.")
        table_sweep("mmtt_bench/sweep_pca.json", r"PCA dim $d_{\text{text}}$", "05",
                    "MI estimator results for different PCA dimensions.")
        table_sweep("mmtt_bench/sweep_lookback.json", r"$n_{\text{lookback}}$", "06",
                    "MI estimator results for different numbers of lookback steps, with no "
                    "time-series tokenisation.")
        table_sweep("mmtt_bench/sweep_patch.json", "patch, stride, lookback", "07",
                    "MI estimator results under PatchTST tokenisation.")
        table_10()
        table_12()
    if args.which in ("all", "timemmd", "fintexts"):
        print("Real-world:")
        table_13()

    print(f"\nTables in {OUT_TABLES.relative_to(ROOT)}/")
    print("Figures are not yet wired up — pending the final figure selection.")


if __name__ == "__main__":
    main()
