# benchmark/viz.py
"""
Visualisation

All plot functions accept save_path=None (show) or a filepath (save).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr

plt.rcParams.update({'font.size': 14})

CAT_COLORS = {
    'correct':              '#1D9E75',  # teal-green
    'incorrect':            '#D85A30',  # red-orange
    'irrelevant':           '#7F8C8D',  # grey
    'synergy':              '#8E44AD',  # purple
    'no_text':              '#4A90D9',  # steel blue — time series only baseline
    # incorrect mixtures: bright teal → golden-yellow → amber
    'mixture_25_incorrect': '#2EC49A',  # bright teal   (75% correct, 25% incorrect)
    'mixture_50_incorrect': '#C4B020',  # golden yellow (50 / 50)
    'mixture_75_incorrect': '#D4821A',  # amber orange  (25% correct, 75% incorrect)
    # irrelevant mixtures: dark green → dark blue → dark purple
    'mixture_25_irrelevant': '#14704D', # dark forest green
    'mixture_50_irrelevant': '#1B4D8F', # dark navy blue
    'mixture_75_irrelevant': '#5C1A7A', # dark purple
}

CAT_LABELS = {
    'correct':               'Correct',
    'incorrect':             'Incorrect',
    'irrelevant':            'Irrelevant',
    'no_text':               'Time series only',
    'mixture_25_incorrect':  r'Mix 25% incorrect',
    'mixture_50_incorrect':  r'Mix 50% incorrect',
    'mixture_75_incorrect':  r'Mix 75% incorrect',
    'mixture_25_irrelevant': r'Mix 25% irrelevant',
    'mixture_50_irrelevant': r'Mix 50% irrelevant',
    'mixture_75_irrelevant': r'Mix 75% irrelevant',
}

PHASE_LABELS = {
    "ascending_zero":   "Asc. zero",
    "peak":             "Peak",
    "descending_zero":  "Desc. zero",
    "trough":           "Trough",
    "drop_to_zero":     "Drop→constant",
    "recover_from_zero":"Rec. constant",
}


METRIC_LABELS = {
    "text":        r"$I(X_{\rm text};\,Y)$",
    "ts":          r"$I(X_{\rm ts};\,Y)$",
    "joint":       r"$I(X_{\rm ts}, X_{\rm text};\,Y)$",
    "conditional": r"$I(Y;\,X_{\rm text}\mid X_{\rm ts})$",
}

# The submetric key that represents "unique text contribution" for each metric family.
CONDITIONAL_KEY = {
    "mutual_information": "conditional",
    "pid":                "unique_text", # synergy
    "v_information":      "v_conditional",
    "mine": "conditional",
    "infonce": "conditional",
    "cca": "conditional",
}
CONDITIONAL_LABEL = {
    "mutual_information": r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$" + "\n(nats)",
    "pid":                r"Unique($X_{\rm text}$)" + "\n(nats)",
    "v_information":      r"$\Delta R^2$" + r"$(X_{\rm text} \mid X_{\rm ts})$",
    "mine": r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
    "infonce": r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
    "cca": r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
}
METRIC_DISPLAY = {
    "mutual_information": "KSG MI",
    "pid":                "PID",
    "v_information":      "V-information",
    "mine": "MINE",
    "infonce":                "InfoNCE",
    "cca":      "CCA",
}

MODEL_DISPLAY = {
    'sentence-transformers/average_word_embeddings_glove.840B.300d': 'GloVe',
    'sentence-transformers/all-MiniLM-L6-v2' : 'all-MiniLM',
    'sentence-transformers/all-distilroberta-v1': 'DistilRoBERTa'
}

# ---------------------------------------------------------------------------
# Compare plots — model performance vs information metrics
# ---------------------------------------------------------------------------

PLOT_POS = {
    "mutual_information": (0, 0),
    "pid":                (1, 0),
    "v_information":      (1, 1),
    "mine":               (0, 1),
    "infonce":            (0, 2),
    "cca":                (1, 2),
}

INFO_METRIC_LABEL = {
    "mutual_information": r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
    "pid":                r"Unique($X_{\rm text}$)",
    "v_information":      r"$\Delta R^2$ $(X_{\rm text} \mid X_{\rm ts})$",
    "mine":               r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
    "infonce":            r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
    "cca":                r"$I(X_{\rm text} \to Y \mid X_{\rm ts})$",
}

PERF_METRIC_LABEL = {
    "delta_r2":   r"$\Delta R^2$ ( f($X_{\rm joint}$) − f($X_{\rm ts}$) )",
    "delta_rmse": r"$\Delta$ RMSE ($f(X_{\rm joint}) − f(X_{\rm ts})$)",
    "delta_mae":  r"$\Delta$ MAE ($f(X_{\rm joint}) − f(X_{\rm ts})$)",
    "delta_mse":  r"$\Delta$ MSE ($f(X_{\rm joint}) − f(X_{\rm ts})$)",
    "r2":         r"$R^2$",
    "rmse":       r"RMSE",
    "mae":        r"MAE",
    "mse":        r"MSE",
}

MODEL_MARKERS = {
    # sklearn baselines
    'ridge': '^',
    'mlp':   'o',
    'svr':   'X',
    'gbr':   'D',
    'rf':    '*',
    'knn':   'P',
    # Transformer backbone models (TaTS + CFA)
    'Autoformer':              's',
    'Crossformer':             'p',
    'DLinear':                 'h',
    'FEDformer':               'H',
    'FiLM':                    'v',
    'Informer':                '<',
    'PatchTST':                '>',
    'Transformer':             '1',
    'iTransformer':            '2',
    'Reformer':                '3',
    'TSMixer':                 '4',
    'TiDE':                    '8',
    'Nonstationary_Transformer': 'd',
    # CFA fusion methods — naive additive  (triangles)
    'first-additive':          '^',
    'middle-additive':         'v',
    'last-additive':           '<',
    # CFA fusion methods — naive concat  (polygons)
    'first-concat':            '>',
    'middle-concat':           's',
    'last-concat':             'p',
    # CFA fusion methods — constrained  (special shapes)
    'film':                    'D',
    'cfa':                     '*',
    'gating':                  'P',
    'orthogonal':              'X',
}

MODEL_COLORS = {
    # sklearn baselines
    'ridge': '#1f77b4',
    'mlp':   '#ff7f0e',
    'svr':   '#9deb30',
    'gbr':   '#e429bf',
    'rf':    '#9467bd',
    'knn':   '#8c564b',
    # Transformer backbone models (TaTS + CFA)
    'Autoformer':              '#d62728',
    'Crossformer':             '#e377c2',
    'DLinear':                 '#17becf',
    'FEDformer':               '#bcbd22',
    'FiLM':                    '#7f7f7f',
    'Informer':                '#aec7e8',
    'PatchTST':                '#ffbb78',
    'Transformer':             '#98df8a',
    'iTransformer':            '#ff9896',
    'Reformer':                '#2ecc71',
    'TSMixer':                 '#3498db',
    'TiDE':                    '#e74c3c',
    'Nonstationary_Transformer': '#9b59b6',
    # CFA fusion methods — naive additive
    'first-additive':          '#e67e22',
    'middle-additive':         '#d35400',
    'last-additive':           '#f39c12',
    # CFA fusion methods — naive concat
    'first-concat':            '#2980b9',
    'middle-concat':           '#1abc9c',
    'last-concat':             '#16a085',
    # CFA fusion methods — constrained
    'film':                    '#27ae60',
    'cfa':                     '#c0392b',
    'gating':                  '#8e44ad',
    'orthogonal':              '#2c3e50',
}

MODEL_LINESTYLES = {
    # sklearn baselines
    'ridge': '-',
    'mlp':   '--',
    'svr':   '-.',
    'gbr':   ':',
    'rf':    (0, (3, 1, 1, 1)),
    'knn':   (0, (5, 2)),
    # Transformer backbone models (TaTS + CFA)
    'Autoformer':              (0, (5, 5)),
    'Crossformer':             (0, (1, 1)),
    'DLinear':                 (0, (5, 1)),
    'FEDformer':               (0, (3, 5, 1, 5)),
    'FiLM':                    (0, (1, 5)),
    'Informer':                (0, (3, 1, 1, 1, 1, 1)),
    'PatchTST':                (0, (5, 2, 1, 2)),
    'Transformer':             (0, (3, 3)),
    'iTransformer':            (0, (1, 2, 5, 2)),
    'Reformer':                (0, (4, 2, 4, 2)),
    'TSMixer':                 (0, (2, 2)),
    'TiDE':                    (0, (6, 2)),
    'Nonstationary_Transformer': (0, (4, 1, 1, 1, 1, 1)),
    # CFA fusion methods — naive additive  (solid/dashed/dash-dot)
    'first-additive':          '-',
    'middle-additive':         '--',
    'last-additive':           '-.',
    # CFA fusion methods — naive concat  (dense dot patterns)
    'first-concat':            ':',
    'middle-concat':           (0, (5, 2)),
    'last-concat':             (0, (5, 5)),
    # CFA fusion methods — constrained  (complex dash patterns)
    'film':                    (0, (1, 1)),
    'cfa':                     (0, (3, 1, 1, 1)),
    'gating':                  (0, (5, 1, 1, 1)),
    'orthogonal':              (0, (3, 3)),
}

INFO_LINESTYLES = {
    'mutual_information': '-',
    'mine':               ':',
    'infonce':            '-.',
    'pid':                '--',
    'v_information':      (0, (3, 1, 1, 1, 1, 1)),
    'cca':                (0, (5, 10)),
}

INFO_COLORS = {
    'mutual_information': '#1f77b4',  # blue
    'mine':               '#ff7f0e',  # orange
    'infonce':            '#2ca02c',  # green
    'cca':                '#d62728',  # red
    'v_information':      '#9467bd',  # purple
    'pid':                '#8c564b',  # brown
}

# x-axis position for each category: % of correct annotations
CAT_X_MAP = {
    'correct':               100,
    'incorrect':               0,
    'irrelevant':              0,
    'mixture_25_incorrect':   75,
    'mixture_50_incorrect':   50,
    'mixture_75_incorrect':   25,
    'mixture_25_irrelevant':  75,
    'mixture_50_irrelevant':  50,
    'mixture_75_irrelevant':  25,
}

def plot_mi_per_category(
    category_data: dict,
    results:       dict,
    metrics:       list,
    save_path:     str = None,
) -> tuple[dict, dict]:
    """
    Bar chart of bootstrapped MI ± std per annotation category.

    Parameters
    ----------
    category_data : {cat: {'ts', 'y', 'x'}} — used only for cat ordering
    results       : output of MutualInformationMetric.compute() or
                    a compatible metric — same dict structure
    save_path     : filepath to save, or None to plt.show()

    Returns
    -------
    (means, stds) dicts — {metric: {cat: value}}
    """
    cats    = list(category_data.keys())

    means = {m: {c: np.mean(results[m][c]) for c in cats} for m in metrics}
    stds  = {m: {c: np.std(results[m][c])  for c in cats} for m in metrics}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=False)
    fig.patch.set_facecolor('white')

    x_pos = np.arange(len(cats))
    width = 0.6

    for ax, metric in zip(axes, metrics):
        m_means = [means[metric][c] for c in cats]
        m_stds  = [stds[metric][c]  for c in cats]
        colors  = [CAT_COLORS.get(c, '#888888') for c in cats]

        ax.bar(x_pos, m_means, width=width, color=colors,
               alpha=0.85, zorder=3)
        ax.errorbar(x_pos, m_means, yerr=m_stds, fmt='none',
                    color='#333333', capsize=4, capthick=1.0,
                    elinewidth=1.0, zorder=4)
        ax.axhline(0, color='#aaaaaa', lw=0.8, zorder=2)

        if metric == 'conditional':
            ax.axhspan(-0.05, 0.05, color='#eeeeee', zorder=1,
                       label='≈ zero band')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [CAT_LABELS.get(c, c) for c in cats],
            rotation=35, ha='right', fontsize=8,
        )
        ax.set_ylabel('MI (nats)', fontsize=9)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, pad=8)
        ax.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    fig.suptitle('MI per annotation category',
                 fontsize=11, fontweight='normal', y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)

    return means, stds

# plot_2d_embeddings 
# Shows whether categories cluster in the text embedding space.

def plot_2d_embeddings(
    train_data:    dict,
    val_data:      dict,
    save_path:     str = None,
) -> None:
    """
    Two figures (train / val), each with two panels:
      left  — X_text embeddings (2D PCA on text features only)
      right — X_joint embeddings (2D PCA on text + time-series features)

    Points are coloured by annotation category.  A 1-std confidence
    ellipse is drawn for each category.
    """
    from sklearn.decomposition import PCA

    def _make_arrays(category_data):
        all_x, all_ts, labels = [], [], []
        for cat, vals in category_data.items():
            for j in range(len(vals['x'])):
                all_x.append(vals['x'][j])
                all_ts.append(vals['ts'][j])
                labels.append(cat)
        return np.array(all_x), np.array(all_ts), np.array(labels)

    def _scatter_panel(ax, coords, labels, ev, title):
        unique_groups = list(dict.fromkeys(labels))
        for group in unique_groups:
            mask = labels == group
            pts  = coords[mask]
            col  = CAT_COLORS.get(group, '#888888')
            lbl  = CAT_LABELS.get(group, group)
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=col, alpha=0.7, s=28,
                       edgecolors='white', linewidths=0.5,
                       label=lbl, zorder=3)
        ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='normal')
        ax.legend(fontsize=8, frameon=True, markerscale=1.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    for split_name, category_data in [('train', train_data), ('val', val_data)]:
        all_x, all_ts, labels = _make_arrays(category_data)
        all_joint = np.hstack([all_x, all_ts])

        pca_text  = PCA(n_components=2, random_state=42).fit(all_x)
        pca_joint = PCA(n_components=2, random_state=42).fit(all_joint)

        coords_text  = pca_text.transform(all_x)
        coords_joint = pca_joint.transform(all_joint)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'{split_name.capitalize()} split', fontsize=12, y=1.01)
        fig.patch.set_facecolor('white')

        _scatter_panel(axes[0], coords_text,  labels,
                       pca_text.explained_variance_ratio_,
                       'Text embeddings (2D PCA)')
        _scatter_panel(axes[1], coords_joint, labels,
                       pca_joint.explained_variance_ratio_,
                       'Joint embeddings — text + TS (2D PCA)')

        plt.tight_layout()

        if save_path:
            base, ext = (save_path.rsplit('.', 1) if '.' in save_path
                         else (save_path, 'png'))
            path = f'{base}_{split_name}.{ext}'
        else:
            path = None
        _save_or_show(path)

def plot_pmi_by_phase(pmi_results, title="Mean PMI by phase and text category",
                      ylabel="Mean PMI", figsize=(11, 4.5), savepath=None):
    # ── fix key if user accidentally used 'incorrect' twice ──────────────────
    cats = list(pmi_results.keys())
    phases = list(next(iter(pmi_results.values())).keys())
    phase_labels = [PHASE_LABELS.get(p, p) for p in phases]

    n_cats   = len(cats)
    n_phases = len(phases)
    width    = 0.22 if len(cats)==3 else 0.05
    offsets  = np.linspace(-(n_cats - 1) / 2 * width,
                            (n_cats - 1) / 2 * width, n_cats)
    x = np.arange(n_phases)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for offset, cat in zip(offsets, cats):
        means = np.array([pmi_results[cat][p]["mean"] for p in phases])
        stds  = np.array([pmi_results[cat][p]["std"]  for p in phases])
        ns    = np.array([pmi_results[cat][p]["n"]    for p in phases])
        sems  = stds / np.sqrt(ns)          # use SEM for error bars; swap for std if preferred

        color = CAT_COLORS.get(cat, "#888")
        ax.bar(x + offset, means, width * 0.9,
               color=color, alpha=0.85, label=CAT_LABELS.get(cat, cat), zorder=3)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="#333", capsize=4, capthick=1,
                    elinewidth=1, zorder=4)

    ax.axhline(0, color="#aaa", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee", zorder=0)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def _apply_sci_x(ax) -> None:
    """Scientific notation on x-axis when values fall outside [1e-3, 1e3]."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(fmt)


def plot_info_vs_delta(
    tables:      dict,
    info_metrics: list[str],
    scores_dict:  dict,
    perf_metric:  str = "delta_r2",
    title:        str = "Information Metric vs Change in Model Performance",
    save_path=None,
) -> None:
    """Scatter: info metric (x) vs performance metric (y), one panel per info metric."""
    n_info = len(info_metrics)
    if not tables or n_info == 0:
        return

    n_rows, n_cols = (2, 3) if n_info > 3 else (1, n_info)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 5.0 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for _row, (model_name, df) in enumerate(tables.items()):
        for col, info_m in enumerate(info_metrics):
            ax = axes[PLOT_POS[info_m][0]][PLOT_POS[info_m][1]] if n_rows > 1 else axes[0][col]
            if info_m not in df.columns or perf_metric not in df.columns:
                ax.set_visible(False)
                continue
            for cat in df.index:
                x_val = df.loc[cat, info_m]
                y_val = df.loc[cat, perf_metric]
                if np.isnan(x_val) or np.isnan(y_val):
                    continue
                ax.scatter(x_val, y_val, facecolors='none',
                           edgecolors=CAT_COLORS.get(cat, "#888"),
                           zorder=3, s=150,
                           label=CAT_LABELS.get(cat, cat),
                           marker=MODEL_MARKERS.get(model_name, 'o'))
                if _row == 1:
                    ax.annotate(CAT_LABELS.get(cat, cat), (x_val, y_val),
                                textcoords="offset points", xytext=(6, 4),
                                color=CAT_COLORS.get(cat, "#888"))
            ax.axhline(0, color="#aaa", lw=0.8, ls="--")
            ax.axvline(0, color="#aaa", lw=0.8, ls="--")
            ax.set_xlabel(INFO_METRIC_LABEL.get(info_m, info_m))
            ax.set_ylabel(PERF_METRIC_LABEL.get(perf_metric, perf_metric))
            ax.set_title(METRIC_DISPLAY.get(info_m, info_m), pad=6)
            _apply_sci_x(ax)
            ax.grid(lw=0.4, alpha=0.4)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontweight="normal", y=1.01)
    def _score_label(m):
        s = scores_dict.get(m)
        return f"{m}  score={round(s, 4)}" if s is not None else m
    model_legend = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color='black',
               label=_score_label(m),
               markerfacecolor='w', markersize=10)
        for m in MODEL_MARKERS if m in scores_dict
    ]
    fig.legend(handles=model_legend, handlelength=0, loc='outside lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=6,
               title='Model / fusion method with baseline score (time series only)')
    plt.tight_layout()
    _save_or_show(save_path)


def plot_correlation_trends(
    tables:       dict,
    info_metrics: list[str],
    scores_dict:  dict,
    perf_metric:  str = "delta_r2",
    title:        str = "Correlation Trend: Information vs Model Performance",
    save_path=None,
    show_ts_only:    bool = True,
    show_error_bars: bool = True,
) -> None:
    """
    Per info-metric panel: category-coloured scatter points + linear trend line
    per model. Shared y-axis across all panels.
    """
    n_info = len(info_metrics)
    if not tables or n_info == 0:
        return

    n_rows, n_cols = (2, 3) if n_info > 3 else (1, n_info)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 5.0 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    all_ys_global = []
    drawn_axes    = []

    for col, info_m in enumerate(info_metrics):
        ax = axes[PLOT_POS[info_m][0]][PLOT_POS[info_m][1]] if n_rows > 1 else axes[0][col]

        for model_name, df in tables.items():
            if info_m not in df.columns or perf_metric not in df.columns:
                continue
            std_col = f'{info_m}_std'
            xs, ys, x_stds, valid_cats = [], [], [], []
            for cat in df.index:
                x_val = df.loc[cat, info_m]
                y_val = df.loc[cat, perf_metric]
                if np.isnan(x_val) or np.isnan(y_val):
                    continue
                xs.append(x_val); ys.append(y_val); valid_cats.append(cat)
                x_stds.append(df.loc[cat, std_col] if std_col in df.columns else np.nan)
            if len(xs) < 2:
                continue

            xs = np.array(xs); ys = np.array(ys); x_stds = np.array(x_stds)

            # Drop any remaining non-finite values (can appear after array conversion)
            finite = np.isfinite(xs) & np.isfinite(ys)
            xs      = xs[finite]
            ys      = ys[finite]
            x_stds  = x_stds[finite]
            valid_cats = [c for c, ok in zip(valid_cats, finite) if ok]
            if len(xs) < 2:
                continue

            all_ys_global.extend(ys.tolist())

            for i, cat in enumerate(valid_cats):
                color = CAT_COLORS.get(cat, "#888")
                ax.scatter(xs[i], ys[i], facecolors='none',
                           edgecolors=color,
                           zorder=4, s=150, marker=MODEL_MARKERS.get(model_name, 'o'))
                if show_error_bars and np.isfinite(x_stds[i]):
                    ax.errorbar(xs[i], ys[i], xerr=x_stds[i], fmt='none',
                                ecolor=color, elinewidth=1.2, capsize=3, zorder=3)

            # Skip trend line if xs has no variance (polyfit is undefined)
            if xs.max() == xs.min():
                continue

            coeffs = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            all_ys_global.extend(y_line.tolist())

            rho, _ = spearmanr(xs, ys)
            ax.plot(x_line, y_line, color='black', lw=1.8,
                    linestyle=MODEL_LINESTYLES.get(model_name, '-'),
                    zorder=3, label=f"{model_name}  ρ={rho:+.2f}")

        ax.set_xlabel(INFO_METRIC_LABEL.get(info_m, info_m))
        ax.set_ylabel(PERF_METRIC_LABEL.get(perf_metric, perf_metric))
        ax.set_title(METRIC_DISPLAY.get(info_m, info_m), pad=6)
        _apply_sci_x(ax)
        ax.grid(lw=0.4, alpha=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        drawn_axes.append(ax)

    # Shared y-axis
    if all_ys_global:
        pad = (max(all_ys_global) - min(all_ys_global)) * 0.08 or 0.05
        for ax in drawn_axes:
            ax.set_ylim(min(all_ys_global) - pad, max(all_ys_global) + pad)
            # hard coding sometimes:
            #ax.set_ylim(0, 5000000000)

    # Time-series-only baseline: scatter point at x=0 per model, per panel
    if show_ts_only:
        for model_name, df in tables.items():
            if 'no_text' not in df.index:
                continue
            y_ts = df.loc['no_text', perf_metric]
            if not np.isfinite(y_ts):
                continue
            for ax in drawn_axes:
                ax.scatter(0, y_ts, facecolors='none',
                           edgecolors=CAT_COLORS['no_text'],
                           zorder=4, s=150,
                           marker=MODEL_MARKERS.get(model_name, 'o'))

    def _score_label(m):
        s = scores_dict.get(m)
        return f"{m}  score={round(s, 4)}" if s is not None else m
    model_legend = [
        Line2D([0], [0], color='black',
               linestyle=MODEL_LINESTYLES.get(m, '-'),
               marker=MODEL_MARKERS.get(m, 'o'), markerfacecolor='w',
               markersize=9, lw=1.8,
               label=_score_label(m))
        for m in tables if m in scores_dict
    ]
    cat_legend = [
        Line2D([0], [0], marker='o', markerfacecolor='white',
               markeredgecolor=CAT_COLORS[cat], linestyle='None',
               markersize=9, label=CAT_LABELS.get(cat, cat))
        for cat in CAT_COLORS
        if cat != 'no_text' and any(cat in df.index for df in tables.values())
    ]
    if show_ts_only and any('no_text' in df.index for df in tables.values()):
        cat_legend.append(
            Line2D([0], [0], marker='o', markerfacecolor='white',
                   markeredgecolor=CAT_COLORS['no_text'], linestyle='None',
                   markersize=9, label=CAT_LABELS['no_text'])
        )
    fig.legend(handles=model_legend, loc='outside lower center',
               bbox_to_anchor=(0.35, -0.2), ncol=max(1, len(tables)//4), handlelength=3,
               #bbox_to_anchor=(0.35, -0.15), ncol=max(1, len(tables)//3), handlelength=3,
               title=f'Model (trend line style) with baseline {PERF_METRIC_LABEL.get(perf_metric, perf_metric)}')
    fig.legend(handles=cat_legend, loc='outside lower center',
               bbox_to_anchor=(0.88, -0.18),#bbox_to_anchor=(0.85, -0.15), 
               ncol=1, title='Category (point colour)')
    plt.tight_layout()
    _save_or_show(save_path)


def plot_delta_bars(
    tables:      dict,
    perf_metric: str = "delta_r2",
    title:       str = "",
    save_path=None,
) -> None:
    """Grouped bar chart: performance metric per category, one group per model."""
    model_names = list(tables.keys())
    if not model_names:
        return

    sample_df = next(iter(tables.values()))
    cats = [c for c in sample_df.index
            if not np.isnan(sample_df.loc[c, perf_metric])]
    if not cats:
        cats = list(sample_df.index)

    n_models = len(model_names)
    x        = np.arange(len(cats))
    width    = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(cats)), 4.5))
    fig.patch.set_facecolor("white")

    for i, model_name in enumerate(model_names):
        df     = tables[model_name]
        means  = [df.loc[c, perf_metric] if c in df.index else np.nan for c in cats]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, label=model_name, alpha=0.82, zorder=3)

    ax.axhline(0, color="#444", lw=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS.get(c, c) for c in cats], rotation=30, ha="right")
    ax.set_ylabel(PERF_METRIC_LABEL.get(perf_metric, perf_metric))
    ax.set_title(title or f"{perf_metric} by category and model")
    ax.legend(frameon=True)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_model_mixture(
    tables:          dict,
    info_metrics:    list[str],
    categories:      list[str],
    perf_metric:     str  = "delta_r2",
    title:           str  = "",
    save_path=None,
    show_error_bars: bool = True,
) -> None:
    """
    2×2 grid showing how performance and information metrics degrade as
    annotation quality drops from 100% correct toward 0%.

    Layout
    ------
    Row 0 — model performance (perf_metric) vs % correct, one column per
            degradation type (incorrect / irrelevant).
    Row 1 — information metrics vs % correct, same column layout.
            Info metric values are model-independent so only one series per
            metric family is drawn, distinguished by line style.
    """
    if not tables:
        return

    subcats = ["incorrect", "irrelevant"]

    def _ymean(df, col, cat):
        v = df.loc[cat, col]
        return float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else float(v)

    def _ystd(df, col, cat):
        std_col = f"{col}_std"
        if std_col in df.columns:
            v = df.loc[cat, std_col]
            return float(v) if np.isfinite(float(v)) else 0.0
        v = df.loc[cat, col]
        return float(np.std(v)) if isinstance(v, (list, np.ndarray)) and len(v) > 1 else 0.0

    def _draw_series(ax, df, metric, subcat, marker, color, linestyle, line_color="black"):
        mix_cats = sorted(
            [c for c in categories if subcat in c and c in CAT_X_MAP],
            key=lambda c: CAT_X_MAP[c],
        )
        # Collect all finite (x, y_mean, y_std, point_color) tuples
        pts = []
        for cat in mix_cats + ["correct"]:
            if cat not in df.index or cat not in CAT_X_MAP:
                continue
            y = _ymean(df, metric, cat)
            if not np.isfinite(y):
                continue
            pts.append((CAT_X_MAP[cat], y, _ystd(df, metric, cat),
                        CAT_COLORS["correct"] if cat == "correct" else color))
        if not pts:
            return
        pts.sort()
        xs    = [p[0] for p in pts]
        ys    = np.array([p[1] for p in pts])
        stds  = np.array([p[2] for p in pts])

        # Shaded band
        if show_error_bars and stds.any():
            ax.fill_between(xs, ys - stds, ys + stds,
                            alpha=0.15, color=color, zorder=2)
        # Scatter points
        for x, y, _, pt_color in pts:
            ax.scatter(x, y, facecolors="none", edgecolors=pt_color,
                       zorder=4, s=150, marker=marker)
        # Connecting line
        if len(pts) >= 2:
            ax.plot(xs, ys, color=line_color, alpha=0.6, lw=1.5,
                    linestyle=linestyle, zorder=3)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 10.0), squeeze=False)
    fig.patch.set_facecolor("white")

    # ── Row 0: model performance ───────────────────────────────────────────
    for col, subcat in enumerate(subcats):
        ax = axes[0][col]
        for model_name, df in tables.items():
            if perf_metric not in df.columns:
                continue
            _draw_series(ax, df, perf_metric, subcat,
                         marker=MODEL_MARKERS.get(model_name, "o"),
                         color='black',#CAT_COLORS.get(subcat, "#888"),
                         linestyle=MODEL_LINESTYLES.get(model_name, "-"))
        ax.axhline(0, color="#aaa", lw=0.8, ls="--")
        ax.set_title(f"Replacing with {subcat.capitalize()} Annotations", pad=6)
        ax.set_xlabel(r"% of Correct Annotations")
        ax.set_ylabel(PERF_METRIC_LABEL.get(perf_metric, perf_metric))
        ax.grid(lw=0.4, alpha=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # ── Row 1: information metrics (model-independent) ─────────────────────
    ref_df = next(iter(tables.values()))
    for col, subcat in enumerate(subcats):
        ax = axes[1][col]
        for info_m in info_metrics:
            if info_m not in ref_df.columns:
                continue
            ic = INFO_COLORS.get(info_m, "#888")
            _draw_series(ax, ref_df, info_m, subcat,
                         marker="o",
                         color=ic,
                         linestyle="-",
                         line_color=ic)
        ax.axhline(0, color="#aaa", lw=0.8, ls="--")
        ax.set_xlabel(r"% of Correct Annotations")
        ax.set_ylabel("Information metric value")
        ax.grid(lw=0.4, alpha=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # ── Legends ────────────────────────────────────────────────────────────
    model_legend = [
        Line2D([0], [0], marker=MODEL_MARKERS.get(m, "o"), color="black",
               markerfacecolor="w", markersize=9, lw=1.5,
               linestyle=MODEL_LINESTYLES.get(m, "-"), label=m)
        for m in tables
    ]
    metric_legend = [
        Line2D([0], [0], color=INFO_COLORS.get(m, "#888"), lw=1.5,
               linestyle="-",
               label=METRIC_DISPLAY.get(m, m))
        for m in info_metrics
    ]
    # cat_legend = [
    #     Line2D([0], [0], marker='o', markerfacecolor='white',
    #            markeredgecolor=CAT_COLORS[cat], linestyle='None',
    #            markersize=9, label=CAT_LABELS.get(cat, cat))
    #     for cat in ["correct", "incorrect", "irrelevant"]
    # ]

    # fig.legend(handles=cat_legend, loc='outside lower center',
    #            bbox_to_anchor=(0.35, -0.12), ncol=1, title='Category')
    

    fig.legend(handles=model_legend, loc="outside lower left", handlelength=3,
               bbox_to_anchor=(0.0, -0.12), ncol=3, title="Model")
    fig.legend(handles=metric_legend, loc="outside lower right",
               bbox_to_anchor=(1.0, -0.12), ncol=len(info_metrics)/2, title="Info metric")
    # if title:
    #     fig.suptitle(title, y=1.01)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(path: str | None) -> None:
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved → {path}")
        plt.close()
    else:
        plt.show()


def _mi_metrics(results: dict) -> list[str]:
    """Return the submetric names present in a results dict."""
    candidates = [
        "text", "ts", "joint", "conditional",          # MI
        "redundancy", "unique_ts", "unique_text", "synergy",  # PID
        "v_text", "v_ts", "v_joint", "v_conditional",  # V-information
    ]
    return [m for m in candidates if m in results and isinstance(results[m], dict)]


# ---------------------------------------------------------------------------
# Single-experiment plots
# ---------------------------------------------------------------------------

def plot_single_bar(results: dict, title: str = "", save_path=None):
    """Bar chart of mean ± std per category for each metric."""
    metrics = _mi_metrics(results)
    cats    = list(next(iter(results[metrics[0]].values().__class__,
                             results[metrics[0]]).keys())
                   if False else results[metrics[0]].keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    x_pos = np.arange(len(cats))

    for ax, metric in zip(axes, metrics):
        mean_key = f"{metric}_mean"
        std_key  = f"{metric}_std"

        if mean_key in results:
            means = [results[mean_key][c] for c in cats]
            stds  = [results[std_key][c]  for c in cats]
        else:
            means = [float(np.mean(results[metric][c])) for c in cats]
            stds  = [float(np.std(results[metric][c]))  for c in cats]

        colors = [CAT_COLORS.get(c, "#888") for c in cats]
        ax.bar(x_pos, means, width=0.6, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                    color="#333", capsize=4, capthick=1, elinewidth=1, zorder=4)
        ax.axhline(0, color="#aaa", lw=0.8, zorder=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([CAT_LABELS.get(c, c) for c in cats],
                           rotation=35, ha="right")
        ax.set_ylabel("MI (nats)" if "MI" not in metric else metric)
        ax.set_title(METRIC_LABELS.get(metric, metric), pad=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(title or "MI per annotation category",
                 fontweight="normal", y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)


def print_single_table(results: dict, label: str = ""):
    """Print a compact table of mean ± std per category / metric."""
    metrics = _mi_metrics(results)
    cats    = list(results[metrics[0]].keys())

    header = f"{'':25}" + "".join(f"{m:>14}" for m in metrics)
    print(f"\n{'─' * len(header)}")
    if label:
        print(label)
    print(header)
    print("─" * len(header))

    for cat in cats:
        row = f"{CAT_LABELS.get(cat, cat):25}"
        for m in metrics:
            mean_key = f"{m}_mean"
            if mean_key in results:
                mean = results[mean_key][cat]
                std  = results[f"{m}_std"][cat]
            else:
                vals = results[m][cat]
                mean, std = float(np.mean(vals)), float(np.std(vals))
            row += f"  {mean:+.4f}±{std:.3f}"
        print(row)

    print("─" * len(header))


# ---------------------------------------------------------------------------
# Sweep plots (grouped bar charts — one group per sweep value, one bar per category)
# ---------------------------------------------------------------------------

def _sweep_bar_data(runs: list, metric: str, cat: str):
    """Extract (means, stds) arrays across sweep runs for one metric + category."""
    mean_key = f"{metric}_mean"
    std_key  = f"{metric}_std"
    means, stds = [], []
    for run in runs:
        r = run["results"]
        if mean_key in r:
            means.append(r[mean_key][cat])
            stds.append(r[std_key][cat])
        else:
            vals = r[metric][cat]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    return np.array(means), np.array(stds)


def _sweep_grouped_bars(ax, runs, values, metric, cats, ylabel=""):
    """Draw grouped bars on ax: groups = sweep values, bars = categories."""
    n_vals = len(values)
    n_cats = len(cats)
    width  = 0.7 / n_cats
    x      = np.arange(n_vals)

    for i, cat in enumerate(cats):
        means, stds = _sweep_bar_data(runs, metric, cat)
        offset = (i - n_cats / 2 + 0.5) * width
        col = CAT_COLORS.get(cat, "#888")
        ax.bar(x + offset, means, width * 0.9, color=col, alpha=0.85,
               label=CAT_LABELS.get(cat, cat), zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="#333", capsize=3, capthick=1, elinewidth=1, zorder=4)

    ax.axhline(0, color="#aaa", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values], rotation=35, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel,)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def plot_sweep_lines(data: dict, metric: str = "conditional",
                     title: str = "", save_path=None):
    """Bar chart: sweep values on x-axis, one bar per annotation category per sweep value."""
    runs   = data["runs"]
    param  = data["sweep_param"]
    values = data["sweep_values"]
    cats   = list(runs[0]["results"][metric].keys())

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(values)), 4.8))
    fig.patch.set_facecolor("white")

    _sweep_grouped_bars(ax, runs, values, metric, cats, ylabel="MI (nats)")
    ax.set_xlabel(param,)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} — {param}")
    ax.legend( frameon=True)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_sweep_all_metrics(data: dict, title: str = "", save_path=None):
    """One panel per MI metric, grouped bar chart per sweep value."""
    runs    = data["runs"]
    param   = data["sweep_param"]
    values  = data["sweep_values"]
    metrics = _mi_metrics(runs[0]["results"])
    cats    = list(runs[0]["results"][metrics[0]].keys())

    panel_w = max(4.5, 1.2 * len(values))
    fig, axes = plt.subplots(1, len(metrics), figsize=(panel_w * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, metrics):
        _sweep_grouped_bars(ax, runs, values, metric, cats, ylabel="MI (nats)")
        ax.set_xlabel(param)
        ax.set_xticklabels([MODEL_DISPLAY.get(value, value) for value in values])
        ax.set_title(METRIC_LABELS.get(metric, metric), pad=8)
        ax.legend( frameon=True)

    fig.suptitle(title or f"MI sensitivity to {param}",
                 fontweight="normal", y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)


def print_sweep_table(data: dict):
    """Print best value per category across the sweep."""
    runs    = data["runs"]
    param   = data["sweep_param"]
    values  = data["sweep_values"]
    metrics = _mi_metrics(runs[0]["results"])
    cats    = list(runs[0]["results"][metrics[0]].keys())

    print(f"\n{'─' * 80}")
    print(f"Sweep: {param} over {values}")
    print(f"{'─' * 80}")

    for metric in metrics:
        mean_key = f"{metric}_mean"
        print(f"\n  {METRIC_LABELS.get(metric, metric)}")
        header = f"  {'':25}" + "".join(f"{str(v):>10}" for v in values)
        print(header)
        for cat in cats:
            row = f"  {CAT_LABELS.get(cat, cat):25}"
            for run in runs:
                r = run["results"]
                if mean_key in r:
                    row += f"  {r[mean_key][cat]:+.4f}"
                else:
                    row += f"  {float(np.mean(r[metric][cat])):+.4f}"
            print(row)

    print(f"\n{'─' * 80}")


# ---------------------------------------------------------------------------
# Overlay: compare multiple sweeps on one plot
# ---------------------------------------------------------------------------

def plot_overlay(datasets: list[dict], labels: list[str],
                 metric: str = "conditional", save_path=None):
    """
    Compare multiple sweep results — grouped bars where groups are sweep values
    and each bar is a different label (run/config). Shows the 'correct' category
    only so the comparison stays readable.
    """
    OVERLAY_COLORS = ["#1D9E75", "#D85A30", "#2980B9", "#7F8C8D",
                      "#8E44AD", "#E67E22", "#C0392B"]

    param  = datasets[0]["sweep_param"]
    values = datasets[0]["sweep_values"]

    # Detect which category key is present in the first run
    first_results = datasets[0]["runs"][0]["results"]
    cat = next((c for c in ("correct", "relevant_correct") if c in first_results.get(metric, {})),
               list(first_results.get(metric, {}).keys())[0])

    n_vals   = len(values)
    n_labels = len(labels)
    width    = 0.7 / n_labels
    x        = np.arange(n_vals)
    mean_key = f"{metric}_mean"
    std_key  = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_vals), 4.8))
    fig.patch.set_facecolor("white")

    for i, (data, label) in enumerate(zip(datasets, labels)):
        means, stds = _sweep_bar_data(data["runs"], metric, cat)
        offset = (i - n_labels / 2 + 0.5) * width
        col = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        ax.bar(x + offset, means, width * 0.9, color=col, alpha=0.85,
               label=label, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="#333", capsize=3, capthick=1, elinewidth=1, zorder=4)

    ax.axhline(0, color="#aaa", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values], rotation=35, ha="right")
    ax.set_xlabel(param)
    ax.set_ylabel("MI (nats)")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {CAT_LABELS.get(cat, cat)}",)
    ax.legend(frameon=True)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Multi-metric conditional comparison (default single-file view)
# ---------------------------------------------------------------------------

def plot_conditional_comparison(
    results_by_metric: dict,
    title: str = "",
    save_path=None,
):
    """
    One panel per metric showing only the conditional submetric,
    with bars per annotation category (mean ± std).

    Conditional keys used:
        mutual_information → conditional
        pid                → unique_text
        v_information      → v_conditional
    """
    metric_names = list(results_by_metric.keys())
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, mname in zip(axes, metric_names):
        results  = results_by_metric[mname]
        ckey     = CONDITIONAL_KEY.get(mname, "conditional")
        # Fall back gracefully if key missing
        if ckey not in results:
            ax.set_visible(False)
            continue

        cats     = list(results[ckey].keys())
        x_pos    = np.arange(len(cats))
        mean_key = f"{ckey}_mean"
        std_key  = f"{ckey}_std"

        if mean_key in results:
            means = [results[mean_key][c] for c in cats]
            stds  = [results[std_key][c]  for c in cats]
        else:
            means = [float(np.mean(results[ckey][c])) for c in cats]
            stds  = [float(np.std(results[ckey][c]))  for c in cats]

        colors = [CAT_COLORS.get(c, "#888") for c in cats]
        ax.bar(x_pos, means, width=0.6, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                    color="#333", capsize=4, capthick=1, elinewidth=1, zorder=4)
        ax.axhline(0, color="#aaa", lw=0.8, zorder=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([CAT_LABELS.get(c, c) for c in cats],
                           rotation=35, ha="right")
        ax.set_ylabel(CONDITIONAL_LABEL.get(mname, ckey))
        ax.set_title(METRIC_DISPLAY.get(mname, mname), pad=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(title or "Conditional text contribution per annotation category",
                 fontweight="normal", y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Multi-metric conditional sweep (default sweep view)
# ---------------------------------------------------------------------------

def plot_conditional_sweep(
    runs_by_metric: dict,
    sweep_param:    str,
    sweep_values:   list,
    title:          str = "",
    save_path=None,
):
    """
    One panel per metric, grouped bar chart: sweep values on x-axis,
    one bar per annotation category. Shows only the conditional submetric.
    """
    metric_names = list(runs_by_metric.keys())
    n = len(metric_names)
    panel_w = max(4.5, 1.2 * len(sweep_values))
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, 4.8))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, mname in zip(axes, metric_names):
        runs  = runs_by_metric[mname]
        ckey  = CONDITIONAL_KEY.get(mname, "conditional")

        first_results = runs[0]["results"]
        if ckey not in first_results:
            ax.set_visible(False)
            continue
        cats = list(first_results[ckey].keys())

        _sweep_grouped_bars(ax, runs, sweep_values, ckey, cats,
                            ylabel=CONDITIONAL_LABEL.get(mname, ckey))
        ax.set_xlabel(sweep_param)
        ax.set_title(METRIC_DISPLAY.get(mname, mname), pad=8)
        ax.legend(frameon=True)

    fig.suptitle(title or "Conditional text contribution — " + sweep_param,
                 fontweight="normal", y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)