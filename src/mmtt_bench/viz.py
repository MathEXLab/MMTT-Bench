# benchmark/viz.py
"""
Visualisation

All plot functions accept save_path=None (show) or a filepath (save).
"""

import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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
    'sentence-transformers/all-distilroberta-v1': 'DistilRoBERTa',
    "iTransformer|first-additive": "iTransformer",
    "FEDformer|first-additive": "FEDformer",
    "Informer|first-additive": "Informer",
    "Nonstationary_Transformer|first-additive": "Nonstationary_Transformer"
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
    "iTransformer|first-additive": '>',
    "FEDformer|first-additive": 'H',
    "Informer|first-additive": '<',
    "Nonstationary_Transformer|first-additive": 'd',
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
    # LLM models
    'TimeLLM': '^',
    'GPT4MTS':   'o',
    'DirectPrompt':   'D',
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
    "iTransformer|first-additive": (0, (1, 2, 5, 2)),
    "FEDformer|first-additive" :(0, (3, 5, 1, 5)),
    "Informer|first-additive" : (0, (3, 1, 1, 1, 1, 1)),
    "Nonstationary_Transformer|first-additive": (0, (4, 1, 1, 1, 1, 1)),
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
    # LLM
    'TimeLLM': '-',
    'GPT4MTS':   '--',
    'DirectPrompt':   ':',
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
def _apply_sci_x(ax) -> None:
    """Scientific notation on x-axis when values fall outside [1e-3, 1e3]."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(fmt)


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
        if "|" in m:
            m = MODEL_DISPLAY.get(m, m)
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
               bbox_to_anchor=(0.35, -0.2), ncol=max(3, len(tables)//4), handlelength=3,
               #bbox_to_anchor=(0.35, -0.15), ncol=max(1, len(tables)//3), handlelength=3,
               title=f'Model (trend line style) with baseline {PERF_METRIC_LABEL.get(perf_metric, perf_metric)}')
    fig.legend(handles=cat_legend, loc='outside lower center',
               bbox_to_anchor=(0.88, -0.18),#bbox_to_anchor=(0.85, -0.15), 
               ncol=1, title='Category (point colour)')
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


# ---------------------------------------------------------------------------
# Sweep plots (grouped bar charts — one group per sweep value, one bar per category)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Overlay: compare multiple sweeps on one plot
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-metric conditional comparison (default single-file view)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-metric conditional sweep (default sweep view)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure 2 — representation-pipeline robustness
#
# Both panels were originally built in plot.ipynb from hand-transcribed numbers.
# They are derived from the sweep result JSONs here so the figure cannot drift
# from Tables 4-7.
# ---------------------------------------------------------------------------

# Verdict codes, in the order the colormap below expects them.
ORDER_HELD      = 2   # correct > incorrect > irrelevant  (the designed ordering)
ORDER_SWAPPED   = 0   # correct > irrelevant > incorrect  (incorrect/irrelevant blind)
ORDER_NEGATIVE  = 1   # some category is negative by more than its bootstrap std

_ORDER_COLORS = ['#B4B2A9', '#E24B4A', '#1D9E75']   # swapped, negative, held
_ORDER_LABELS = {
    ORDER_HELD:     'Correct > Incorrect > Irrelevant',
    ORDER_SWAPPED:  'Correct > Irrelevant > Incorrect',
    ORDER_NEGATIVE: r'MI < 0 and |MI| > bootstrap std',
}


def _cond_mean_std(md: dict, est: str, cat: str) -> tuple[float, float]:
    """Conditional MI and its bootstrap std, in each estimator's own terms."""
    if est == 'pid':
        v = np.asarray(md['unique_text'][cat], dtype=float)
        return float(v.mean()), float(v.std())
    pre = 'v_' if est == 'v_information' else ''
    return (float(md[f'{pre}conditional_mean'][cat]),
            float(md[f'{pre}conditional_std'][cat]))


def ordering_verdict(md: dict, est: str) -> int:
    """Classify one (estimator, setting) cell for the Figure 2 grid.

    Negative-by-more-than-noise takes precedence: an estimate that has gone
    negative is not reporting an ordering at all, it is reporting a broken
    estimate, and that is the more useful thing to show.
    """
    vals = {c: _cond_mean_std(md, est, c) for c in ('correct', 'incorrect', 'irrelevant')}
    if any(m < 0 and abs(m) > s for m, s in vals.values()):
        return ORDER_NEGATIVE
    c, i, r = (vals[k][0] for k in ('correct', 'incorrect', 'irrelevant'))
    return ORDER_HELD if c > i > r else ORDER_SWAPPED


def plot_ordering_grid(panels: list[dict], estimators: list[str] | None = None,
                       ax_row=None, save_path=None):
    """Figure 2, left: did the category ordering survive each pipeline setting?

    `panels` is one dict per sweep axis:
        {'name': 'PCA dim', 'runs': <sweep JSON 'runs' list>,
         'labels': ['4','8','16','32'], 'chosen': 2}
    where `chosen` is the index of the setting used in the paper, boxed in the
    plot. Verdicts are derived per cell by ordering_verdict().
    """
    estimators = estimators or ['mutual_information', 'mine', 'infonce',
                                'cca', 'v_information', 'pid']
    cmap = ListedColormap(_ORDER_COLORS)

    made_fig = ax_row is None
    if made_fig:
        fig, ax_row = plt.subplots(
            1, len(panels), figsize=(5.5, 2.4), sharey=True,
            gridspec_kw={'width_ratios': [len(p['labels']) for p in panels]})
        fig.patch.set_facecolor('white')
    ax_row = np.atleast_1d(ax_row)

    for ax, panel in zip(ax_row, panels):
        status = np.array([[ordering_verdict(rec['results_by_metric'][e], e)
                            if e in rec['results_by_metric'] else ORDER_SWAPPED
                            for rec in panel['runs']]
                           for e in estimators], dtype=float)
        ax.imshow(status, cmap=cmap, vmin=0, vmax=2, aspect='auto')
        ax.set_xticks(range(len(panel['labels'])))
        ax.set_xticklabels(panel['labels'], fontsize=8)
        ax.set_xlabel(panel['name'], fontsize=9)
        ax.set_yticks(range(len(estimators)))
        ax.set_yticklabels([METRIC_DISPLAY.get(e, e) for e in estimators], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(panel['labels']), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(estimators), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if panel.get('chosen') is not None:      # box the setting the paper uses
            ax.add_patch(Rectangle((panel['chosen'] - 0.5, -0.5), 1, len(estimators),
                                   fill=False, edgecolor='black', lw=1.6, zorder=5))

    if made_fig:
        handles = [Patch(facecolor=_ORDER_COLORS[v], label=_ORDER_LABELS[v])
                   for v in (ORDER_HELD, ORDER_SWAPPED, ORDER_NEGATIVE)]
        handles.append(Patch(facecolor='white', edgecolor='black', label='Final parameter choice'))
        ax_row[0].figure.legend(handles=handles, frameon=False, fontsize=7,
                                loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)
        _save_or_show(save_path)


def plot_embedding_sensitivity(runs: list[dict], labels: list[str],
                               estimators: list[str] | None = None,
                               category: str = 'correct', ax=None, save_path=None):
    """Figure 2, right: conditional MI for one category across text encoders.

    Estimators are dodged horizontally within each encoder group so points and
    error bars stay separable. Pass the oracle/dummy run as the final record to
    reproduce the paper's upper-bound column.
    """
    estimators = estimators or ['mutual_information', 'mine', 'infonce',
                                'cca', 'v_information', 'pid']
    made_fig = ax is None
    if made_fig:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        fig.patch.set_facecolor('white')

    x = np.arange(len(labels), dtype=float)
    dodge = np.linspace(-0.3, 0.3, len(estimators))
    for k, est in enumerate(estimators):
        ys, es = [], []
        for rec in runs:
            md = rec['results_by_metric']
            m, s = _cond_mean_std(md[est], est, category) if est in md else (np.nan, np.nan)
            ys.append(m); es.append(s)
        ax.errorbar(x + dodge[k], ys, yerr=es, fmt='o', markersize=4.5,
                    color=INFO_COLORS.get(est, '#888'), elinewidth=1.1, capsize=2,
                    label=METRIC_DISPLAY.get(est, est))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Conditional MI (nats)', fontsize=9)
    ax.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if made_fig:
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Figure 5 — real-world MI estimates
# ---------------------------------------------------------------------------

BASELINE = "no_text"                                   # the series-only reference
CATEGORIES = ["correct", "incorrect", "irrelevant"]     # text corpora, in plot order
ESTIMATORS = ["mutual_information", "mine", "infonce",
              "cca", "v_information", "pid"]
DATASET_LABEL = {"PublicHealth": "Public\nHealth", "SocialGood": "Social\nGood",
                 "FinTexTS": "FinTexTS\n(10 tickers)"}
GRID = "#D8D8D8"                                        # shaded-band fill


def median_ci(v, n_boot: int = 2000, seed: int = 0):
    """Median with a 95% percentile-bootstrap interval and the bootstrap standard
    error of the median. Resampling configurations (not runs) is the right unit:
    the spread being reported is across architecture x fusion choices, which is
    what a practitioner faces when picking one."""
    v = np.asarray(v, dtype=float)
    if not len(v):
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return (float(np.median(v)), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)), float(np.std(boots)))


def _grid_cell(ax) -> None:
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])

def plot_realworld_mi_grid(rows, data, quantity: str = "joint",
                           show_mse: bool = True, fill_all: bool = False,
                           save_path=None):
    """Figure 5 — MI estimates on the real-world datasets.

    Datasets run down the page, estimators across it. Each cell shows the three
    text corpora against that estimator's series-only value (blue dashed rule);
    distance from the rule is the information text adds beyond the series.

    Every cell carries its own y-scale, and that is forced rather than chosen: a
    row is one dataset and would otherwise have to hold KSG (~2 nats) and
    V-information (~0.03 nats) on one axis. The MSE column is a separate column
    for the same reason -- percent error and nats are different quantities, and
    two scales on one axis is the one chart construction that reliably misleads.

    `rows` and `data` come from plot.load_realworld_results().
    """
    n_ds, n_est = len(rows), len(ESTIMATORS)
    pos = np.arange(len(CATEGORIES), dtype=float)

    n_col = n_est + (1 if show_mse else 0)
    fig = plt.figure(figsize=(1.30 * n_col + 1.9, 0.92 * n_ds + 1.7))
    gs = fig.add_gridspec(n_ds, n_col,
                          width_ratios=([1.22] if show_mse else []) + [1.0] * n_est,
                          hspace=0.16, wspace=0.34)
    fig.patch.set_facecolor("white")

    def label_row(ax, r):
        ax.set_ylabel(DATASET_LABEL.get(r["name"], r["name"]).replace("\n", " "),
                      color='black', fontsize=9, rotation=0, ha="right", va="center",
                      labelpad=10)

    for i, r in enumerate(rows):
        # ── left column: what the models did ────────────────────────────────
        ax = fig.add_subplot(gs[i, 0]) if show_mse else None
        if ax is not None:
            ax.axhline(0, color=CAT_COLORS[BASELINE], lw=1.6, zorder=3)
            for k, cat in enumerate(CATEGORIES):
                v = r["mse"].get(cat)
                if v is None or not len(v):
                    continue
                med, lo, hi, _ = median_ci(v)
                q25, q75 = np.percentile(v, [25, 75])
                ax.vlines(pos[k], q25, q75, color=CAT_COLORS[cat], lw=1.3,
                          alpha=0.55, zorder=4)
                ax.errorbar(pos[k], med, yerr=[[med - lo], [hi - med]], fmt="none",
                            ecolor=CAT_COLORS[cat], elinewidth=1.6, capsize=2, zorder=5)
                ax.scatter(pos[k], med, s=40, facecolors=CAT_COLORS[cat],
                           edgecolors="white", linewidths=0.8, zorder=6)
            ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
            _grid_cell(ax)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            label_row(ax, r)
            if i == 0:
                ax.set_title("Downstream MSE\n(% vs no-text)", fontsize=9, color='black',
                             pad=6, fontweight="bold")

        # ── estimator columns: what the estimators reported ─────────────────
        for j, est in enumerate(ESTIMATORS):
            ax = fig.add_subplot(gs[i, j + (1 if show_mse else 0)])
            entry = data.get(est, {}).get(i, {})
            # The series-only value is a reference level, not an observation to
            # be compared alongside the corpora, so it is drawn as a rule.
            base = entry.get(BASELINE)
            if base and np.isfinite(base[0]):
                ax.axhline(base[0], color=CAT_COLORS[BASELINE], lw=1.2,
                           ls=(0, (3, 2)), alpha=0.9, zorder=2)
            for k, cat in enumerate(CATEGORIES):
                v = entry.get(cat)
                if v is None or not np.isfinite(v[0]):
                    continue
                col = CAT_COLORS[cat]
                err = v[1] if np.isfinite(v[1]) else 0.0
                ax.errorbar(pos[k], v[0], yerr=err, fmt="none", ecolor=col,
                            elinewidth=1.3, capsize=2, alpha=0.85, zorder=4)
                filled = True if fill_all else v[2]
                ax.scatter(pos[k], v[0], s=40,
                           facecolors=col if filled else "white",
                           edgecolors="white" if filled else col,
                           linewidths=0.8 if filled else 1.3, zorder=5)
            ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
            # A cell whose values are all identical (KSG's exact zeros) autoscales
            # to a degenerate range and renders blank. Give it a readable window.
            seen = [entry[c][0] for c in CATEGORIES
                    if c in entry and np.isfinite(entry[c][0])]
            if seen and max(seen) - min(seen) < 1e-12:
                c0 = seen[0]
                half = max(abs(c0) * 0.5, 0.05)
                ax.set_ylim(c0 - half, c0 + half)
            _grid_cell(ax)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            if j == 0 and not show_mse:
                label_row(ax, r)
            if i == 0:
                ax.set_title(METRIC_DISPLAY.get(est, est), fontsize=9,
                             color='black', pad=6, fontweight="bold")

    handles = [plt.Line2D([], [], color=CAT_COLORS[BASELINE], lw=1.8, ls=(0, (3, 2)),
                          label="No text (series only)")]
    handles += [plt.Line2D([], [], marker="o", ls="", markersize=7,
                           markerfacecolor=CAT_COLORS[c], markeredgecolor="white",
                           label=c.capitalize()) for c in CATEGORIES]
    if not fill_all:
        handles.append(plt.Line2D([], [], marker="o", ls="", markersize=7,
                                  markerfacecolor="white", markeredgecolor='#555555',
                                  label="within 1.96σ of baseline"))
    fig.legend(handles=handles, frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.995), labelcolor='black', fontsize=9,
               handlelength=1.0, columnspacing=1.3, handletextpad=0.35)

    caption = (
        ("Each cell shows the three text corpora against that estimator's series-only value, drawn as a blue "
         "dashed rule; distance from the rule is the information text adds beyond the series. Marks are the "
         "raw joint estimate I(X_ts, X_text; Y) in nats with +/-1 bootstrap sigma. " if quantity == "joint" else
         "Each cell shows the conditional information each text corpus adds beyond the series, "
         "I(X_text; Y | X_ts), read from the estimator's paired-resample field, in nats with +/-1 bootstrap "
         "sigma. The blue rule at zero is what a corpus adding nothing would score. ")
        + ("Left column: median percent change in MSE against no-text with its 95% bootstrap interval, thin "
           "bar showing the interquartile range across every architecture x fusion configuration; the blue "
           "rule at zero is the no-text baseline. " if show_mse else "")
        + ("" if fill_all else
           "Filled marks differ from the rule by more than 1.96 standard deviations on the paired "
           "per-resample difference; hollow marks do not. ")
        + "Every cell has its own y-scale — estimators are not comparable in absolute units"
        + (" and nats are not comparable with percent error" if show_mse else "")
        + " — so compare points within a cell, and patterns across cells, never absolute heights."
    )
    fig.text(0.0, -0.012, textwrap.fill(caption, 130), fontsize=8, color='#555555', va="top")

    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Figure 1 — the signal and its annotations
# ---------------------------------------------------------------------------

# Where each callout box sits, in axes fractions: two correct annotations (one
# at each transition direction), one incorrect and one irrelevant. The three
# directional ones sit above the axes; irrelevant drops below, where it has room
# and where its lack of any arrow into a transition point reads clearly.
_FIG1_CALLOUTS = [
    ("drop_to_constant",      "correct",    1, (0.16, 1.24)),
    ("recover_from_constant", "incorrect",  1, (0.50, 1.24)),
    ("recover_from_constant", "correct",    0, (0.84, 1.24)),
    ("drop_to_constant",      "irrelevant", 8, (0.50, -0.30)),
]


def _flattened_sine(t_dense, y_dense, x):
    """The underlying sine with the flat segments cut in, for the dotted guide.

    The dense signal only carries annotation points, so the constant stretches
    between them are recovered by extending each flat value forward to the next
    annotation point whenever the gap is a full quarter-cycle.
    """
    y = np.sin(x)
    for v in (-0.5, 0.5):
        held = [t for t, yy in zip(t_dense, y_dense) if yy == v]
        flat = []
        for i in range(len(held) - 1):
            if round(held[i + 1] - held[i], 2) == round(np.pi / 2, 2):
                flat += [round(held[i] + k * np.pi / 8, 2) for k in range(4)]
            else:
                flat.append(round(held[i], 2))
        y = [v if round(xx, 2) in flat else yy for xx, yy in zip(x, y)]
    return y


def plot_annotation_example(raw: dict, n_periods: int = 12, callouts: bool = True,
                            wrap_width: int = 22, save_path=None):
    """Figure 1 — example text annotations against the signal.

    `raw` is a loaded split JSON (metadata.t / metadata.y plus points). Blue
    markers are annotation points on the sine; orange are transition points,
    where the signal has dropped to the constant and the series alone carries no
    information about the next value, so correct text becomes the only source.
    """
    t_dense, y_dense = raw["metadata"]["t"], raw["metadata"]["y"]
    x = np.arange(0, 50, np.pi / 8)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.plot(x, _flattened_sine(t_dense, y_dense, x), linestyle=":", color="#4A90D9", lw=1.2)
    ax.scatter(t_dense, y_dense, s=22, color="#4A90D9", label="signal", zorder=3)

    trans = [p for p in raw["points"]
             if p["phase"] in ("drop_to_constant", "recover_from_constant")]
    ax.scatter([p["t"] for p in trans], [p["y"] for p in trans],
               s=22, color="#E08214", label="transition", zorder=4)

    if callouts:
        for phase, cat, idx, pos in _FIG1_CALLOUTS:
            # Restrict to the plotted window first: indexing the full split can
            # select a point beyond xlim, whose arrow would then point off-plot.
            pts = [p for p in raw["points"]
                   if p["phase"] == phase and p["category"] == cat
                   and 0 <= p["t"] <= n_periods * np.pi]
            if not pts:
                continue
            pt = pts[min(idx, len(pts) - 1)]
            ax.annotate(
                textwrap.fill(pt["samples"][0]["text"], width=wrap_width),
                xy=(pt["t"], pt["y"]), xycoords="data",
                xytext=pos, textcoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=CAT_COLORS[cat], lw=2),
                arrowprops=dict(arrowstyle="->", color="#444",
                                connectionstyle="arc3,rad=0.15"),
                fontsize=9, va="center", ha="center")

    ax.set_xlim(0, n_periods * np.pi)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(top=0.70, bottom=0.30)
    if save_path:
        # bbox_inches="tight" crops the callout that sits below the axes, so the
        # figure is saved with its own margins instead.
        fig.savefig(save_path, dpi=300, facecolor="white")
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()
