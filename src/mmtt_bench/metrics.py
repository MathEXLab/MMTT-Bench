# benchmark/metrics.py
"""
Metric computation — registered by name in METRICS.

Result dict keys (produced by MutualInformationMetric.compute and compatible
metrics):
    'text'          {cat: [bootstrap_samples]}
    'ts'            {cat: [bootstrap_samples]}
    'joint'         {cat: [bootstrap_samples]}
    'conditional'   {cat: [bootstrap_samples]}
    'text_mean'     {cat: float}
    'text_std'      {cat: float}
    ... (same pattern for ts, joint, conditional)
    'explained_variance'  {cat: float}
    'd_conservative'      {cat: int}
    'd_liberal'           {cat: int}

KSG mutual information estimator
---------------------------------
Implements the Kraskov-Stögbauer-Grassberger k-NN estimator for I(X; Y).

Matches sklearn's _estimate_mi internals:
    - nn.kneighbors() with no argument (self-query, excludes query point)
    - np.nextafter(radius[:, -1], 0)  for strict less-than boundary
    - KDTree directly for marginal counts

Reference: Kraskov, Stögbauer & Grassberger, Phys Rev E 69, 066138 (2004)
"""

import warnings
import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree


def ksg_mi(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
) -> float:
    """
    Estimate I(X; Y) using the KSG k-NN estimator.

    X : (N, dx) or (N,) — continuous
    Y : (N, dy) or (N,) — continuous
    k : number of neighbours

    Returns MI in nats, clipped to 0.0 (MI is non-negative).
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X.reshape(len(X), -1)
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y.reshape(len(Y), -1)

    if len(X) != len(Y):
        raise ValueError(
            f"X and Y must have the same number of rows, "
            f"got X={len(X)}, Y={len(Y)}"
        )

    N  = len(X)
    XY = np.hstack([X, Y])

    # k-th neighbour distance in joint space
    # kneighbors() with no argument = self-query (excludes the point itself)
    nn     = NearestNeighbors(n_neighbors=k, metric='chebyshev').fit(XY)
    radius = np.nextafter(nn.kneighbors()[0][:, -1], 0)

    # Neighbour counts in marginal spaces
    nx = KDTree(X, metric='chebyshev').query_radius(
        X, radius, count_only=True) - 1
    ny = KDTree(Y, metric='chebyshev').query_radius(
        Y, radius, count_only=True) - 1

    mi = (digamma(k)
          + digamma(N)
          - np.mean(digamma(nx + 1))
          - np.mean(digamma(ny + 1)))

    return float(max(mi, 0.0))

def _detect_n_samples(ts: np.ndarray) -> int:
    """
    Infer n_samples from the number of consecutive identical rows in ts.
    ts has shape (N * n_samples, n_patches) where every n_samples rows
    are identical (same annotation point, different text surface form).
    """
    if len(ts) == 0:
        return 1
    first_row = ts[0]
    n = 1
    while n < len(ts) and np.allclose(ts[n], first_row):
        n += 1
    return n

def _pool_by_annotation_point(
    x:  np.ndarray,   # (N * n_samples, embed_dim)
    ts: np.ndarray,   # (N * n_samples, ts_dim)    — repeated rows
    y:  np.ndarray,   # (N * n_samples, y_dim)     — repeated rows
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse n_samples rows per annotation point down to one row each.

    Works for both univariate and 3-D signals:
        ts_dim = n_patches           (1-D signal)
               = n_patches * C       (C-channel signal, channels concatenated)
        y_dim  = H                   (1-D, horizon H)
               = H * C               (C-channel, flattened by data._finalise)

    Text embeddings (x) are mean-pooled across surface forms.
    Time-series (ts) and target (y) use the first of each repeated block —
    the values are identical across the block by construction.

    Returns
    -------
    x_pooled  : (N, embed_dim) — mean of n_samples surface-form embeddings
    ts_pooled : (N, ts_dim)    — first row of each n_samples block
    y_pooled  : (N, y_dim)     — first row of each n_samples block
    """
    N_total = len(ts)

    if N_total % n_samples != 0:
        raise ValueError(
            f"Total rows {N_total} not evenly divisible by n_samples={n_samples}. "
            "Some annotation points may have been skipped during preprocessing."
        )

    n_points  = N_total // n_samples
    x_pooled  = x.reshape(n_points, n_samples, -1).mean(axis=1)
    ts_pooled = ts.reshape(n_points, n_samples, -1)[:, 0, :]
    y_pooled  = y.reshape(n_points, n_samples, -1)[:, 0, :]
    return x_pooled, ts_pooled, y_pooled

def _prepare_pooled(
    category_data: dict,
    pca_text: PCA = None,
) -> tuple[dict, PCA]:
    """
    Pool annotation points to one row each and fit (or reuse) a shared PCA.

    Returns
    -------
    pooled   : {cat: (x_pooled, ts_pooled, y_pooled)}
    pca_text : PCA fitted on pooled text embeddings, or the passed-in PCA
    """
    pooled = {}
    for cat, vals in category_data.items():
        x_p, ts_p, y_p = _pool_by_annotation_point(
            np.array(vals['x']),
            np.array(vals['ts']),
            np.array(vals['y']),
            vals['n_samples'][0],
        )
        pooled[cat] = (x_p, ts_p, y_p)

    if pca_text is None:
        all_x = np.vstack([x_p for x_p, _, _ in pooled.values()])
        pca_text = PCA(random_state=42).fit(all_x)

    return pooled, pca_text


class MutualInformationMetric:
    """
    KSG mutual information at four levels per annotation category.

    Pools annotation points across all categories before the category loop,
    uses a shared pooled ts baseline for the conditional term, bootstrap
    resamples rows with replacement, and computes mean/std after the
    bootstrap loop.

    Parameters
    ----------
    n_neighbours : KSG k
    n_bootstrap  : bootstrap resamples
    seed         : rng seed — pass rng externally to reuse the same
                   Generator across multiple pca_dim values
    """

    def __init__(self,
                 n_neighbours: int = 3,
                 n_bootstrap: int = 20,
                 seed: int = 42,
                **_kwargs,
                 ):
        self.n_neighbours = n_neighbours
        self.n_bootstrap  = n_bootstrap
        self.seed         = seed
    def compute(
        self,
        category_data: dict,
        pca_dim:       int,
        rng:           np.random.Generator = None,
        pca_text:      PCA = None,
        **_kwargs,
    ) -> dict:
        """
        Compute bootstrapped MI per category.

        Parameters
        ----------
        category_data : {cat: {'ts': ndarray, 'y': ndarray, 'x': ndarray}}
        pca_dim       : number of PCA dims for text embeddings
        rng           : numpy Generator — if None, creates one with self.seed.
                        Pass the same rng instance across pca_dim values to
                        keep bootstrap resamples comparable.
        pca_text      : pre-fitted PCA — if None, fits on pooled x.
                        Pass a fitted PCA to reuse the same embedding space
                        across pca_dim values.

        Returns
        -------
        Results dict with keys: text, ts, joint, conditional (bootstrap lists),
        <key>_mean, <key>_std, explained_variance, d_conservative, d_liberal
        — all keyed by category.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        pooled, pca_text = _prepare_pooled(category_data, pca_text)

        metrics     = ['text', 'ts', 'joint', 'conditional']
        all_metrics = (metrics
                       + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std"  for m in metrics]
                       + ['explained_variance', 'd_conservative', 'd_liberal'])
        results = {m: {c: [] for c in category_data} for m in all_metrics}

        cumvar         = np.cumsum(pca_text.explained_variance_ratio_)

        for cat, vals in category_data.items():
            x_pooled, X_ts, Y = pooled[cat]

            X_text = pca_text.transform(x_pooled)[:, :pca_dim]
            X_joint = np.hstack([X_ts, X_text])
            N       = len(Y)
            d_conservative = int(np.sqrt(N / 2))
            d_liberal      = N // 5
            if pca_dim > d_liberal:
                warnings.warn(
                    f"pca_dim={pca_dim} > liberal bound d={d_liberal} "
                    f"(N={N}). MI estimates may be unreliable. "
                    "Consider reducing pca_dim.",
                    UserWarning
                )


            for _ in range(self.n_bootstrap):
                idx    = rng.choice(N, size=N, replace=True)
                y_boot = Y[idx]

                mi_text  = ksg_mi(X_text[idx],  y_boot, self.n_neighbours)
                mi_ts    = ksg_mi(X_ts[idx],    y_boot, self.n_neighbours)
                mi_joint = ksg_mi(X_joint[idx], y_boot, self.n_neighbours)

                results['text'][cat].append(mi_text)
                results['ts'][cat].append(mi_ts)
                results['joint'][cat].append(mi_joint)
                results['conditional'][cat].append(mi_joint - mi_ts)

            for m in metrics:
                results[f"{m}_mean"][cat] = float(np.mean(results[m][cat]))
                results[f"{m}_std"][cat]  = float(np.std(results[m][cat]))

            results['explained_variance'][cat] = float(
                cumvar[min(pca_dim - 1, len(cumvar) - 1)]
            )
            results['d_conservative'][cat] = d_conservative
            results['d_liberal'][cat]      = d_liberal

        return results

# PID
class PIDMetric:
    """
    Partial Information Decomposition via the PID repo (pliang279/PID).

    Decomposes I(Y ; X_ts, X_text) into four atoms:
        redundancy : information carried by BOTH modalities
        unique_ts  : information carried only by time series
        unique_text: information carried only by text
        synergy    : information only available from the combination

    Requires:
        pip install cvxpy
        git clone https://github.com/pliang279/PID  (add to PYTHONPATH)

    PID requires discrete inputs. Continuous variables are discretised:
        X_ts   → KMeans cluster labels  (n_clusters_ts  clusters)
        X_text → KMeans cluster labels  (n_clusters_text clusters)
        Y      → rounded to n_bins_y equally-spaced bins


    Parameters
    ----------
    n_clusters_ts   : KMeans clusters for ts discretisation
    n_clusters_text : KMeans clusters for text discretisation
    n_bins_y        : bins for Y discretisation (3 gives labels {-1, 0, 1})
    seed            : random seed for KMeans
    """

    def __init__(
        self,
        n_clusters_ts:    int  = 4,    # 4 phase positions on sine wave
        n_clusters_text:  int  = 3,
        n_bins_y:         int  = 3,
        seed:             int  = 42,
        **_kwargs,
    ):
        self.n_clusters_ts    = n_clusters_ts
        self.n_clusters_text  = n_clusters_text
        self.n_bins_y         = n_bins_y
        self.seed             = seed

    def _discretise(self, X, n_clusters, n_seeds=10):
        from sklearn.cluster import KMeans
        X = np.nan_to_num(X).reshape(X.shape[0], -1)
        # Use best inertia seed for stability
        best = min(
            (KMeans(n_clusters=min(n_clusters, len(X)),
                    random_state=s, n_init=10).fit(X)
            for s in range(n_seeds)),
            key=lambda m: m.inertia_,
        )
        return best.labels_.reshape(-1, 1)

    def _discretise_y(self, Y, strategy='sign', channel: int = 0):
        """
        Discretise a continuous target array to integer labels.

        For 3-D Rössler, Y has shape (N, H*C) where the columns are
        [x_0…x_{H-1}, y_0…y_{H-1}, z_0…z_{H-1}].  The `channel`
        parameter selects which future step index to use (default 0 =
        first step of the x-component, the primary observable).

        PID requires a single discrete scalar target, so only one
        column of Y is used regardless of dimensionality.
        """
        Y = Y[:, channel] if Y.ndim > 1 else Y.ravel()
        if strategy == 'sign':
            raw = np.sign(Y).astype(int)          # {-1, 0, +1} or {-1, +1}
            uniq = np.sort(np.unique(raw))         # contiguous 0-indexed labels
            return np.searchsorted(uniq, raw)
        elif strategy == 'quantile':
            from sklearn.preprocessing import KBinsDiscretizer
            kbd = KBinsDiscretizer(n_bins=self.n_bins_y,
                                encode='ordinal',
                                strategy='quantile')
            return kbd.fit_transform(Y.reshape(-1,1)).ravel().astype(int)
        elif strategy == 'discrete':
            # we know values are close to 1,0,-5,-1
            midpoints = [1.2, 0.2, -0.2, -0.7, -1.2]
            # np.digitize returns 0-indexed bin IDs (0 to 3)
            bin_indices = np.digitize(Y, midpoints)
            return bin_indices
        elif strategy == 'energy':
            # we know values are close to 1,0,-5,-1
            midpoints = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            # np.digitize returns 0-indexed bin IDs (0 to 3)
            bin_indices = np.digitize(Y, midpoints)
            return bin_indices
        else:  # equal-width fallback
            bins = np.linspace(Y.min()-1e-9, Y.max()+1e-9, self.n_bins_y+1)
            return np.digitize(Y, bins[1:-1])

    def compute(
        self,
        category_data: dict,
        pca_dim:       int  = 4,
        pca_text:      object = None,
        strategy:      str  = 'sign',
        n_bootstrap:   int  = 1,
        y_channel:     int  = 0,   # which column of Y to discretise;
                                    # 0 = first future step of x-component
        **_kwargs,
    ) -> dict:
        """
        Compute PID per annotation category.

        The KMeans codebook is fitted once on all pooled data and held fixed
        across bootstrap iterations, so resampling reflects genuine sampling
        variance rather than discretisation instability.

        When n_bootstrap=1 (default) the full data is used without resampling,
        returning a single-element list per category (compatible with
        build_eval_table's np.mean/np.std handling).

        Parameters
        ----------
        category_data : output of benchmark.data.build_category_data()
        pca_dim       : PCA dims to apply to text before clustering
        pca_text      : pre-fitted PCA (fitted externally on pooled x)
                        if None, fits on pooled data here
        n_bootstrap   : bootstrap resamples; 1 = point estimate only

        Returns
        -------
        {
          'redundancy':  {cat: [float, ...]},
          'unique_ts':   {cat: [float, ...]},
          'unique_text': {cat: [float, ...]},
          'synergy':     {cat: [float, ...]},
        }
        """
        pooled, pca_text = _prepare_pooled(category_data, pca_text)
        rng = np.random.default_rng(self.seed)

        # ── FIT TS AND TEXT KMEANS ONCE ON ALL CATEGORIES POOLED ─────────
        # Codebook is fixed for all bootstrap iterations so that variance in
        # resampled estimates reflects data uncertainty, not cluster instability.
        all_X_ts = np.vstack([ts_p for _, ts_p, _ in pooled.values()])
        all_X_text = np.vstack([
            pca_text.transform(x_p)[:, :pca_dim]
            for x_p, _, _ in pooled.values()
        ])
        from sklearn.cluster import KMeans
        ts_kmeans = min(
            (KMeans(n_clusters=min(self.n_clusters_ts, len(all_X_ts)),
                    random_state=s, n_init=10).fit(all_X_ts)
            for s in range(10)),
            key=lambda m: m.inertia_,
        )
        text_kmeans = min(
            (KMeans(n_clusters=min(self.n_clusters_text, len(all_X_text)),
                    random_state=s, n_init=10).fit(all_X_text)
            for s in range(10)),
            key=lambda m: m.inertia_,
        )

        pid_keys = ['redundancy', 'unique_ts', 'unique_text', 'synergy']
        results  = {k: {} for k in pid_keys}

        # Fit Y discretisation once on pooled Y so bins are consistent across categories
        all_Y = np.vstack([y_p for _, _, y_p in pooled.values()])
        y_all_disc = self._discretise_y(all_Y, strategy=strategy, channel=y_channel)
        cat_sizes = {c: len(pooled[c][2]) for c in category_data}
        offset = 0
        y_disc_per_cat = {}
        for c in category_data:
            n = cat_sizes[c]
            y_disc_per_cat[c] = y_all_disc[offset:offset + n]
            offset += n

        import importlib.util, pathlib
        _rus_path = pathlib.Path(__file__).parent.parent.parent / "PID" / "synthetic" / "rus.py"
        _spec = importlib.util.spec_from_file_location("rus", _rus_path)
        _rus  = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_rus)

        for cat in category_data:
            x_pooled, X_ts, Y = pooled[cat]
            X_text = pca_text.transform(x_pooled)[:, :pca_dim]

            # Predict cluster labels using fixed codebook
            x1_full = ts_kmeans.predict(X_ts).ravel()
            x2_full = text_kmeans.predict(X_text).ravel()
            y_full  = y_disc_per_cat[cat]
            n       = len(x1_full)

            cat_samples: dict[str, list] = {k: [] for k in pid_keys}

            for b in range(n_bootstrap):
                if n_bootstrap == 1:
                    idx = np.arange(n)
                else:
                    idx = rng.integers(0, n, size=n)

                P, _ = _rus.convert_data_to_distribution(
                    x1_full[idx], x2_full[idx], y_full[idx]
                )
                raw = _rus.get_measure(P)

                cat_samples['redundancy'].append(float(raw['redundancy']))
                cat_samples['unique_ts'].append(float(raw['unique1']))
                cat_samples['unique_text'].append(float(raw['unique2']))
                cat_samples['synergy'].append(float(raw['synergy']))

            for k in pid_keys:
                results[k][cat] = cat_samples[k]

            print(f"  {cat}: R={np.mean(cat_samples['redundancy']):.4f}  "
                  f"U_ts={np.mean(cat_samples['unique_ts']):.4f}  "
                  f"U_text={np.mean(cat_samples['unique_text']):.4f}  "
                  f"S={np.mean(cat_samples['synergy']):.4f}"
                  + (f"  (n_bootstrap={n_bootstrap})" if n_bootstrap > 1 else ""))

        return results


# V-Information
class VInformationMetric:
    """
    V-usable information (Xu et al., ICLR 2020) using Ridge regression
    as the predictor family V.

    Under squared loss with a linear predictor family, V-information
    specialises to the coefficient of determination (R²):

        I_V(X -> Y)         = H_V(Y) - H_V(Y|X)
                            = (1 - R²_null) - (1 - R²_X)
                            = R²_X

    where R²_null = 0 (mean predictor).  The conditional V-information

        I_V(X_text -> Y | X_ts) = R²(X_ts, X_text) - R²(X_ts)

    measures the unique forecasting gain from text beyond the time series.
    This is the V-information analogue of the KSG conditional MI F term,
    but is estimated via regression rather than k-NN density, making it
    reliable at small N and in high dimensions without PCA constraints.

    Unlike KSG, V-information is representation-dependent by design —
    it measures information that a specific model class can exploit,
    not the Shannon-theoretic information in the distribution.

    Parameters
    ----------
    n_bootstrap : bootstrap resamples for variance estimation
    seed        : rng seed
    alpha       : Ridge regularisation strength (default 1.0)
    cv          : cross-validation folds for RidgeCV (default 5)
    """

    def __init__(
        self,
        n_bootstrap: int   = 20,
        seed:        int   = 42,
        alphas:       list = [0.01, 0.1, 1.0, 10.0, 100.0],
        cv:             int = 5,
        **_kwargs,   # absorbs MI-only kwargs (n_neighbours, rng) from Experiment
    ):
        self.n_bootstrap = n_bootstrap
        self.seed        = seed
        self.alphas       = alphas
        self.cv          = cv

    # ── Internal helpers ──────────────────────────────────────────────────

    def _r2(
        self,
        X:   np.ndarray,   # (N, d)
        Y:   np.ndarray,   # (N,) univariate  OR  (N, K) multivariate
        rng: np.random.Generator,
        idx: np.ndarray,   # bootstrap indices into 0..N-1
    ) -> float:
        """
        In-bag R² of a Ridge regression on the bootstrap resample.

        For univariate Y (N,) or (N, 1): standard scalar R².
        For multivariate Y (N, K) — e.g. 3-D Rössler with K = H*C:
            fits a single multi-output RidgeCV, then returns the
            mean per-output R² so the scale is unchanged relative to
            the univariate case.
        """
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        X_b = X[idx]
        Y_b = Y[idx]

        # Normalise shape: keep 2-D for multi-output, squeeze to 1-D if scalar
        multivariate = Y_b.ndim > 1 and Y_b.shape[1] > 1
        if not multivariate:
            Y_b = Y_b.ravel()

        # Degenerate case — all target values identical
        variances = np.var(Y_b, axis=0) if multivariate else np.array([np.var(Y_b)])
        if np.all(variances < 1e-12):
            return 0.0

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X_b)

        reg = RidgeCV(alphas=self.alphas)
        reg.fit(X_s, Y_b)
        Y_pred = reg.predict(X_s)

        if multivariate:
            # Per-output R², averaged — keeps the same [0, 1] scale
            ss_res = np.sum((Y_b - Y_pred) ** 2, axis=0)
            ss_tot = np.sum((Y_b - Y_b.mean(axis=0)) ** 2, axis=0)
            valid  = ss_tot > 1e-12
            r2_per = np.where(valid, 1.0 - ss_res / np.where(valid, ss_tot, 1.0), 0.0)
            return float(max(0.0, float(np.mean(r2_per))))
        else:
            ss_res = np.sum((Y_b - Y_pred) ** 2)
            ss_tot = np.sum((Y_b - Y_b.mean()) ** 2)
            return float(max(0.0, 1.0 - ss_res / ss_tot))

    # ── Public API — mirrors MutualInformationMetric.compute() ───────────

    def compute(
        self,
        category_data: dict,
        pca_dim:       int,
        rng:           np.random.Generator = None,
        pca_text:      PCA = None,
        **_kwargs,
    ) -> dict:
        """
        Compute bootstrapped V-information per annotation category.

        Signature mirrors MutualInformationMetric.compute().

        Result dict keys (same pattern as MutualInformationMetric):
            'v_text'       {cat: [bootstrap R² samples]}
            'v_ts'         {cat: [bootstrap R² samples]}
            'v_joint'      {cat: [bootstrap R² samples]}
            'v_conditional'{cat: [bootstrap R²_joint - R²_ts samples]}
            '<key>_mean'   {cat: float}
            '<key>_std'    {cat: float}
            'explained_variance' {cat: float}
            'd_conservative'     {cat: int}
            'd_liberal'          {cat: int}

        Parameters
        ----------
        category_data : {cat: {'ts', 'y', 'x', 'n_samples'}}
        pca_dim       : PCA dims for text embeddings
        rng           : numpy Generator (created from self.seed if None)
        pca_text      : pre-fitted PCA; fitted on pooled data if None
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        pooled, pca_text = _prepare_pooled(category_data, pca_text)

        metrics     = ['v_text', 'v_ts', 'v_joint', 'v_conditional']
        all_metrics = (metrics
                       + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std"  for m in metrics]
                       + ['explained_variance', 'd_conservative', 'd_liberal'])
        results = {m: {c: [] for c in category_data} for m in all_metrics}

        cumvar = np.cumsum(pca_text.explained_variance_ratio_)

        for cat in category_data:
            x_pooled, X_ts, Y = pooled[cat]
            X_text  = pca_text.transform(x_pooled)[:, :pca_dim]
            X_joint = np.hstack([X_ts, X_text])
            # Keep Y 2-D throughout — _r2 handles both univariate (N,1) and
            # multivariate (N, H*C) via multi-output ridge + mean R².
            N = len(Y)

            d_conservative = int(np.sqrt(N / 2))
            d_liberal      = N // 5

            for _ in range(self.n_bootstrap):
                idx = rng.choice(N, size=N, replace=True)

                r2_text  = self._r2(X_text,  Y, rng, idx)
                r2_ts    = self._r2(X_ts,    Y, rng, idx)
                r2_joint = self._r2(X_joint, Y, rng, idx)

                results['v_text'][cat].append(r2_text)
                results['v_ts'][cat].append(r2_ts)
                results['v_joint'][cat].append(r2_joint)
                # Conditional V-info: unique text contribution beyond ts
                # Mirrors KSG conditional: mi_joint - mi_ts from same resample
                results['v_conditional'][cat].append(
                    r2_joint - r2_ts
                )

            for m in metrics:
                results[f"{m}_mean"][cat] = float(np.mean(results[m][cat]))
                results[f"{m}_std"][cat]  = float(np.std(results[m][cat]))

            results['explained_variance'][cat] = float(
                cumvar[min(pca_dim - 1, len(cumvar) - 1)]
            )
            results['d_conservative'][cat] = d_conservative
            results['d_liberal'][cat]      = d_liberal

        return results




# ---------------------------------------------------------------------------
# MINE — Mutual Information Neural Estimation
# ---------------------------------------------------------------------------

class MINEMetric:
    """
    Neural MI estimator (Belghazi et al., 2018) via the mine-pytorch library.

    Drop-in replacement for MutualInformationMetric — same compute()
    signature, same result-dict keys, same PCA text reduction.

    Differences from KSG:
        - Re-trains a statistics network for each MI term.  This is
          expensive, so n_bootstrap defaults to 1 (single estimate).
          Set n_bootstrap > 1 to get variance estimates at the cost
          of n_bootstrap × 3 training runs per category.
        - Returns MI in nats (same as KSG).
        - Not sensitive to the curse of dimensionality — works well
          at higher pca_dim values where KSG degrades.

    Parameters
    ----------
    n_bootstrap      : resamples (1 = point estimate only, no variance)
    hidden_dim       : width of each hidden layer in the statistics network
    n_layers         : number of hidden layers (minimum 1)
    mine_iters       : gradient-descent iterations per MI estimate
    mine_batch_size  : mini-batch size during MINE optimisation
    loss             : 'mine' (EMA), 'mine_biased', or 'fdiv'
    lr               : Adam learning rate
    mine_pytorch_path: explicit path to the mine-pytorch repo root;
                       if None, auto-detects relative to this file
    seed             : rng seed for bootstrap resampling
    """

    def __init__(
        self,
        n_bootstrap:       int   = 1,
        hidden_dim:        int   = 100,
        n_layers:          int   = 2,
        mine_iters:        int   = 500,
        mine_batch_size:   int   = 64,
        loss:              str   = 'mine',
        lr:                float = 1e-4,
        mine_pytorch_path: str   = None,
        seed:              int   = 42,
        **_kwargs,
    ):
        self.n_bootstrap       = n_bootstrap
        self.hidden_dim        = hidden_dim
        self.n_layers          = n_layers
        self.mine_iters        = mine_iters
        self.mine_batch_size   = mine_batch_size
        self.loss              = loss
        self.lr                = lr
        self.mine_pytorch_path = mine_pytorch_path
        self.seed              = seed

    # ── Import helpers ────────────────────────────────────────────────────

    def _import_mine(self):
        """
        Lazily import the Mine class, injecting the repo path and stubbing
        out torchvision / pytorch_lightning which Mine doesn't actually use
        but imports at module level.
        """
        import sys
        from unittest.mock import MagicMock
        from pathlib import Path

        for mod in ['torchvision', 'torchvision.datasets',
                    'torchvision.transforms', 'pytorch_lightning']:
            sys.modules.setdefault(mod, MagicMock())

        path = self.mine_pytorch_path
        if path is None:
            # Auto-detect: mine-pytorch lives next to src/ in the repo root
            candidate = Path(__file__).resolve().parent.parent.parent / 'mine-pytorch'
            if candidate.exists():
                path = str(candidate)

        if path and path not in sys.path:
            sys.path.insert(0, path)

        from mine.models.mine import Mine
        return Mine

    # ── Network builder ───────────────────────────────────────────────────

    def _build_net(self, input_dim: int):
        """Statistics network T: R^input_dim → R^1."""
        import torch.nn as nn
        layers = [nn.Linear(input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(self.hidden_dim, 1)]
        return nn.Sequential(*layers)

    # ── Single MI estimate ────────────────────────────────────────────────

    def _estimate_mi(self, Mine, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Train a fresh MINE estimator on (X, Y) and return I(X; Y) in nats.

        A new network is built for every call so each bootstrap resample
        and each MI term (text / ts / joint) gets an independent estimate.
        """
        import torch
        import torch.optim as optim

        # Standardise to zero-mean unit-variance so exp() in EMALoss doesn't overflow.
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        net = self._build_net(x.shape[1] + y.shape[1])
        estimator = Mine(T=net, loss=self.loss, method='concat')
        opt = optim.Adam(estimator.parameters(), lr=self.lr)

        mi = estimator.optimize(
            x, y,
            iters=self.mine_iters,
            batch_size=self.mine_batch_size,
            opt=opt,
        )
        return float(mi.item() if hasattr(mi, 'item') else mi)

    # ── Public API (mirrors MutualInformationMetric.compute) ─────────────

    def compute(
        self,
        category_data: dict,
        pca_dim:       int,
        rng:           np.random.Generator = None,
        pca_text:      PCA = None,
        **_kwargs,
    ) -> dict:
        """
        Compute MINE MI estimates per annotation category.

        Parameters and return value are identical to
        MutualInformationMetric.compute() — drop-in replacement.

        Result keys
        -----------
        text / ts / joint / conditional        : lists of bootstrap samples
        <key>_mean / <key>_std                 : float
        explained_variance / d_conservative / d_liberal
        """
        Mine = self._import_mine()

        if rng is None:
            rng = np.random.default_rng(self.seed)

        pooled, pca_text = _prepare_pooled(category_data, pca_text)

        metrics     = ['text', 'ts', 'joint', 'conditional']
        all_metrics = (metrics
                       + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std"  for m in metrics]
                       + ['explained_variance', 'd_conservative', 'd_liberal'])
        results = {m: {c: [] for c in category_data} for m in all_metrics}
        cumvar  = np.cumsum(pca_text.explained_variance_ratio_)

        for cat, vals in category_data.items():
            x_pooled, X_ts, Y = pooled[cat]
            X_text  = pca_text.transform(x_pooled)[:, :pca_dim]
            X_joint = np.hstack([X_ts, X_text])
            N       = len(Y)

            d_conservative = int(np.sqrt(N / 2))
            d_liberal      = N // 5

            for b in range(self.n_bootstrap):
                # For n_bootstrap=1 use all data; for >1 bootstrap resample
                idx = (rng.choice(N, size=N, replace=True)
                       if self.n_bootstrap > 1 else np.arange(N))

                mi_text  = self._estimate_mi(Mine, X_text[idx],  Y[idx])
                mi_ts    = self._estimate_mi(Mine, X_ts[idx],    Y[idx])
                mi_joint = self._estimate_mi(Mine, X_joint[idx], Y[idx])

                results['text'][cat].append(mi_text)
                results['ts'][cat].append(mi_ts)
                results['joint'][cat].append(mi_joint)
                results['conditional'][cat].append(mi_joint - mi_ts)

            for m in metrics:
                results[f"{m}_mean"][cat] = float(np.mean(results[m][cat]))
                results[f"{m}_std"][cat]  = float(np.std(results[m][cat]))

            results['explained_variance'][cat] = float(
                cumvar[min(pca_dim - 1, len(cumvar) - 1)]
            )
            results['d_conservative'][cat] = d_conservative
            results['d_liberal'][cat]      = d_liberal

        return results

class InfoNCEMetric:
    """
    InfoNCE mutual information estimator (Oord et al., 2018).

    InfoNCE provides a lower bound on MI via a contrastive objective:

        I(X; Y) >= E[log f(x_i, y_i) - log(1/N * sum_j f(x_i, y_j))]

    where f(x, y) = exp(T(x, y)) and T is a learned critic network that
    scores alignment between x and y.  The bound tightens as N (batch
    size) increases, and saturates at log(N) nats — so batch_size must
    satisfy log(batch_size) >> true MI.  At your benchmark MI values
    (0.3 – 1.2 nats), batch_size=64 gives a ceiling of ~4.2 nats which
    is well clear.

    Unlike KSG, InfoNCE:
        - Does not require PCA dimensionality reduction for reliability
        - Avoids the chain-rule inconsistency that makes KSG conditional
          MI negative on the Rössler attractor
        - Scales gracefully to higher pca_dim values
        - Returns a LOWER BOUND on MI, not an unbiased estimate — so
          values will be slightly below KSG on well-estimated problems

    Architecture: a two-layer MLP critic T(x, y) = MLP([x; y]) -> R
    trained with Adam to maximise the InfoNCE objective.

    Parameters
    ----------
    n_bootstrap    : bootstrap resamples (1 = point estimate, no variance)
    hidden_dim     : hidden layer width of the critic MLP
    n_layers       : number of hidden layers (minimum 1)
    n_iters        : gradient steps per MI estimate
    batch_size     : contrastive batch size — ceiling on MI estimate is
                     log(batch_size) nats.  Must satisfy batch_size <= N.
    lr             : Adam learning rate
    seed           : rng seed for bootstrap resampling and torch init

    Reference
    ---------
    Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with
    Contrastive Predictive Coding. arXiv:1807.03748.
    """

    def __init__(
        self,
        n_bootstrap:  int   = 1,
        hidden_dim:   int   = 128,
        n_layers:     int   = 2,
        n_iters:      int   = 1000,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        seed:         int   = 42,
        **_kwargs,    # absorbs metric-specific kwargs not used by this class
    ):
        self.n_bootstrap = n_bootstrap
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.n_iters     = n_iters
        self.batch_size  = batch_size
        self.lr          = lr
        self.seed        = seed

    # ── Critic network ────────────────────────────────────────────────────

    def _build_critic(self, input_dim: int):
        """
        Two-stream critic T: R^input_dim -> R.

        Concatenation critic: T([x; y]) with shared MLP.
        The output scores how well x and y align — high score means
        the pair is likely a positive (drawn from the joint p(x,y))
        rather than a negative (x from marginal, y independently drawn).
        """
        import torch.nn as nn

        layers = [nn.Linear(input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers)

    # ── Core InfoNCE objective ────────────────────────────────────────────

    def _infonce_loss(self, critic, x, y):
        """
        Compute the InfoNCE lower bound on I(X; Y).

        For a batch of N pairs (x_i, y_i):
            - Positive scores: T(x_i, y_i) for each i          shape (N,)
            - All scores:      T(x_i, y_j) for all i,j         shape (N, N)
            - InfoNCE = mean_i [ T(x_i,y_i) - log sum_j exp(T(x_i,y_j)) ]
                      = mean_i [ T(x_i,y_i) - logsumexp_j T(x_i,y_j) ]

        This is equivalent to the diagonal of a cross-entropy loss over
        an N-way classification problem, where the 'label' for row i is
        position i (the correct positive).

        Returns the InfoNCE estimate (positive float, in nats).
        """
        import torch

        N = x.shape[0]

        # Score all (x_i, y_j) pairs: tile x and y to form N^2 combinations
        # x_rep: (N, N, dx) — each x_i repeated N times across dim 1
        # y_rep: (N, N, dy) — all y_j repeated N times across dim 0
        x_rep = x.unsqueeze(1).expand(N, N, -1)   # (N, N, dx)
        y_rep = y.unsqueeze(0).expand(N, N, -1)   # (N, N, dy)
        xy    = torch.cat([x_rep, y_rep], dim=-1)  # (N, N, dx+dy)

        # Scores: (N, N) — scores[i, j] = T(x_i, y_j)
        scores = critic(xy.reshape(N * N, -1)).reshape(N, N)

        # InfoNCE: for each row i, the positive is at column i
        # Loss = -mean [ scores[i,i] - logsumexp(scores[i,:]) ]
        # Negate because we maximise the bound = minimise this loss
        log_denom = torch.logsumexp(scores, dim=1)   # (N,) log sum_j exp(T(x_i,y_j))
        infonce   = (scores.diagonal() - log_denom).mean()

        # Add log(N) to convert from the biased estimator to the unbiased
        # lower bound: InfoNCE_bound = E[...] + log(N)
        # (Oord et al. 2018 Eq. 4 — the log(1/N) term becomes log(N))
        return infonce + np.log(N)

    # ── Single MI estimate ────────────────────────────────────────────────

    def _estimate_mi(
        self,
        X: np.ndarray,   # (N, dx)
        Y: np.ndarray,   # (N, dy)
        torch_seed: int = 0,
    ) -> float:
        """
        Train a fresh critic on (X, Y) and return InfoNCE MI estimate.

        Trains for self.n_iters steps using mini-batches of self.batch_size.
        Returns the final InfoNCE value averaged over the last 10% of
        training iterations for stability (avoids single-step noise).
        """
        import torch
        import torch.optim as optim

        torch.manual_seed(torch_seed)

        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        if Y_t.ndim == 1:
            Y_t = Y_t.unsqueeze(-1)

        N       = X_t.shape[0]
        bs      = min(self.batch_size, N)
        critic  = self._build_critic(X_t.shape[1] + Y_t.shape[1])
        opt     = optim.Adam(critic.parameters(), lr=self.lr)

        # Collect estimates from the last 10% of iterations for stability
        warmup      = int(self.n_iters * 0.9)
        tail_values = []

        for step in range(self.n_iters):
            idx   = torch.randperm(N)[:bs]
            x_b   = X_t[idx]
            y_b   = Y_t[idx]

            opt.zero_grad()
            mi_est = self._infonce_loss(critic, x_b, y_b)
            loss   = -mi_est   # maximise InfoNCE = minimise negative InfoNCE
            loss.backward()
            opt.step()

            if step >= warmup:
                tail_values.append(mi_est.item())

        return float(np.mean(tail_values)) if tail_values else float(mi_est.item())

    # ── Public API — mirrors MutualInformationMetric.compute() exactly ────

    def compute(
        self,
        category_data: dict,
        pca_dim:       int,
        rng:           np.random.Generator = None,
        pca_text:      PCA = None,
        **_kwargs,
    ) -> dict:
        """
        Compute InfoNCE MI estimates per annotation category.

        Signature and return structure are identical to
        MutualInformationMetric.compute() — drop-in replacement.

        Computes four quantities matching the KSG Venn diagram:
            text        : I(X_text; Y)             = B + C
            ts          : I(X_ts; Y)               = A + C
            joint       : I(X_ts, X_text; Y)       = A + B + C
            conditional : joint - ts               = B  (unique text)

        The conditional computation uses both terms from the same
        training run via the same random seed, which is more consistent
        than KSG where bias differs between joint and marginal problems.

        Parameters
        ----------
        category_data : {cat: {'ts', 'y', 'x', 'n_samples'}}
        pca_dim       : PCA dims for text embeddings
        rng           : numpy Generator (created if None)
        pca_text      : pre-fitted PCA (fitted on pooled data if None)

        Returns
        -------
        Same dict structure as MutualInformationMetric.compute():
            'text', 'ts', 'joint', 'conditional' : {cat: [bootstrap samples]}
            '<key>_mean', '<key>_std'             : {cat: float}
            'explained_variance'                  : {cat: float}
            'd_conservative', 'd_liberal'         : {cat: int}
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        pooled, pca_text = _prepare_pooled(category_data, pca_text)

        metrics     = ['text', 'ts', 'joint', 'conditional']
        all_metrics = (metrics
                       + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std"  for m in metrics]
                       + ['explained_variance', 'd_conservative', 'd_liberal'])
        results = {m: {c: [] for c in category_data} for m in all_metrics}
        cumvar  = np.cumsum(pca_text.explained_variance_ratio_)

        for cat in category_data:
            x_pooled, X_ts, Y = pooled[cat]
            X_text  = pca_text.transform(x_pooled)[:, :pca_dim]
            X_joint = np.hstack([X_ts, X_text])
            Y_flat  = Y.ravel() if Y.ndim == 1 else Y
            N       = len(Y_flat)

            d_conservative = int(np.sqrt(N / 2))
            d_liberal      = N // 5

            # Warn if batch_size exceeds N — InfoNCE requires bs <= N
            if self.batch_size > N:
                warnings.warn(
                    f"batch_size={self.batch_size} > N={N} for category "
                    f"'{cat}'. Reducing to N.",
                    UserWarning,
                )

            for b in range(self.n_bootstrap):
                # For n_bootstrap=1 use all data; otherwise bootstrap resample
                if self.n_bootstrap > 1:
                    idx    = rng.choice(N, size=N, replace=True)
                    x_text_b  = X_text[idx]
                    x_ts_b    = X_ts[idx]
                    x_joint_b = X_joint[idx]
                    y_b       = Y_flat[idx]
                else:
                    x_text_b  = X_text
                    x_ts_b    = X_ts
                    x_joint_b = X_joint
                    y_b       = Y_flat

                # Use b as the torch seed so each resample is deterministic
                # but independent — same pattern as MINE implementation
                torch_seed = self.seed * 1000 + b

                mi_text  = self._estimate_mi(x_text_b,  y_b, torch_seed)
                mi_ts    = self._estimate_mi(x_ts_b,    y_b, torch_seed + 1)
                mi_joint = self._estimate_mi(x_joint_b, y_b, torch_seed + 2)

                results['text'][cat].append(mi_text)
                results['ts'][cat].append(mi_ts)
                results['joint'][cat].append(mi_joint)
                results['conditional'][cat].append(mi_joint - mi_ts)

            for m in metrics:
                results[f"{m}_mean"][cat] = float(np.mean(results[m][cat]))
                results[f"{m}_std"][cat]  = float(np.std(results[m][cat]))

            results['explained_variance'][cat] = float(
                cumvar[min(pca_dim - 1, len(cumvar) - 1)]
            )
            results['d_conservative'][cat] = d_conservative
            results['d_liberal'][cat]      = d_liberal

        return results

class CCAMutualInformationMetric:
    """
    Mutual information estimator based on Canonical Correlation Analysis.

    Assumes the joint distribution p(X, Y) is multivariate Gaussian and
    estimates MI from the canonical correlations rho_1, ..., rho_k:

        I(X; Y) = -0.5 * sum_k log(1 - rho_k^2)

    This is the closed-form MI for jointly Gaussian variables.  CCA is
    model-based and deterministic — no training, no hyperparameters, and
    very low sample complexity.  It is the most effective estimator when
    the Gaussianity assumption holds, and remains competitive even when
    it is mildly violated (Czyż et al., NeurIPS 2023).

    Limitations:
        - Only captures linear dependence (same as V-information).
        - Fails when the joint distribution is strongly non-Gaussian
          (heavy tails, multimodal, spiral structure).
        - Regularisation is required when N < d_X + d_Y to avoid
          singular covariance matrices.

    The result dict mirrors MutualInformationMetric exactly:
        'text' / 'ts' / 'joint' / 'conditional'   {cat: [bootstrap samples]}
        '<key>_mean' / '<key>_std'                  {cat: float}
        'explained_variance'                        {cat: float}
        'd_conservative' / 'd_liberal'              {cat: int}

    Parameters
    ----------
    n_bootstrap : bootstrap resamples for variance estimation.
                  Set to 1 for a single point estimate (CCA is fast).
    eps         : ridge regularisation added to diagonal of covariance
                  to handle near-singular matrices (default 1e-5).
    seed        : rng seed for bootstrap resampling.

    References
    ----------
    Kay (1992) "Feature discovery under contextual supervision using
        mutual information."
    Brillinger (2004) "Some data analyses using mutual information."
    Czyż et al. (2023) "Beyond Normal: On the Evaluation of Mutual
        Information Estimators." NeurIPS 2023.
    """

    def __init__(
        self,
        n_bootstrap: int   = 20,
        eps:         float = 1e-5,
        seed:        int   = 42,
        **_kwargs,    # absorbs metric-specific kwargs not used by this class
    ):
        self.n_bootstrap = n_bootstrap
        self.eps         = float(eps)   # guard: YAML may deliver this as str '1e-5'
        self.seed        = seed

    # ── Core CCA MI estimator ─────────────────────────────────────────────

    def _cca_mi(
        self,
        X: np.ndarray,   # (N, dx) — continuous
        Y: np.ndarray,   # (N, dy) — continuous
    ) -> float:
        """
        Estimate I(X; Y) from canonical correlations under Gaussian assumption.

            I(X; Y) = -0.5 * sum_k log(1 - rho_k^2)

        where rho_k^2 are eigenvalues of Cxx^{-1} Cxy Cyy^{-1} Cxy^T,
        i.e., the squared canonical correlations between X and Y.

        Returns MI in nats, clipped to 0.0.
        """
        X = X.reshape(len(X), -1)
        Y = Y.reshape(len(Y), -1) if Y.ndim > 1 else Y.reshape(-1, 1)

        N, dx = X.shape
        dy    = Y.shape[1]

        # Centre (mean subtraction — CCA is translation-invariant)
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Joint covariance via concatenation
        XY  = np.hstack([X, Y])
        Cov = (XY.T @ XY) / (N - 1)

        Cxx = Cov[:dx, :dx] + self.eps * np.eye(dx)
        Cyy = Cov[dx:, dx:] + self.eps * np.eye(dy)
        Cxy = Cov[:dx, dx:]

        # Squared canonical correlations = eigenvalues of
        # Cxx^{-1} Cxy Cyy^{-1} Cxy^T
        # Use Cholesky for numerical stability when N >> d
        try:
            Lxx = np.linalg.cholesky(Cxx)   # Cxx = Lxx Lxx^T
            Lyy = np.linalg.cholesky(Cyy)   # Cyy = Lyy Lyy^T
            # W = Lxx^{-1} Cxy Lyy^{-T}  -->  W W^T has eigenvalues rho^2
            W    = np.linalg.solve(Lxx, Cxy)               # (dx, dy)
            W    = np.linalg.solve(Lyy, W.T).T             # (dx, dy)
            rho2 = np.linalg.svd(W, compute_uv=False) ** 2
        except np.linalg.LinAlgError:
            # Fallback: eigh-based pseudoinverse — avoids SVD entirely so it
            # succeeds when np.linalg.svd / pinv fail on ill-conditioned data.
            def _eigh_pinv(A: np.ndarray) -> np.ndarray:
                vals, vecs = np.linalg.eigh(A)
                tol = max(vals.max() * A.shape[0] * np.finfo(float).eps * 10, 1e-12)
                keep = vals > tol
                return (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T

            Cxx_inv = _eigh_pinv(Cxx)
            Cyy_inv = _eigh_pinv(Cyy)
            M       = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
            # eigvalsh exploits symmetry of M — more stable than eigvals
            rho2    = np.real(np.linalg.eigvalsh(M))

        # Clip to [0, 1-eps] to avoid log(0) at rho^2 = 1
        rho2 = np.clip(rho2, 0.0, 1.0 - 1e-9)

        # MI under joint Gaussianity (Kay 1992, Brillinger 2004)
        mi = -0.5 * float(np.sum(np.log(1.0 - rho2)))
        return max(mi, 0.0)

    # ── Public API — mirrors MutualInformationMetric.compute() exactly ────

    def compute(
        self,
        category_data: dict,
        pca_dim:       int,
        rng:           np.random.Generator = None,
        pca_text:      PCA = None,
        **_kwargs,
    ) -> dict:
        """
        Compute bootstrapped CCA MI estimates per annotation category.

        Signature and return structure are identical to
        MutualInformationMetric.compute() — drop-in replacement.

        Computes four quantities mapping to the Venn diagram:
            text        : I(X_text; Y)              = B + C
            ts          : I(X_ts; Y)                = A + C
            joint       : I(X_ts, X_text; Y)        = A + B + C
            conditional : joint - ts                = B (unique text)

        Unlike KSG, CCA estimates are deterministic for a given resample
        (no k-NN hyperparameter sensitivity), and the chain-rule identity
            I(X_ts, X_text; Y) = I(X_ts; Y) + I(X_text; Y | X_ts)
        holds approximately under the Gaussian assumption, so conditional
        MI estimates are more self-consistent than KSG on complex attractors.

        Parameters
        ----------
        category_data : {cat: {'ts', 'y', 'x', 'n_samples'}}
        pca_dim       : PCA dims for text embeddings
        rng           : numpy Generator (created if None)
        pca_text      : pre-fitted PCA (fitted on pooled data if None)

        Returns
        -------
        Same dict structure as MutualInformationMetric.compute().
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        pooled, pca_text = _prepare_pooled(category_data, pca_text)

        metrics     = ['text', 'ts', 'joint', 'conditional']
        all_metrics = (metrics
                       + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std"  for m in metrics]
                       + ['explained_variance', 'd_conservative', 'd_liberal'])
        results = {m: {c: [] for c in category_data} for m in all_metrics}
        cumvar  = np.cumsum(pca_text.explained_variance_ratio_)

        for cat in category_data:
            x_pooled, X_ts, Y = pooled[cat]
            X_text  = pca_text.transform(x_pooled)[:, :pca_dim]
            X_joint = np.hstack([X_ts, X_text])

            # CCA requires Y to be 2D: (N, dy)
            Y_2d = Y if Y.ndim > 1 else Y.reshape(-1, 1)
            N    = len(Y_2d)

            d_conservative = int(np.sqrt(N / 2))
            d_liberal      = N // 5

            # Warn if pca_dim + ts_dim approaches N — covariance singular
            total_dim = X_joint.shape[1] + Y_2d.shape[1]
            if total_dim >= N:
                warnings.warn(
                    f"CCA: total input dimension ({total_dim}) >= N ({N}) "
                    f"for category '{cat}'. Covariance matrix is rank-deficient; "
                    f"eps regularisation ({self.eps}) may not be sufficient. "
                    f"Reduce pca_dim or increase N.",
                    UserWarning,
                )

            for _ in range(self.n_bootstrap):
                idx = rng.choice(N, size=N, replace=True)

                mi_text  = self._cca_mi(X_text[idx],  Y_2d[idx])
                mi_ts    = self._cca_mi(X_ts[idx],    Y_2d[idx])
                mi_joint = self._cca_mi(X_joint[idx], Y_2d[idx])

                results['text'][cat].append(mi_text)
                results['ts'][cat].append(mi_ts)
                results['joint'][cat].append(mi_joint)
                # Conditional MI: unique text contribution beyond ts
                # CCA satisfies the chain rule approximately under the Gaussian
                # assumption, so this subtraction is more reliable than KSG
                results['conditional'][cat].append(mi_joint - mi_ts)

            for m in metrics:
                results[f"{m}_mean"][cat] = float(np.mean(results[m][cat]))
                results[f"{m}_std"][cat]  = float(np.std(results[m][cat]))

            results['explained_variance'][cat] = float(
                cumvar[min(pca_dim - 1, len(cumvar) - 1)]
            )
            results['d_conservative'][cat] = d_conservative
            results['d_liberal'][cat]      = d_liberal

        return results

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS = {
    'mutual_information': MutualInformationMetric,
    'pid':                PIDMetric,
    'v_information':      VInformationMetric,
    'mine':               MINEMetric,
    'infonce':            InfoNCEMetric,
    'cca': CCAMutualInformationMetric,
}
