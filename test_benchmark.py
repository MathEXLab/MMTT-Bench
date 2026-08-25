"""
Pytest tests for the sine-wave benchmark.

Run with:  pytest test_benchmark.py -v
"""

import numpy as np
import pytest

from src.mmtt_bench.tokenizers import TS_TOKENIZERS
from src.mmtt_bench.data import (
    _attach_lookback_and_future,
    build_category_data_dummy_embedding,
)
from src.mmtt_bench.metrics import ksg_mi, _pool_by_annotation_point


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sine_dense(n=80):
    t = np.linspace(0, 4 * np.pi, n)
    return t, np.sin(t)


def _make_data(t, y, indices):
    """Wrap arrays into the dict format expected by _attach_lookback_and_future."""
    return {
        "metadata": {"t": t.tolist(), "y": y.tolist()},
        "points": [
            {"t": round(float(t[i]), 4), "category": "correct",
             "samples": [{"text": "test"}]}
            for i in indices
        ],
    }


def _attach(data, horizon, lookback, tok_key="identity", patch_len=4, stride=2):
    tok = TS_TOKENIZERS[tok_key](
        patch_len=patch_len, stride=stride, lookback_steps=lookback
    ).build()
    return _attach_lookback_and_future(data, horizon, lookback, tok)


def _t_index(t, t_val):
    """Return the index of a (rounded) t value in the dense array."""
    t_rounded = [round(v, 4) for v in t]
    return t_rounded.index(round(t_val, 4))


# data.py — y_lookback and y_future shapes and values

class TestLookbackFutureShapes:
    """Dimensionality of y_lookback and y_future under different param combos."""

    def setup_method(self):
        self.t, self.y = _sine_dense(n=80)
        # Use indices well into the array so lookback is always fully available
        self.data = _make_data(self.t, self.y, [20, 30, 40, 50])

    def _attached_points(self, **kw):
        return [p for p in _attach(self.data, **kw)["points"] if "y_future" in p]

    # --- y_future length ---

    def test_y_future_length_horizon_1(self):
        for pt in self._attached_points(horizon=1, lookback=8):
            assert len(pt["y_future"]) == 1

    def test_y_future_length_horizon_4(self):
        for pt in self._attached_points(horizon=4, lookback=8):
            assert len(pt["y_future"]) == 4

    # --- y_lookback shape by tokenizer ---

    def test_identity_lookback_shape(self):
        for pt in self._attached_points(horizon=1, lookback=8, tok_key="identity"):
            assert pt["y_lookback"].shape == (8,)

    def test_identity_lookback_shape_different_size(self):
        for pt in self._attached_points(horizon=1, lookback=16, tok_key="identity"):
            assert pt["y_lookback"].shape == (16,)

    def test_patch_mean_lookback_shape(self):
        lookback, patch_len, stride = 8, 4, 2
        expected = (lookback - patch_len) // stride + 1   # = 3
        for pt in self._attached_points(
            horizon=1, lookback=lookback, tok_key="patch_mean",
            patch_len=patch_len, stride=stride,
        ):
            assert pt["y_lookback"].shape == (expected,)

    def test_patch_mean_lookback_shape_stride_1(self):
        lookback, patch_len, stride = 8, 2, 1
        expected = (lookback - patch_len) // stride + 1   # = 7
        for pt in self._attached_points(
            horizon=1, lookback=lookback, tok_key="patch_mean",
            patch_len=patch_len, stride=stride,
        ):
            assert pt["y_lookback"].shape == (expected,)


class TestLookbackFutureValues:
    """Correctness of lookback/future window contents."""

    def setup_method(self):
        self.t, self.y = _sine_dense(n=80)

    def test_y_future_is_strictly_future_not_current(self):
        """y_future[0] must be y[idx+1], not y[idx]."""
        data = _make_data(self.t, self.y, [20, 30, 40])
        data = _attach(data, horizon=1, lookback=8)
        for pt in data["points"]:
            if "y_future" not in pt:
                continue
            idx = _t_index(self.t, pt["t"])
            assert pt["y_future"][0] == pytest.approx(self.y[idx + 1])

    def test_y_future_multi_step_is_contiguous(self):
        """y_future should be y[idx+1 : idx+1+horizon]."""
        data = _make_data(self.t, self.y, [30])
        data = _attach(data, horizon=4, lookback=8)
        for pt in data["points"]:
            if "y_future" not in pt:
                continue
            idx = _t_index(self.t, pt["t"])
            expected = list(self.y[idx + 1 : idx + 5])
            assert list(pt["y_future"]) == pytest.approx(expected)

    def test_identity_lookback_includes_current_point(self):
        """Last element of the lookback window is y[idx], the annotation point.

        The window ends at the annotation point inclusive: y(t) is an observation
        available at forecast time, and the annotation is generated from y(t) and
        dy(t). Excluding it would hand the text a value X_ts does not have, and
        I(X_text; Y | X_ts) would then be inflated by that asymmetry rather than
        by anything predictive. See data.attach_lookback_future.
        """
        data = _make_data(self.t, self.y, [20, 30])
        data = _attach(data, horizon=1, lookback=8, tok_key="identity")
        for pt in data["points"]:
            if "y_lookback" not in pt:
                continue
            idx = _t_index(self.t, pt["t"])
            assert pt["y_lookback"][-1] == pytest.approx(self.y[idx])

    def test_identity_lookback_window_content(self):
        """Full lookback window should match y[idx+1-lookback : idx+1]."""
        lookback = 8
        data = _make_data(self.t, self.y, [30])
        data = _attach(data, horizon=1, lookback=lookback, tok_key="identity")
        for pt in data["points"]:
            if "y_lookback" not in pt:
                continue
            idx = _t_index(self.t, pt["t"])
            expected = self.y[idx + 1 - lookback : idx + 1]
            np.testing.assert_allclose(pt["y_lookback"], expected)

    def test_left_padding_near_start(self):
        """Point at idx=2 with lookback=8 has 5 leading zeros.

        The window is y[idx+1-lookback : idx+1] clipped at zero, so idx=2 yields
        three real values (y[0..2]) left-padded to length 8.
        """
        t, y = _sine_dense(n=80)
        data = _make_data(t, y, [2])
        data = _attach(data, horizon=1, lookback=8, tok_key="identity")
        for pt in data["points"]:
            if "y_lookback" not in pt:
                continue
            assert np.all(pt["y_lookback"][:5] == 0.0), "first 5 values should be padding"
            np.testing.assert_allclose(pt["y_lookback"][-3:], y[0:3])

    def test_point_beyond_end_is_skipped(self):
        """A point with no room for y_future is silently dropped."""
        t, y = _sine_dense(n=20)
        data = _make_data(t, y, [len(t) - 1])   # no y[idx+1] exists
        data = _attach(data, horizon=1, lookback=4)
        assert "y_future" not in data["points"][0]

    def test_patch_mean_values(self):
        """PatchMean should equal the mean of each patch window."""
        t, y = _sine_dense(n=80)
        lookback, patch_len, stride = 8, 4, 2
        data = _make_data(t, y, [30])
        data = _attach(data, horizon=1, lookback=lookback,
                       tok_key="patch_mean", patch_len=patch_len, stride=stride)
        for pt in data["points"]:
            if "y_lookback" not in pt:
                continue
            idx = _t_index(t, pt["t"])
            window = y[idx + 1 - lookback : idx + 1]   # inclusive of y(t), as above
            n_patches = (lookback - patch_len) // stride + 1
            expected = np.array([
                window[i * stride : i * stride + patch_len].mean()
                for i in range(n_patches)
            ])
            np.testing.assert_allclose(pt["y_lookback"], expected, rtol=1e-5)


# data.py — build_category_data_dummy_embedding

def _make_multi_category_data(t, y, points):
    """points: list of (t_val, category) pairs, one text sample each."""
    return {
        "metadata": {"t": t.tolist(), "y": y.tolist()},
        "points": [
            {"t": round(float(t_val), 4), "category": cat,
             "samples": [{"text": "test"}]}
            for t_val, cat in points
        ],
    }


class TestDummyEmbedding:
    """_attach_dummy_embeddings / build_category_data_dummy_embedding."""

    def setup_method(self):
        self.t, self.y = _sine_dense(n=80)
        self.tok = TS_TOKENIZERS["identity"](
            patch_len=4, stride=2, lookback_steps=8
        ).build()

    def _build(self, points, seed=0):
        data = _make_multi_category_data(self.t, self.y, points)
        return build_category_data_dummy_embedding(
            data, self.tok, horizon_steps=1, lookback_steps=8, seed=seed
        )

    def test_correct_embeds_true_future_value(self):
        idx = 30
        cat_data = self._build([(self.t[idx], "correct")])
        expected = self.y[idx + 1]
        np.testing.assert_allclose(cat_data["correct"]["x"], [[expected]])

    def test_irrelevant_is_always_zero(self):
        cat_data = self._build([
            (self.t[20], "irrelevant"), (self.t[40], "irrelevant"),
        ])
        np.testing.assert_array_equal(cat_data["irrelevant"]["x"], [[0.0], [0.0]])

    def test_incorrect_is_valid_but_not_true_value(self):
        valid = {-1.0, -0.5, 0.0, 1.0}
        for idx in [10, 20, 30, 40, 50, 60]:
            cat_data = self._build([(self.t[idx], "incorrect")], seed=idx)
            true_value = self.y[idx + 1]
            value = cat_data["incorrect"]["x"][0, 0]
            assert value in valid
            assert not np.isclose(value, true_value)

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            self._build([(self.t[30], "bogus")])

    def test_row_counts_match_across_categories(self):
        idx = 30
        cat_data = self._build([
            (self.t[idx], "correct"),
            (self.t[idx], "incorrect"),
            (self.t[idx], "irrelevant"),
        ])
        for cat in ("correct", "incorrect", "irrelevant"):
            assert cat_data[cat]["x"].shape == (1, 1)
            assert cat_data[cat]["ts"].shape[0] == 1
            assert cat_data[cat]["y"].shape[0] == 1


# metrics.py — _pool_by_annotation_point

class TestPool:

    def test_output_shapes(self):
        N, n_samples, embed_dim, n_patches, H = 10, 3, 16, 4, 1
        rng = np.random.default_rng(0)
        x  = rng.standard_normal((N * n_samples, embed_dim))
        ts = np.tile(rng.standard_normal((N, n_patches)), (n_samples, 1)).reshape(N * n_samples, n_patches)
        y  = np.tile(rng.standard_normal((N, H)),         (n_samples, 1)).reshape(N * n_samples, H)
        x_p, ts_p, y_p = _pool_by_annotation_point(x, ts, y, n_samples)
        assert x_p.shape  == (N, embed_dim)
        assert ts_p.shape == (N, n_patches)
        assert y_p.shape  == (N, H)

    def test_x_is_mean_pooled(self):
        N, n_samples, embed_dim = 5, 4, 8
        rng = np.random.default_rng(1)
        x  = rng.standard_normal((N * n_samples, embed_dim))
        ts = np.ones((N * n_samples, 2))
        y  = np.ones((N * n_samples, 1))
        x_p, _, _ = _pool_by_annotation_point(x, ts, y, n_samples)
        for i in range(N):
            expected = x[i * n_samples : (i + 1) * n_samples].mean(axis=0)
            np.testing.assert_allclose(x_p[i], expected)

    def test_ts_and_y_take_first_of_block(self):
        N, n_samples = 5, 3
        rng = np.random.default_rng(2)
        ts = rng.standard_normal((N * n_samples, 2))
        y  = rng.standard_normal((N * n_samples, 1))
        x  = np.ones((N * n_samples, 4))
        _, ts_p, y_p = _pool_by_annotation_point(x, ts, y, n_samples)
        for i in range(N):
            np.testing.assert_array_equal(ts_p[i], ts[i * n_samples])
            np.testing.assert_array_equal(y_p[i],  y[i  * n_samples])

    def test_n_samples_1_is_identity(self):
        rng = np.random.default_rng(3)
        x  = rng.standard_normal((8, 6))
        ts = rng.standard_normal((8, 3))
        y  = rng.standard_normal((8, 1))
        x_p, ts_p, y_p = _pool_by_annotation_point(x, ts, y, 1)
        np.testing.assert_array_equal(x_p,  x)
        np.testing.assert_array_equal(ts_p, ts)
        np.testing.assert_array_equal(y_p,  y)

    def test_raises_on_misaligned_rows(self):
        x  = np.ones((7, 4))
        ts = np.ones((7, 2))
        y  = np.ones((7, 1))
        with pytest.raises(ValueError, match="not evenly divisible"):
            _pool_by_annotation_point(x, ts, y, 3)


# metrics.py — ksg_mi

class TestKsgMI:

    def test_always_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            X = rng.standard_normal((60, 1))
            Y = rng.standard_normal((60, 1))
            assert ksg_mi(X, Y) >= 0.0

    def test_independent_gaussians_near_zero(self):
        """MI of independent variables should be very small (clipped to 0)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 1))
        Y = rng.standard_normal((300, 1))
        assert ksg_mi(X, Y, k=5) < 0.05

    def test_identical_inputs_is_high(self):
        """MI(X, X) should be well above zero — roughly H(X) for a Gaussian."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((150, 1))
        assert ksg_mi(X, X, k=3) > 0.5

    def test_perfect_linear_relationship_is_high(self):
        """MI(X, aX+b) should be high regardless of scale/offset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((150, 1))
        # KSG with Chebyshev is scale-sensitive so MI(X, 2X) != MI(X, X)
        # exactly, but a perfect linear relationship should give high MI.
        assert ksg_mi(X, 2 * X + 1, k=3) > 0.5

    def test_higher_correlation_higher_mi(self):
        """Stronger linear relationship → higher MI estimate."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 1))
        Y_weak   = X + rng.standard_normal((200, 1)) * 5.0
        Y_strong = X + rng.standard_normal((200, 1)) * 0.1
        assert ksg_mi(X, Y_strong, k=5) > ksg_mi(X, Y_weak, k=5)

    def test_accepts_1d_arrays(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal(80)
        Y = rng.standard_normal(80)
        result = ksg_mi(X, Y)
        assert isinstance(result, float)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            ksg_mi(np.ones((10, 1)), np.ones((11, 1)))

    def test_multidimensional_inputs(self):
        """Should work with (N, d) arrays for d > 1."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 3))
        Y = rng.standard_normal((100, 2))
        result = ksg_mi(X, Y, k=3)
        assert isinstance(result, float)
        assert result >= 0.0


# metrics.py — PIDMetric (skipped if PID repo not present)

def _pid_available():
    import pathlib
    p = pathlib.Path(__file__).parent / "PID" / "synthetic" / "rus.py"
    return p.exists()


@pytest.mark.skipif(not _pid_available(), reason="PID repo not installed")
class TestPIDMetric:
    """PIDMetric decomposes I(Y; X_ts, X_text) into four non-negative atoms.

    The public entry point is compute(category_data), which takes the same
    {category: {'ts', 'x', 'y'}} structure the rest of the pipeline uses.
    """

    def _category_data(self, N=120, seed=0):
        """One category with an informative X_ts and a pure-noise X_text."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 4 * np.pi, N)
        Y = np.sin(t)
        return {
            "correct": {
                "ts": Y.reshape(-1, 1) + rng.standard_normal((N, 1)) * 0.05,
                "x":  rng.standard_normal((N, 4)),     # text carries nothing
                "y":  Y.reshape(-1, 1),
                "n_samples": np.ones(N, dtype=int),
            }
        }

    def _atoms(self, **kwargs):
        from src.mmtt_bench.metrics import PIDMetric
        res = PIDMetric(**kwargs).compute(self._category_data(), pca_dim=2,
                                          strategy="quantile", n_bootstrap=1)
        return {atom: res[atom]["correct"] for atom in
                ("redundancy", "unique_ts", "unique_text", "synergy")}

    def test_output_keys(self):
        assert set(self._atoms()) == {"redundancy", "unique_ts", "unique_text", "synergy"}

    def test_values_in_valid_range(self):
        for atom, v in self._atoms().items():
            v = np.mean(v)
            assert v >= -1e-6, f"PID atom {atom} is negative: {v}"

    def test_informative_ts_has_high_unique_ts(self):
        """X_ts tracks Y and X_text is noise, so unique_ts should dominate."""
        atoms = self._atoms()
        assert np.mean(atoms["unique_ts"]) > np.mean(atoms["unique_text"])

    def test_discretise_y_bin_count(self):
        """The default 'sign' strategy labels by sign, not into n_bins_y bins."""
        from src.mmtt_bench.metrics import PIDMetric
        pid = PIDMetric(n_bins_y=3)
        labels = pid._discretise_y(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
        assert len(labels) == 5
        assert set(labels).issubset({0, 1, 2})

    def test_discretise_y_quantile_spreads_bins(self):
        """'quantile' is the strategy that honours n_bins_y; uniform Y fills all."""
        from src.mmtt_bench.metrics import PIDMetric
        pid = PIDMetric(n_bins_y=4)
        labels = pid._discretise_y(np.linspace(-1, 1, 200), strategy="quantile")
        assert len(set(labels)) == 4

    def test_discretise_y_sign_gives_two_labels(self):
        """Sign of a symmetric ramp takes two values, whatever n_bins_y says."""
        from src.mmtt_bench.metrics import PIDMetric
        pid = PIDMetric(n_bins_y=4)
        labels = pid._discretise_y(np.linspace(-1, 1, 200))
        assert len(set(labels)) == 2


# metrics.py — VInformationMetric._r2

class TestVInformationR2:
    """Unit tests for the _r2 helper — verifies R² on simple known models."""

    def _metric(self):
        from src.mmtt_bench.metrics import VInformationMetric
        return VInformationMetric(n_bootstrap=1, seed=0, cv=3)

    def test_perfect_linear_gives_high_r2(self):
        """R² should be ≈1 for Y = aX + b (exact linear relationship)."""
        rng = np.random.default_rng(0)
        N = 300
        X = rng.standard_normal((N, 1))
        Y = 3.0 * X.ravel() + 1.0
        r2 = self._metric()._r2(X, Y, rng, np.arange(N))
        assert r2 > 0.95, f"Expected R² > 0.95 for perfect linear, got {r2:.3f}"

    def test_independent_gaussian_gives_low_r2(self):
        """R² should be ≈0 when X and Y are independent standard normals."""
        rng = np.random.default_rng(42)
        N = 400
        X = rng.standard_normal((N, 1))
        Y = rng.standard_normal(N)
        r2 = self._metric()._r2(X, Y, rng, np.arange(N))
        assert r2 < 0.05, f"Expected R² < 0.05 for independent Gaussians, got {r2:.3f}"

    def test_gaussian_shaped_nonlinear(self):
        """Ridge (linear) cannot fit Y = exp(-X²/2) well; R² should be well below 1.
        Cross-val R² can be slightly negative for near-zero fits, so lower bound is -0.1."""
        rng = np.random.default_rng(7)
        N = 400
        X = rng.standard_normal((N, 1))
        Y = np.exp(-0.5 * X.ravel() ** 2)
        r2 = self._metric()._r2(X, Y, rng, np.arange(N))
        assert -0.1 <= r2 < 0.85, f"Expected near-zero R² for Gaussian nonlinearity, got {r2:.3f}"

    def test_constant_y_returns_zero(self):
        """Degenerate constant Y → 0.0 without error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        Y = np.ones(60)
        assert self._metric()._r2(X, Y, rng, np.arange(60)) == 0.0

    def test_bootstrap_subset_indices(self):
        """_r2 with bootstrap indices (with replacement) still returns a float in [−ε, 1]."""
        rng = np.random.default_rng(1)
        N = 200
        X = rng.standard_normal((N, 2))
        Y = X[:, 0] * 2 + rng.standard_normal(N) * 0.1
        idx = rng.choice(N, size=N, replace=True)
        r2 = self._metric()._r2(X, Y, rng, idx)
        assert isinstance(r2, float)
        assert r2 > -0.05


# metrics.py — VInformationMetric.compute

def _make_vinf_category_data(X_ts, X_text, Y, n_samples=1):
    """Pack arrays into the category_data format expected by VInformationMetric.compute."""
    N = len(Y)
    return {
        "cat": {
            "x":        X_text,
            "ts":       X_ts,
            "y":        Y.reshape(-1, 1),
            "n_samples": [n_samples] * N,
        }
    }


class TestVInformationMetric:
    """Integration tests for VInformationMetric.compute()."""

    _EXPECTED_SCALAR_KEYS = {
        "v_text_mean", "v_ts_mean", "v_joint_mean", "v_conditional_mean",
        "v_text_std",  "v_ts_std",  "v_joint_std",  "v_conditional_std",
        "explained_variance", "d_conservative", "d_liberal",
    }
    _EXPECTED_LIST_KEYS = {"v_text", "v_ts", "v_joint", "v_conditional"}

    def _metric(self, n_bootstrap=5):
        from src.mmtt_bench.metrics import VInformationMetric
        return VInformationMetric(n_bootstrap=n_bootstrap, seed=0, cv=3)

    def _linear_data(self, N=200, seed=0):
        """Y = ts_feature + small noise; text is pure noise."""
        rng = np.random.default_rng(seed)
        X_ts   = rng.standard_normal((N, 2))
        Y      = X_ts[:, 0] * 2.0 + rng.standard_normal(N) * 0.1
        X_text = rng.standard_normal((N, 4))
        return X_ts, X_text, Y

    def _gaussian_noise_data(self, N=200, seed=1):
        """X_ts, X_text, Y all independent standard normals."""
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((N, 2)),
            rng.standard_normal((N, 4)),
            rng.standard_normal(N),
        )

    # ── Structure tests ───────────────────────────────────────────────────

    def test_output_keys_present(self):
        X_ts, X_text, Y = self._linear_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        for key in self._EXPECTED_SCALAR_KEYS | self._EXPECTED_LIST_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_bootstrap_list_lengths(self):
        n_bootstrap = 4
        X_ts, X_text, Y = self._linear_data()
        result = self._metric(n_bootstrap).compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        for key in self._EXPECTED_LIST_KEYS:
            assert len(result[key]["cat"]) == n_bootstrap, (
                f"{key} should have {n_bootstrap} bootstrap samples"
            )

    def test_v_conditional_is_r2_joint_minus_r2_ts(self):
        """v_conditional = r2_joint − r2_ts; can be negative when text adds noise."""
        X_ts, X_text, Y = self._linear_data()
        result = self._metric(n_bootstrap=10).compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        joints = result["v_joint"]["cat"]
        tss    = result["v_ts"]["cat"]
        conds  = result["v_conditional"]["cat"]
        for r2j, r2t, vc in zip(joints, tss, conds):
            assert vc == pytest.approx(r2j - r2t, abs=1e-9)

    def test_mean_std_consistent(self):
        """*_mean and *_std should match numpy computations on the sample lists."""
        X_ts, X_text, Y = self._linear_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        for base in ["v_text", "v_ts", "v_joint", "v_conditional"]:
            samples = result[base]["cat"]
            assert result[f"{base}_mean"]["cat"] == pytest.approx(np.mean(samples), abs=1e-9)
            assert result[f"{base}_std"]["cat"]  == pytest.approx(np.std(samples),  abs=1e-9)

    # ── Correctness: linear model ─────────────────────────────────────────

    def test_linear_ts_signal_gives_high_v_ts(self):
        """When Y = f(X_ts), v_ts_mean should be high (R² close to 1)."""
        X_ts, X_text, Y = self._linear_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        assert result["v_ts_mean"]["cat"] > 0.8, (
            f"Expected v_ts_mean > 0.8 for linear ts signal, "
            f"got {result['v_ts_mean']['cat']:.3f}"
        )

    def test_linear_ts_signal_text_noise_low_v_text(self):
        """When text is pure noise, v_text_mean should be low."""
        X_ts, X_text, Y = self._linear_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        assert result["v_text_mean"]["cat"] < 0.3, (
            f"Expected v_text_mean < 0.3 for noise text, "
            f"got {result['v_text_mean']['cat']:.3f}"
        )

    # ── Correctness: Gaussian (null) model ───────────────────────────────

    def test_gaussian_null_v_ts_near_zero(self):
        """Independent Gaussian data → v_ts_mean should be near zero."""
        X_ts, X_text, Y = self._gaussian_noise_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        assert result["v_ts_mean"]["cat"] < 0.15, (
            f"Expected v_ts_mean < 0.15 for null Gaussian model, "
            f"got {result['v_ts_mean']['cat']:.3f}"
        )

    def test_gaussian_null_v_text_near_zero(self):
        """Independent Gaussian data → v_text_mean should be near zero."""
        X_ts, X_text, Y = self._gaussian_noise_data()
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        assert result["v_text_mean"]["cat"] < 0.15, (
            f"Expected v_text_mean < 0.15 for null Gaussian model, "
            f"got {result['v_text_mean']['cat']:.3f}"
        )

    # ── Correctness: text adds unique information ─────────────────────────

    def test_informative_text_raises_v_conditional(self):
        """When text carries unique signal not in ts, v_conditional_mean > 0."""
        rng = np.random.default_rng(3)
        N = 300
        X_ts   = rng.standard_normal((N, 2))
        X_text = rng.standard_normal((N, 4))
        # Y depends on both ts and text
        Y = X_ts[:, 0] + X_text[:, 0] * 2.0 + rng.standard_normal(N) * 0.1
        result = self._metric().compute(
            _make_vinf_category_data(X_ts, X_text, Y), pca_dim=2
        )
        assert result["v_conditional_mean"]["cat"] > 0.1, (
            f"Expected v_conditional_mean > 0.1 when text is informative, "
            f"got {result['v_conditional_mean']['cat']:.3f}"
        )
