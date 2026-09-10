# MMTT-Bench: Multiodal Text Time Series Benchmark

This repository is the official implementation of [When Does Text Inform? Benchmarking Information-Theoretic Metrics for Multimodal Time-Series Forecasting](Under Review).

A synthetic benchmark for measuring whether natural language annotations carry information about future time series values, and whether regression models can exploit that information. The benchmark combines controlled signal generation with ground-truth annotation categories to give reproducible measurements via information theory and supervised regression.

Each annotation point is assigned one of three base categories:

| Category | Description |
|---|---|
| **Correct** | Accurately describes the upcoming signal direction and magnitude |
| **Incorrect** | Deliberately contradicts the true signal direction |
| **Irrelevant** | Factually true but carries no directional information |

The benchmark measures two complementary things:

1. **Information content** — how much each annotation category contains about the future signal, estimated via KSG mutual information, PID, V-information, MINE, InfoNCE, and CCA.
2. **Regression benefit** — how much a forecasting model improves when given the annotation as an additional input, measured as the performance delta between a joint (time series + text) model and a time-series-only baseline.

A correctly calibrated result shows *correct* scoring highest on both axes, *irrelevant* near zero on the information axis, and *incorrect* at or below zero on the performance delta axis.

---

## Installation

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13.

```bash
# 1. Install all pinned dependencies from the lock file
uv sync

# 2. Install PyTorch into the uv-managed environment
uv pip install torch torchvision 
```
For a different CUDA version or CPU-only, see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right `--index-url`. The `uv.lock` file pins every other dependency exactly, so `uv sync` always produces an identical environment.

Optional dependencies for neural MI estimators:

```bash
# MINE
git clone https://github.com/gtegner/mine-pytorch mine-pytorch

# PID
pip install cvxpy
git clone https://github.com/pliang279/PID PID
```

---

## Reproducing the paper

Three scripts run every experiment, then a fourth turns the saved results into the tables and figures.

```bash
./reproduce/01_mmtt_bench.sh                        # synthetic — no downloads needed
MM_TSF_ROOT=../MM-TSFlib/data ./reproduce/02_timemmd.sh
FINTEXTS_ROOT=../FinTexTS      ./reproduce/03_fintexts.sh

python reproduce/make_artifacts.py                  # tables from saved results
python plot.py --figure1 -o results/figures         # figures
```

`reproduce/make_artifacts.py --list` prints which artifact comes from which script and which result file.

### 1. MMTT-Bench (synthetic)

```bash
./reproduce/01_mmtt_bench.sh
```

Self-contained: `datasets/benchmark_splits/` is in the repository, and the
script derives everything else from it — the nine mixture corpora, all three
coupling axes (corpus-level ρ, per-annotation content noise, temporal jitter),
and the four noisy-signal variants. It then computes MI, runs the four
representation sweeps, the shuffle control, and trains the downstream models.

Produces Tables 4–12 and Figures 2, 3, 4 and 6.

| variable | effect |
|---|---|
| `QUICK=1` | skip MINE, InfoNCE and the transformers — minutes instead of hours |
| `SKIP_DATA=1` | reuse the corpora already in `datasets/` |
| `PY=...` | interpreter to use (default `python`) |

MINE dominates the runtime: roughly 53 s per MI estimate on CPU at N = 1279,
20 estimates per result. Use `QUICK=1` for a smoke test.

### 2. Time-MMD

```bash
git clone https://github.com/AdityaLab/MM-TSFlib ../MM-TSFlib
MM_TSF_ROOT=../MM-TSFlib/data ./reproduce/02_timemmd.sh
```

The Time-MMD CSVs are not redistributed here. Six domains are used; Economy and
Security are excluded, as they have too few points.

Time-MMD mixes sampling frequencies and the CFA window settings differ by
frequency, so MI runs twice: `final_configs/timemmd.yaml` covers the monthly
domains (seq_len 8, pred_len 6) and `timemmd_weekly.yaml` the weekly ones
(seq_len 36, pred_len 12).

### 3. FinTexTS

```bash
FINTEXTS_ROOT=/path/to/FinTexTS ./reproduce/03_fintexts.sh
```

Ten tickers, matching the paper: AMD, BA, COST, DIS, GOOGL, INTC, NFLX, NVDA,
T, TSLA. Override with `TICKERS="AMD BA"`. Each ticker is treated as an
independent dataset and the estimates are pooled at the end, because the
tickers share a calendar but not a data-generating process.

FinTexTS carries 21 per-field texts per annotation point. `train_transformer.py`
embeds each separately and mean-pools, matching FinTexTS Sec 4.1 rather than
concatenating and embedding once, and uses ModernBERT for its 8192-token
context — BERT would truncate the filing fields at 512.

Together, 2 and 3 produce Tables 13, 17 and 18, and Figure 5.

---

## Tables and figures

### Tables

```bash
python reproduce/make_artifacts.py                    # everything available
python reproduce/make_artifacts.py --which mmtt_bench # one dataset family
python reproduce/make_artifacts.py --list             # the artifact manifest
```

Reads only saved results, so it runs in seconds and can be re-run freely.
LaTeX goes to `paper_artifacts/tables/`. Missing inputs are reported and skipped
rather than raising, so a partial run still produces its share.

### Figures

Every figure has one command. All of them write into `--output-dir`.

```bash
python plot.py --figure1 -o results/figures    # signal with example annotations
python plot.py --figure2 -o results/figures    # ordering grid + encoder sensitivity
python plot.py --figure5 -o results/figures    # real-world MI grid

python plot.py --metrics results/mmtt_bench/metrics.json \
               --models  results/mmtt_bench/sklearn.json \
               -o results/figures --perf-metric mse    # Figures 3 and 4
```

`--figure2` takes the sweep JSONs explicitly, or finds them in
`results/mmtt_bench/` when given none. `--figure5` accepts `--fig5-quantity
conditional` to plot `I(X_text; Y | X_ts)` against a rule at zero instead of the
joint estimate against the series-only value.

Drawing lives in `src/mmtt_bench/viz.py`, loading in `plot.py`. Retired
plotting code is kept in `to-delete/` with a note on what replaced it.

## Running a single experiment

The reproduce scripts are thin wrappers over two entry points, both of which
take a config from `final_configs/`.

```bash
# mutual information
python run.py final_configs/sine.yaml -o results/mmtt_bench/metrics.json

# downstream models
python train.py --config final_configs/sine.yaml -o results/mmtt_bench/sklearn.json
python train_transformer.py --config final_configs/sine_transformers.yaml \
                            -o results/mmtt_bench/transformers
```

A config with a `datasets:` list runs once per dataset and writes one JSON per
dataset into the output directory. A config with a `sweep:` block varies one
parameter — or several coupled parameters, as `sweeps/patch.yaml` does for
patch length, stride and lookback — and writes all points to one file.

### Regenerating the synthetic corpora

```bash
python src/mmtt_bench/data_generation/sine_generator.py \
    --signal-type noisy_sine_with_constants --noise-std 0.1 --output-dir datasets/my_split
python src/mmtt_bench/data_generation/mixture.py  --source datasets/benchmark_splits
python src/mmtt_bench/data_generation/coupling.py --source datasets/benchmark_splits --mode jitter
```

`coupling.py` takes `--mode rho|noise|jitter`. All three write a new split
directory alongside the source, adding one category per level, which `run.py`
and `train.py` pick up automatically.

---


## Configuration reference

### Top-level fields

| Field | Default | Description |
|---|---|---|
| `data_path` | — | Path to the train split JSON |
| `embedding_type` | `sentence_transformer` | `sentence_transformer` or `huggingface` |
| `model_name` | `all-distilroberta-v1` | HuggingFace sentence-transformer model identifier |
| `ts_tokenizer` | `identity` | `identity`, `patch`, or `patch_mean` |
| `lookback_steps` | `2` | Past dense time steps included in each lookback window |
| `horizon_steps` | `1` | Steps ahead to predict (Y target length) |
| `patch_len` | `1` | Patch length for `patch` / `patch_mean` tokenizers |
| `stride` | `1` | Stride between successive patches |
| `shuffle` | `false` | Randomises text–signal alignment as a null baseline |
| `shuffle_seed` | `42` | RNG seed for shuffle |

### `compute_kwargs` — passed to every metric's `.compute()` call

| Field | Default | Description |
|---|---|---|
| `pca_dim` | `16` | PCA dimensions kept from text embeddings before metric computation |
| `strategy` | `equal-width` | Y-discretisation strategy for PID: `sign`, `quantile`, `equal-width`, `discrete`, `energy` |
| `y_channel` | `0` | Column of Y used for PID discretisation (use the horizon step the text describes, e.g. `11` for a 12-step horizon) |
| `n_bootstrap` | `1` | Bootstrap resamples passed to metrics that use it (does not apply to KSG — use `n_subsample_reps` there) |

### `metrics` — per-metric `__init__` kwargs

**`mutual_information` (KSG)**

| Field | Default | Description |
|---|---|---|
| `n_neighbours` | `3` | KSG k-nearest-neighbour count |
| `n_subsample_reps` | `20` | Draws per subsample size for Holmes & Nemenman variance estimation |

**`mine`**

| Field | Default | Description |
|---|---|---|
| `mine_iters` | `500` | Gradient steps per MI estimate |
| `mine_batch_size` | `256` | Mini-batch size |
| `lr` | `0.0001` | Adam learning rate |
| `n_bootstrap` | `1` | Bootstrap resamples (1 = full data, no resampling) |

**`infonce`**

| Field | Default | Description |
|---|---|---|
| `n_iters` | `500` | Gradient steps per MI estimate |
| `batch_size` | `64` | Mini-batch size (sets ceiling MI ≈ log(`batch_size`) nats) |
| `lr` | `0.001` | Adam learning rate |
| `hidden_dim` | `128` | Hidden layer width of the critic network |
| `n_bootstrap` | `1` | Bootstrap resamples |

**`cca`**

| Field | Default | Description |
|---|---|---|
| `n_bootstrap` | `1` | Bootstrap resamples |
| `eps` | `1e-5` | Tikhonov regularisation added to covariance matrices |

**`v_information`**

| Field | Default | Description |
|---|---|---|
| `cv` | `5` | Cross-validation folds for RidgeCV |
| `n_bootstrap` | `1` | Bootstrap resamples |

**`pid`**

| Field | Default | Description |
|---|---|---|
| `n_clusters_ts` | `4` | KMeans clusters for ts discretisation |
| `n_clusters_text` | `5` | KMeans clusters for text discretisation |
| `n_bins_y` | `4` | Y bins (strategy set via `compute_kwargs.strategy`) |
| `n_bootstrap` | `1` | Bootstrap resamples with fixed KMeans codebook |

### `sweep` block

| Field | Description |
|---|---|
| `param` | Parameter to vary — any top-level key, any `compute_kwargs` key, or any key inside a metric's block |
| `values` | List of values to sweep over |

---

## Dataset format

Each split file is a JSON with the following structure:

```json
{
  "metadata": {
    "t": [0.0, 0.785, 1.571, "..."],
    "y": [0.0, 0.707, 1.0,   "..."],
    "n_samples": 1
  },
  "points": [
    {
      "t":        0.785,
      "y":        0.707,
      "phase":    "ascending_zero",
      "category": "correct",
      "samples":  [{"text": "The signal is rising steeply towards its peak."}]
    }
  ]
}
```

`metadata.t` and `metadata.y` are the dense signal arrays used to reconstruct the lookback window and future target for each annotation point. `points` lists one entry per annotation, with `phase` recording the local signal behaviour at that time step.

---

## Extending the benchmark

### New signal types

Add a generator to `src/mmtt-bench/data_generation/` following the pattern in `sine_generator.py`. Register the signal function and its analytical derivative in the `SIGNAL_FUNCTIONS` and `SIGNAL_DERIVATIVES` dicts, then add a corresponding config to `final_configs/`.

### New text embedding models

Change `model_name` in any config to any HuggingFace sentence-transformer identifier. To sweep over models:

```yaml
sweep:
  param: model_name
  values:
    - sentence-transformers/all-MiniLM-L6-v2
    - sentence-transformers/all-distilroberta-v1
    - sentence-transformers/average_word_embeddings_glove.840B.300d
```

### New metrics

Add a class to `src/mmtt-bench/metrics.py` implementing `.compute(category_data, pca_dim, rng, pca_text, **kwargs) -> dict`, register it in the `METRICS` dict, and include it in the `metrics:` block of any config. The result dict must follow the same key convention as `MutualInformationMetric` — see the module docstring for the expected structure.

---
<!-- 
## Citation

```bibtex
@article{,
  title   = {Can LLMs Use Text to Improve Time Series Forecasting?},
  author  = {},
  journal = {},
  year    = {2025},
}
``` -->

---

## License

MIT
