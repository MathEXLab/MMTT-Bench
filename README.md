# Multimodal Time Series Benchmark

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

## Repository structure

```
run.py                              # Compute information-theoretic metrics → JSON
train.py                            # Train and evaluate regression models → JSON
plot.py                             # Visualise metrics and compare metrics vs models
final_configs/                      # YAML configs for each experiment (best parameters)
src/
  benchmark/
    data.py                         # Data loading and preprocessing pipeline
    metrics.py                      # Metric implementations (KSG, PID, V-info, …)
    tokenizers.py                   # Text and time-series tokenizers
    viz.py                          # Shared plotting functions
    data_generation/
      sine_generator.py             # Generate synthetic sine-wave benchmark splits
      mixture.py                    # Augment splits with mixture annotation categories
      timemmd.py                    # Convert TimeMMD CSV dataset to benchmark format
datasets/benchmark_splits*/                  # Generated dataset splits (train / val / test JSON)
results/                            # Experiment outputs (JSON, CSV, plots)
```

---

## Installation

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13.

```bash
# 1. Install all pinned dependencies from the lock file
uv sync

# 2. Install PyTorch into the uv-managed environment (requires CUDA 12.8)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For a different CUDA version or CPU-only, see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right `--index-url`. The `uv.lock` file pins every other dependency exactly, so `uv sync` always produces an identical environment.

Optional dependencies for neural MI estimators:

```bash
# MINE
git clone https://github.com/MasanoriYamada/mine-pytorch mine-pytorch

# PID
pip install cvxpy
git clone https://github.com/pliang279/PID PID
```

---

## Experiment workflow

The full experimental pipeline proceeds in six steps. Steps 1–3 establish the base result on the synthetic dataset; steps 4–5 test robustness; step 6 applies the benchmark to real-world data.

---

### Step 1 — Generate the synthetic dataset

All data generation scripts must be run from the `data_generation/` directory because they import local template modules.

```bash
cd src/benchmark/data_generation
python sine_generator.py --output-dir ../../../benchmark_splits
```

This produces three JSON splits (`train.json`, `val.json`, `test.json`) inside `benchmark_splits/`. Each split records the dense signal arrays and the annotation points with their ground-truth category labels and text samples.

Key parameters (edit directly in `sine_generator.py`):

| Parameter | Default | Description |
|---|---|---|
| `signal_type` | `sine_with_constants` | `sine` or `sine_with_constants` |
| `constant` | `-0.5` | Value substituted during constant segments |
| `transition_mask` | `0.3` | Fraction of t-values replaced by the constant |
| `n_signals` | `512` | Number of sinusoidal cycles to generate |
| `annotation_step` | `period / 4` | Time between annotation points |
| `annotations_per_point` | `1` | Text samples generated per annotation point |

Train / val / test splits follow a cycle-based scheme: the first 62.5% of cycles for training, the next 18.75% for validation, and the remaining 18.75% for test.

---

### Step 2 — Compute metrics and train regression models

Use `final_configs/sine.yaml` for the best-validated parameters. To inspect or adjust them, see the configuration reference at the end of this document.

**Information-theoretic metrics:**

```bash
python run.py final_configs/sine.yaml --output results/final/sine_metrics.json
```

**Regression models:**

```bash
python train.py \
    --config  final_configs/sine.yaml \
    --output  results/final/sine_model.json
```

`run.py` writes a JSON with type `"single"` containing all bootstrap samples so plots can be regenerated without re-running. `train.py` writes both a structured JSON (for `plot.py`) and a `results.csv` summary table.

Both scripts accept the same YAML config. All fields can be overridden on the command line; see `python run.py --help` and `python train.py --help`.

---

### Step 3 — Visualise and compare results

```bash
python plot.py \
    --metrics results/final/sinezero_metrics.json \
    --models  results/final/sinezero_model.json \
    --output-dir results/final/plots/
```

This produces all plots in one pass and prints Spearman and Kendall rank correlations to stdout. Saved outputs:

- **Scatter plots** — info metric value (x) vs performance delta (y), one point per category, one panel per metric
- **Correlation trend plots** — linear trend lines per model with shared y-axis across panels
- **Delta bar charts** — grouped bars showing the per-category improvement per model

Use `--perf-metric` to select which performance metric to plot (default: `delta_r2`):

```bash
python plot.py \
    --metrics results/final/sinezero_metrics.json \
    --models  results/final/sinezero_model.json \
    --perf-metric r2 \
    --output-dir results/final/plots/
```

Use `--model-filter` to restrict to specific regressors:

```bash
python plot.py \
    --metrics results/final/sinezero_metrics.json \
    --models  results/final/sinezero_model.json \
    --model-filter ridge mlp
```

---

### Step 4 — Mixture categories

Mixture categories linearly interpolate between `correct` and a contaminating category (`incorrect` or `irrelevant`), letting the benchmark test whether information metrics track annotation quality monotonically.

**Generate mixture splits:**

```bash
cd src/benchmark/data_generation
python mixture.py \
    --source     ../../../benchmark_splits_sinehalf \
    --output-dir ../../../benchmark_splits_sinehalf_mixtures
```

This adds six new categories to the split — `mixture_25_incorrect`, `mixture_50_incorrect`, `mixture_75_incorrect`, and the corresponding `irrelevant` variants — where the number indicates the contamination percentage. The original three categories are preserved unchanged.

Default contamination levels (25%, 50%, 75%) and types (`incorrect`, `irrelevant`) can be customised:

```bash
python mixture.py \
    --source     ../../../benchmark_splits_sinehalf \
    --output-dir ../../../benchmark_splits_sinehalf_mixtures \
    --levels 10 25 50 75 90 \
    --types  incorrect irrelevant \
    --seed   42
```

**Run metrics and models on the mixture dataset:**

```bash
python run.py final_configs/mixture.yaml --output results/final/mixture_metrics.json

python train.py \
    --config  final_configs/mixture.yaml \
    --output  results/final/mixture_model.json
```

**Plot:**

```bash
python plot.py results/final/mixture_metrics.json --output-dir results/final/plots/

python plot.py \
    --metrics results/final/mixture_metrics.json \
    --models  results/final/mixture_model.json \
    --output-dir results/final/plots/
```

---

### Step 5 — Shuffle baseline

Shuffling reassigns text embeddings to random time points within each category, breaking any text–signal alignment. All metrics should fall to approximately zero on shuffled data; non-zero values indicate a numerical artefact.

Add to any YAML config:

```yaml
shuffle:      true
shuffle_seed: 42
```

Or pass `--shuffle` on the command line:

```bash
python run.py final_configs/sine.yaml \
    --output results/final/sineshuffle_metrics.json \
    --shuffle

python train.py \
    --config  final_configs/sine.yaml \
    --output  results/final/sineshuffle_model.json \
    --shuffle
```

Shuffling is applied after tokenization and applies equally to all categories including mixture categories. Compare the shuffled result against the unshuffled result from Step 2 to confirm that positive MI estimates reflect genuine text–signal alignment.

---

### Step 6 — TimeMMD real-world dataset

TimeMMD provides real-world time series paired with news and search text across several domains. This step converts the raw CSVs into the benchmark split format and runs the same pipeline as Steps 2–3.

**Obtain the raw data** from the [TimeMMD repository](https://github.com/AdityaLab/MM-TSFlib) and place the CSVs in a local directory (e.g. `timemmd_dataset/`).

**Generate benchmark splits:**

```bash
python src/benchmark/data_generation/timemmd.py \
    --input-dir  timemmd_dataset/ \
    --output-dir timemmd_splits/
```

Each CSV (e.g. `agriculture.csv`) produces two JSON files: `agriculture_train.json` and `agriculture_val.json`. Datasets with fewer than 400 training rows are skipped with a warning. The `--seed` flag controls the RNG used to sample irrelevant text from other domains (default: 42).

**Run metrics and models:**

```bash
python run.py final_configs/timemmd.yaml \
    --output results/final/timemmd_metrics.json

python train.py \
    --config  final_configs/timemmd.yaml \
    --output  results/final/timemmd_model.json
```

**Plot:**

```bash
python plot.py results/final/timemmd_metrics.json \
    --output-dir results/final/plots/

python plot.py \
    --metrics results/final/timemmd_metrics.json \
    --models  results/final/timemmd_model.json \
    --output-dir results/final/plots/
```

---

## Parameter sweeps

Use a `sweep:` block in any config to vary one parameter across all metrics in a single run:

```yaml
data_path:      benchmark_splits_sinehalf/train.json
model_name:     sentence-transformers/all-distilroberta-v1
ts_tokenizer:   identity
lookback_steps: 2
patch_len:      1
stride:         1

compute_kwargs:
  n_bootstrap: 20

metrics:
  cca: {}
  v_information:
    n_bootstrap: 20

sweep:
  param:  pca_dim
  values: [2, 4, 8, 16, 32]
```

```bash
python run.py sweep_config.yaml --output results/pca_sweep.json
python plot.py results/pca_sweep.json --output-dir results/plots/
```

The `param` field can be any top-level config key, any key inside `compute_kwargs`, or any key inside a specific metric's kwargs block. Category data is only rebuilt when the swept parameter affects data loading.

---

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `data_path` | — | Path to the train split JSON |
| `embedding_type` | `sentence_transformer` | `sentence_transformer` or `huggingface` |
| `model_name` | `all-distilroberta-v1` | HuggingFace model identifier |
| `ts_tokenizer` | `identity` | `identity`, `patch`, or `patch_mean` |
| `lookback_steps` | `2` | Past time steps in each lookback window |
| `horizon_steps` | `1` | Steps ahead to predict |
| `patch_len` | `1` | Patch length (for `patch` / `patch_mean` tokenizers) |
| `stride` | `1` | Stride between patches |
| `compute_kwargs.pca_dim` | `16` | PCA dimensions applied to text embeddings before metric computation |
| `compute_kwargs.n_bootstrap` | `20` | Bootstrap resampling iterations |
| `shuffle` | `false` | If true, randomises text–signal alignment as a null baseline |
| `shuffle_seed` | `42` | RNG seed for shuffling |

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

Add a generator to `src/benchmark/data_generation/` following the pattern in `sine_generator.py`. Register the signal function and its analytical derivative in the `SIGNAL_FUNCTIONS` and `SIGNAL_DERIVATIVES` dicts, then add a corresponding config to `final_configs/`.

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

Add a class to `src/benchmark/metrics.py` implementing `.compute(category_data, pca_dim, rng, pca_text, **kwargs) -> dict`, register it in the `METRICS` dict, and include it in the `metrics:` block of any config. The result dict must follow the same key convention as `MutualInformationMetric` — see the module docstring for the expected structure.

---

## Citation

```bibtex
@article{,
  title   = {Can LLMs Use Text to Improve Time Series Forecasting?},
  author  = {},
  journal = {},
  year    = {2025},
}
```

---

## License

MIT
