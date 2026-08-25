#!/usr/bin/env bash
# 01_mmtt_bench.sh — every MMTT-Bench result in the paper, from a clean checkout.
#
# Needs no downloads: datasets/benchmark_splits is in the repository, and every
# other synthetic corpus (mixture, coupling, noisy signals) is derived from it
# here. Reproduces:
#
#   Table 4   conditional MI per text encoder          (sweeps/embedding.yaml)
#   Table 5   conditional MI per PCA dimension         (sweeps/pca.yaml)
#   Table 6   conditional MI per lookback length       (sweeps/lookback.yaml)
#   Table 7   conditional MI per PatchTST tokenisation (sweeps/patch.yaml)
#   Table 8   conditional MI, final parameters         (sine.yaml)
#   Table 9   downstream MSE, 5 sklearn + 9 transformers x 10 fusions
#   Table 10  shuffle control                          (sineshuffle.yaml)
#   Table 11  observation-noise robustness             (sine_noise.yaml)
#   Table 12  temporal jitter                          (coupling, jitter mode)
#   Figure 2  parameter sweeps        (from Tables 4-7)
#   Figure 3  MI vs downstream MSE    (from Table 8 + Table 9)
#   Figure 4  mixture corpora         (mixture.yaml + mixture models)
#   Figure 6  embedding visualisation
#
# Usage:
#   ./reproduce/01_mmtt_bench.sh            # everything (hours: MINE dominates)
#   QUICK=1 ./reproduce/01_mmtt_bench.sh    # skip neural estimators + transformers
#   SKIP_DATA=1 ./reproduce/01_mmtt_bench.sh   # reuse existing datasets/
set -euo pipefail
cd "$(dirname "$0")/.."

# OpenBLAS spawns more threads than it was built for on this many cores and
# segfaults inside numpy; cap it before anything imports numpy.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-python}"
OUT=results/mmtt_bench
GEN=src/mmtt_bench/data_generation
mkdir -p "$OUT"

say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

# ── Datasets ────────────────────────────────────────────────────────────────
if [[ -z "${SKIP_DATA:-}" ]]; then
  say "Deriving synthetic corpora from datasets/benchmark_splits"
  # Figure 4: nine corpora, correct text replaced at 25/50/75% by each contaminant.
  $PY $GEN/mixture.py  --source datasets/benchmark_splits
  # Table 12 and the coupling axes: rho (corpus-level), noise (per-annotation
  # content), jitter (per-annotation timestamp displacement).
  for mode in rho noise jitter; do
    $PY $GEN/coupling.py --source datasets/benchmark_splits --mode "$mode"
  done
  # Table 11: the same annotations over a signal with observation noise added.
  for s in 0.05 0.1 0.2 0.4; do
    $PY $GEN/sine_generator.py --signal-type noisy_sine_with_constants \
        --noise-std "$s" --output-dir "datasets/benchmark_splits_noise$s"
  done
fi

# ── Mutual information ──────────────────────────────────────────────────────
say "MI: final parameters (Table 8) and shuffle control (Table 10)"
$PY run.py final_configs/sine.yaml        -o "$OUT/metrics.json"
$PY run.py final_configs/sineshuffle.yaml -o "$OUT/shuffle.json"

say "MI: representation-pipeline sweeps (Tables 4-7, Figure 2)"
for sweep in embedding pca lookback patch; do
  $PY run.py "final_configs/sweeps/$sweep.yaml" -o "$OUT/sweep_$sweep.json"
done

say "MI: perturbation axes (Tables 11-12)"
$PY run.py final_configs/sine_noise.yaml -o "$OUT/noise.json"
for mode in rho noise jitter; do
  $PY run.py "final_configs/coupling_$mode.yaml" -o "$OUT/coupling_$mode.json"
done

say "MI: mixture corpora (Figure 4)"
$PY run.py final_configs/mixture.yaml -o "$OUT/mixture.json"

# ── Downstream models ───────────────────────────────────────────────────────
say "Models: scikit-learn regressors (Table 9)"
$PY train.py --config final_configs/sine.yaml    -o "$OUT/sklearn.json"
$PY train.py --config final_configs/mixture.yaml -o "$OUT/mixture_sklearn.json"
for mode in rho noise jitter; do
  $PY train.py --config "final_configs/coupling_$mode.yaml" -o "$OUT/coupling_${mode}_sklearn.json"
done

if [[ -z "${QUICK:-}" ]]; then
  say "Models: transformer backbones x fusion strategies (Table 9)"
  $PY train_transformer.py --config final_configs/sine_transformers.yaml \
      -o "$OUT/transformers"
fi

say "Done — results in $OUT/"
echo "Next: python reproduce/make_artifacts.py --which mmtt_bench"
