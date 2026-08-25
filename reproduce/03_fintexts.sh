#!/usr/bin/env bash
# 03_fintexts.sh — FinTexTS results (Tables 13, 17, 18; Figure 5, FinTexTS row).
#
# Requires the FinTexTS release (100 tickers, one HuggingFace datasets dir per
# ticker). Not redistributed here:
#
#   FINTEXTS_ROOT=/path/to/FinTexTS ./reproduce/03_fintexts.sh
#
# Ten tickers are used, matching the paper: AMD BA COST DIS GOOGL INTC NFLX
# NVDA T TSLA. Each ticker is treated as an independent dataset and the
# estimates are pooled at the end (z-scores Stouffer-pooled), because the
# tickers share a calendar but not a data-generating process.
#
# FinTexTS carries 21 per-field texts per point; train_transformer.py embeds
# each separately and mean-pools, matching FinTexTS Sec 4.1 rather than
# concatenating and embedding once. ModernBERT is used for its 8192-token
# context — BERT would truncate the filing fields at 512.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-python}"
FINTEXTS_ROOT="${FINTEXTS_ROOT:-../FinTexTS}"
OUT=results/fintexts
TICKERS="${TICKERS:-AMD BA COST DIS GOOGL INTC NFLX NVDA T TSLA}"
mkdir -p "$OUT"
say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

if [[ -z "${SKIP_DATA:-}" ]]; then
  say "Building FinTexTS splits from $FINTEXTS_ROOT"
  [[ -d "$FINTEXTS_ROOT" ]] || { echo "ERROR: $FINTEXTS_ROOT not found — see header." >&2; exit 1; }
  $PY src/mmtt_bench/data_generation/fintexts.py \
      --source "$FINTEXTS_ROOT" --output-dir datasets/fintexts_splits
fi

say "MI estimates, per ticker (Table 17, Figure 5)"
# fintexts.yaml declares all ten tickers as a `datasets:` list; run.py writes
# one JSON per ticker into the output directory.
$PY run.py final_configs/fintexts.yaml -o "$OUT/mi"

say "Downstream models, per ticker (Table 18)"
for tk in $TICKERS; do
  $PY train_transformer.py --config final_configs/fintexts_transformer.yaml \
      --dataset "$tk" -o "$OUT/transformer/$tk"
done

say "Done — results in $OUT/"
echo "Next: python reproduce/make_artifacts.py --which fintexts"
