#!/usr/bin/env bash
# 02_timemmd.sh — Time-MMD results (Tables 13, 17, 18; Figure 5, Time-MMD rows).
#
# Requires the Time-MMD CSVs, which are not redistributed here. Clone
# MM-TSFlib and point --mm-tsf-root at its data directory:
#
#   git clone https://github.com/AdityaLab/MM-TSFlib ../MM-TSFlib
#   MM_TSF_ROOT=../MM-TSFlib/data ./reproduce/02_timemmd.sh
#
# Six domains are used. Economy and Security are excluded: they have fewer than
# 300 training samples per category, below what KSG needs at d = 16, and no
# usable validation split.
#
# Time-MMD mixes sampling frequencies, and the CFA window settings differ by
# frequency, so MI is run twice with different lookback/horizon rather than once
# — monthly domains at seq_len 8 / pred_len 6, weekly at 36 / 12.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-python}"
MM_TSF_ROOT="${MM_TSF_ROOT:-../MM-TSFlib/data}"
OUT=results/timemmd
mkdir -p "$OUT"
say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

if [[ -z "${SKIP_DATA:-}" ]]; then
  say "Building Time-MMD splits from $MM_TSF_ROOT"
  [[ -d "$MM_TSF_ROOT" ]] || { echo "ERROR: $MM_TSF_ROOT not found — see header." >&2; exit 1; }
  $PY src/mmtt_bench/data_generation/timemmd.py \
      --mm-tsf-root "$MM_TSF_ROOT" --output-dir datasets/timemmd_cfa_splits
fi

say "MI estimates (Table 17, Figure 5)"
# Each config declares its domains as a `datasets:` list, so run.py loops over
# them and writes one JSON per domain into the output directory.
$PY run.py final_configs/timemmd.yaml        -o "$OUT/mi"   # monthly: Agriculture, Climate, SocialGood, Traffic
$PY run.py final_configs/timemmd_weekly.yaml -o "$OUT/mi"   # weekly:  Energy, PublicHealth

say "Downstream models: 9 backbones x 10 fusion strategies (Table 18)"
$PY train_transformer.py --config final_configs/timemmd_transformer.yaml -o "$OUT/transformer"

say "Done — results in $OUT/"
echo "Next: python reproduce/make_artifacts.py --which timemmd"
