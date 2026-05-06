#!/usr/bin/env bash
# run_finals.sh — Run metrics and training for every final config.
#
# Configs may set data_path as a single path or a YAML list:
#
#   data_path: benchmark_splits/train.json          # single
#
#   data_path:                                       # list
#     - timemmd_dataset/agriculture_train.json
#     - timemmd_dataset/economy_train.json
#
# When data_path is a list, each dataset is run independently and outputs are
# named  results/final/<config>_<dataset_stem>_metrics.json  etc.
#
# After this completes, use compare.py to analyse and plot the results:
#   python compare.py \
#       --metrics results/final/sine_metrics.json \
#       --models  results/final/sine_model.json
#
# Usage:
#   ./run_finals.sh                    # all configs
#   ./run_finals.sh sine rossler       # specific configs by name
#   SKIP_TRAIN=1 ./run_finals.sh       # metrics only (no training)
#   SKIP_METRICS=1 ./run_finals.sh     # training only

set -euo pipefail

# Prevent OpenBLAS from spawning more threads than it was compiled to handle.
# Without this, numpy/scipy BLAS calls across many cores cause a segfault.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RESULTS_DIR="results/final"
CONFIGS_DIR="final_configs"

mkdir -p "$RESULTS_DIR"

# Build list of configs to run
if [[ $# -gt 0 ]]; then
    NAMES=("$@")
else
    NAMES=()
    for cfg in "$CONFIGS_DIR"/*.yaml; do
        NAMES+=("$(basename "$cfg" .yaml)")
    done
fi

echo "Running final evaluations for: ${NAMES[*]}"
echo "Output directory: $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helper: expand a config's data_path into (stem, tmp_config_path) pairs.
# Prints lines of the form:  STEM<TAB>CONFIG_PATH
# When data_path is a single value, CONFIG_PATH == the original file.
# When data_path is a list, a temp config is written for each entry.
# Caller must delete any temp files; they are named /tmp/finalcfg_*.yaml.
# ---------------------------------------------------------------------------
expand_data_paths() {
    local config="$1"
    python3 - "$config" <<'PYEOF'
import sys, yaml, tempfile, os
from pathlib import Path

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

dp = cfg.get("data_path", "")

if isinstance(dp, list):
    for path in dp:
        stem = Path(path).stem          # e.g. agriculture_train
        # write a temp config with a single data_path
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="finalcfg_", delete=False
        )
        single = {**cfg, "data_path": path}
        yaml.dump(single, tmp, default_flow_style=False)
        tmp.close()
        print(f"{stem}\t{tmp.name}")
else:
    stem = Path(config_path).stem       # e.g. rossler
    print(f"{stem}\t{config_path}")
PYEOF
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for NAME in "${NAMES[@]}"; do
    CONFIG="$CONFIGS_DIR/${NAME}.yaml"

    if [[ ! -f "$CONFIG" ]]; then
        echo "  [skip] $CONFIG not found"
        continue
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "  Config: $NAME"
    echo "════════════════════════════════════════════════════════════"

    # Collect (stem, cfg_path) pairs; track any temp files for cleanup
    declare -a STEMS=()
    declare -a CFGS=()
    declare -a TMPS=()

    while IFS=$'\t' read -r stem cfg_path; do
        STEMS+=("$stem")
        CFGS+=("$cfg_path")
        # mark as temp if it lives in /tmp
        if [[ "$cfg_path" == /tmp/* ]]; then
            TMPS+=("$cfg_path")
        fi
    done < <(expand_data_paths "$CONFIG")

    for i in "${!STEMS[@]}"; do
        STEM="${STEMS[$i]}"
        RUN_CFG="${CFGS[$i]}"

        # If the stem already equals the config name (single data_path case)
        # keep output as <name>; otherwise suffix with the dataset stem.
        if [[ "$STEM" == "$NAME" ]]; then
            OUT_STEM="$NAME"
        else
            OUT_STEM="${NAME}_${STEM}"
        fi

        echo "  Dataset: $STEM"

        # ── Step 1: Information-theoretic metrics ────────────────────────
        if [[ -z "${SKIP_METRICS:-}" ]]; then
            METRICS_OUT="$RESULTS_DIR/${OUT_STEM}_metrics.json"
            echo "  [metrics] → $METRICS_OUT"
            python run.py "$RUN_CFG" --output "$METRICS_OUT"
        fi

        # ── Step 2: Regression models ─────────────────────────────────────
        if [[ -z "${SKIP_TRAIN:-}" ]]; then
            MODEL_OUT="$RESULTS_DIR/${OUT_STEM}_model.json"
            echo "  [train]   → $MODEL_OUT"

            python train.py \
                --config "$RUN_CFG" \
                --output "$MODEL_OUT"
        fi

        echo ""
    done

    # Clean up any temp configs generated for list data_paths
    for tmp in "${TMPS[@]+"${TMPS[@]}"}"; do
        rm -f "$tmp"
    done

    unset STEMS CFGS TMPS
done

echo "════════════════════════════════════════════════════════════"
echo "Done. Results written to $RESULTS_DIR/"
echo ""
echo "To compare metrics vs model performance, run e.g.:"
echo "  python plot.py \\"
echo "      --metrics $RESULTS_DIR/sine_metrics.json \\"
echo "      --models  $RESULTS_DIR/sine_model.json \\"
echo "      --output-dir $RESULTS_DIR/plots/"
echo ""
echo "════════════════════════════════════════════════════════════"
