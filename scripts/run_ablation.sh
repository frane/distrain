#!/usr/bin/env bash
# Distrain v0.2 ablation experiment runner.
#
# Usage:
#   ./scripts/run_ablation.sh <experiment_name> [options]
#
# Options:
#   --pipeline <block|unstructured>  Compression pipeline (default: block)
#   --retention <float>              Fixed retention fraction (default: adaptive)
#   --quantization <int8_block|int8_tensor|bf16>  Quantization mode
#   --importance                     Enable importance-weighted selection
#   --rebasing                       Enable delta rebasing
#   --rebasing-threshold <int>       Staleness threshold for rebasing (default: 3)
#   --rebasing-coefficient <float>   Drift subtraction coefficient (default: 0.5)
#   --tokens <int>                   Target token count (default: 1000000)
#   --nodes <int>                    Number of nodes (default: 3)
#   --preset <tiny|small>            Model preset (default: tiny)
#
# Example:
#   ./scripts/run_ablation.sh block_vs_unstructured --pipeline block --tokens 500000

set -euo pipefail

EXPERIMENT_NAME="${1:?Usage: $0 <experiment_name> [options]}"
shift

# Defaults
PIPELINE="block"
RETENTION=""
QUANTIZATION="int8_block"
USE_IMPORTANCE="false"
ENABLE_REBASING="true"
REBASING_THRESHOLD=3
REBASING_COEFFICIENT=0.5
TARGET_TOKENS=1000000
NUM_NODES=3
PRESET="tiny"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --retention) RETENTION="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --importance) USE_IMPORTANCE="true"; shift ;;
        --rebasing) ENABLE_REBASING="true"; shift ;;
        --rebasing-threshold) REBASING_THRESHOLD="$2"; shift 2 ;;
        --rebasing-coefficient) REBASING_COEFFICIENT="$2"; shift 2 ;;
        --tokens) TARGET_TOKENS="$2"; shift 2 ;;
        --nodes) NUM_NODES="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "=== Ablation Experiment: ${EXPERIMENT_NAME} ==="
echo "Pipeline:     ${PIPELINE}"
echo "Retention:    ${RETENTION:-adaptive}"
echo "Quantization: ${QUANTIZATION}"
echo "Importance:   ${USE_IMPORTANCE}"
echo "Rebasing:     ${ENABLE_REBASING} (threshold=${REBASING_THRESHOLD}, coeff=${REBASING_COEFFICIENT})"
echo "Tokens:       ${TARGET_TOKENS}"
echo "Nodes:        ${NUM_NODES}"
echo "Preset:       ${PRESET}"
echo "Results:      ${RESULTS_DIR}/"
echo ""

# Save experiment config
cat > "${RESULTS_DIR}/config.json" <<CONFIGEOF
{
    "experiment": "${EXPERIMENT_NAME}",
    "pipeline": "${PIPELINE}",
    "retention": ${RETENTION:-null},
    "quantization": "${QUANTIZATION}",
    "use_importance": ${USE_IMPORTANCE},
    "enable_rebasing": ${ENABLE_REBASING},
    "rebasing_threshold": ${REBASING_THRESHOLD},
    "rebasing_coefficient": ${REBASING_COEFFICIENT},
    "target_tokens": ${TARGET_TOKENS},
    "num_nodes": ${NUM_NODES},
    "preset": "${PRESET}",
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CONFIGEOF

echo "Config saved to ${RESULTS_DIR}/config.json"
echo ""
echo "To run this experiment:"
echo ""
echo "1. Start coordinator:"
echo "   ENABLE_REBASING=${ENABLE_REBASING} \\"
echo "   REBASING_THRESHOLD=${REBASING_THRESHOLD} \\"
echo "   REBASING_COEFFICIENT=${REBASING_COEFFICIENT} \\"
echo "   cargo run -p distrain-coordinator"
echo ""
echo "2. Bootstrap v0 checkpoint:"
echo "   cargo run -p distrain-node -- bootstrap --preset ${PRESET}"
echo ""
echo "3. Start ${NUM_NODES} nodes with ablation config in node.toml:"
echo "   compression_pipeline = \"${PIPELINE}\""
if [[ -n "${RETENTION}" ]]; then
echo "   compression_retention = ${RETENTION}"
fi
echo "   quantization_mode = \"${QUANTIZATION}\""
echo "   use_importance = ${USE_IMPORTANCE}"
echo ""
echo "4. Monitor until ${TARGET_TOKENS} tokens trained"
echo "5. Download results: loss history from R2 stats/training_history.jsonl"
echo "6. Save to ${RESULTS_DIR}/"
