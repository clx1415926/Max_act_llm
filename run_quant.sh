#!/bin/bash
# Parallel runner for quant_sanity_check.py
# Already-done models are skipped automatically by the script itself.
# Usage:
#   bash run_quant.sh               # run all
#   bash run_quant.sh --summary_only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/quant"
mkdir -p "${LOG_DIR}"

PYTHON="python"
QUANT="${SCRIPT_DIR}/quant_sanity_check.py"
QUANT_DIR="${SCRIPT_DIR}/results/quant"

# ── GPU selection helper ────────────────────────────────────────────────────
# Prints the index of a free GPU (memory.used < threshold_mb), excluding list.
get_free_gpu() {
    local threshold_mb=${1:-2000}
    local excluded=" ${2:-} "
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n \
        | while IFS=',' read -r idx mem_used; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            if [ "$mem_used" -lt "$threshold_mb" ] && [[ "$excluded" != *" $idx "* ]]; then
                echo "$idx"
                break
            fi
        done
}

# ── Summary only ────────────────────────────────────────────────────────────
if [ "${1:-}" = "--summary_only" ]; then
    echo "[summary] aggregating all *_quant_sanity.json ..."
    ${PYTHON} "${QUANT}" --summary_only
    exit 0
fi

# ── All models: run in parallel, one per GPU ────────────────────────────────
ALL_MODELS=(
    "Qwen3.5-0.8B" "gemma-3-4b-it"
    "Qwen2.5-7B" "Qwen3-8B" "Qwen3.5-9B"
    "Qwen3.5-35B-A3B" "Qwen3-30B-A3B" "Qwen3-32B"
)

echo "================================================================"
echo "Running all models in parallel (one per GPU)"
echo "================================================================"

declare -A PID_GPU
declare -A PID_NAME
USED_GPUS=""
queue=("${ALL_MODELS[@]}")

while [ ${#queue[@]} -gt 0 ] || [ ${#PID_GPU[@]} -gt 0 ]; do
    # Reap finished jobs
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rc=0; wait "$pid" || rc=$?
            gpu="${PID_GPU[$pid]}"; name="${PID_NAME[$pid]}"
            [ $rc -eq 0 ] \
                && echo "[DONE] ${name} (GPU ${gpu})" \
                || echo "[FAIL] ${name} (GPU ${gpu}) — see ${LOG_DIR}/${name}.log"
            USED_GPUS=" $USED_GPUS "
            USED_GPUS="${USED_GPUS// $gpu / }"
            USED_GPUS=$(echo "$USED_GPUS" | xargs)
            unset "PID_GPU[$pid]"
            unset "PID_NAME[$pid]"
        fi
    done

    # Launch into free GPUs
    launched=0
    while [ ${#queue[@]} -gt 0 ]; do
        gpu_idx=$(get_free_gpu 2000 "$USED_GPUS")
        [ -z "$gpu_idx" ] && break

        model_name="${queue[0]}"; queue=("${queue[@]:1}")

        # Skip if already done
        if [ -f "${QUANT_DIR}/${model_name}_quant_sanity.json" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        logfile="${LOG_DIR}/${model_name}.log"
        echo "[LAUNCH] ${model_name} → GPU ${gpu_idx}  (log: ${logfile})"
        nohup ${PYTHON} "${QUANT}" \
            --models "${model_name}" \
            --gpu_id "${gpu_idx}" \
            --no_summary \
            > "${logfile}" 2>&1 &
        new_pid=$!
        PID_GPU[$new_pid]=$gpu_idx
        PID_NAME[$new_pid]="$model_name"
        USED_GPUS=$(echo "$USED_GPUS $gpu_idx" | xargs)
        launched=$((launched + 1))
        sleep 3   # let nvidia-smi reflect the new allocation
    done

    [ $launched -eq 0 ] && [ ${#PID_GPU[@]} -gt 0 ] && sleep 5
done

echo ""
echo "================================================================"
echo "Generating summary ..."
echo "================================================================"
${PYTHON} "${QUANT}" --summary_only
echo ""
echo "Done. Results in: ${QUANT_DIR}"
