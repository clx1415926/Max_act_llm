#!/bin/bash
###############################################################################
# run_all_experiments.sh — 8-GPU unified runner for two supplementary
# experiments:
#
#   Exp A (Quant Sanity Check):
#     Measure INT-8 activation quantization error across representative models
#     at two clipping policies (max_abs vs clip_q0.999).  Runs quant_sanity_check.py
#     per model in parallel, then aggregates the summary.
#
#   Exp B (Bootstrap Stability):
#     Re-run analyze_model.py on category-proportional 1k/2k sub-samples
#     (5 repeats each) for 4 representative models to verify that the global
#     peak activation is not driven by a single outlier sample.
#
# Usage:
#   bash run_all_experiments.sh           # run both experiments
#   bash run_all_experiments.sh --quant   # run Exp A only
#   bash run_all_experiments.sh --bootstrap # run Exp B only
#
# Key environment overrides (all optional):
#   BITS=8 CLIP_QUANTILE=0.999
#   CALIBRATION_SAMPLES=128 EVAL_SAMPLES=256 QUANT_SEQ_LEN=4096
#   BOOTSTRAP_SIZES="1000 2000" BOOTSTRAP_REPEATS=5 BOOTSTRAP_SEQ_LEN=32768
#   FREE_MEM_THRESHOLD_MB=2000   — GPU considered "free" below this used-mem
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# ─── Parameters ──────────────────────────────────────────────────────────────
BITS="${BITS:-8}"
CLIP_QUANTILE="${CLIP_QUANTILE:-0.999}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
QUANT_SEQ_LEN="${QUANT_SEQ_LEN:-4096}"
LAYER_POLICY="${LAYER_POLICY:-peak}"

BOOTSTRAP_SIZES="${BOOTSTRAP_SIZES:-1000 2000}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-5}"
BOOTSTRAP_SEQ_LEN="${BOOTSTRAP_SEQ_LEN:-32768}"

FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-2000}"

QUANT_OUT="${SCRIPT_DIR}/results/quant"
BOOTSTRAP_ROOT="${SCRIPT_DIR}/results/bootstrap"
BOOTSTRAP_DATA="${SCRIPT_DIR}/datasets/bootstrap"

# ─── Stage selector ──────────────────────────────────────────────────────────
RUN_QUANT=1
RUN_BOOTSTRAP=1
for arg in "$@"; do
    case "$arg" in
        --quant)      RUN_QUANT=1; RUN_BOOTSTRAP=0 ;;
        --bootstrap)  RUN_QUANT=0; RUN_BOOTSTRAP=1 ;;
    esac
done

# ─── Model definitions ───────────────────────────────────────────────────────
# Quant sanity models
QUANT_MODELS=("Qwen3.5-0.8B" "gemma-3-4b-it" "Qwen3-30B-A3B" "Qwen3-32B")

# Bootstrap models: FAMILY|MODEL_NAME|MODEL_PATH|SERIES_KEY|BATCH_SIZE
BOOTSTRAP_MODELS=(
    "Qwen3.5|Qwen3.5-0.8B|${SCRIPT_DIR}/models/Qwen3.5/Qwen3.5-0.8B|qwen3.5|8"
    "gemma3|gemma-3-4b-it|${SCRIPT_DIR}/models/gemma3/gemma-3-4b-it|gemma3|4"
    "Qwen3|Qwen3-30B-A3B|${SCRIPT_DIR}/models/Qwen3/Qwen3-30B-A3B|qwen3|1"
    "Qwen3|Qwen3-32B|${SCRIPT_DIR}/models/Qwen3/Qwen3-32B|qwen3|1"
)

# ─── GPU Scheduler ───────────────────────────────────────────────────────────
declare -A PID_GPU   # pid -> gpu_index
declare -A PID_NAME  # pid -> job_name
USED_GPUS=""

get_free_gpu() {
    local excluded=" ${1:-} "
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n \
        | while IFS=',' read -r idx mem_used; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            if [ "$mem_used" -lt "$FREE_MEM_THRESHOLD_MB" ] \
               && [[ "$excluded" != *" $idx "* ]]; then
                echo "$idx"
                break
            fi
        done
}

reap_finished() {
    local pid rc gpu name
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rc=0; wait "$pid" || rc=$?
            gpu="${PID_GPU[$pid]}"; name="${PID_NAME[$pid]}"
            if [ $rc -eq 0 ]; then
                echo "[DONE] ${name} (GPU ${gpu})"
            else
                echo "[FAIL] ${name} (GPU ${gpu}) exit=${rc}"
            fi
            local tmp=" $USED_GPUS "
            tmp="${tmp// $gpu / }"
            USED_GPUS=$(echo "$tmp" | xargs)
            unset "PID_GPU[$pid]" "PID_NAME[$pid]"
        fi
    done
}

wait_all() {
    while [ "${#PID_GPU[@]}" -gt 0 ]; do
        reap_finished
        sleep 5
    done
}

# Launch a command in background on an auto-selected free GPU.
# Args: job_name  log_file  cmd_string
launch() {
    local job_name="$1"
    local log_file="$2"
    local cmd="$3"
    local gpu_idx

    while true; do
        reap_finished
        gpu_idx=$(get_free_gpu "$USED_GPUS")
        [ -n "$gpu_idx" ] && break
        echo "[WAIT] ${job_name} — no free GPU (threshold ${FREE_MEM_THRESHOLD_MB} MB); retry in 5s"
        sleep 5
    done

    echo "[LAUNCH] ${job_name} → GPU ${gpu_idx}"
    mkdir -p "$(dirname "$log_file")"
    CUDA_VISIBLE_DEVICES="${gpu_idx}" bash -c "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    PID_GPU[$pid]="$gpu_idx"
    PID_NAME[$pid]="$job_name"
    USED_GPUS=$(echo "$USED_GPUS $gpu_idx" | xargs)
    sleep 2   # let nvidia-smi catch the memory allocation
}

# ─── Stage 0: Bootstrap dataset creation (CPU, fast) ─────────────────────────
if [ "$RUN_BOOTSTRAP" -eq 1 ]; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Stage 0: Creating bootstrap sub-datasets"
    echo "══════════════════════════════════════════════════"
    python "${SCRIPT_DIR}/make_bootstrap_datasets.py" \
        --sizes ${BOOTSTRAP_SIZES} \
        --repeats "${BOOTSTRAP_REPEATS}" \
        --output_dir "${BOOTSTRAP_DATA}"
fi

# ─── Stage 1: Quant sanity check — one process per model ─────────────────────
if [ "$RUN_QUANT" -eq 1 ]; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Stage 1: Quant sanity check (Exp A)"
    echo "══════════════════════════════════════════════════"
    mkdir -p "${QUANT_OUT}/logs"
    for model_name in "${QUANT_MODELS[@]}"; do
        out_json="${QUANT_OUT}/${model_name}_quant_sanity.json"
        if [ -f "$out_json" ]; then
            echo "[SKIP] quant-${model_name} — already done"
            continue
        fi
        log_file="${QUANT_OUT}/logs/${model_name}.log"
        cmd="python '${SCRIPT_DIR}/quant_sanity_check.py' \
  --models '${model_name}' \
  --bits '${BITS}' \
  --clip_quantile '${CLIP_QUANTILE}' \
  --calibration_samples '${CALIBRATION_SAMPLES}' \
  --eval_samples '${EVAL_SAMPLES}' \
  --max_seq_len '${QUANT_SEQ_LEN}' \
  --layer_policy '${LAYER_POLICY}' \
  --gpu_id 0 \
  --no_summary \
  --output_dir '${QUANT_OUT}'"
        launch "quant-${model_name}" "$log_file" "$cmd"
    done
fi

# ─── Stage 2: Bootstrap stability — (size × repeat × model) tasks ────────────
if [ "$RUN_BOOTSTRAP" -eq 1 ]; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Stage 2: Bootstrap stability (Exp B)"
    echo "══════════════════════════════════════════════════"
    for size in ${BOOTSTRAP_SIZES}; do
        for repeat in $(seq 0 $((BOOTSTRAP_REPEATS - 1))); do
            for entry in "${BOOTSTRAP_MODELS[@]}"; do
                IFS='|' read -r family model_name model_path series_key batch_size <<< "$entry"
                data_path="${BOOTSTRAP_DATA}/eval_diverse_bootstrap_${size}_r${repeat}_${series_key}.jsonl"
                output_dir="${BOOTSTRAP_ROOT}/${size}/r${repeat}/${family}"
                json_output="${output_dir}/json/${model_name}_activation_stats.json"
                log_file="${output_dir}/logs/${model_name}.log"

                if [ -f "$json_output" ]; then
                    echo "[SKIP] bootstrap-${model_name}-${size}-r${repeat}"
                    continue
                fi
                if [ ! -d "$model_path" ]; then
                    echo "[SKIP] bootstrap-${model_name}-${size}-r${repeat} — model path missing"
                    continue
                fi
                if [ ! -f "$data_path" ]; then
                    echo "[SKIP] bootstrap-${model_name}-${size}-r${repeat} — data missing: ${data_path}"
                    continue
                fi
                mkdir -p "${output_dir}/json" "${output_dir}/logs"
                cmd="python '${SCRIPT_DIR}/analyze_model.py' \
  --model_path '${model_path}' \
  --data_path '${data_path}' \
  --output_dir '${output_dir}' \
  --max_samples '${size}' \
  --max_seq_len '${BOOTSTRAP_SEQ_LEN}' \
  --batch_size '${batch_size}' \
  --gpu_id 0"
                launch "boot-${model_name}-${size}-r${repeat}" "$log_file" "$cmd"
            done
        done
    done
fi

# ─── Wait for all background jobs ────────────────────────────────────────────
echo ""
echo "Waiting for all jobs to finish..."
wait_all

# ─── Stage 3: Quant summary aggregation ──────────────────────────────────────
if [ "$RUN_QUANT" -eq 1 ]; then
    echo ""
    echo "── Aggregating quant sanity summary ──"
    python "${SCRIPT_DIR}/quant_sanity_check.py" \
        --summary_only \
        --output_dir "${QUANT_OUT}"
fi

echo ""
echo "All experiments complete."
echo "  Quant results:     ${QUANT_OUT}/"
echo "  Bootstrap results: ${BOOTSTRAP_ROOT}/"
