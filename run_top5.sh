#!/bin/bash
###############################################################################
# run_top5.sh — Top-5 activation analysis pipeline
#
# Usage:
#   bash run_top5.sh           # Run analyze + plot
#   bash run_top5.sh --analyze # Only run analysis
#   bash run_top5.sh --plot    # Only generate plots
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/root/paddlejob/workspace/env_run/clx"
MODELS_DIR="${BASE_DIR}/models"
DATA_DIR="${BASE_DIR}/datasets"
RESULTS_DIR="${SCRIPT_DIR}/results"

MAX_SAMPLES=2000
MAX_SEQ_LEN=32768

# Model definitions: SERIES|MODEL_NAME|MODEL_PATH|DATASET_FILE|BATCH_SIZE|SIZE_CLASS
MODELS=(
    "Qwen2.5|Qwen2.5-1.5b|${MODELS_DIR}/Qwen2.5/Qwen2.5-1.5b|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|32|small"
    "Qwen2.5|Qwen2.5-7B|${MODELS_DIR}/Qwen2.5/Qwen2.5-7B|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|16|medium"
    "Qwen2.5|Qwen2.5-32B|${MODELS_DIR}/Qwen2.5/Qwen2.5-32B|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|4|xlarge"
    "Qwen3|Qwen3-1.7B|${MODELS_DIR}/Qwen3/Qwen3-1.7B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|32|small"
    "Qwen3|Qwen3-8B|${MODELS_DIR}/Qwen3/Qwen3-8B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|16|medium"
    "Qwen3|Qwen3-32B|${MODELS_DIR}/Qwen3/Qwen3-32B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|4|xlarge"
    "Qwen3|Qwen3-30B-A3B|${MODELS_DIR}/Qwen3/Qwen3-30B-A3B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|4|xlarge"
    "gemma2|gemma-2-2b|${MODELS_DIR}/gemma2/gemma-2-2b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|32|small"
    "gemma2|gemma-2-9b|${MODELS_DIR}/gemma2/gemma-2-9b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|16|medium"
    "gemma2|gemma-2-27b|${MODELS_DIR}/gemma2/gemma-2-27b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|1|xlarge"
    # gpt-oss series (MoE 20B, dequant to bf16 ~40GB, needs small batch due to eager attention)
    "gpt_oss|gpt-oss-20b|${MODELS_DIR}/gpt-oss/gpt-oss-20b|${DATA_DIR}/eval_diverse_5k_gpt_oss.jsonl|2|medium"
)

# ─── GPU selection helper ──────────────────────────────────────────────────
# Returns the GPU index with the least memory usage (most free memory).
# Threshold: only pick GPUs with memory used < threshold_mb (default 1000 MB).

get_free_gpu() {
    local threshold_mb=${1:-1000}
    # Query all GPUs, sorted by memory used ascending
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n \
        | while IFS=',' read -r idx mem_used; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            if [ "$mem_used" -lt "$threshold_mb" ]; then
                echo "$idx"
                return 0
            fi
        done
    # No GPU below threshold found — return the one with least usage
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | xargs
}

# Wait for a free GPU (polls every 10s until one is available)
wait_for_free_gpu() {
    local threshold_mb=${1:-1000}
    while true; do
        local gpu
        gpu=$(get_free_gpu "$threshold_mb")
        if [ -n "$gpu" ]; then
            echo "$gpu"
            return 0
        fi
        sleep 10
    done
}

# ─── Parse arguments ────────────────────────────────────────────────────────
DO_ANALYZE=false
DO_PLOT=false

if [ $# -eq 0 ]; then
    DO_ANALYZE=true
    DO_PLOT=true
else
    for arg in "$@"; do
        case $arg in
            --analyze) DO_ANALYZE=true ;;
            --plot)    DO_PLOT=true ;;
            *)         echo "Unknown arg: $arg"; exit 1 ;;
        esac
    done
fi

# ─── Analyze ─────────────────────────────────────────────────────────────────
if $DO_ANALYZE; then
    echo "============================================================"
    echo "Top-5 Activation Analysis"
    echo "============================================================"

    # ── Phase 1: Small models ────────────────────────────────────────
    echo ""
    echo "── Phase 1: Small models (parallel on free GPUs) ──"
    PIDS=()
    NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "small" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_top5_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_top5.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}_top5.log" 2>&1 &
        PIDS+=($!)
        NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2  # brief delay to let GPU memory register
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#PIDS[@]} small model(s): ${NAMES[*]}"
        FAIL=0
        for i in "${!PIDS[@]}"; do
            if wait ${PIDS[$i]}; then
                echo "[DONE] ${NAMES[$i]}"
            else
                echo "[FAIL] ${NAMES[$i]}"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some small models failed."
    fi

    # ── Phase 2: Medium models ───────────────────────────────────────
    echo ""
    echo "── Phase 2: Medium models (parallel on free GPUs) ──"
    PIDS=()
    NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "medium" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_top5_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_top5.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}_top5.log" 2>&1 &
        PIDS+=($!)
        NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#PIDS[@]} medium model(s): ${NAMES[*]}"
        FAIL=0
        for i in "${!PIDS[@]}"; do
            if wait ${PIDS[$i]}; then
                echo "[DONE] ${NAMES[$i]}"
            else
                echo "[FAIL] ${NAMES[$i]}"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some medium models failed."
    fi

    # ── Phase 3: XLarge models ───────────────────────────────────────
    echo ""
    echo "── Phase 3: XLarge models (parallel on free GPUs) ──"
    PIDS=()
    NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "xlarge" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_top5_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_top5.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}_top5.log" 2>&1 &
        PIDS+=($!)
        NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#PIDS[@]} xlarge model(s): ${NAMES[*]}"
        FAIL=0
        for i in "${!PIDS[@]}"; do
            if wait ${PIDS[$i]}; then
                echo "[DONE] ${NAMES[$i]}"
            else
                echo "[FAIL] ${NAMES[$i]}"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some xlarge models failed."
    fi
    echo ""
fi

# ─── Plot ────────────────────────────────────────────────────────────────────
if $DO_PLOT; then
    echo "============================================================"
    echo "Generating top-5 activation plots"
    echo "============================================================"
    python3 "${SCRIPT_DIR}/plot_top5.py" --all --results_dir "${RESULTS_DIR}"
    echo ""
fi

echo "============================================================"
echo "All done!"
echo "Results directory: ${RESULTS_DIR}"
echo "============================================================"
ls -la "${RESULTS_DIR}"/*/ 2>/dev/null || true
