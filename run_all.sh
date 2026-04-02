#!/bin/bash
###############################################################################
# run_all.sh — Full pipeline: convert datasets + analyze all models + plot
#
# Usage:
#   bash run_all.sh           # Run everything
#   bash run_all.sh --convert # Only convert datasets
#   bash run_all.sh --analyze # Only run activation analysis
#   bash run_all.sh --plot    # Only generate plots
#
# GPU selection: dynamically picks the GPU with least memory usage,
# avoiding GPUs already occupied by other tasks.
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/root/paddlejob/workspace/env_run/clx"
MODELS_DIR="${BASE_DIR}/models"
DATA_DIR="${BASE_DIR}/datasets"
RESULTS_DIR="${SCRIPT_DIR}/results"

MAX_SAMPLES=2000
MAX_SEQ_LEN=32768

# ─── Model definitions ──────────────────────────────────────────────────────
MODELS=(
    # Qwen2.5 series
    "Qwen2.5|Qwen2.5-1.5b|${MODELS_DIR}/Qwen2.5/Qwen2.5-1.5b|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|32|small"
    "Qwen2.5|Qwen2.5-7B|${MODELS_DIR}/Qwen2.5/Qwen2.5-7B|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|16|medium"
    "Qwen2.5|Qwen2.5-32B|${MODELS_DIR}/Qwen2.5/Qwen2.5-32B|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|4|xlarge"
    # Qwen3 series
    "Qwen3|Qwen3-1.7B|${MODELS_DIR}/Qwen3/Qwen3-1.7B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|32|small"
    "Qwen3|Qwen3-8B|${MODELS_DIR}/Qwen3/Qwen3-8B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|16|medium"
    "Qwen3|Qwen3-32B|${MODELS_DIR}/Qwen3/Qwen3-32B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|4|xlarge"
    "Qwen3|Qwen3-30B-A3B|${MODELS_DIR}/Qwen3/Qwen3-30B-A3B|${DATA_DIR}/eval_diverse_5k_qwen3.jsonl|4|xlarge"
    # gemma2 series
    "gemma2|gemma-2-2b|${MODELS_DIR}/gemma2/gemma-2-2b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|32|small"
    "gemma2|gemma-2-9b|${MODELS_DIR}/gemma2/gemma-2-9b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|16|medium"
    "gemma2|gemma-2-27b|${MODELS_DIR}/gemma2/gemma-2-27b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|1|xlarge"
    # gpt-oss series (MoE 20B, dequant to bf16 ~40GB, needs small batch due to eager attention)
    "gpt_oss|gpt-oss-20b|${MODELS_DIR}/gpt-oss/gpt-oss-20b|${DATA_DIR}/eval_diverse_5k_gpt_oss.jsonl|2|medium"
)

# ─── GPU selection helper ──────────────────────────────────────────────────
# Returns the GPU index with memory used below threshold_mb.
# Falls back to the GPU with least memory usage.

get_free_gpu() {
    local threshold_mb=${1:-1000}
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
    # Fallback: least-used GPU
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | xargs
}

# Wait for a free GPU (polls every 10s)
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
DO_CONVERT=false
DO_ANALYZE=false
DO_PLOT=false

if [ $# -eq 0 ]; then
    DO_CONVERT=true
    DO_ANALYZE=true
    DO_PLOT=true
else
    for arg in "$@"; do
        case $arg in
            --convert) DO_CONVERT=true ;;
            --analyze) DO_ANALYZE=true ;;
            --plot)    DO_PLOT=true ;;
            *)         echo "Unknown arg: $arg"; exit 1 ;;
        esac
    done
fi

# ─── Step 1: Convert datasets ───────────────────────────────────────────────
if $DO_CONVERT; then
    echo "============================================================"
    echo "Step 1: Converting datasets to target encodings"
    echo "============================================================"
    python3 "${SCRIPT_DIR}/convert_dataset.py" --all
    echo ""
fi

# ─── Step 2: Analyze each model ─────────────────────────────────────────────
if $DO_ANALYZE; then
    echo "============================================================"
    echo "Step 2: Running activation analysis for all models"
    echo "============================================================"

    # ── Phase 1: Small models ──────────────────────────────────────────
    echo ""
    echo "── Phase 1: Small models (parallel on free GPUs) ──"
    SMALL_PIDS=()
    SMALL_NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "small" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_activation_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_model.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}.log" 2>&1 &
        SMALL_PIDS+=($!)
        SMALL_NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#SMALL_PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#SMALL_PIDS[@]} small model(s): ${SMALL_NAMES[*]}"
        FAIL=0
        for i in "${!SMALL_PIDS[@]}"; do
            if wait ${SMALL_PIDS[$i]}; then
                echo "[DONE] ${SMALL_NAMES[$i]}"
            else
                echo "[FAIL] ${SMALL_NAMES[$i]} — check log for details"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some small models failed. Continuing..."
    fi

    # ── Phase 2: Medium models ─────────────────────────────────────────
    echo ""
    echo "── Phase 2: Medium models (parallel on free GPUs) ──"
    MED_PIDS=()
    MED_NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "medium" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_activation_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_model.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}.log" 2>&1 &
        MED_PIDS+=($!)
        MED_NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#MED_PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#MED_PIDS[@]} medium model(s): ${MED_NAMES[*]}"
        FAIL=0
        for i in "${!MED_PIDS[@]}"; do
            if wait ${MED_PIDS[$i]}; then
                echo "[DONE] ${MED_NAMES[$i]}"
            else
                echo "[FAIL] ${MED_NAMES[$i]} — check log for details"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some medium models failed. Continuing..."
    fi

    # ── Phase 3: XLarge models ─────────────────────────────────────────
    echo ""
    echo "── Phase 3: XLarge models (parallel on free GPUs) ──"
    XL_PIDS=()
    XL_NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"
        [ "$size_class" != "xlarge" ] && continue

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_activation_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 "${SCRIPT_DIR}/analyze_model.py" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}.log" 2>&1 &
        XL_PIDS+=($!)
        XL_NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#XL_PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#XL_PIDS[@]} xlarge model(s): ${XL_NAMES[*]}"
        FAIL=0
        for i in "${!XL_PIDS[@]}"; do
            if wait ${XL_PIDS[$i]}; then
                echo "[DONE] ${XL_NAMES[$i]}"
            else
                echo "[FAIL] ${XL_NAMES[$i]} — check log for details"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some xlarge models failed."
    fi
    echo ""
fi

# ─── Step 3: Generate plots ─────────────────────────────────────────────────
if $DO_PLOT; then
    echo "============================================================"
    echo "Step 3: Generating activation distribution plots"
    echo "============================================================"
    python3 "${SCRIPT_DIR}/plot_activations.py" --all --results_dir "${RESULTS_DIR}"
    echo ""
fi

echo "============================================================"
echo "All done!"
echo "Results directory: ${RESULTS_DIR}"
echo "============================================================"
ls -la "${RESULTS_DIR}"/*/ 2>/dev/null || true
