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
    "Qwen2.5|Qwen2.5-1.5B|${MODELS_DIR}/Qwen2.5/Qwen2.5-1.5B|${DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl|32|small"
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
    local found
    found=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n \
        | while IFS=',' read -r idx mem_used; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            if [ "$mem_used" -lt "$threshold_mb" ]; then
                echo "$idx"
                break
            fi
        done)
    if [ -n "$found" ]; then
        echo "$found"
    else
        # Fallback: GPU with least memory usage
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | xargs
    fi
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

    # Track analyzed models
    ANALYZED_MODELS=()
    SKIPPED_MODELS=()
    FAILED_MODELS=()

    # ── Launch all models in parallel, one per free GPU ───────────────
    # Models are launched immediately as free GPUs are found, regardless of
    # size class. Each model occupies one GPU (single-GPU for <70 GB models,
    # auto for larger). A 45-second sleep between launches gives the previous
    # model enough time to claim its GPU memory before we check for the next
    # free card.
    echo ""
    echo "── Launching all models in parallel (one per free GPU) ──"
    ALL_PIDS=()
    ALL_NAMES=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs size_class <<< "$entry"

        output_dir="${RESULTS_DIR}/${series}"
        json_output="${output_dir}/json/${model_name}_activation_stats.json"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done"
            SKIPPED_MODELS+=("${model_name} (already processed)")
            continue
        fi

        if [ ! -d "$model_path" ]; then
            echo "[SKIP] ${model_name} — model path not found: ${model_path}"
            SKIPPED_MODELS+=("${model_name} (model not found)")
            continue
        fi

        if [ ! -f "$data_path" ]; then
            echo "[SKIP] ${model_name} — data path not found: ${data_path}"
            SKIPPED_MODELS+=("${model_name} (data not found)")
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
        ALL_PIDS+=($!)
        ALL_NAMES+=("${model_name}(GPU${GPU_IDX})")
        ANALYZED_MODELS+=("${model_name}")
        # Wait for this GPU's memory to be claimed before picking the next card.
        # 45 s covers even large models (32B+ takes ~30 s to start allocating).
        sleep 45
    done

    # ── Wait for all launched jobs ─────────────────────────────────────
    if [ ${#ALL_PIDS[@]} -gt 0 ]; then
        echo ""
        echo "Waiting for ${#ALL_PIDS[@]} model(s): ${ALL_NAMES[*]}"
        FAIL=0
        for i in "${!ALL_PIDS[@]}"; do
            model_name_only="${ALL_NAMES[$i]%(*}"
            if wait ${ALL_PIDS[$i]}; then
                echo "[DONE] ${ALL_NAMES[$i]}"
            else
                echo "[FAIL] ${ALL_NAMES[$i]} — check log for details"
                FAILED_MODELS+=("${model_name_only}")
                ANALYZED_MODELS=("${ANALYZED_MODELS[@]/$model_name_only}")
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some models failed. Check logs." || true
    fi
    echo ""
fi
·
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
echo "============================================================"
echo ""

# ─── Summary Report ─────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "                     EXECUTION SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

# Models analysis summary
if $DO_ANALYZE; then
    echo "📊 Models Analysis:"
    echo "──────────────────────────────────────────────────────────"
    
    # Successfully analyzed models
    if [ ${#ANALYZED_MODELS[@]} -gt 0 ]; then
        # Filter out empty entries
        ANALYZED_FILTERED=()
        for model in "${ANALYZED_MODELS[@]}"; do
            [ -n "$model" ] && ANALYZED_FILTERED+=("$model")
        done
        
        if [ ${#ANALYZED_FILTERED[@]} -gt 0 ]; then
            echo "✓ Successfully analyzed (${#ANALYZED_FILTERED[@]} models):"
            for model in "${ANALYZED_FILTERED[@]}"; do
                echo "  • $model"
            done
            echo ""
        fi
    fi
    
    # Skipped models
    if [ ${#SKIPPED_MODELS[@]} -gt 0 ]; then
        echo "⊘ Skipped (${#SKIPPED_MODELS[@]} models):"
        for model in "${SKIPPED_MODELS[@]}"; do
            echo "  • $model"
        done
        echo ""
    fi
    
    # Failed models
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        # Filter out empty entries
        FAILED_FILTERED=()
        for model in "${FAILED_MODELS[@]}"; do
            [ -n "$model" ] && FAILED_FILTERED+=("$model")
        done
        
        if [ ${#FAILED_FILTERED[@]} -gt 0 ]; then
            echo "✗ Failed (${#FAILED_FILTERED[@]} models):"
            for model in "${FAILED_FILTERED[@]}"; do
                echo "  • $model"
            done
            echo ""
        fi
    fi
fi

# Generated files summary
echo "📁 Generated Files:"
echo "──────────────────────────────────────────────────────────"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# List JSON files
JSON_COUNT=$(find "${RESULTS_DIR}" -name "*_activation_stats.json" 2>/dev/null | wc -l)
if [ $JSON_COUNT -gt 0 ]; then
    echo "JSON Statistics Files ($JSON_COUNT):"
    find "${RESULTS_DIR}" -name "*_activation_stats.json" 2>/dev/null | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  • $(basename "$file") [$size]"
    done
    echo ""
fi

# List log files
LOG_COUNT=$(find "${RESULTS_DIR}" -name "*.log" 2>/dev/null | wc -l)
if [ $LOG_COUNT -gt 0 ]; then
    echo "Log Files ($LOG_COUNT):"
    find "${RESULTS_DIR}" -name "*.log" 2>/dev/null | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  • $(basename "$file") [$size]"
    done
    echo ""
fi

# List plot files
PLOT_COUNT=$(find "${RESULTS_DIR}" -path "*/plots/*" -type f 2>/dev/null | wc -l)
if [ $PLOT_COUNT -gt 0 ]; then
    echo "Plot Files ($PLOT_COUNT):"
    find "${RESULTS_DIR}" -path "*/plots/*" -type f 2>/dev/null | head -20 | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  • $(basename "$file") [$size]"
    done
    if [ $PLOT_COUNT -gt 20 ]; then
        echo "  ... and $((PLOT_COUNT - 20)) more plot files"
    fi
    echo ""
fi

# Directory structure
echo "📂 Results Directory Structure:"
echo "──────────────────────────────────────────────────────────"
ls -lh "${RESULTS_DIR}"/*/ 2>/dev/null | grep -E "^(d|total)" || echo "  (No subdirectories found)"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "Summary report complete!"
echo "════════════════════════════════════════════════════════════"

