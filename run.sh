#!/bin/bash
###############################################################################
# run.sh — Unified pipeline: convert datasets + analyze all models + plot
#
# Modes (select via --mode, default: both):
#   all   — full activation stats  (analyze_model.py + plot_activations.py)
#   top5  — top-5 activation stats (analyze_top5.py  + plot_top5.py)
#   both  — run all then top5 sequentially
#
# Stages (filter with flags; default = all applicable stages):
#   --convert   Only convert datasets        (all mode only)
#   --analyze   Only run activation analysis
#   --plot      Only generate plots
#
# Examples:
#   bash run.sh                        # mode=both, all stages
#   bash run.sh --mode all             # full activation stats only
#   bash run.sh --mode top5 --analyze  # top5 analysis only (no plot)
#   bash run.sh --mode both --plot     # plots for both modes
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
# Format: SERIES|MODEL_NAME|MODEL_PATH|DATASET_FILE|BATCH_SIZE|SIZE_CLASS
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
    # ling series (MoE 256 experts, ~32GB per checkpoint, must run sequentially)
    "ling|Ling-mini-base-2.0-5T|${MODELS_DIR}/ling/Ling-mini-base-2.0-5T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xxlarge"
    "ling|Ling-mini-base-2.0-10T|${MODELS_DIR}/ling/Ling-mini-base-2.0-10T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xxlarge"
    "ling|Ling-mini-base-2.0-15T|${MODELS_DIR}/ling/Ling-mini-base-2.0-15T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xxlarge"
    "ling|Ling-mini-base-2.0-20T|${MODELS_DIR}/ling/Ling-mini-base-2.0-20T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xxlarge"
)

# ─── GPU selection helper ──────────────────────────────────────────────────
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

# ─── Run one analysis phase (small / medium / xlarge) ─────────────────────
# Args: $1=size_class  $2=analyze_script  $3=json_suffix  $4=log_suffix
run_phase() {
    local size_class="$1"
    local analyze_script="$2"
    local json_suffix="$3"
    local log_suffix="$4"

    echo ""
    echo "── Phase: ${size_class} models (${analyze_script##*/}) ──"
    local PIDS=()
    local NAMES=()

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs sc <<< "$entry"
        [ "$sc" != "$size_class" ] && continue

        local output_dir="${RESULTS_DIR}/${series}"
        local json_output="${output_dir}/json/${model_name}${json_suffix}"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done (${json_suffix})"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[LAUNCH] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python "${analyze_script}" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}${log_suffix}" 2>&1 &
        PIDS+=($!)
        NAMES+=("${model_name}(GPU${GPU_IDX})")
        sleep 2
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Waiting for ${#PIDS[@]} ${size_class} model(s): ${NAMES[*]}"
        local FAIL=0
        for i in "${!PIDS[@]}"; do
            if wait ${PIDS[$i]}; then
                echo "[DONE] ${NAMES[$i]}"
            else
                echo "[FAIL] ${NAMES[$i]} — check log for details"
                FAIL=1
            fi
        done
        [ $FAIL -ne 0 ] && echo "WARNING: Some ${size_class} models failed. Continuing..."
    fi
}

# ─── Run xxlarge models sequentially (one at a time, too large to parallelize)
# Args: $1=analyze_script  $2=json_suffix  $3=log_suffix
run_phase_sequential() {
    local analyze_script="$1"
    local json_suffix="$2"
    local log_suffix="$3"

    echo ""
    echo "── Phase: xxlarge models / sequential (${analyze_script##*/}) ──"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs sc <<< "$entry"
        [ "$sc" != "xxlarge" ] && continue

        local output_dir="${RESULTS_DIR}/${series}"
        local json_output="${output_dir}/json/${model_name}${json_suffix}"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done (${json_suffix})"
            continue
        fi

        GPU_IDX=$(wait_for_free_gpu 1000)
        echo "[RUN] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python "${analyze_script}" \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --output_dir "${output_dir}" \
            --max_samples ${MAX_SAMPLES} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --batch_size ${bs} \
            --gpu_id 0 \
            > "${output_dir}/logs/${model_name}${log_suffix}" 2>&1
        if [ $? -eq 0 ]; then
            echo "[DONE] ${model_name}"
        else
            echo "[FAIL] ${model_name} — check log for details"
        fi
    done
}

# ─── Parse arguments ────────────────────────────────────────────────────────
MODE="both"
DO_CONVERT=false
DO_ANALYZE=false
DO_PLOT=false
EXPLICIT_STAGE=false

for arg in "$@"; do
    case $arg in
        --mode)    :;;  # value handled below
        all|top5|both)
            MODE="$arg" ;;
        --convert) DO_CONVERT=true; EXPLICIT_STAGE=true ;;
        --analyze) DO_ANALYZE=true; EXPLICIT_STAGE=true ;;
        --plot)    DO_PLOT=true; EXPLICIT_STAGE=true ;;
        *)
            # Handle --mode <value>
            if [ "${PREV_ARG:-}" = "--mode" ]; then
                MODE="$arg"
            else
                echo "Unknown arg: $arg"; exit 1
            fi
            ;;
    esac
    PREV_ARG="$arg"
done

# If no stage flags given, enable all applicable stages
if ! $EXPLICIT_STAGE; then
    DO_CONVERT=true
    DO_ANALYZE=true
    DO_PLOT=true
fi

# ─── Step 1: Convert datasets (only for "all" and "both" modes) ───────────
if $DO_CONVERT && [ "$MODE" != "top5" ]; then
    echo "============================================================"
    echo "Step 1: Converting datasets to target encodings"
    echo "============================================================"
    python "${SCRIPT_DIR}/convert_dataset.py" --all
    echo ""
fi

# ─── Step 2: Analyze ─────────────────────────────────────────────────────
if $DO_ANALYZE; then
    # --- Mode: all (full activation stats) ---
    if [ "$MODE" = "all" ] || [ "$MODE" = "both" ]; then
        echo "============================================================"
        echo "Running full activation analysis (analyze_model.py)"
        echo "============================================================"
        SCRIPT="${SCRIPT_DIR}/analyze_model.py"
        JSUFFIX="_activation_stats.json"
        LSUFFIX=".log"
        for sc in small medium xlarge; do
            run_phase "$sc" "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
        done
        run_phase_sequential "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
        echo ""
    fi

    # --- Mode: top5 ---
    if [ "$MODE" = "top5" ] || [ "$MODE" = "both" ]; then
        echo "============================================================"
        echo "Running top-5 activation analysis (analyze_top5.py)"
        echo "============================================================"
        SCRIPT="${SCRIPT_DIR}/analyze_top5.py"
        JSUFFIX="_top5_stats.json"
        LSUFFIX="_top5.log"
        for sc in small medium xlarge; do
            run_phase "$sc" "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
        done
        run_phase_sequential "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
        echo ""
    fi
fi

# ─── Step 3: Generate plots ──────────────────────────────────────────────
if $DO_PLOT; then
    if [ "$MODE" = "all" ] || [ "$MODE" = "both" ]; then
        echo "============================================================"
        echo "Generating activation distribution plots"
        echo "============================================================"
        python "${SCRIPT_DIR}/plot_activations.py" --all --results_dir "${RESULTS_DIR}"
        echo ""
    fi

    if [ "$MODE" = "top5" ] || [ "$MODE" = "both" ]; then
        echo "============================================================"
        echo "Generating top-5 activation plots"
        echo "============================================================"
        python "${SCRIPT_DIR}/plot_top5.py" --all --results_dir "${RESULTS_DIR}"
        echo ""
    fi
fi

echo "============================================================"
echo "All done!"
echo "Results directory: ${RESULTS_DIR}"
echo "============================================================"
ls -la "${RESULTS_DIR}"/*/ 2>/dev/null || true
