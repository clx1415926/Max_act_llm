#!/bin/bash
###############################################################################
# run.sh — Unified pipeline: convert datasets + analyze all models + plot
#
# analyze_model.py emits both _activation_stats.json and _top5_stats.json
# in a single forward pass, so there is no separate top5 analysis stage.
# --plots selects which plot families to generate.
#
# Plot families (select via --plots, default: both):
#   full  — full activation plots only  (plot_activations.py)
#   top5  — top-5 activation plots only (plot_top5.py)
#   both  — generate both plot families
#
# Stages (filter with flags; default = all applicable stages):
#   --convert   Only convert datasets
#   --analyze   Only run activation analysis (produces both JSONs)
#   --plot      Only generate plots
#
# Examples:
#   bash run.sh                        # all stages, both plot families
#   bash run.sh --plots full           # only full-activation plots at plot stage
#   bash run.sh --analyze              # analyze only (both JSONs)
#   bash run.sh --plot --plots top5    # only top5 plots
#   bash run.sh stop                   # kill all running pipeline processes
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
MODELS_DIR="${BASE_DIR}/models"
DATA_DIR="${BASE_DIR}/datasets"
RESULTS_DIR="${SCRIPT_DIR}/results"
NEWLY_FULL=()   # models analyzed (full) this run
NEWLY_TOP5=()   # models analyzed (top5) this run

MAX_SAMPLES=5000
MAX_SEQ_LEN=32768

# ─── Model definitions ──────────────────────────────────────────────────────
# Format: SERIES|MODEL_NAME|MODEL_PATH|DATASET_FILE|BATCH_SIZE|SIZE_CLASS
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
    "gemma2|gemma-2b|${MODELS_DIR}/gemma2/gemma-2b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|32|small"
    "gemma2|gemma-2-9b|${MODELS_DIR}/gemma2/gemma-2-9b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|16|medium"
    "gemma2|gemma-2-27b|${MODELS_DIR}/gemma2/gemma-2-27b|${DATA_DIR}/eval_diverse_5k_gemma2.jsonl|1|xlarge"
    # gpt-oss series (MoE 20B, dequant to bf16 ~40GB, needs small batch due to eager attention)
    "gpt_oss|gpt-oss-20b|${MODELS_DIR}/gpt-oss/gpt-oss-20b|${DATA_DIR}/eval_diverse_5k_gpt_oss.jsonl|2|medium"
    # ling series (MoE 256 experts, ~32GB per checkpoint, fits easily on 80GB GPUs)
    "ling|Ling-mini-5T|${MODELS_DIR}/ling/Ling-mini-5T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xlarge"
    "ling|Ling-mini-10T|${MODELS_DIR}/ling/Ling-mini-10T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xlarge"
    "ling|Ling-mini-15T|${MODELS_DIR}/ling/Ling-mini-15T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xlarge"
    "ling|Ling-mini-20T|${MODELS_DIR}/ling/Ling-mini-20T|${DATA_DIR}/eval_diverse_5k_ling.jsonl|2|xlarge"
    # Qwen3.5 series (multimodal VLM, text-backbone analyzed; hybrid linear+full attention)
    "Qwen3.5|Qwen3.5-0.8B|${MODELS_DIR}/Qwen3.5/Qwen3.5-0.8B|${DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl|16|small"
    "Qwen3.5|Qwen3.5-9B|${MODELS_DIR}/Qwen3.5/Qwen3.5-9B|${DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl|8|medium"
    "Qwen3.5|Qwen3.5-27B|${MODELS_DIR}/Qwen3.5/Qwen3.5-27B|${DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl|2|xlarge"
    "Qwen3.5|Qwen3.5-35B-A3B|${MODELS_DIR}/Qwen3.5/Qwen3.5-35B-A3B|${DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl|2|xlarge"
    # gemma3 series (multimodal VLM, text-backbone via text_config)
    "gemma3|gemma-3-4b-it|${MODELS_DIR}/gemma3/gemma-3-4b-it|${DATA_DIR}/eval_diverse_5k_gemma3.jsonl|16|small"
    "gemma3|gemma-3-27b-it|${MODELS_DIR}/gemma3/gemma-3-27b-it|${DATA_DIR}/eval_diverse_5k_gemma3.jsonl|2|xlarge"
    # Qwen2.5-vl series (multimodal VLM, text-backbone at top-level config; nested under Qwen2.5-vl/models/)
    "Qwen2.5-vl|Qwen2.5-VL-3B|${MODELS_DIR}/Qwen2.5-vl/models/Qwen2.5-VL-3B|${DATA_DIR}/eval_diverse_5k_qwen2.5-vl.jsonl|16|small"
    "Qwen2.5-vl|Qwen2.5-VL-7B|${MODELS_DIR}/Qwen2.5-vl/models/Qwen2.5-VL-7B|${DATA_DIR}/eval_diverse_5k_qwen2.5-vl.jsonl|8|medium"
    "Qwen2.5-vl|Qwen2.5-VL-32B|${MODELS_DIR}/Qwen2.5-vl/models/Qwen2.5-VL-32B|${DATA_DIR}/eval_diverse_5k_qwen2.5-vl.jsonl|4|xlarge"
)

# ─── GPU selection helper ──────────────────────────────────────────────────
# Args: $1=threshold_mb (default 1000), $2=space-separated exclude list
# Prints a free GPU index or empty string if none eligible. No fallback.
get_free_gpu() {
    local threshold_mb=${1:-1000}
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


# ─── Scheduler: one task per GPU, reuse cards as models finish ─────────────
# xxlarge is excluded here and handled by run_phase_sequential instead.
# Args: $1=analyze_script  $2=json_suffix  $3=log_suffix
run_phase() {
    local analyze_script="$1"
    local json_suffix="$2"
    local log_suffix="$3"

    echo ""
    echo "── Scheduling models (${analyze_script##*/}) ──"

    # Build queue of eligible entries (skip xxlarge / done / missing paths)
    local -a queue=()
    local entry series model_name model_path data_path bs sc
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r series model_name model_path data_path bs sc <<< "$entry"
        [ "$sc" = "xxlarge" ] && continue

        local output_dir="${RESULTS_DIR}/${series}"
        local json_output="${output_dir}/json/${model_name}${json_suffix}"
        if [ -f "$json_output" ]; then
            echo "[SKIP] ${model_name} — already done (${json_suffix})"; continue
        fi
        if [ ! -d "$model_path" ]; then
            echo "[SKIP] ${model_name} — model path not found: ${model_path}"; continue
        fi
        if [ ! -f "$data_path" ]; then
            echo "[SKIP] ${model_name} — data path not found: ${data_path}"; continue
        fi
        queue+=("$entry")
    done

    if [ ${#queue[@]} -eq 0 ]; then
        echo "Nothing to run."
        return 0
    fi
    echo "Queue size: ${#queue[@]} model(s)"

    # Scheduler state
    declare -A PID_GPU    # pid -> gpu index
    declare -A PID_NAME   # pid -> model name
    local USED_GPUS=""    # space-separated list of GPUs currently in use

    while [ ${#queue[@]} -gt 0 ] || [ ${#PID_GPU[@]} -gt 0 ]; do
        # 1) Reap any finished jobs, release their GPUs
        local pid rc gpu name
        for pid in "${!PID_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                rc=0
                wait "$pid" || rc=$?
                gpu="${PID_GPU[$pid]}"
                name="${PID_NAME[$pid]}"
                if [ $rc -eq 0 ]; then
                    echo "[DONE] ${name} (GPU ${gpu})"
                    [ "$json_suffix" = "_activation_stats.json" ] && NEWLY_FULL+=("$name") || true
                    [ "$json_suffix" = "_top5_stats.json" ]       && NEWLY_TOP5+=("$name") || true
                else
                    echo "[FAIL] ${name} (GPU ${gpu}) — check log"
                fi
                # Remove gpu from USED_GPUS
                USED_GPUS=" $USED_GPUS "
                USED_GPUS="${USED_GPUS// $gpu / }"
                USED_GPUS=$(echo "$USED_GPUS" | xargs)
                unset "PID_GPU[$pid]"
                unset "PID_NAME[$pid]"
            fi
        done

        # 2) Launch as many as free GPUs allow
        local launched_this_pass=0
        while [ ${#queue[@]} -gt 0 ]; do
            local gpu_idx
            gpu_idx=$(get_free_gpu 1000 "$USED_GPUS")
            [ -z "$gpu_idx" ] && break   # no free card; wait and reap again

            entry="${queue[0]}"
            queue=("${queue[@]:1}")
            IFS='|' read -r series model_name model_path data_path bs sc <<< "$entry"
            local output_dir="${RESULTS_DIR}/${series}"
            mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"

            echo "[LAUNCH] ${model_name} → GPU ${gpu_idx}"
            CUDA_VISIBLE_DEVICES=${gpu_idx} python "${analyze_script}" \
                --model_path "${model_path}" \
                --data_path "${data_path}" \
                --output_dir "${output_dir}" \
                --max_samples ${MAX_SAMPLES} \
                --max_seq_len ${MAX_SEQ_LEN} \
                --batch_size ${bs} \
                --gpu_id 0 \
                > "${output_dir}/logs/${model_name}${log_suffix}" 2>&1 &
            local new_pid=$!
            PID_GPU[$new_pid]=$gpu_idx
            PID_NAME[$new_pid]="$model_name"
            USED_GPUS=$(echo "$USED_GPUS $gpu_idx" | xargs)
            launched_this_pass=$((launched_this_pass + 1))

            # Brief pause so nvidia-smi updates before we poll again.
            # This is only needed if the queue has more items than free GPUs,
            # but costs nothing in the common case.
            sleep 2
        done

        # 3) If nothing launched this pass, wait for a job to finish
        if [ $launched_this_pass -eq 0 ] && [ ${#PID_GPU[@]} -gt 0 ]; then
            sleep 5
        fi
    done
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

        if [ ! -d "$model_path" ]; then
            echo "[SKIP] ${model_name} — model path not found: ${model_path}"
            continue
        fi

        if [ ! -f "$data_path" ]; then
            echo "[SKIP] ${model_name} — data path not found: ${data_path}"
            continue
        fi

        GPU_IDX=""
        while [ -z "$GPU_IDX" ]; do
            GPU_IDX=$(get_free_gpu 1000)
            [ -z "$GPU_IDX" ] && { echo "[WAIT] ${model_name} — no free GPU, retrying in 10 s"; sleep 10; }
        done
        echo "[RUN] ${model_name} → GPU ${GPU_IDX}"
        mkdir -p "${output_dir}/json" "${output_dir}/logs" "${output_dir}/plots"
        if CUDA_VISIBLE_DEVICES=${GPU_IDX} python "${analyze_script}" \
                --model_path "${model_path}" \
                --data_path "${data_path}" \
                --output_dir "${output_dir}" \
                --max_samples ${MAX_SAMPLES} \
                --max_seq_len ${MAX_SEQ_LEN} \
                --batch_size ${bs} \
                --gpu_id 0 \
                > "${output_dir}/logs/${model_name}${log_suffix}" 2>&1; then
            echo "[DONE] ${model_name}"
            [ "$json_suffix" = "_activation_stats.json" ] && NEWLY_FULL+=("$model_name") || true
            [ "$json_suffix" = "_top5_stats.json" ]       && NEWLY_TOP5+=("$model_name") || true
        else
            echo "[FAIL] ${model_name} — check log for details"
        fi
    done
}

# ─── stop: terminate all processes launched by this pipeline ───────────────
# Only matches processes whose cmdline is:
#   - python  <SCRIPT_DIR>/<one of our scripts>.py ...
#   - bash    <SCRIPT_DIR>/run.sh ...
# This avoids killing unrelated processes that merely reference the directory
# (editors, log tailers, nvitop, other projects' run.sh, etc.).
# Kills the whole process tree (children first via pkill -P, then parents).
if [ "${1:-}" = "stop" ]; then
    echo "Stopping pipeline processes under: ${SCRIPT_DIR}"
    MY_PID=$$
    MY_PPID=$PPID

    # Scripts we own; extend here if new entry points are added.
    OWN_SCRIPTS=(
        analyze_model.py
        convert_dataset.py
        build_eval_diverse.py
        plot_activations.py
        plot_top5.py
        scan_top5.py
        verify_existence.py
    )

    # Build regex: python[3]? <SCRIPT_DIR>/(a|b|c).py
    SCRIPT_RE="$(IFS='|'; echo "${OWN_SCRIPTS[*]}")"
    # Escape regex metacharacters in SCRIPT_DIR (dots are the main concern)
    DIR_RE="$(printf '%s' "${SCRIPT_DIR}" | sed 's/[].[^$*\\/]/\\&/g')"
    PY_PATTERN="^python[0-9.]* ${DIR_RE}/(${SCRIPT_RE})( |$)"

    mapfile -t PY_PIDS < <(
        pgrep -af "${PY_PATTERN}" 2>/dev/null \
            | awk -v self="$MY_PID" -v parent="$MY_PPID" \
                  '$1 != self && $1 != parent {print $1}'
    )
    # For bash run.sh we can't rely on absolute path alone (users often run
    # `bash run.sh` from this directory), so resolve each candidate's script
    # path against its /proc/<pid>/cwd and keep only those that resolve into
    # this directory.
    SH_PIDS=()
    while read -r pid cmd; do
        [ -z "$pid" ] && continue
        [ "$pid" = "$MY_PID" ] || [ "$pid" = "$MY_PPID" ] && continue
        # Extract the script argument (first non-flag arg after bash)
        script_arg=$(awk '{for(i=2;i<=NF;i++){if($i !~ /^-/){print $i; exit}}}' <<<"$cmd")
        [ -z "$script_arg" ] && continue
        case "$script_arg" in
            /*) abs="$script_arg" ;;
            *)  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
                abs="${cwd}/${script_arg}" ;;
        esac
        # Normalize and compare
        abs_real=$(readlink -f "$abs" 2>/dev/null) || continue
        [ "$abs_real" = "${SCRIPT_DIR}/run.sh" ] && SH_PIDS+=("$pid")
    done < <(pgrep -af 'bash .*run\.sh' 2>/dev/null)

    ALL_PIDS=("${SH_PIDS[@]}" "${PY_PIDS[@]}")

    if [ ${#ALL_PIDS[@]} -eq 0 ]; then
        echo "No related processes found."
        exit 0
    fi

    echo "Matched PIDs (will be terminated):"
    for pid in "${ALL_PIDS[@]}"; do
        ps -p "$pid" -o pid=,cmd= 2>/dev/null || true
    done

    # TERM children first, then parents; SIGKILL survivors after grace period.
    for pid in "${ALL_PIDS[@]}"; do
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill  -TERM     "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill  -KILL     "$pid" 2>/dev/null || true
        fi
    done

    echo "Done."
    exit 0
fi

# ─── Parse arguments ────────────────────────────────────────────────────────
PLOTS="both"          # which plot families to emit: full | top5 | both
DO_CONVERT=false
DO_ANALYZE=false
DO_PLOT=false
EXPLICIT_STAGE=false
PREV_ARG=""

for arg in "$@"; do
    case $arg in
        --plots)   :;;  # value handled below
        full|top5|both)
            if [ "${PREV_ARG:-}" = "--plots" ]; then
                PLOTS="$arg"
            else
                echo "Unknown arg: $arg (did you forget --plots?)"; exit 1
            fi
            ;;
        --convert) DO_CONVERT=true; EXPLICIT_STAGE=true ;;
        --analyze) DO_ANALYZE=true; EXPLICIT_STAGE=true ;;
        --plot)    DO_PLOT=true; EXPLICIT_STAGE=true ;;
        *)
            echo "Unknown arg: $arg"; exit 1
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

# ─── Step 1: Convert datasets ───────────────────────────────────────────────
if $DO_CONVERT; then
    echo "============================================================"
    echo "Step 1: Converting datasets to target encodings"
    echo "============================================================"
    python "${SCRIPT_DIR}/convert_dataset.py" --all
    echo ""
fi

# ─── Step 2: Analyze (single pass produces both JSONs) ──────────────────────
if $DO_ANALYZE; then
    echo "============================================================"
    echo "Running activation analysis (analyze_model.py)"
    echo "============================================================"
    SCRIPT="${SCRIPT_DIR}/analyze_model.py"
    JSUFFIX="_activation_stats.json"
    LSUFFIX=".log"
    run_phase "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
    run_phase_sequential "$SCRIPT" "$JSUFFIX" "$LSUFFIX"
    echo ""
fi

# ─── Step 3: Generate plots ─────────────────────────────────────────────────
if $DO_PLOT; then
    if [ "$PLOTS" = "full" ] || [ "$PLOTS" = "both" ]; then
        echo "============================================================"
        echo "Generating activation distribution plots"
        echo "============================================================"
        python "${SCRIPT_DIR}/plot_activations.py" --all --results_dir "${RESULTS_DIR}"
        echo ""
    fi

    if [ "$PLOTS" = "top5" ] || [ "$PLOTS" = "both" ]; then
        echo "============================================================"
        echo "Generating top-5 activation plots"
        echo "============================================================"
        python "${SCRIPT_DIR}/plot_top5.py" --all --results_dir "${RESULTS_DIR}"
        echo ""
    fi
fi

echo "============================================================"
echo "All done!"
echo "============================================================"
echo ""

# ─── Concise summary ─────────────────────────────────────────────────────
_in_array() { local e; for e in "${@:2}"; do [ "$e" = "$1" ] && return 0; done; return 1; }

NEW_LINES=(); PREV_LINES=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r series model_name model_path data_path bs sc <<< "$entry"
    jdir="${RESULTS_DIR}/${series}/json"
    tags=""
    [ -f "${jdir}/${model_name}_activation_stats.json" ] && tags="${tags} [full]"
    [ -f "${jdir}/${model_name}_top5_stats.json" ]       && tags="${tags} [top5]"
    [ -z "$tags" ] && continue
    if _in_array "$model_name" "${NEWLY_FULL[@]}" || _in_array "$model_name" "${NEWLY_TOP5[@]}"; then
        NEW_LINES+=("  ${model_name}${tags}")
    else
        PREV_LINES+=("  ${model_name}${tags}")
    fi
done

if [ ${#NEW_LINES[@]} -gt 0 ]; then
    echo "Analyzed this run:"
    for l in "${NEW_LINES[@]}"; do echo "$l"; done
    echo ""
fi
if [ ${#PREV_LINES[@]} -gt 0 ]; then
    echo "Previously analyzed:"
    for l in "${PREV_LINES[@]}"; do echo "$l"; done
    echo ""
fi

echo "Generated plots:"
find "${RESULTS_DIR}" -name "*.png" 2>/dev/null \
    | sort | while read -r f; do
        ts=$(date -r "$f" "+%m-%d %H:%M" 2>/dev/null || echo "??")
        echo "  [${ts}] $(realpath --relative-to="${RESULTS_DIR}" "$f")"
    done || true
