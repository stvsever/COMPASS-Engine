#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 5: Multi-Group Phenotype Validation Batch
# =============================================================================
#
# PURPOSE: Runs balanced cohorts across multiple annotated phenotype
#          groups through the COMPASS pipeline, with crash-safe per-participant
#          result persistence and optional post-hoc analysis.
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#
#   # 1. Default: 5 disorder groups × 40 (20 cases + 20 controls) = 200 total
#   bash hpc/05_submit_batch.sh
#
#   # 2. Custom group size (e.g., 10 cases + 10 controls per group = 100 total):
#   PER_GROUP_SIZE=20 bash hpc/05_submit_batch.sh
#
#   # 3. Subset of disorder groups:
#   DISORDER_GROUPS="MAJOR_DEPRESSIVE_DISORDER,ANXIETY_DISORDERS" bash hpc/05_submit_batch.sh
#
#   # 4. Run ALL participants from the targets file:
#   PER_GROUP_SIZE=ALL bash hpc/05_submit_batch.sh
#
#   # 5. Skip post-hoc analysis:
#   RUN_ANALYSIS=0 bash hpc/05_submit_batch.sh
#
# =============================================================================

#SBATCH --job-name=compass_batch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/compass_batch_%j.out
#SBATCH --error=logs/compass_batch_%j.err

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_system"
LOG_DIR="${PROJECT_DIR}/logs"

CONTAINER_IMAGE="${HOME}/compass_containers/pytorch_24.01.sif"
VENV_DIR="${HOME}/compass_venv"
MODELS_DIR="${HOME}/compass_models"

DATA_DIR="${PROJECT_DIR}/../data/__FEATURES__/HPC_data"
TARGETS_FILE="${PROJECT_DIR}/../data/__TARGETS__/cases_controls_with_specific_subtypes.txt"
RESULTS_DIR="${PROJECT_DIR}/../results"

MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-14B-AWQ"
EMBEDDING_MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-Embedding-8B"

# Tunables (override via env if needed)
: "${MAX_TOKENS:=60000}"
: "${GPU_MEM_UTIL:=0.95}"
: "${MAX_AGENT_INPUT:=auto}"
: "${MAX_AGENT_OUTPUT:=16000}"
: "${MAX_TOOL_INPUT:=auto}"
: "${MAX_TOOL_OUTPUT:=8000}"
: "${LOCAL_ENGINE:=vllm}"
: "${LOCAL_DTYPE:=auto}"
: "${LOCAL_QUANT:=awq_marlin}"
: "${LOCAL_KV_CACHE_DTYPE:=auto}"
: "${LOCAL_ENFORCE_EAGER:=1}"
: "${LOCAL_ATTN:=auto}"
: "${PREFLIGHT_AUDIT:=1}"

# ─── Multi-Disorder Configuration ──────────────────────────────────────────
# Disorder groups to process (comma-separated). Processed in listed order.
: "${DISORDER_GROUPS:=MAJOR_DEPRESSIVE_DISORDER,ANXIETY_DISORDERS,SUBSTANCE_USE_DISORDERS,SLEEP_WAKE_DISORDERS,BIPOLAR_AND_MANIC_DISORDERS}"

# Per-group size: total participants per disorder group (half cases, half controls).
# Set to "ALL" to use every available participant for each group.
: "${PER_GROUP_SIZE:=40}"

# Comparator label used for binary runs (can be overridden via CONTROL_LABEL).
FIXED_CONTROL="${CONTROL_LABEL:-non-target comparator phenotype profile}"

# Prediction task mode for main.py. This script defaults to binary validation.
# Supported values mirror main.py: binary, multiclass, regression_univariate,
# regression_multivariate, hierarchical.
: "${PREDICTION_TYPE:=binary}"
# Optional mode-specific extras:
: "${CLASS_LABELS:=}"
: "${REGRESSION_OUTPUT:=}"
: "${REGRESSION_OUTPUTS:=}"
: "${TASK_SPEC_FILE:=}"
: "${TASK_SPEC_JSON:=}"
# Optional runtime guidance passed to COMPASS agents/tools
: "${GLOBAL_INSTRUCTION:=}"
: "${ORCHESTRATOR_INSTRUCTION:=}"
: "${EXECUTOR_INSTRUCTION:=}"
: "${TOOLS_INSTRUCTION:=}"
: "${INTEGRATOR_INSTRUCTION:=}"
: "${PREDICTOR_INSTRUCTION:=}"
: "${CRITIC_INSTRUCTION:=}"
: "${COMMUNICATOR_INSTRUCTION:=}"

# Post-hoc analysis toggle
: "${RUN_ANALYSIS:=1}"

is_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

resolve_token_budgets() {
    local ctx="$1"
    local reserve=2048
    local min_in=4096
    local min_out=1024
    local sync_tool_input=0

    if ! is_int "${MAX_AGENT_OUTPUT}"; then
        echo "⚠  Invalid MAX_AGENT_OUTPUT='${MAX_AGENT_OUTPUT}' -> using 16000"
        MAX_AGENT_OUTPUT=16000
    fi
    if ! is_int "${MAX_TOOL_OUTPUT}"; then
        echo "⚠  Invalid MAX_TOOL_OUTPUT='${MAX_TOOL_OUTPUT}' -> using 8000"
        MAX_TOOL_OUTPUT=8000
    fi

    local max_agent_out=$((ctx - reserve))
    if (( max_agent_out < min_out )); then
        max_agent_out="${min_out}"
    fi
    if (( MAX_AGENT_OUTPUT > max_agent_out )); then
        echo "⚠  Clamping MAX_AGENT_OUTPUT ${MAX_AGENT_OUTPUT} -> ${max_agent_out} (context=${ctx})"
        MAX_AGENT_OUTPUT="${max_agent_out}"
    fi

    local max_tool_out=$((ctx - reserve))
    if (( max_tool_out < min_out )); then
        max_tool_out="${min_out}"
    fi
    if (( MAX_TOOL_OUTPUT > max_tool_out )); then
        echo "⚠  Clamping MAX_TOOL_OUTPUT ${MAX_TOOL_OUTPUT} -> ${max_tool_out} (context=${ctx})"
        MAX_TOOL_OUTPUT="${max_tool_out}"
    fi

    local agent_in_cap=$((ctx - MAX_AGENT_OUTPUT - reserve))
    if (( agent_in_cap < min_in )); then
        agent_in_cap="${min_in}"
    fi
    local tool_in_cap=$((ctx - MAX_TOOL_OUTPUT - reserve))
    if (( tool_in_cap < min_in )); then
        tool_in_cap="${min_in}"
    fi

    if [[ -z "${MAX_AGENT_INPUT}" || "${MAX_AGENT_INPUT,,}" == "auto" || "${MAX_AGENT_INPUT}" == "0" ]]; then
        MAX_AGENT_INPUT=$((ctx * 75 / 100))
    fi
    if [[ -z "${MAX_TOOL_INPUT}" || "${MAX_TOOL_INPUT,,}" == "auto" || "${MAX_TOOL_INPUT}" == "0" ]]; then
        MAX_TOOL_INPUT="${MAX_AGENT_INPUT}"
        sync_tool_input=1
    fi

    if ! is_int "${MAX_AGENT_INPUT}"; then
        echo "⚠  Invalid MAX_AGENT_INPUT='${MAX_AGENT_INPUT}' -> using auto"
        MAX_AGENT_INPUT=$((ctx * 75 / 100))
    fi
    if ! is_int "${MAX_TOOL_INPUT}"; then
        echo "⚠  Invalid MAX_TOOL_INPUT='${MAX_TOOL_INPUT}' -> using MAX_AGENT_INPUT"
        MAX_TOOL_INPUT="${MAX_AGENT_INPUT}"
        sync_tool_input=1
    fi

    if (( MAX_AGENT_INPUT > agent_in_cap )); then
        echo "⚠  Clamping MAX_AGENT_INPUT ${MAX_AGENT_INPUT} -> ${agent_in_cap} (context=${ctx}, agent_out=${MAX_AGENT_OUTPUT})"
        MAX_AGENT_INPUT="${agent_in_cap}"
    fi
    if (( MAX_TOOL_INPUT > tool_in_cap )); then
        echo "⚠  Clamping MAX_TOOL_INPUT ${MAX_TOOL_INPUT} -> ${tool_in_cap} (context=${ctx}, tool_out=${MAX_TOOL_OUTPUT})"
        MAX_TOOL_INPUT="${tool_in_cap}"
    fi

    if (( MAX_AGENT_INPUT < min_in )); then
        MAX_AGENT_INPUT="${min_in}"
    fi
    if (( MAX_TOOL_INPUT < min_in )); then
        MAX_TOOL_INPUT="${min_in}"
    fi

    if (( sync_tool_input == 1 )); then
        MAX_TOOL_INPUT="${MAX_AGENT_INPUT}"
        if (( MAX_TOOL_INPUT > tool_in_cap )); then
            MAX_TOOL_INPUT="${tool_in_cap}"
        fi
        if (( MAX_TOOL_INPUT < min_in )); then
            MAX_TOOL_INPUT="${min_in}"
        fi
    fi
}

# ─── Auto-Submit to Compute Node ───────────────────────────────────────────
CURRENT_HOST="$(hostname)"
if [[ "${CURRENT_HOST}" == login* ]]; then
    mkdir -p "${LOG_DIR}"
    echo "⚠  Login node detected. Apptainer is only on compute nodes."
    echo "   Auto-submitting this script as a Slurm job..."
    echo ""

    JOB_ID="$(sbatch --parsable \
        --job-name="compass_batch" \
        --output="${LOG_DIR}/compass_batch_%j.out" \
        --error="${LOG_DIR}/compass_batch_%j.err" \
        --partition=main \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=64G \
        --gres=gpu:l40s:1 \
        --time=168:00:00 \
        --chdir="${PROJECT_DIR}" \
        --export=ALL \
        "$0")"

    echo "✓ Batch job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  Monitor:"
    echo "    tail -f ${LOG_DIR}/compass_batch_${JOB_ID}.out"
    echo ""
    echo "  Errors:"
    echo "    cat ${LOG_DIR}/compass_batch_${JOB_ID}.err"
    echo ""
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# FROM HERE: Running on a COMPUTE node (GPU allocated)
# ═════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

# Results directory structure
PARTICIPANT_RUNS_DIR="${RESULTS_DIR}/participant_runs"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"
MATRIX_DIR="${ANALYSIS_DIR}/binary_confusion_matrix"
DETAILS_DIR="${ANALYSIS_DIR}/details"

mkdir -p "${PARTICIPANT_RUNS_DIR}"
mkdir -p "${MATRIX_DIR}"
mkdir -p "${DETAILS_DIR}"

echo "============================================="
echo " COMPASS HPC — Multi-Disorder Batch Run"
echo "============================================="
echo ""
echo "Date:           $(date)"
echo "Target File:    ${TARGETS_FILE}"
echo "Disorder Groups:${DISORDER_GROUPS}"
echo "Per-Group Size: ${PER_GROUP_SIZE}"
echo "Prediction Type:${PREDICTION_TYPE}"
echo "Regression output: ${REGRESSION_OUTPUT:-<none>}"
echo "Regression outputs: ${REGRESSION_OUTPUTS:-<none>}"
echo "Requested ctx:  ${MAX_TOKENS} tokens"
echo "Budget request: agent(in=${MAX_AGENT_INPUT}, out=${MAX_AGENT_OUTPUT}) tool(in=${MAX_TOOL_INPUT}, out=${MAX_TOOL_OUTPUT})"
echo "Results Dir:    ${RESULTS_DIR}"
echo "Run Analysis:   ${RUN_ANALYSIS}"
echo ""

# ─── Validate Targets File ─────────────────────────────────────────────────
if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "✗ ERROR: Target file not found at ${TARGETS_FILE}"
    exit 1
fi

# ─── Build Global Participant Queue (per-disorder balanced) ────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Building balanced participant queue"
echo "═══════════════════════════════════════════════════════════"

TMP_QUEUE=$(mktemp)
TMP_DISORDER_MAP=$(mktemp)  # Maps EID -> DISORDER_GROUP
BATCH_MANIFEST=$(mktemp)

cleanup_tmp_files() {
    rm -f "${TMP_QUEUE}" "${TMP_DISORDER_MAP}" "${BATCH_MANIFEST}"
}
trap cleanup_tmp_files EXIT

TOTAL_QUEUED=0
IFS=',' read -ra DISORDER_ARRAY <<< "${DISORDER_GROUPS}"

for DISORDER in "${DISORDER_ARRAY[@]}"; do
    DISORDER=$(echo "${DISORDER}" | xargs)  # Trim whitespace
    echo ""
    echo "─── ${DISORDER} ───"

    CASE_TMP=$(mktemp)
    CTRL_TMP=$(mktemp)

    # Extract CASE and CONTROL IDs for this disorder
    awk -v d="${DISORDER}" 'BEGIN{IGNORECASE=1}
         $0 ~ d && /CASE/ {print $1}' "${TARGETS_FILE}" > "${CASE_TMP}"
    awk -v d="${DISORDER}" 'BEGIN{IGNORECASE=1}
         $0 ~ d && /CONTROL/ {print $1}' "${TARGETS_FILE}" > "${CTRL_TMP}"

    AVAILABLE_CASES=$(wc -l < "${CASE_TMP}" | tr -d ' ')
    AVAILABLE_CONTROLS=$(wc -l < "${CTRL_TMP}" | tr -d ' ')
    echo "  Available: ${AVAILABLE_CASES} Cases / ${AVAILABLE_CONTROLS} Controls"

    if (( AVAILABLE_CASES == 0 || AVAILABLE_CONTROLS == 0 )); then
        echo "  ⚠ Skipping ${DISORDER}: no cases or controls found."
        rm -f "${CASE_TMP}" "${CTRL_TMP}"
        continue
    fi

    if [[ "${PER_GROUP_SIZE}" == "ALL" ]]; then
        TAKE_CASES=${AVAILABLE_CASES}
        TAKE_CONTROLS=${AVAILABLE_CONTROLS}
    else
        if ! is_int "${PER_GROUP_SIZE}"; then
            echo "✗ ERROR: PER_GROUP_SIZE must be an integer or ALL"
            exit 1
        fi
        HALF=$((PER_GROUP_SIZE / 2))
        TAKE_CASES=${HALF}
        TAKE_CONTROLS=${HALF}
        if (( AVAILABLE_CASES < HALF )); then
            echo "  ⚠ Only ${AVAILABLE_CASES} cases available (wanted ${HALF})"
            TAKE_CASES=${AVAILABLE_CASES}
        fi
        if (( AVAILABLE_CONTROLS < HALF )); then
            echo "  ⚠ Only ${AVAILABLE_CONTROLS} controls available (wanted ${HALF})"
            TAKE_CONTROLS=${AVAILABLE_CONTROLS}
        fi
    fi

    echo "  Selected: ${TAKE_CASES} Cases / ${TAKE_CONTROLS} Controls"

    # Filter out participants whose data directory does not exist
    FILTERED_CASE_TMP=$(mktemp)
    FILTERED_CTRL_TMP=$(mktemp)

    while read -r eid; do
        if [[ -d "${DATA_DIR}/participant_ID${eid}" ]]; then
            echo "${eid}" >> "${FILTERED_CASE_TMP}"
        fi
    done < "${CASE_TMP}"

    while read -r eid; do
        if [[ -d "${DATA_DIR}/participant_ID${eid}" ]]; then
            echo "${eid}" >> "${FILTERED_CTRL_TMP}"
        fi
    done < "${CTRL_TMP}"

    ACTUAL_CASES=$(wc -l < "${FILTERED_CASE_TMP}" | tr -d ' ')
    ACTUAL_CONTROLS=$(wc -l < "${FILTERED_CTRL_TMP}" | tr -d ' ')

    if (( ACTUAL_CASES < TAKE_CASES )); then
        echo "  ⚠ Only ${ACTUAL_CASES} cases have data on disk (wanted ${TAKE_CASES})"
        TAKE_CASES=${ACTUAL_CASES}
    fi
    if (( ACTUAL_CONTROLS < TAKE_CONTROLS )); then
        echo "  ⚠ Only ${ACTUAL_CONTROLS} controls have data on disk (wanted ${TAKE_CONTROLS})"
        TAKE_CONTROLS=${ACTUAL_CONTROLS}
    fi

    head -n "${TAKE_CASES}" "${FILTERED_CASE_TMP}" >> "${TMP_QUEUE}"
    head -n "${TAKE_CONTROLS}" "${FILTERED_CTRL_TMP}" >> "${TMP_QUEUE}"

    # Record disorder mapping
    head -n "${TAKE_CASES}" "${FILTERED_CASE_TMP}" | while read -r eid; do
        echo "${eid}|${DISORDER}" >> "${TMP_DISORDER_MAP}"
    done
    head -n "${TAKE_CONTROLS}" "${FILTERED_CTRL_TMP}" | while read -r eid; do
        echo "${eid}|${DISORDER}" >> "${TMP_DISORDER_MAP}"
    done

    GROUP_TOTAL=$((TAKE_CASES + TAKE_CONTROLS))
    TOTAL_QUEUED=$((TOTAL_QUEUED + GROUP_TOTAL))
    echo "  ✓ Queued ${GROUP_TOTAL} for ${DISORDER}"

    rm -f "${CASE_TMP}" "${CTRL_TMP}" "${FILTERED_CASE_TMP}" "${FILTERED_CTRL_TMP}"
done

QUEUE_SIZE=$(wc -l < "${TMP_QUEUE}" | tr -d ' ')
if (( QUEUE_SIZE == 0 )); then
    echo "✗ ERROR: Participant queue is empty after selection."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Total queued: ${QUEUE_SIZE} participants"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Detect Model Max Context Length ───────────────────────────────────────
echo "Detecting model max context length..."
MODEL_CFG="${MODEL_NAME}/config.json"
TOKENIZER_CFG="${MODEL_NAME}/tokenizer_config.json"
DETECTED_MAX=""
if [[ -f "${MODEL_CFG}" || -f "${TOKENIZER_CFG}" ]]; then
    DETECTED_MAX="$(apptainer exec "${CONTAINER_IMAGE}" python3 - "${MODEL_CFG}" "${TOKENIZER_CFG}" <<'PY'
import json, os, sys
model_cfg = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
tok_cfg = sys.argv[2] if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]) else None

def collect(path, keys):
    vals = []
    if not path:
        return vals
    try:
        cfg = json.load(open(path, "r"))
    except Exception:
        return vals
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            vals.append(v)
    return vals

model_vals = collect(
    model_cfg,
    ["max_position_embeddings", "max_sequence_length", "max_seq_len", "max_seq_length", "seq_length"],
)
tokenizer_vals = collect(
    tok_cfg,
    ["model_max_length", "max_position_embeddings", "max_sequence_length", "max_seq_len", "max_seq_length", "seq_length"],
)

if model_vals:
    print(min(model_vals))
elif tokenizer_vals:
    print(min(tokenizer_vals))
else:
    print("")
PY
)"
fi
if [[ -n "${DETECTED_MAX}" ]]; then
    echo "✓ Detected model/tokenizer limit: ${DETECTED_MAX}"
    if (( MAX_TOKENS > DETECTED_MAX )); then
        echo "⚠  Clamping MAX_TOKENS ${MAX_TOKENS} -> ${DETECTED_MAX}"
        MAX_TOKENS="${DETECTED_MAX}"
    fi
else
    echo "⚠  Could not detect model max length; continuing with MAX_TOKENS=${MAX_TOKENS}"
fi
resolve_token_budgets "${MAX_TOKENS}"
echo "Resolved runtime token profile:"
echo "  Context window: ${MAX_TOKENS}"
echo "  Agent budget:   in=${MAX_AGENT_INPUT}, out=${MAX_AGENT_OUTPUT}"
echo "  Tool budget:    in=${MAX_TOOL_INPUT}, out=${MAX_TOOL_OUTPUT}"
echo ""

# ─── Initialize Batch Manifest ────────────────────────────────────────────
echo "[" > "${BATCH_MANIFEST}"
FIRST_ENTRY=1

# ─── Run Loop (sequential, crash-safe per-participant) ─────────────────────
START_TIME=${SECONDS}
COUNTER=0
SUCCESSES=0
FAILURES=0
SKIPPED=0

while read -r PARTICIPANT_ID; do
    COUNTER=$((COUNTER + 1))

    # Look up disorder group for this participant
    DISORDER_GROUP=$(grep "^${PARTICIPANT_ID}|" "${TMP_DISORDER_MAP}" | head -1 | cut -d'|' -f2)
    if [[ -z "${DISORDER_GROUP}" ]]; then
        DISORDER_GROUP="UNKNOWN"
    fi

    echo "================================================================"
    echo " [${COUNTER}/${QUEUE_SIZE}] Participant ${PARTICIPANT_ID} (${DISORDER_GROUP})"
    echo "================================================================"

    PARTICIPANT_DIR="${DATA_DIR}/participant_ID${PARTICIPANT_ID}"

    # Check if data exists
    if [[ ! -d "${PARTICIPANT_DIR}" ]]; then
        echo "  ⚠ Directory not found: ${PARTICIPANT_DIR} (Skipping)"
        SKIPPED=$((SKIPPED + 1))
        # Manifest entry
        if (( FIRST_ENTRY == 0 )); then echo "," >> "${BATCH_MANIFEST}"; fi
        FIRST_ENTRY=0
        echo "  {\"eid\":\"${PARTICIPANT_ID}\",\"disorder\":\"${DISORDER_GROUP}\",\"status\":\"SKIPPED\",\"reason\":\"data_not_found\"}" >> "${BATCH_MANIFEST}"
        continue
    fi

    # Determine target from file lookup
    FULL_TARGET_LINE=$(grep "^${PARTICIPANT_ID}" "${TARGETS_FILE}" | head -1)

    # Leak Protection: Strip CASE/CONTROL literals and parentheses
    SPECIFIC_TARGET=$(echo "${FULL_TARGET_LINE}" | cut -d'|' -f2- | sed -E 's/\bCASE\b//g; s/\bCONTROL\b//g; s/[()]//g' | xargs)

    echo "  Disorder:       ${DISORDER_GROUP}"
    echo "  Ground truth:   $(echo "${FULL_TARGET_LINE}" | grep -oE "CASE|CONTROL")"
    echo "  Engine Target:  '${SPECIFIC_TARGET}'"
    echo "  Engine Control: '${FIXED_CONTROL}'"

    P_START=${SECONDS}

    # ── Run main.py inside Apptainer ──
    apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${MODELS_DIR}:${MODELS_DIR}" \
    --bind "${HOME}:${HOME}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env HF_HOME="${MODELS_DIR}/hf_cache" \
    --env TRANSFORMERS_CACHE="${MODELS_DIR}/hf_cache" \
    --env EMBEDDING_MODEL="${EMBEDDING_MODEL_NAME}" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env PYTHONUNBUFFERED="1" \
    --env MODEL_NAME="${MODEL_NAME}" \
    --env MAX_TOKENS="${MAX_TOKENS}" \
    --env GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    --env LOCAL_KV_CACHE_DTYPE="${LOCAL_KV_CACHE_DTYPE}" \
    "${CONTAINER_IMAGE}" \
    bash -lc "
        source '${VENV_DIR}/bin/activate'
        cd '${PROJECT_DIR}'

        # CUDA driver shim
        if [[ -f '/usr/local/cuda/compat/lib/libcuda.so.1' ]]; then
            export COMPASS_LIBCUDA_PATH=\"\${HOME}/.cache/compass_libcuda\"
            mkdir -p \"\${COMPASS_LIBCUDA_PATH}\"
            ln -sf '/usr/local/cuda/compat/lib/libcuda.so.1' \"\${COMPASS_LIBCUDA_PATH}/libcuda.so.1\"
            ln -sf '/usr/local/cuda/compat/lib/libcuda.so.1' \"\${COMPASS_LIBCUDA_PATH}/libcuda.so\"
            export TRITON_LIBCUDA_PATH=\"\${COMPASS_LIBCUDA_PATH}\"
            export CUDA_HOME='/usr/local/cuda'
            export CUDA_PATH='/usr/local/cuda'
            export LD_LIBRARY_PATH=\"\${COMPASS_LIBCUDA_PATH}:/usr/local/cuda/compat/lib:\${LD_LIBRARY_PATH:-}\"
            export LIBRARY_PATH=\"\${COMPASS_LIBCUDA_PATH}:/usr/local/cuda/compat/lib:\${LIBRARY_PATH:-}\"
            export LD_PRELOAD=\"\${COMPASS_LIBCUDA_PATH}/libcuda.so\${LD_PRELOAD:+:\${LD_PRELOAD}}\"
        fi
        python3 - <<'PY'
import ctypes
try:
    ctypes.CDLL('libcuda.so')
    print('libcuda.so dynamic load: OK')
except Exception as e:
    print('libcuda.so dynamic load: WARN:', e)
PY

        # Compute Flags
        LOCAL_ENGINE_FLAG='${LOCAL_ENGINE}'
        if [[ \"\${LOCAL_ENGINE_FLAG}\" == 'auto' ]]; then LOCAL_ENGINE_FLAG='vllm'; fi

        EXTRA_FLAGS=''
        if [[ '${LOCAL_ENFORCE_EAGER}' == '1' ]]; then EXTRA_FLAGS='--local_enforce_eager'; fi
        if [[ '${LOCAL_QUANT}' != 'None' ]]; then EXTRA_FLAGS=\"\${EXTRA_FLAGS} --local_quant ${LOCAL_QUANT}\"; fi

        if [[ '${PREFLIGHT_AUDIT}' == '1' ]]; then
            echo '--- Dataflow preflight audit ---'
            python3 main.py \
                '${PARTICIPANT_DIR}' \
                --prediction_type '${PREDICTION_TYPE}' \
                --target_label '${SPECIFIC_TARGET}' \
                --control_label '${FIXED_CONTROL}' \
                ${CLASS_LABELS:+--class_labels '${CLASS_LABELS}'} \
                ${REGRESSION_OUTPUT:+--regression_output '${REGRESSION_OUTPUT}'} \
                ${REGRESSION_OUTPUTS:+--regression_outputs '${REGRESSION_OUTPUTS}'} \
                ${TASK_SPEC_FILE:+--task_spec_file '${TASK_SPEC_FILE}'} \
                ${TASK_SPEC_JSON:+--task_spec_json '${TASK_SPEC_JSON}'} \
                ${GLOBAL_INSTRUCTION:+--global_instruction '${GLOBAL_INSTRUCTION}'} \
                ${ORCHESTRATOR_INSTRUCTION:+--orchestrator_instruction '${ORCHESTRATOR_INSTRUCTION}'} \
                ${EXECUTOR_INSTRUCTION:+--executor_instruction '${EXECUTOR_INSTRUCTION}'} \
                ${TOOLS_INSTRUCTION:+--tools_instruction '${TOOLS_INSTRUCTION}'} \
                ${INTEGRATOR_INSTRUCTION:+--integrator_instruction '${INTEGRATOR_INSTRUCTION}'} \
                ${PREDICTOR_INSTRUCTION:+--predictor_instruction '${PREDICTOR_INSTRUCTION}'} \
                ${CRITIC_INSTRUCTION:+--critic_instruction '${CRITIC_INSTRUCTION}'} \
                ${COMMUNICATOR_INSTRUCTION:+--communicator_instruction '${COMMUNICATOR_INSTRUCTION}'} \
                --backend local \
                --model '${MODEL_NAME}' \
                --max_tokens ${MAX_TOKENS} \
                --local_engine \${LOCAL_ENGINE_FLAG} \
                --local_dtype ${LOCAL_DTYPE} \
                --local_kv_cache_dtype ${LOCAL_KV_CACHE_DTYPE} \
                --local_gpu_mem_util ${GPU_MEM_UTIL} \
                --local_max_model_len ${MAX_TOKENS} \
                --max_agent_input ${MAX_AGENT_INPUT} \
                --max_agent_output ${MAX_AGENT_OUTPUT} \
                --max_tool_input ${MAX_TOOL_INPUT} \
                --max_tool_output ${MAX_TOOL_OUTPUT} \
                --local_trust_remote_code \
                \${EXTRA_FLAGS} \
                --audit \
                --quiet || {
                    echo '✗ Dataflow preflight audit failed for ${PARTICIPANT_ID}'
                    exit 1
                }
            echo '✓ Dataflow preflight audit passed'
            echo ''
        fi

        python3 main.py \
            '${PARTICIPANT_DIR}' \
            --prediction_type '${PREDICTION_TYPE}' \
            --target_label '${SPECIFIC_TARGET}' \
            --control_label '${FIXED_CONTROL}' \
            ${CLASS_LABELS:+--class_labels '${CLASS_LABELS}'} \
            ${REGRESSION_OUTPUT:+--regression_output '${REGRESSION_OUTPUT}'} \
            ${REGRESSION_OUTPUTS:+--regression_outputs '${REGRESSION_OUTPUTS}'} \
            ${TASK_SPEC_FILE:+--task_spec_file '${TASK_SPEC_FILE}'} \
            ${TASK_SPEC_JSON:+--task_spec_json '${TASK_SPEC_JSON}'} \
            ${GLOBAL_INSTRUCTION:+--global_instruction '${GLOBAL_INSTRUCTION}'} \
            ${ORCHESTRATOR_INSTRUCTION:+--orchestrator_instruction '${ORCHESTRATOR_INSTRUCTION}'} \
            ${EXECUTOR_INSTRUCTION:+--executor_instruction '${EXECUTOR_INSTRUCTION}'} \
            ${TOOLS_INSTRUCTION:+--tools_instruction '${TOOLS_INSTRUCTION}'} \
            ${INTEGRATOR_INSTRUCTION:+--integrator_instruction '${INTEGRATOR_INSTRUCTION}'} \
            ${PREDICTOR_INSTRUCTION:+--predictor_instruction '${PREDICTOR_INSTRUCTION}'} \
            ${CRITIC_INSTRUCTION:+--critic_instruction '${CRITIC_INSTRUCTION}'} \
            ${COMMUNICATOR_INSTRUCTION:+--communicator_instruction '${COMMUNICATOR_INSTRUCTION}'} \
            --backend local \
            --model '${MODEL_NAME}' \
            --max_tokens ${MAX_TOKENS} \
            --local_engine \${LOCAL_ENGINE_FLAG} \
            --local_dtype ${LOCAL_DTYPE} \
            --local_kv_cache_dtype ${LOCAL_KV_CACHE_DTYPE} \
            --local_gpu_mem_util ${GPU_MEM_UTIL} \
            --local_max_model_len ${MAX_TOKENS} \
            --max_agent_input ${MAX_AGENT_INPUT} \
            --max_agent_output ${MAX_AGENT_OUTPUT} \
            --max_tool_input ${MAX_TOOL_INPUT} \
            --max_tool_output ${MAX_TOOL_OUTPUT} \
            --local_trust_remote_code \
            \${EXTRA_FLAGS} \
            --detailed_log \
            --quiet
    "
    RUN_EXIT=$?

    P_ELAPSED=$((SECONDS - P_START))

    # ── Crash-safe: copy results immediately ──
    # main.py writes output to ${RESULTS_DIR}/participant_ID{eid}/ by default.
    # We ensure a copy lives in participant_runs/ for analysis.
    SOURCE_OUTPUT="${RESULTS_DIR}/participant_ID${PARTICIPANT_ID}"
    TARGET_OUTPUT="${PARTICIPANT_RUNS_DIR}/participant_ID${PARTICIPANT_ID}"
    if [[ -d "${SOURCE_OUTPUT}" ]]; then
        # Move or copy into the persistent runs directory
        if [[ "${SOURCE_OUTPUT}" != "${TARGET_OUTPUT}" ]]; then
            mkdir -p "${TARGET_OUTPUT}"
            cp -r "${SOURCE_OUTPUT}/"* "${TARGET_OUTPUT}/" 2>/dev/null || true
        fi
    fi

    # Manifest entry
    if (( FIRST_ENTRY == 0 )); then echo "," >> "${BATCH_MANIFEST}"; fi
    FIRST_ENTRY=0

    if [[ ${RUN_EXIT} -ne 0 ]]; then
        echo "  ✗ FAILED (exit ${RUN_EXIT}) — ${P_ELAPSED}s"
        FAILURES=$((FAILURES + 1))
        echo "  {\"eid\":\"${PARTICIPANT_ID}\",\"disorder\":\"${DISORDER_GROUP}\",\"status\":\"FAILED\",\"exit_code\":${RUN_EXIT},\"duration_s\":${P_ELAPSED}}" >> "${BATCH_MANIFEST}"
    else
        echo "  ✓ SUCCESS — ${P_ELAPSED}s"
        SUCCESSES=$((SUCCESSES + 1))
        echo "  {\"eid\":\"${PARTICIPANT_ID}\",\"disorder\":\"${DISORDER_GROUP}\",\"status\":\"SUCCESS\",\"duration_s\":${P_ELAPSED}}" >> "${BATCH_MANIFEST}"
    fi
    echo ""

done < "${TMP_QUEUE}"

# Close manifest JSON
echo "]" >> "${BATCH_MANIFEST}"

# Save manifest to results
cp "${BATCH_MANIFEST}" "${RESULTS_DIR}/batch_manifest.json"

BATCH_ELAPSED=$((SECONDS - START_TIME))

echo "═══════════════════════════════════════════════════════════"
echo " Batch Processing Complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  End time:   $(date)"
echo "  Wall time:  ${BATCH_ELAPSED}s"
echo "  Success:    ${SUCCESSES}"
echo "  Failed:     ${FAILURES}"
echo "  Skipped:    ${SKIPPED}"
echo "  Total:      ${COUNTER}"
echo ""

# ─── Post-Hoc Analysis (if annotated dataset available) ───────────────────
if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo "═══════════════════════════════════════════════════════════"
    echo " Running Post-Hoc Validation Analysis"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    VALIDATION_SCRIPT_DIR="${PROJECT_DIR}/utils/validation/with_annotated_dataset"
    DISORDER_GROUPS_ARG="${DISORDER_GROUPS}"

    if [[ -f "${VALIDATION_SCRIPT_DIR}/compute_confusion_matrix.py" ]]; then
        echo "─── Computing Confusion Matrices ───"
        apptainer exec \
            --nv \
            --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
            --bind "${HOME}:${HOME}" \
            "${CONTAINER_IMAGE}" \
            bash -lc "
                source '${VENV_DIR}/bin/activate'
                cd '${PROJECT_DIR}'
                python3 utils/validation/with_annotated_dataset/compute_confusion_matrix.py \
                    --results_dir '${PARTICIPANT_RUNS_DIR}' \
                    --targets_file '${TARGETS_FILE}' \
                    --output_dir '${MATRIX_DIR}' \
                    --disorder_groups '${DISORDER_GROUPS_ARG}'
            " || echo "  ⚠ Confusion matrix computation failed"
        echo ""
    fi

    if [[ -f "${VALIDATION_SCRIPT_DIR}/detailed_analysis.py" ]]; then
        echo "─── Computing Detailed Analysis ───"
        apptainer exec \
            --nv \
            --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
            --bind "${HOME}:${HOME}" \
            "${CONTAINER_IMAGE}" \
            bash -lc "
                source '${VENV_DIR}/bin/activate'
                cd '${PROJECT_DIR}'
                python3 utils/validation/with_annotated_dataset/detailed_analysis.py \
                    --results_dir '${PARTICIPANT_RUNS_DIR}' \
                    --targets_file '${TARGETS_FILE}' \
                    --output_dir '${DETAILS_DIR}' \
                    --disorder_groups '${DISORDER_GROUPS_ARG}'
            " || echo "  ⚠ Detailed analysis computation failed"
        echo ""
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo " All Done"
echo " Results:  ${RESULTS_DIR}"
echo " Manifest: ${RESULTS_DIR}/batch_manifest.json"
echo "═══════════════════════════════════════════════════════════"
