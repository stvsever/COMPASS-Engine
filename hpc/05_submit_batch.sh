#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 5: Run Clinical Validation Batch (Dynamic Selection)
# =============================================================================
#
# PURPOSE: Runs a specific subset of participants (cases & controls) through the pipeline.
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#   # 1. To run the default balanced subset (10 MDD Cases + 10 MDD Controls ; for pilot test):
#   bash hpc/05_submit_batch.sh
#
#   # 2. To run ALL participants in the file:
#   BATCH_SIZE=ALL bash hpc/05_submit_batch.sh
#
#   # 3. To run a specific custom size:
#   BATCH_SIZE=50 bash hpc/05_submit_batch.sh
#
# =============================================================================

#SBATCH --job-name=compass_batch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
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
: "${PREFLIGHT_AUDIT:=1}"  # 1 = run fast offline dataflow audit before each full run
: "${BATCH_SIZE:=20}"  # Default: 10 Cases + 10 Controls = 20 total. Set to "ALL" for everything.

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

    # Keep tool input aligned with agent input when tool input is auto/invalid.
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
        --time=48:00:00 \
        --chdir="${PROJECT_DIR}" \
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

echo "============================================="
echo " COMPASS HPC — Batch Run (Dynamic Selection)"
echo "============================================="
echo ""
echo "Date:         $(date)"
echo "Target File:  ${TARGETS_FILE}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Requested ctx:${MAX_TOKENS} tokens"
echo "Budget request: agent(in=${MAX_AGENT_INPUT}, out=${MAX_AGENT_OUTPUT}) tool(in=${MAX_TOOL_INPUT}, out=${MAX_TOOL_OUTPUT})"
echo "KV cache dtype request: ${LOCAL_KV_CACHE_DTYPE}"
echo ""

# ─── Dynamic Participant Selection ───────────────────────────────────────────
if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "✗ ERROR: Target file not found at ${TARGETS_FILE}"
    echo "  Ensure you have synced the data directory."
    exit 1
fi

echo "Parsing participants..."
# Create a temporary list of IDs
TMP_LIST=$(mktemp)
CASE_TMP=$(mktemp)
CTRL_TMP=$(mktemp)

cleanup_tmp_files() {
    rm -f "${TMP_LIST}" "${CASE_TMP}" "${CTRL_TMP}"
}
trap cleanup_tmp_files EXIT

if [[ "${BATCH_SIZE}" == "ALL" ]]; then
    echo "  - Mode: ALL participants"
    awk '{print $1}' "${TARGETS_FILE}" > "${TMP_LIST}"
else
    # Validate integer batch size for balanced mode.
    if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
        echo "✗ ERROR: BATCH_SIZE must be an integer or ALL (received: ${BATCH_SIZE})"
        exit 1
    fi
    if (( BATCH_SIZE < 2 )); then
        echo "✗ ERROR: BATCH_SIZE must be >= 2 for balanced selection (received: ${BATCH_SIZE})"
        exit 1
    fi
    if (( BATCH_SIZE % 2 != 0 )); then
        echo "⚠  WARNING: BATCH_SIZE=${BATCH_SIZE} is odd. Using $((BATCH_SIZE - 1)) for balanced split."
        BATCH_SIZE=$((BATCH_SIZE - 1))
    fi

    # Calculate split (half cases, half controls).
    HALF_BATCH=$((BATCH_SIZE / 2))
    echo "  - Mode: Balanced Subset (${HALF_BATCH} Cases / ${HALF_BATCH} Controls)"

    # Extract IDs with relaxed matching to avoid brittle literal-pattern failures.
    # We only require lines to contain MAJOR_DEPRESSIVE_DISORDER plus CASE/CONTROL.
    awk 'BEGIN{IGNORECASE=1}
         /MAJOR_DEPRESSIVE_DISORDER/ && /CASE/ {print $1}' "${TARGETS_FILE}" > "${CASE_TMP}"
    awk 'BEGIN{IGNORECASE=1}
         /MAJOR_DEPRESSIVE_DISORDER/ && /CONTROL/ {print $1}' "${TARGETS_FILE}" > "${CTRL_TMP}"

    AVAILABLE_CASES=$(wc -l < "${CASE_TMP}")
    AVAILABLE_CONTROLS=$(wc -l < "${CTRL_TMP}")
    echo "  - Available candidates: ${AVAILABLE_CASES} Cases / ${AVAILABLE_CONTROLS} Controls"

    if (( AVAILABLE_CASES == 0 || AVAILABLE_CONTROLS == 0 )); then
        echo "✗ ERROR: No selectable CASE/CONTROL rows found for MAJOR_DEPRESSIVE_DISORDER."
        echo "  Check format in: ${TARGETS_FILE}"
        exit 1
    fi

    TAKE_CASES=${HALF_BATCH}
    TAKE_CONTROLS=${HALF_BATCH}
    if (( AVAILABLE_CASES < HALF_BATCH )); then
        echo "⚠  WARNING: Requested ${HALF_BATCH} Cases but only ${AVAILABLE_CASES} available."
        TAKE_CASES=${AVAILABLE_CASES}
    fi
    if (( AVAILABLE_CONTROLS < HALF_BATCH )); then
        echo "⚠  WARNING: Requested ${HALF_BATCH} Controls but only ${AVAILABLE_CONTROLS} available."
        TAKE_CONTROLS=${AVAILABLE_CONTROLS}
    fi

    echo "  - Selecting: ${TAKE_CASES} Cases / ${TAKE_CONTROLS} Controls"
    head -n "${TAKE_CASES}" "${CASE_TMP}" > "${TMP_LIST}"
    head -n "${TAKE_CONTROLS}" "${CTRL_TMP}" >> "${TMP_LIST}"
fi

QUEUE_SIZE=$(wc -l < "${TMP_LIST}")
if (( QUEUE_SIZE == 0 )); then
    echo "✗ ERROR: Participant queue is empty after selection."
    exit 1
fi
echo "  - Queue size: ${QUEUE_SIZE}"
echo ""
cat "${TMP_LIST}"
echo ""

# ─── Run Loop ──────────────────────────────────────────────────────────────
# We run them sequentially in this single job (no parallel sruns to avoid VRAM collision)

# Detect model max length once
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

START_TIME=${SECONDS}
COUNTER=0

while read -r PARTICIPANT_ID; do
    COUNTER=$((COUNTER + 1))
    echo "----------------------------------------------------------------"
    echo " Processing ${COUNTER}/${QUEUE_SIZE}: ${PARTICIPANT_ID}"
    echo "----------------------------------------------------------------"
    
    PARTICIPANT_DIR="${DATA_DIR}/participant_ID${PARTICIPANT_ID}"
    
    # Check if exists
    if [[ ! -d "${PARTICIPANT_DIR}" ]]; then
        echo "  ⚠ Directory not found: ${PARTICIPANT_DIR} (Skipping)"
        continue
    fi

    # Determine target based on file lookup.
    FULL_TARGET_LINE=$(grep "^${PARTICIPANT_ID}" "${TARGETS_FILE}")

    # Leak Protection: Strip CASE/CONTROL literals and parentheses
    # Ensures the engine is blinded to the ground truth label.
    SPECIFIC_TARGET=$(echo "${FULL_TARGET_LINE}" | cut -d'|' -f2- | sed -E 's/\bCASE\b//g; s/\bCONTROL\b//g; s/[()]//g' | xargs)
    
    # HARDCODED CONTROL baseline
    FIXED_CONTROL="possible brain-implicated pathology, but NOT psychiatric"

    echo "  Leaked label:   $(echo "${FULL_TARGET_LINE}" | grep -oE "CASE|CONTROL")" # Log internally in .out
    echo "  Engine Target:  '${SPECIFIC_TARGET}'"
    echo "  Engine Control: '${FIXED_CONTROL}'"
    
    # Run main.py for this participant
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
        
        # CUDA driver shim: some stacks require libcuda.so (not only libcuda.so.1).
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
                --target '${SPECIFIC_TARGET}' \
                --control '${FIXED_CONTROL}' \
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
                    echo '✗ Dataflow preflight audit failed; aborting batch before full LLM run.'
                    exit 1
                }
            echo '✓ Dataflow preflight audit passed'
            echo ''
        fi

        python3 main.py \
            '${PARTICIPANT_DIR}' \
            --target '${SPECIFIC_TARGET}' \
            --control '${FIXED_CONTROL}' \
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
    " || echo "  ✗ Failed processing ${PARTICIPANT_ID}"

done < "${TMP_LIST}"
ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "============================================="
echo " Batch Completed"
echo " End time:  $(date)"
echo " Wall time: ${ELAPSED}s"
echo "============================================="
