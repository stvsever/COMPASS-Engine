#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 4: Run a Batch (Subset of Participants)
# =============================================================================
#
# PURPOSE: Runs a subset of participants through COMPASS pipeline using batch_run.py.
# Run AFTER validating with 03_submit_single.sh.
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#   bash hpc/05_submit_batch.sh
#   # auto-submits from login node
#
# TIMING: ~10-30 min per participant × 10 participants ≈ 2-5 hours total.
# Time limit set to 10 hours for safety.
#
# =============================================================================

#SBATCH --job-name=compass_batch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=10:00:00
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

MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-14B-AWQ"
EMBEDDING_MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-Embedding-8B"

# Tunables (override via env if needed)
: "${MAX_TOKENS:=32768}"
: "${GPU_MEM_UTIL:=0.95}"
: "${MAX_AGENT_INPUT:=24000}"
: "${MAX_AGENT_OUTPUT:=8000}"
: "${MAX_TOOL_INPUT:=16000}"
: "${MAX_TOOL_OUTPUT:=8000}"
: "${LOCAL_ENGINE:=vllm}"            # vllm|transformers|auto
: "${LOCAL_DTYPE:=auto}"
: "${LOCAL_QUANT:=awq_marlin}"
: "${LOCAL_ENFORCE_EAGER:=1}"        # 1=on, 0=off
: "${LOCAL_ATTN:=auto}"

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
        --time=10:00:00 \
        --chdir="${PROJECT_DIR}" \
        "$0")"

    echo "✓ Batch job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  Monitor:"
    echo "    tail -f ${LOG_DIR}/compass_batch_${JOB_ID}.out"
    echo ""
    echo "  Errors (do NOT run as bash):"
    echo "    cat ${LOG_DIR}/compass_batch_${JOB_ID}.err"
    echo ""
    echo "  Queue:"
    echo "    squeue -u $(whoami)"
    echo ""
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# FROM HERE: Running on a COMPUTE node (GPU allocated)
# ═════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

# ─── Pre-flight Checks ───────────────────────────────────────────────────
echo "============================================="
echo " COMPASS HPC — Batch Run (v2)"
echo "============================================="
echo ""
echo "SCRIPT_PATH:  $0"
echo "SCRIPT_SHA:   $(sha256sum "$0" | awk '{print $1}')"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "PWD:          $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time Limit:   10 hours"
echo ""
echo "Container:    ${CONTAINER_IMAGE}"
echo "Venv:         ${VENV_DIR}"
echo "Model:        ${MODEL_NAME}"
echo "Embed model:  ${EMBEDDING_MODEL_NAME}"
echo "Runtime:      local/${LOCAL_ENGINE} (quant=${LOCAL_QUANT}, eager=${LOCAL_ENFORCE_EAGER})"
echo "Context:      ${MAX_TOKENS} tokens"
echo "Agent budget: in=${MAX_AGENT_INPUT}, out=${MAX_AGENT_OUTPUT}"
echo "Tool budget:  in=${MAX_TOOL_INPUT}, out=${MAX_TOOL_OUTPUT}"
echo "Data:         ${DATA_DIR}"
echo ""

# Validate prerequisites
PREREQ_ERROR=0
for CHECK_PATH in "${CONTAINER_IMAGE}" "${VENV_DIR}" "${MODEL_NAME}" "${EMBEDDING_MODEL_NAME}" "${PROJECT_DIR}/main.py" "${PROJECT_DIR}/utils/batch_run.py"; do
    if [ ! -e "${CHECK_PATH}" ]; then
        echo "✗ ERROR: ${CHECK_PATH} not found!"
        PREREQ_ERROR=1
    fi
done

if ! command -v apptainer >/dev/null 2>&1; then
    echo "✗ ERROR: apptainer not found in PATH on this node."
    PREREQ_ERROR=1
fi

if [ ! -x "${VENV_DIR}/bin/python3" ]; then
    echo "✗ ERROR: ${VENV_DIR}/bin/python3 not found/executable."
    PREREQ_ERROR=1
fi

if [ ${PREREQ_ERROR} -ne 0 ]; then
    echo ""
    echo "  Fix: Run hpc/02_setup_environment.sh and hpc/03_download_models.sh first."
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "⚠ WARNING: Data directory not found at ${DATA_DIR}"
    echo "  batch_run.py may fail if DATA_ROOT is not set correctly."
fi

# ─── GPU Information ──────────────────────────────────────────────────────
echo "─── GPU Information ─────────────────────────────────────"
nvidia-smi 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# ─── Detect model max length and clamp ─────────────────────────────────────
MODEL_CFG="${MODEL_NAME}/config.json"
TOKENIZER_CFG="${MODEL_NAME}/tokenizer_config.json"
DETECTED_MAX=""

if [[ -f "${MODEL_CFG}" || -f "${TOKENIZER_CFG}" ]]; then
    DETECTED_MAX="$(apptainer exec "${CONTAINER_IMAGE}" python3 - "${MODEL_CFG}" "${TOKENIZER_CFG}" <<'PY'
import json, os, sys
paths = [p for p in sys.argv[1:] if p and os.path.isfile(p)]
keys = [
    "max_position_embeddings",
    "max_sequence_length",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "model_max_length",
]
vals = []
for p in paths:
    try:
        cfg = json.load(open(p, "r"))
    except Exception:
        continue
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            vals.append(v)
print(max(vals) if vals else "")
PY
)"
fi

if [[ -n "${DETECTED_MAX}" ]]; then
    echo "✓ Detected model/tokenizer limit: ${DETECTED_MAX}"
    if (( MAX_TOKENS > DETECTED_MAX )); then
        echo "⚠  Clamping MAX_TOKENS from ${MAX_TOKENS} -> ${DETECTED_MAX}"
        MAX_TOKENS="${DETECTED_MAX}"
    fi
else
    echo "⚠  Could not detect model max length; continuing with MAX_TOKENS=${MAX_TOKENS}"
fi
echo ""

# ─── Run Batch ─────────────────────────────────────────────────────────────
echo "─── Starting COMPASS Batch Pipeline ─────────────────────"
echo "  Start time: $(date)"
echo ""
START_TIME=${SECONDS}

mkdir -p "${MODELS_DIR}/hf_cache"

# Apptainer is at /usr/bin/apptainer on compute nodes (no module load needed)
set +e
apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${MODELS_DIR}:${MODELS_DIR}" \
    --bind "${HOME}:${HOME}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env HF_HOME="${MODELS_DIR}/hf_cache" \
    --env TRANSFORMERS_CACHE="${MODELS_DIR}/hf_cache" \
    --env EMBEDDING_MODEL="${EMBEDDING_MODEL_NAME}" \
    --env DATA_ROOT="${DATA_DIR}" \
    --env PYTHONUNBUFFERED="1" \
    "${CONTAINER_IMAGE}" \
    bash -lc "
        set -euo pipefail
        source ${VENV_DIR}/bin/activate
        cd ${PROJECT_DIR}

        # Triton on some HPC images expects libcuda.so (not only libcuda.so.1).
        if [[ -f '/usr/local/cuda/compat/lib/libcuda.so.1' ]]; then
            export TRITON_LIBCUDA_PATH=\"\${HOME}/.cache/triton_libcuda\"
            mkdir -p \"\${TRITON_LIBCUDA_PATH}\"
            ln -sf '/usr/local/cuda/compat/lib/libcuda.so.1' \"\${TRITON_LIBCUDA_PATH}/libcuda.so.1\"
            ln -sf '/usr/local/cuda/compat/lib/libcuda.so.1' \"\${TRITON_LIBCUDA_PATH}/libcuda.so\"
            export LD_LIBRARY_PATH=\"\${TRITON_LIBCUDA_PATH}:\${LD_LIBRARY_PATH:-}\"
        fi

        echo '--- Runtime Information ---'
        echo 'Python:  \$(python3 --version)'
        echo 'PyTorch: \$(python3 -c \"import torch; print(torch.__version__)\")'
        echo 'CUDA:    \$(python3 -c \"import torch; print(torch.cuda.is_available())\")'
        echo 'vLLM:    \$(python3 -c \"import vllm; print(vllm.__version__)\" 2>/dev/null || echo \"not found\")'
        echo 'DataRoot:' \"${DATA_DIR}\"
        echo ''

        ENFORCE_FLAG=''
        if [[ '${LOCAL_ENFORCE_EAGER}' == '1' ]]; then
            ENFORCE_FLAG='--local_enforce_eager'
        fi

        echo 'Batch mode: sequential participant processing on single GPU'
        echo ''

        # Run batch pipeline with detailed per-participant logs
        python3 utils/batch_run.py \\
            --backend local \\
            --model '${MODEL_NAME}' \\
            --max_tokens ${MAX_TOKENS} \\
            --max_agent_input ${MAX_AGENT_INPUT} \\
            --max_agent_output ${MAX_AGENT_OUTPUT} \\
            --max_tool_input ${MAX_TOOL_INPUT} \\
            --max_tool_output ${MAX_TOOL_OUTPUT} \\
            --local_engine '${LOCAL_ENGINE}' \\
            --local_dtype '${LOCAL_DTYPE}' \\
            --local_quant '${LOCAL_QUANT}' \\
            --local_gpu_mem_util ${GPU_MEM_UTIL} \\
            --local_max_model_len ${MAX_TOKENS} \\
            --local_attn '${LOCAL_ATTN}' \\
            --local_trust_remote_code \\
            \${ENFORCE_FLAG}
    "

EXIT_CODE=$?
set -e
ELAPSED=$((SECONDS - START_TIME))

# ─── Post-run Summary ────────────────────────────────────────────────────
echo ""
echo "============================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo " ✓ COMPASS batch completed successfully!"
    echo "   Results: ${PROJECT_DIR}/../results/"
else
    echo " ✗ COMPASS batch exited with code ${EXIT_CODE}"
    echo "   Check:"
    echo "     ${LOG_DIR}/compass_batch_${SLURM_JOB_ID:-unknown}.out"
    echo "     ${LOG_DIR}/compass_batch_${SLURM_JOB_ID:-unknown}.err"
    echo ""
    echo "   Tip: use 'cat' or 'tail', don't run the .err as a script."
fi
echo "============================================="
echo "End time:  $(date)"
echo "Wall time: ${ELAPSED}s ($((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m)"
echo ""

exit ${EXIT_CODE}
