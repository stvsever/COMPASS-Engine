#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 3: Run Single Participant (Test Job)
# =============================================================================
# - Tries vLLM first (if LOCAL_ENGINE=auto), falls back to Transformers if vLLM init fails
# - Prints vLLM traceback to STDOUT so it shows up in .out logs
# - Uses DYNAMIC TARGET LOOKUP from file for strict blinding (same as batch logic)
# =============================================================================

#SBATCH --job-name=compass_single
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/compass_single_%j.out
#SBATCH --error=logs/compass_single_%j.err

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_system"
LOG_DIR="${PROJECT_DIR}/logs"

CONTAINER_IMAGE="${HOME}/compass_containers/pytorch_24.01.sif"
VENV_DIR="${HOME}/compass_venv"
MODELS_DIR="${HOME}/compass_models"

MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-14B-AWQ"
EMBEDDING_MODEL_NAME="${MODELS_DIR}/Qwen_Qwen3-Embedding-8B"

# Data Config
DATA_DIR="${PROJECT_DIR}/../data/__FEATURES__/HPC_data"
# EXPLICIT SHARED PATH
TARGETS_FILE="${PROJECT_DIR}/../data/__TARGETS__/cases_controls_with_specific_subtypes.txt"

#
# Example participant ID (placeholder).
# Replace this with a real participant ID present under DATA_DIR (folder: participant_ID<id>),
# or override at submit time:
#   PARTICIPANT_ID=01 bash hpc/04_submit_single.sh
#
: "${PARTICIPANT_ID:=01}"
PARTICIPANT_DIR="${DATA_DIR}/participant_ID${PARTICIPANT_ID}"

# Tunables (override via env if needed)
: "${MAX_TOKENS:=60000}"
: "${GPU_MEM_UTIL:=0.95}"
: "${QUIET:=0}"                  # 1 = pass --quiet
: "${LOCAL_ENGINE:=auto}"         # auto|vllm|transformers
: "${PREFLIGHT_VLLM:=1}"          # 1 = run vLLM preflight if engine allows (recommended)
: "${PREFLIGHT_AUDIT:=1}"         # 1 = run fast offline dataflow audit before full run
: "${MAX_AGENT_INPUT:=auto}"
: "${MAX_AGENT_OUTPUT:=16000}"
: "${MAX_TOOL_INPUT:=auto}"
: "${MAX_TOOL_OUTPUT:=8000}"
: "${LOCAL_KV_CACHE_DTYPE:=auto}"

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
        --job-name="compass_single" \
        --output="${LOG_DIR}/compass_single_%j.out" \
        --error="${LOG_DIR}/compass_single_%j.err" \
        --partition=main \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=64G \
        --gres=gpu:l40s:1 \
        --time=04:00:00 \
        --chdir="${PROJECT_DIR}" \
        "$0")"

    echo "✓ Smoke test job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  Monitor:"
    echo "    tail -f ${LOG_DIR}/compass_single_${JOB_ID}.out"
    echo ""
    echo "  Errors (do NOT run as bash):"
    echo "    cat ${LOG_DIR}/compass_single_${JOB_ID}.err"
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

echo "============================================="
echo " COMPASS HPC — Single Participant Test"
echo "============================================="
echo ""
echo "SCRIPT_PATH:  $0"
echo "SCRIPT_SHA:   $(sha256sum "$0" | awk '{print $1}')"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "PWD:          $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LOCAL_ENGINE: ${LOCAL_ENGINE}"
echo "Requested ctx:${MAX_TOKENS} tokens"
echo "GPU mem util: ${GPU_MEM_UTIL}"
echo "Budget request: agent(in=${MAX_AGENT_INPUT}, out=${MAX_AGENT_OUTPUT}) tool(in=${MAX_TOOL_INPUT}, out=${MAX_TOOL_OUTPUT})"
echo "KV cache dtype request: ${LOCAL_KV_CACHE_DTYPE}"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
    echo "GPU: nvidia-smi not found"
fi
echo ""

echo "Container:    ${CONTAINER_IMAGE}"
echo "Venv:         ${VENV_DIR}"
echo "Model:        ${MODEL_NAME}"
echo "Embed model:  ${EMBEDDING_MODEL_NAME}"
echo "Participant:  ${PARTICIPANT_DIR}"
echo "Target File:  ${TARGETS_FILE}"
echo ""

# ─── Preconditions ──────────────────────────────────────────────────────────
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "✗ ERROR: Container not found at ${CONTAINER_IMAGE}"
    echo "  Run: bash hpc/02_setup_environment.sh"
    exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "✗ ERROR: apptainer not found in PATH on this node. PATH=${PATH}"
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
    echo "✗ ERROR: venv python not found at ${VENV_DIR}/bin/python3"
    echo "  Run: bash hpc/02_setup_environment.sh"
    exit 1
fi
if [[ ! -f "${PROJECT_DIR}/main.py" ]]; then
    echo "✗ ERROR: main.py not found at ${PROJECT_DIR}/main.py"
    exit 1
fi
if [[ ! -d "${MODEL_NAME}" ]]; then
    echo "✗ ERROR: Model dir not found: ${MODEL_NAME}"
    echo "  Run: bash hpc/03_download_models.sh"
    exit 1
fi
if [[ ! -d "${EMBEDDING_MODEL_NAME}" ]]; then
    echo "✗ ERROR: Embedding model dir not found: ${EMBEDDING_MODEL_NAME}"
    echo "  Run: bash hpc/03_download_models.sh"
    exit 1
fi
if [[ ! -d "${PARTICIPANT_DIR}" ]]; then
    echo "✗ ERROR: Participant dir not found: ${PARTICIPANT_DIR}"
    echo "  Looking in: ${DATA_DIR}"
    ls -d "${DATA_DIR}"/participant* 2>/dev/null | head -10 || true
    exit 1
fi

# ─── Dynamic Target Lookup & Blinding ────────────────────────────────────────
# This logic mirrors hpc/05_submit_batch.sh exactly to avoid discrepancy
if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "✗ ERROR: Target file not found at ${TARGETS_FILE}"
    echo "  Ensure you have synced the data directory."
    exit 1
fi

FULL_TARGET_LINE=$(grep "^${PARTICIPANT_ID}" "${TARGETS_FILE}")
if [[ -z "${FULL_TARGET_LINE}" ]]; then
    echo "✗ ERROR: Participant ID ${PARTICIPANT_ID} not found in target file."
    exit 1
fi

# Leak Protection: Strip CASE/CONTROL literals and parentheses
# Ensures the engine is blinded to the ground truth label.
SPECIFIC_TARGET=$(echo "${FULL_TARGET_LINE}" | cut -d'|' -f2- | sed -E 's/\bCASE\b//g; s/\bCONTROL\b//g; s/[()]//g' | xargs)

# HARDCODED CONTROL baseline
FIXED_CONTROL="possible brain-implicated pathology, but NOT the psychiatric target phenotype"

echo "  Leaked label:   $(echo "${FULL_TARGET_LINE}" | grep -oE "CASE|CONTROL")" # Log internally in .out
echo "  Engine Target:  '${SPECIFIC_TARGET}'"
echo "  Engine Control: '${FIXED_CONTROL}'"
echo ""

# ─── Detect model max length and clamp ───────────────────────────────────────
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
        echo "⚠  Clamping MAX_TOKENS from ${MAX_TOKENS} → ${DETECTED_MAX}"
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

# ─── Decide engine and run ───────────────────────────────────────────────────
echo "─── Starting COMPASS Pipeline ─────────────────────────────"
echo "Start time: $(date)"
echo ""

START_TIME=${SECONDS}
mkdir -p "${MODELS_DIR}/hf_cache"

# We will compute ACTUAL_ENGINE on-node after optional vLLM preflight
ACTUAL_ENGINE="${LOCAL_ENGINE}"

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
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env PYTHONUNBUFFERED="1" \
    --env LOCAL_ENGINE="${LOCAL_ENGINE}" \
    --env MODEL_NAME="${MODEL_NAME}" \
    --env MAX_TOKENS="${MAX_TOKENS}" \
    --env GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    --env LOCAL_KV_CACHE_DTYPE="${LOCAL_KV_CACHE_DTYPE}" \
    "${CONTAINER_IMAGE}" \
    bash -lc "
        set -euo pipefail
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

        echo '--- Runtime Information ---'
        python3 - <<'PY'
import torch
import transformers
try:
    import vllm
except Exception as e:
    vllm = None
print('Python OK')
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('vLLM:', getattr(vllm,'__version__','<not importable>'))
print('vLLM file:', getattr(vllm,'__file__','<n/a>'))
try:
    import vllm._C
    print('vLLM CUDA extension: OK')
except Exception as e:
    print('vLLM CUDA extension: NOT OK:', repr(e))
PY
        echo ''

        echo '--- Transformers local model sanity ---'
        python3 - <<'PY'
from transformers import AutoConfig, AutoTokenizer
m='${MODEL_NAME}'
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
print('Loaded config/tokenizer from:', m)
print('config.model_type:', getattr(cfg,'model_type', None))
print('config.max_position_embeddings:', getattr(cfg,'max_position_embeddings', None))
print('tokenizer.model_max_length:', getattr(tok,'model_max_length', None))
print('has quantization_config:', hasattr(cfg,'quantization_config'))
PY
        echo ''

        ACTUAL_ENGINE=\"\${LOCAL_ENGINE}\"

        # vLLM preflight (only if engine is auto or vllm)
        if [[ \"\${LOCAL_ENGINE}\" == 'auto' || \"\${LOCAL_ENGINE}\" == 'vllm' ]]; then
            if [[ \"${PREFLIGHT_VLLM}\" == '1' ]]; then
                echo '--- vLLM preflight (will fallback if LOCAL_ENGINE=auto) ---'
                if python3 - <<'PY'
import inspect, json, os, sys, traceback
from vllm import LLM, SamplingParams

model=os.environ['MODEL_NAME']
max_len=int(os.environ['MAX_TOKENS'])
gpu_mem=float(os.environ['GPU_MEM_UTIL'])
kv_dtype=(os.environ.get('LOCAL_KV_CACHE_DTYPE') or '').strip().lower()

sig = inspect.signature(LLM)
base_kwargs = dict(
    model=model,
    dtype='auto',
    trust_remote_code=True,
    max_model_len=max_len,
    gpu_memory_utilization=gpu_mem,
)
if kv_dtype and kv_dtype not in ('auto', 'none', 'null', '0'):
    if 'kv_cache_dtype' in sig.parameters:
        base_kwargs['kv_cache_dtype'] = kv_dtype

# Determine a quantization hint from model name/config, but keep a safe fallback.
quant_hint = None
if 'AWQ' in model.upper():
    quant_hint = 'awq'
cfg_path = os.path.join(model, 'config.json')
if os.path.isfile(cfg_path):
    try:
        cfg = json.load(open(cfg_path, 'r'))
        qcfg = cfg.get('quantization_config') or {}
        qmethod = qcfg.get('quant_method') or qcfg.get('method')
        if isinstance(qmethod, str) and qmethod.strip():
            quant_hint = qmethod.strip().lower()
    except Exception:
        pass

candidates = [('auto-detect', dict(base_kwargs))]
if 'quantization' in sig.parameters and quant_hint:
    if quant_hint == 'awq':
        candidates.insert(0, ('quantization=awq_marlin', {**base_kwargs, 'quantization': 'awq_marlin'}))
        candidates.insert(1, ('quantization=awq', {**base_kwargs, 'quantization': 'awq'}))
    else:
        candidates.insert(0, (f'quantization={quant_hint}', {**base_kwargs, 'quantization': quant_hint}))
if 'enforce_eager' in sig.parameters:
    candidates = [(label, {**kwargs, 'enforce_eager': True}) for label, kwargs in candidates]

for label, kwargs in candidates:
    try:
        print(f'Trying vLLM init ({label})')
        print('LLM init kwargs:', kwargs)
        llm = LLM(**kwargs)
        test = llm.generate(
            ['Preflight health check'],
            SamplingParams(temperature=0.0, max_tokens=8),
        )
        txt = ''
        try:
            txt = test[0].outputs[0].text if test and test[0].outputs else ''
        except Exception:
            txt = ''
        print('vLLM preflight generate output preview:', repr(str(txt)[:80]))
        print('vLLM preflight: SUCCESS')
        raise SystemExit(0)
    except Exception:
        print(f'vLLM preflight attempt failed ({label})')
        traceback.print_exc(file=sys.stdout)

print('vLLM preflight: FAILED')
raise SystemExit(17)
PY
                then
                    PRE=0
                else
                    PRE=\$?
                fi
                if [[ \$PRE -eq 0 ]]; then
                    ACTUAL_ENGINE='vllm'
                else
                    if [[ \"\${LOCAL_ENGINE}\" == 'vllm' ]]; then
                        echo '✗ LOCAL_ENGINE=vllm but vLLM preflight failed. Exiting.'
                        exit 1
                    fi
                    echo '⚠ vLLM preflight failed → falling back to Transformers.'
                    ACTUAL_ENGINE='transformers'
                fi
                echo \"Selected engine: \${ACTUAL_ENGINE}\"
                echo ''
            fi
        fi

        EXTRA_QUIET=''
        if [[ '${QUIET}' == '1' ]]; then
            EXTRA_QUIET='--quiet'
        fi

        # Map ACTUAL_ENGINE to your CLI.
        # If preflight is disabled and LOCAL_ENGINE=auto, prefer vLLM for HPC AWQ models.
        if [[ \"\${ACTUAL_ENGINE}\" == 'transformers' ]]; then
            LOCAL_ENGINE_FLAG='transformers'
            LOCAL_EXTRA_FLAGS=''
        else
            LOCAL_ENGINE_FLAG='vllm'
            LOCAL_EXTRA_FLAGS='--local_quant awq_marlin --local_enforce_eager'
        fi
        echo \"Runtime engine: \${LOCAL_ENGINE_FLAG} (requested=\${LOCAL_ENGINE}, preflight=${PREFLIGHT_VLLM})\"

        export CUDA_LAUNCH_BLOCKING=1

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
                --local_dtype auto \
                --local_kv_cache_dtype ${LOCAL_KV_CACHE_DTYPE} \
                --local_gpu_mem_util ${GPU_MEM_UTIL} \
                --local_max_model_len ${MAX_TOKENS} \
                --max_agent_input ${MAX_AGENT_INPUT} \
                --max_agent_output ${MAX_AGENT_OUTPUT} \
                --max_tool_input ${MAX_TOOL_INPUT} \
                --max_tool_output ${MAX_TOOL_OUTPUT} \
                --local_trust_remote_code \
                --audit \
                --quiet || {
                    echo '✗ Dataflow preflight audit failed; aborting before full LLM run.'
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
            --local_dtype auto \
            --local_kv_cache_dtype ${LOCAL_KV_CACHE_DTYPE} \
            --local_gpu_mem_util ${GPU_MEM_UTIL} \
            --local_max_model_len ${MAX_TOKENS} \
            --max_agent_input ${MAX_AGENT_INPUT} \
            --max_agent_output ${MAX_AGENT_OUTPUT} \
            --max_tool_input ${MAX_TOOL_INPUT} \
            --max_tool_output ${MAX_TOOL_OUTPUT} \
            --local_trust_remote_code \
            --detailed_log \
            \${LOCAL_EXTRA_FLAGS} \
            \${EXTRA_QUIET}
    "
EXIT_CODE=$?
set -e

ELAPSED=$((SECONDS - START_TIME))

echo ""
echo "============================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo " ✓ COMPASS completed successfully!"
    echo "   Results: ${PROJECT_DIR}/../results/participant_${PARTICIPANT_ID}/"
else
    echo " ✗ COMPASS exited with code ${EXIT_CODE}"
    echo "   Check:"
    echo "     ${LOG_DIR}/compass_single_${SLURM_JOB_ID:-unknown}.out"
    echo "     ${LOG_DIR}/compass_single_${SLURM_JOB_ID:-unknown}.err"
    echo ""
    echo "   Tip: use 'cat' or 'tail', don't run the .err as a script."
fi
echo "============================================="
echo "End time:  $(date)"
echo "Wall time: ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo ""

exit ${EXIT_CODE}
