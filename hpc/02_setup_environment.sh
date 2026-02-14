#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 1: Environment Setup
# =============================================================================
#
# PURPOSE:
#   - Pull NVIDIA PyTorch container
#   - Create Python venv
#   - Install pinned dependencies (numpy 1.x etc.) + project requirements
#
# NOTE (Izaro / Biobizkaia):
#   Apptainer is only available on compute nodes.
#   If run on login node, this script auto-submits itself as a Slurm job.
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#   bash hpc/02_setup_environment.sh
#   # logs: ~/compass_pipeline/multi_agent_system/logs/setup_<JOBID>.out
# =============================================================================

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
CONTAINER_DIR="${HOME}/compass_containers"
CONTAINER_IMAGE="${CONTAINER_DIR}/pytorch_24.01.sif"
DOCKER_IMAGE="docker://nvcr.io/nvidia/pytorch:24.01-py3"

VENV_DIR="${HOME}/compass_venv"
PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_system"
LOG_DIR="${PROJECT_DIR}/logs"

# Hard pins (your “atomic env”)
PIN_NUMPY="1.26.4"
PIN_PANDAS="2.2.2"
PIN_SCIPY="1.13.1"
# Qwen3 AWQ requires newer stacks (Qwen docs recommend vllm>=0.8.5 and transformers>=4.51.0).
PIN_VLLM="0.8.5"
PIN_TRANSFORMERS="4.51.3"

# ─── Header / Fingerprint ───────────────────────────────────────────────────
echo "============================================="
echo " COMPASS HPC — Environment Setup"
echo "============================================="
echo ""
echo "SCRIPT_PATH:  $0"
echo "SCRIPT_SHA:   $(sha256sum "$0" | awk '{print $1}')"
echo "Hostname:     $(hostname)"
echo "User:         $(whoami)"
echo "Home:         ${HOME}"
echo "Date:         $(date)"
echo "PWD:          $(pwd)"
echo ""

# ─── Auto-Submit to Compute Node ───────────────────────────────────────────
CURRENT_HOST="$(hostname)"
if [[ "${CURRENT_HOST}" == login* ]]; then
    echo "⚠  Login node detected. Apptainer is only on compute nodes."
    echo "   Auto-submitting this script as a Slurm job..."
    echo ""

    mkdir -p "${LOG_DIR}"

    JOB_ID="$(sbatch --parsable \
        --job-name="compass_setup" \
        --output="${LOG_DIR}/setup_%j.out" \
        --error="${LOG_DIR}/setup_%j.err" \
        --time=02:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --partition=main \
        --chdir="${PROJECT_DIR}" \
        "$0")"

    echo "✓ Setup job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  Monitor:"
    echo "    tail -f ${LOG_DIR}/setup_${JOB_ID}.out"
    echo ""
    echo "  Queue:"
    echo "    squeue -u $(whoami)"
    echo ""
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# FROM HERE: Running on a COMPUTE node
# ═════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "─── Running on Compute Node: ${CURRENT_HOST} ──────────────"
echo ""

# ─── Sanity: Apptainer ──────────────────────────────────────────────────────
if ! command -v apptainer >/dev/null 2>&1; then
    echo "✗ ERROR: apptainer not found in PATH on this node."
    echo "PATH: ${PATH}"
    exit 1
fi
echo "✓ Apptainer: $(command -v apptainer)"
echo ""

# ─── Step 1/3: Pull Container ───────────────────────────────────────────────
echo "─── Step 1/3: Container ───────────────────────────────────"
echo "  Source: ${DOCKER_IMAGE}"
echo "  Target: ${CONTAINER_IMAGE}"
echo ""

mkdir -p "${CONTAINER_DIR}"
export APPTAINER_CACHEDIR="${CONTAINER_DIR}/.cache"
export APPTAINER_TMPDIR="${CONTAINER_DIR}/.tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

if [[ -f "${CONTAINER_IMAGE}" ]]; then
    echo "✓ Container already exists ($(du -h "${CONTAINER_IMAGE}" | cut -f1))."
else
    echo "  Pulling container (can take a while)..."
    apptainer pull "${CONTAINER_IMAGE}" "${DOCKER_IMAGE}"
    echo "✓ Container downloaded ($(du -h "${CONTAINER_IMAGE}" | cut -f1))."
fi
echo ""

# ─── Step 2/3: Create venv ─────────────────────────────────────────────────
echo "─── Step 2/3: Virtualenv ──────────────────────────────────"
echo "  Venv path: ${VENV_DIR}"
echo ""

rm -rf "${VENV_DIR}"
echo "  Creating venv inside container (keeps container torch/CUDA via --system-site-packages)..."
if apptainer exec "${CONTAINER_IMAGE}" python3 -m venv --system-site-packages "${VENV_DIR}"; then
    :
else
    echo "  ⚠ venv creation failed. Trying --without-pip + bootstrap pip..."
    apptainer exec "${CONTAINER_IMAGE}" python3 -m venv --system-site-packages --without-pip "${VENV_DIR}"
    apptainer exec "${CONTAINER_IMAGE}" bash -lc "curl -sS https://bootstrap.pypa.io/get-pip.py | '${VENV_DIR}/bin/python3'"
fi
echo "✓ Virtual environment created."
echo ""

echo "  Verifying pip in venv..."
apptainer exec "${CONTAINER_IMAGE}" bash -lc "
set -euo pipefail
if '${VENV_DIR}/bin/python3' -m pip --version >/dev/null 2>&1; then
    echo '  ✓ pip is available in venv'
    exit 0
fi

echo '  ⚠ pip missing in venv; bootstrapping...'
if ! '${VENV_DIR}/bin/python3' -m ensurepip --upgrade >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    '${VENV_DIR}/bin/python3' /tmp/get-pip.py
    rm -f /tmp/get-pip.py
fi

'${VENV_DIR}/bin/python3' -m pip --version
"
echo ""

# ─── Step 3/3: Install deps ────────────────────────────────────────────────
echo "─── Step 3/3: Dependencies ────────────────────────────────"
REQUIREMENTS="${PROJECT_DIR}/requirements.txt"
if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "✗ ERROR: requirements.txt not found at: ${REQUIREMENTS}"
    exit 1
fi

export VENV_DIR REQUIREMENTS
export PIN_NUMPY PIN_PANDAS PIN_SCIPY PIN_VLLM PIN_TRANSFORMERS

echo "  Installing into venv..."
apptainer exec "${CONTAINER_IMAGE}" bash -s <<'EOF'
set -euo pipefail

echo "  Container Python: $(python3 --version)"
echo "  Venv Python:      ${VENV_DIR}/bin/python3"

"${VENV_DIR}/bin/python3" -m pip install --upgrade pip setuptools wheel

# 1) Install project requirements first (whatever COMPASS needs)
"${VENV_DIR}/bin/python3" -m pip install --no-cache-dir -r "${REQUIREMENTS}"

# 2) Force the pinned “atomic” stack afterwards (wins over transient upgrades)
"${VENV_DIR}/bin/python3" -m pip install --no-cache-dir --upgrade --force-reinstall \
  "numpy==${PIN_NUMPY}" \
  "pandas==${PIN_PANDAS}" \
  "scipy==${PIN_SCIPY}" \
  "transformers==${PIN_TRANSFORMERS}" \
  "vllm==${PIN_VLLM}" \
  "optimum" "autoawq>=0.2.6" "safetensors" "sentencepiece"

echo ""
echo "  Freeze sanity:"
"${VENV_DIR}/bin/python3" -m pip freeze | egrep -i '^(numpy|pandas|scipy|transformers|vllm|torch|autoawq|optimum|safetensors)=' || true

echo ""
echo "  Import sanity:"
"${VENV_DIR}/bin/python3" - <<PY
import numpy, pandas, scipy
print("VERIFIED_NUMPY:", numpy.__version__)
print("VERIFIED_PANDAS:", pandas.__version__)
print("VERIFIED_SCIPY:", scipy.__version__)
import torch
print("VERIFIED_TORCH:", torch.__version__)
import transformers
print("VERIFIED_TRANSFORMERS:", transformers.__version__)
import vllm
print("VERIFIED_VLLM:", getattr(vllm, "__version__", "unknown"))
print("VLLM_FILE:", vllm.__file__)
PY

echo ""
echo "NOTE: vLLM CUDA extension check is done in Step 3 on a GPU job (guaranteed --nv)."
EOF

echo ""
echo "============================================="
echo " ✓ ENVIRONMENT SETUP COMPLETE"
echo "============================================="
echo ""
echo " Created:"
echo "   Container: ${CONTAINER_IMAGE}"
echo "   Venv:      ${VENV_DIR}"
echo ""
echo " NEXT STEP: Download model weights:"
echo "   bash hpc/03_download_models.sh"
echo ""
