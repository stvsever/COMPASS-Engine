#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 2: Download LLM Models
# =============================================================================
#
# PURPOSE: Downloads model weights from HuggingFace to shared storage.
#
# FIXES IN v2:
#   - Uses local_dir_use_symlinks=True by default so the target dir is NOT empty
#     while downloading (no "download forever into cache then copy at the end").
#   - Forces HF cache into ${MODELS_DIR}/hf_home to keep everything on the same FS.
#   - Enables hf_transfer acceleration when available.
#   - Adds explicit apptainer binds (HOME/MODELS/PROJECT) for reliability.
#   - YaRN patch uses rope_type=yarn (correct key for Qwen3/Transformers>=4.51).
#   - YaRN patch is OPTIONAL (ENABLE_YARN=1). Default is OFF (safer).
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#   bash hpc/03_download_models.sh
#
# OPTIONAL ENV:
#   LOCAL_DIR_USE_SYMLINKS=1|0   (default 1; 0 = real copies, slower)
#   ENABLE_YARN=1|0             (default 0; 1 = patch config.json for YaRN)
#   YARN_FACTOR=2.0             (default 2.0 for ~64k if base is 32k)
#
# =============================================================================

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
MODELS_DIR="${HOME}/compass_models"
CONTAINER_IMAGE="${HOME}/compass_containers/pytorch_24.01.sif"
VENV_DIR="${HOME}/compass_venv"
PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_system"

MODEL_MAIN="Qwen/Qwen3-14B-AWQ"
MODEL_EMBED="Qwen/Qwen3-Embedding-8B"

# Defaults (can be overridden via environment)
: "${LOCAL_DIR_USE_SYMLINKS:=1}"   # 1 = fast, directory populates immediately; 0 = physical copy (slow)
: "${ENABLE_YARN:=0}"              # 1 = patch rope_scaling; default off (recommended)
: "${YARN_FACTOR:=2.0}"            # 2.0 => ~65k if base is 32k

# Derived HF cache locations (keep inside MODELS_DIR)
HF_HOME_DIR="${MODELS_DIR}/hf_home"
HF_HUB_CACHE_DIR="${HF_HOME_DIR}/hub"
TRANSFORMERS_CACHE_DIR="${HF_HOME_DIR}/transformers"

# ─── Header ────────────────────────────────────────────────────────────────
echo "============================================="
echo " COMPASS HPC — Model Download (v2)"
echo "============================================="
echo ""
echo "Host:    $(hostname)"
echo "User:    $(whoami)"
echo "Date:    $(date)"
echo ""
echo "MODELS_DIR: ${MODELS_DIR}"
echo "Symlinks:  ${LOCAL_DIR_USE_SYMLINKS} (1=on, 0=off)"
echo "YaRN:      ${ENABLE_YARN} (factor=${YARN_FACTOR})"
echo ""

# ─── Auto-Submit to Compute Node ───────────────────────────────────────────
CURRENT_HOST="$(hostname)"
if [[ "${CURRENT_HOST}" == login* ]]; then
    echo "⚠  Login node detected. Submitting download job to compute node..."
    echo ""

    mkdir -p "${PROJECT_DIR}/logs"

    JOB_ID="$(sbatch --parsable \
        --job-name="compass_download" \
        --output="${PROJECT_DIR}/logs/download_%j.out" \
        --error="${PROJECT_DIR}/logs/download_%j.err" \
        --time=04:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --partition=main \
        --chdir="${PROJECT_DIR}" \
        "$0")"

    echo "✓ Download job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  Monitor progress:"
    echo "    tail -f ${PROJECT_DIR}/logs/download_${JOB_ID}.out"
    echo ""
    echo "  Check queue:"
    echo "    squeue -u $(whoami)"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# FROM HERE: Running on a COMPUTE node
# ═════════════════════════════════════════════════════════════════════════════

echo "─── Running on Compute Node: ${CURRENT_HOST} ──────────────"
echo ""

# ─── Pre-checks ───────────────────────────────────────────────────────────
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "✗ ERROR: Container not found at ${CONTAINER_IMAGE}"
    echo "  Run 01_setup_environment.sh first."
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
    echo "✗ ERROR: Virtual environment not found or invalid at ${VENV_DIR}"
    echo "  Run 01_setup_environment.sh first."
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "✗ ERROR: apptainer not found in PATH on this node."
    echo "  PATH=${PATH}"
    exit 1
fi

mkdir -p "${MODELS_DIR}" "${HF_HOME_DIR}" "${HF_HUB_CACHE_DIR}" "${TRANSFORMERS_CACHE_DIR}"

# ─── Disk Space Check (check filesystem of MODELS_DIR) ──────────────────────
echo "─── Disk Space Check ──────────────────────────────────────"
AVAIL_KB="$(df "${MODELS_DIR}" 2>/dev/null | tail -1 | awk '{print $4}')"
if [[ -n "${AVAIL_KB}" ]] && [[ "${AVAIL_KB}" -lt 60000000 ]] 2>/dev/null; then
    echo "  ⚠ WARNING: Less than 60 GB available where models will be stored."
    echo "  Available: $(df -h "${MODELS_DIR}" | tail -1 | awk '{print $4}')"
else
    echo "  ✓ Sufficient disk space available"
fi
echo ""

# ─── Download Function ──────────────────────────────────────────────────────
download_model() {
    local model_id="$1"
    local target_dir="$2"

    echo ""
    echo "─── Downloading: ${model_id} ──────────────────────────────"
    echo "  Target: ${target_dir}"
    echo ""

    # If directory is non-empty, assume done
    if [[ -d "${target_dir}" ]] && [[ -n "$(ls -A "${target_dir}" 2>/dev/null)" ]]; then
        echo "✓ Already present (non-empty). Skipping."
        echo "  To re-download:"
        echo "    rm -rf '${target_dir}' '${HF_HUB_CACHE_DIR}'"
        return 0
    fi

    mkdir -p "${target_dir}"

    # Run inside container + venv, with explicit binds and HF cache env.
    # Symlinks ON means target_dir populates quickly (no long final copy).
    apptainer exec \
        --bind "${HOME}:${HOME}" \
        --bind "${MODELS_DIR}:${MODELS_DIR}" \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --env HF_HOME="${HF_HOME_DIR}" \
        --env HF_HUB_CACHE="${HF_HUB_CACHE_DIR}" \
        --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}" \
        --env HF_HUB_ENABLE_HF_TRANSFER="1" \
        --env HF_HUB_DISABLE_PROGRESS_BARS="0" \
        --env MODEL_ID="${model_id}" \
        --env TARGET_DIR="${target_dir}" \
        --env USE_SYMLINKS="${LOCAL_DIR_USE_SYMLINKS}" \
        "${CONTAINER_IMAGE}" \
        bash -lc "
            set -euo pipefail
            source '${VENV_DIR}/bin/activate'

            # Ensure deps once (quiet but not fully silent)
            python3 - <<'PY'
import importlib.util, subprocess, sys
def ensure(pkg):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', pkg])
ensure('huggingface_hub')
# hf_transfer speeds downloads when HF_HUB_ENABLE_HF_TRANSFER=1
ensure('hf_transfer')
PY

            python3 -u - <<'PY'
import os, json, time
from huggingface_hub import snapshot_download

model_id = os.environ['MODEL_ID']
target_dir = os.environ['TARGET_DIR']
use_symlinks = os.environ.get('USE_SYMLINKS','1').strip() not in ('0','false','False')

print(f'Downloading {model_id}...')
print(f'  local_dir: {target_dir}')
print(f'  local_dir_use_symlinks: {use_symlinks}')
print(f'  HF_HOME: {os.environ.get(\"HF_HOME\")}')
print(f'  HF_HUB_CACHE: {os.environ.get(\"HF_HUB_CACHE\")}')

t0 = time.time()
snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=use_symlinks,
    resume_download=True,
)
dt = int(time.time() - t0)

# Minimal sanity: should have config.json and at least one weight file
files = []
for root, _, fn in os.walk(target_dir):
    for f in fn:
        files.append(f)
has_cfg = any(f == 'config.json' for f in files)
has_weights = any(f.endswith('.safetensors') or f.endswith('.bin') for f in files)

print(f'Download finished in {dt}s')
print(f'Files in target: {len(files)}')
print(f'Has config.json: {has_cfg}')
print(f'Has weights:     {has_weights}')

if not has_cfg or not has_weights:
    raise SystemExit('ERROR: target_dir looks incomplete (missing config or weights).')
print('✓ Download complete!')
PY
        "

    echo "✓ Downloaded: ${model_id} → ${target_dir}"
}

# ─── Patch Model Config (for vLLM compatibility & optionally YaRN) ──────────
patch_model_config() {
    local model_dir="$1"
    local yarn_factor="$2"
    local cfg_path="${model_dir}/config.json"

    echo ""
    echo "─── Patching Model Config ─────────────────────────────────"
    echo "  Model dir: ${model_dir}"
    echo "  config:    ${cfg_path}"
    echo ""

    if [[ ! -f "${cfg_path}" ]]; then
        echo "✗ ERROR: config.json not found at ${cfg_path}"
        exit 1
    fi

    # Run inside container + venv
    apptainer exec \
        --bind "${HOME}:${HOME}" \
        --bind "${MODELS_DIR}:${MODELS_DIR}" \
        --env CFG_PATH="${cfg_path}" \
        --env YARN_FACTOR="${yarn_factor}" \
        --env ENABLE_YARN="${ENABLE_YARN}" \
        "${CONTAINER_IMAGE}" \
        bash -lc "
            set -euo pipefail
            source '${VENV_DIR}/bin/activate'
            python3 -u - <<'PY'
import json, os
from pathlib import Path

cfg_path = Path(os.environ['CFG_PATH'])
factor = float(os.environ['YARN_FACTOR'])
enable_yarn = os.environ.get('ENABLE_YARN') == '1'

data = json.loads(cfg_path.read_text())
changed = False

# vLLM (esepcially older/strict builds) asserts 'factor' in rope_scaling.
# Transformers >= 4.51 often populates a 'default' rope_scaling without factor.
rope = data.get('rope_scaling')

if enable_yarn:
    # Full YaRN patch for long context support
    native = 32768
    desired = {
        'rope_type': 'yarn',
        'factor': factor,
        'original_max_position_embeddings': native
    }
    if rope != desired:
        data['rope_scaling'] = desired
        changed = True
        print(f'✓ YaRN enabled: {desired}')
else:
    # Minimal compatibility fix: Ensure 'factor' exists if rope_scaling is present
    # or if we need to force it for Qwen architectures to satisfy vLLM.
    if rope is not None and 'factor' not in rope:
        rope['factor'] = 1.0
        if 'rope_type' not in rope:
            rope['rope_type'] = 'default'
        data['rope_scaling'] = rope
        changed = True
        print(f'✓ Enhanced rope_scaling for vLLM compatibility: {rope}')
    elif rope is None and data.get('model_type') in ('qwen2', 'qwen3'):
        # Force a minimal rope_scaling for Qwen models to prevent vLLM failures
        data['rope_scaling'] = {'factor': 1.0, 'rope_type': 'default'}
        changed = True
        print(f'✓ Created minimal rope_scaling for vLLM compatibility.')

if changed:
    backup = cfg_path.with_suffix('.json.bak')
    if not backup.exists():
        backup.write_text(cfg_path.read_text())
    cfg_path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print('✓ Patched config.json')
else:
    print('✓ Config already compatible')
PY
        "
}

# ─── Download Models ────────────────────────────────────────────────────────
echo "Downloading model weights to: ${MODELS_DIR}"
echo "Estimated total: ~12-18 GB (main) + ~16 GB (embed) ≈ 28-34 GB"
echo ""

MAIN_DIR="${MODELS_DIR}/Qwen_Qwen3-14B-AWQ"
EMBED_DIR="${MODELS_DIR}/Qwen_Qwen3-Embedding-8B"

download_model "${MODEL_MAIN}" "${MAIN_DIR}"
patch_model_config "${MAIN_DIR}" "${YARN_FACTOR}"
download_model "${MODEL_EMBED}" "${EMBED_DIR}"

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " ✓ MODEL DOWNLOAD COMPLETE"
echo "============================================="
echo ""
echo " Models stored in: ${MODELS_DIR}"
du -sh "${MODELS_DIR}"/* 2>/dev/null || echo "  (no models found)"
echo ""
echo " HF cache stored in: ${HF_HOME_DIR}"
du -sh "${HF_HOME_DIR}" 2>/dev/null || true
echo ""
echo " NEXT STEP: Test with one participant:"
echo "   sbatch hpc/04_submit_single.sh"
echo ""
