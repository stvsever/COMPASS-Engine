#!/bin/bash
# =============================================================================
# COMPASS HPC — Step 0: Pre-flight Health Check
# =============================================================================
#
# PURPOSE: Validates the HPC setup before submitting compute jobs.
# Run this FIRST when you login to the cluster.
#
# NOTE: On this cluster (Izaro / Biobizkaia), Apptainer is ONLY available
# on compute nodes (c01-c03, f01-f02, g01), NOT on login01.
# This script checks file existence (works from login node) and warns
# about compute-node-only tools.
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_system
#   bash hpc/01_check_status.sh
#
# =============================================================================

set -euo pipefail

echo "============================================="
echo " COMPASS HPC — Pre-flight Health Check"
echo "============================================="
echo ""
echo "Host:    $(hostname)"
echo "User:    $(whoami)"
echo "Date:    $(date)"
echo ""

# ─── Counters ─────────────────────────────────────────────────────────────
PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" = "pass" ]; then
        echo "  ✓ ${label}"
        PASS=$((PASS + 1))
    elif [ "$result" = "warn" ]; then
        echo "  ⚠ ${label}"
        WARN=$((WARN + 1))
    else
        echo "  ✗ ${label}"
        FAIL=$((FAIL + 1))
    fi
}

normalize_version() {
    echo "$1" | sed -E 's/^([0-9]+(\.[0-9]+){0,2}).*/\1/'
}

version_ge() {
    local have want
    have="$(normalize_version "$1")"
    want="$(normalize_version "$2")"
    if [[ -z "${have}" || -z "${want}" ]]; then
        return 1
    fi
    [[ "$(printf '%s\n%s\n' "${want}" "${have}" | sort -V | head -n1)" == "${want}" ]]
}

# ─── 1. Container Runtime ────────────────────────────────────────────────
# On this cluster, Apptainer is only installed on COMPUTE nodes.
# We check if the binary exists on THIS node, but warn (not fail) if missing.
echo "─── Container Runtime ─────────────────────────────────────"
if command -v apptainer &>/dev/null; then
    check "apptainer available ($(apptainer --version))" "pass"
elif command -v singularity &>/dev/null; then
    check "singularity available ($(singularity --version))" "pass"
else
    HOSTNAME_CHECK=$(hostname)
    if [[ "$HOSTNAME_CHECK" == login* ]]; then
        check "Apptainer not on login node (OK — it's on compute nodes)" "warn"
        echo "       This is normal for the Izaro cluster."
    else
        check "No apptainer/singularity found on $(hostname)" "fail"
    fi
fi

# ─── 2. Container Image (.sif file) ──────────────────────────────────────
echo ""
echo "─── Container Image ───────────────────────────────────────"
CONTAINER="${HOME}/compass_containers/pytorch_24.01.sif"
if [ -f "${CONTAINER}" ]; then
    SIZE=$(du -h "${CONTAINER}" | cut -f1)
    check "Container exists (${SIZE})" "pass"
else
    check "Container missing at ${CONTAINER}" "fail"
    echo "       → Fix: bash hpc/02_setup_environment.sh"
fi

# ─── 3. Virtual Environment ──────────────────────────────────────────────
echo ""
echo "─── Virtual Environment ───────────────────────────────────"
VENV="${HOME}/compass_venv"
if [ -d "${VENV}" ]; then
    check "Venv exists at ${VENV}" "pass"
    if [ -f "${VENV}/bin/python" ]; then
        check "Python binary present" "pass"
    else
        check "Python binary missing in venv" "fail"
    fi
else
    check "Venv not found at ${VENV}" "fail"
    echo "       → Fix: bash hpc/02_setup_environment.sh"
fi

# ─── 3b. Python Package Versions (Qwen3 compatibility) ───────────────────
echo ""
echo "─── Python Package Versions ───────────────────────────────"
if [ -x "${VENV}/bin/python3" ]; then
    # Try to check version, but handle failure if binary can't run on login node
    if "${VENV}/bin/python3" -m pip --version >/dev/null 2>&1; then
        check "pip available in venv" "pass"

        PKG_VERS="$("${VENV}/bin/python3" - <<'PY' 2>/dev/null
import importlib

def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "")
    except Exception:
        return ""

print(ver("transformers"))
print(ver("vllm"))
PY
)"
        TF_VER="$(echo "${PKG_VERS}" | sed -n '1p')"
        VLLM_VER="$(echo "${PKG_VERS}" | sed -n '2p')"

        if [[ -n "${TF_VER}" ]]; then
            if version_ge "${TF_VER}" "4.51.0"; then
                check "transformers ${TF_VER} (>=4.51.0)" "pass"
            else
                check "transformers ${TF_VER} (<4.51.0; Qwen3 unsupported)" "fail"
            fi
        else
            check "transformers not importable" "fail"
        fi

        if [[ -n "${VLLM_VER}" ]]; then
            if version_ge "${VLLM_VER}" "0.8.5"; then
                check "vllm ${VLLM_VER} (>=0.8.5)" "pass"
            else
                check "vllm ${VLLM_VER} (<0.8.5; may fail on Qwen3)" "fail"
            fi
        else
            check "vllm not importable" "fail"
        fi
    else
        # Fallback for login node where containerized python might not execute
        if [ -f "${VENV}/bin/pip" ]; then
            check "pip found (version check requires compute node)" "warn"
            check "transformers found (version check requires compute node)" "warn"
            check "vllm found (version check requires compute node)" "warn"
        else
            check "pip missing in venv" "fail"
            echo "       → Fix: bash hpc/02_setup_environment.sh"
        fi
    fi
else
    check "Skipping package version checks (venv python missing)" "warn"
fi

# ─── 4. Model Weights ────────────────────────────────────────────────────
echo ""
echo "─── Model Weights ────────────────────────────────────────"
MODELS="${HOME}/compass_models"
if [ -d "${MODELS}" ]; then
    for expected_model in "Qwen_Qwen3-14B-AWQ" "Qwen_Qwen3-Embedding-8B"; do
        model_path="${MODELS}/${expected_model}"
        if [ -d "${model_path}" ] && [ "$(ls -A "${model_path}" 2>/dev/null)" ]; then
            size=$(du -sh "${model_path}" 2>/dev/null | cut -f1)
            check "Model: ${expected_model} (${size})" "pass"
        else
            check "Model: ${expected_model} NOT FOUND" "fail"
            echo "       → Fix: bash hpc/03_download_models.sh"
        fi
    done
else
    check "Models directory not found at ${MODELS}" "fail"
    echo "       → Fix: bash hpc/03_download_models.sh"
fi

# ─── 5. Project Source Files ─────────────────────────────────────────────
echo ""
echo "─── Project Source Files ─────────────────────────────────"
PROJECT="${HOME}/compass_pipeline/multi_agent_system"
if [ -d "${PROJECT}" ]; then
    check "Project directory exists" "pass"
    for required in "main.py" "requirements.txt" "config/settings.py" "utils/batch_run.py" "utils/local_llm.py" "utils/llm_client.py"; do
        if [ -f "${PROJECT}/${required}" ]; then
            check "${required}" "pass"
        else
            check "${required} MISSING" "fail"
        fi
    done
else
    check "Project directory not found at ${PROJECT}" "fail"
    echo "       → Fix: Upload project via SCP first"
fi

# ─── 6. Participant Data ─────────────────────────────────────────────────
echo ""
echo "─── Participant Data (V2) ─────────────────────────────────────"
DATA_DIR="${PROJECT}/../data/__FEATURES__/HPC_data"
if [ -d "${DATA_DIR}" ]; then
    PCOUNT=$(find "${DATA_DIR}" -maxdepth 1 -type d -name "participant_*" 2>/dev/null | wc -l | tr -d ' ')
    check "Data directory exists (${PCOUNT} participants found)" "pass"

    SAMPLE=$(find "${DATA_DIR}" -maxdepth 1 -type d -name "participant_*" 2>/dev/null | head -1)
    if [ -n "${SAMPLE}" ]; then
        SAMPLE_NAME=$(basename "${SAMPLE}")
        EXPECTED_FILES=("data_overview.json" "hierarchical_deviation_map.json" "multimodal_data.json" "non_numerical_data.txt")
        MISSING=0
        for f in "${EXPECTED_FILES[@]}"; do
            [ ! -f "${SAMPLE}/${f}" ] && MISSING=$((MISSING + 1))
        done
        if [ ${MISSING} -eq 0 ]; then
            check "Sample ${SAMPLE_NAME}: all 4 data files present" "pass"
        else
            check "Sample ${SAMPLE_NAME}: ${MISSING}/4 files missing" "warn"
        fi
    fi
else
    check "Data directory not found at ${DATA_DIR}" "warn"
    echo "       Upload data to: ${DATA_DIR}/participant_ID{eid}/"
fi

# ─── 7. Slurm Job Scheduler ─────────────────────────────────────────────
echo ""
echo "─── Slurm Status ──────────────────────────────────────────"
if command -v squeue &>/dev/null; then
    MYJOBS=$(squeue -u "$(whoami)" -h 2>/dev/null | wc -l | tr -d ' ')
    check "Slurm available (${MYJOBS} active jobs for this user)" "pass"
    if [ "${MYJOBS}" -gt 0 ]; then
        echo ""
        squeue -u "$(whoami)" --format="%.8i %.16j %.8T %.10M %.6D %R" 2>/dev/null || true
    fi
else
    check "squeue not found (are you on the login node?)" "warn"
fi

# ─── 8. GPU Node Availability ────────────────────────────────────────────
echo ""
echo "─── GPU Node ──────────────────────────────────────────────"
if command -v sinfo &>/dev/null; then
    GPU_INFO=$(sinfo --format="%N %T %G %m" -h 2>/dev/null | grep -i "gpu\|l40s\|g0" | head -3 || echo "")
    if [ -n "${GPU_INFO}" ]; then
        echo "  ${GPU_INFO}"
        check "GPU node information retrieved" "pass"
    else
        # Show all nodes if no GPU-specific info
        sinfo --format="%N %P %T %C" -h 2>/dev/null | head -5
        check "Node information retrieved (no GPU filter match)" "warn"
    fi
else
    check "sinfo not available (Slurm not loaded?)" "warn"
fi

# ─── 9. Disk Space ───────────────────────────────────────────────────────
echo ""
echo "─── Disk Space ────────────────────────────────────────────"
if command -v df &>/dev/null; then
    AVAIL=$(df -h "${HOME}" 2>/dev/null | tail -1 | awk '{print $4}')
    USED_PCT=$(df -h "${HOME}" 2>/dev/null | tail -1 | awk '{print $5}')
    check "Home directory: ${AVAIL} available (${USED_PCT} used)" "pass"
else
    check "Could not check disk space" "warn"
fi

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================="
echo " Results: ✓${PASS} pass / ⚠${WARN} warn / ✗${FAIL} fail"
echo "============================================="

if [ ${FAIL} -gt 0 ]; then
    echo ""
    echo " ✗ Fix the failed checks before submitting jobs."
    echo ""
    echo " Run scripts in order:"
    echo "   0. bash hpc/00_deploy_and_run.sh      # Optional: deploy repo to HPC"
    echo "   1. bash hpc/02_setup_environment.sh   # Container + venv"
    echo "   2. bash hpc/03_download_models.sh     # Model weights"
    echo "   3. sbatch hpc/04_submit_single.sh     # Test run"
    exit 1
elif [ ${WARN} -gt 0 ]; then
    echo ""
    echo " ⚠ Some warnings detected — review above before proceeding."
else
    echo ""
    echo " ✓ All checks passed! Ready to submit jobs."
fi

echo ""
echo " Quick start:"
echo "   sbatch hpc/04_submit_single.sh  # Test with 1 participant"
echo "   sbatch hpc/05_submit_batch.sh   # Full batch"
echo ""
