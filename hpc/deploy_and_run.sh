#!/bin/bash
# =============================================================================
# COMPASS — HPC Deployment Helper
# =============================================================================
# This script automates transferring your code to the Biobizkaia HPC
# and initiating the setup process.
#
# USAGE:
#   bash hpc/deploy_and_run.sh
# =============================================================================

# Configuration
HPC_USER="..."
HPC_HOST="..."
HPC_DIR="~/compass_pipeline"

# Local paths (override via environment if needed)
# - Defaults are derived relative to this repo.
# - Examples:
#     LOCAL_CODE_DIR=/path/to/multi_agent_system LOCAL_DATA_DIR=/path/to/HPC_data bash hpc/deploy_and_run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${LOCAL_CODE_DIR:=${REPO_ROOT}}"
: "${LOCAL_DATA_DIR:=${REPO_ROOT}/../data/__FEATURES__/HPC_data}"

echo "========================================================"
echo " 🚀 COMPASS Deployment to Biobizkaia HPC"
echo "========================================================"
echo "Target: ${HPC_USER}@${HPC_HOST}"
echo ""
echo "Note: You will be prompted for your password multiple times"
echo "      (once for each command) unless you have SSH keys set up."
echo "      Password: ..."
echo "========================================================"
echo ""

# 1. Create remote directory structure
echo "─── [1/3] Creating remote directories..."
ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p ${HPC_DIR}/data/__FEATURES__ 2>/dev/null" || true

# 2. Upload Code
echo ""
echo "─── [2/3] Uploading Project Code..."
# We exclude __pycache__, .git, and local logs to save time/space
# Using rsync if available, else scp
if command -v rsync &> /dev/null; then
    rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.DS_Store' \
        "${LOCAL_CODE_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
else
    scp -r "${LOCAL_CODE_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
fi

# 3. Upload Data (Optional prompt)
echo ""
echo "─── [3/3] Participant Data..."
if [ -d "${LOCAL_DATA_DIR}" ]; then
    echo "Found local data at: ${LOCAL_DATA_DIR}"
    read -p "Upload participant data now? (This might take a while) [y/N]: " upload_data
    if [[ "$upload_data" =~ ^[Yy]$ ]]; then
        echo "Uploading data..."
        if command -v rsync &> /dev/null; then
             rsync -avz "${LOCAL_DATA_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/__FEATURES__/"
        else
             scp -r "${LOCAL_DATA_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/__FEATURES__/"
        fi
    else
        echo "Skipping data upload."
    fi
else
    echo "⚠️  Local data directory not found at: ${LOCAL_DATA_DIR}"
    echo "   Please make sure to upload data manually."
fi

echo ""
echo "========================================================"
echo "Deployment Complete!"
echo "========================================================"
echo "To start the pipeline, run these commands inside the HPC:"
echo ""
echo "  ssh ${HPC_USER}@${HPC_HOST}"
echo "  cd ~/compass_pipeline/multi_agent_system"
echo "  bash hpc/00_check_status.sh"
echo ""
echo "Connecting you now..."
echo "========================================================"
ssh "${HPC_USER}@${HPC_HOST}"
