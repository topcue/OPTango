#!/usr/bin/env bash

# Reference: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # OPTango/env
CONDA_DIR="${SCRIPT_DIR}/miniconda3"
INSTALLER="${SCRIPT_DIR}/miniconda.sh"

echo "[*] Installing Miniconda to: ${CONDA_DIR}"

mkdir -p "${CONDA_DIR}"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	-O "${INSTALLER}"

bash "${INSTALLER}" -b -u -p "${CONDA_DIR}"
rm -f "${INSTALLER}"

source "${CONDA_DIR}/bin/activate"

echo "[*] Miniconda installation complete"

# EOF
