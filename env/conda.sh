#!/usr/bin/bash

SCRIPT_PATH="$(readlink -f "$0")"       # OPTango/env/conda.sh
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"  # OPTango/env
CONDA_DIR="${SCRIPT_DIR}/miniconda3"

source ${CONDA_DIR}/bin/activate

conda activate optango

# EOF
