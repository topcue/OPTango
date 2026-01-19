#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # OPTango/env
CONDA_DIR="${SCRIPT_DIR}/miniconda3"

source ${CONDA_DIR}/bin/activate

conda tos accept

conda create -n optango python=3.12 -y
conda activate optango

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# EOF
