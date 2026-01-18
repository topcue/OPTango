#!/usr/bin/bash

source ${HOME}/miniconda3/bin/activate

conda tos accept

conda create -n optango python=3.12 -y
conda activate optango
# conda install pandas tqdm -y

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# EOF
