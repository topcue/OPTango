#!/usr/bin/bash

# https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

mkdir -p ${HOME}/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ${HOME}/miniconda3/miniconda.sh -b -u -p ${HOME}/miniconda3
rm ${HOME}/miniconda3/miniconda.sh

source ${HOME}/miniconda3/bin/activate

conda init --all

conda config --set auto_activate_base false

# EOF
