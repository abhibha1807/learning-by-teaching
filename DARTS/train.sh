#!/usr/bin/env bash
nvidia-smi
# conda env create -f environment.yml
# conda update -n base -c defaults conda
# source activate yxy
# python arch_search.py
# python train.py
python train.py --arch 'arch_name'
