#!/usr/bin/env bash
nvidia-smi
# conda env create -f environment.yml
# conda update -n base -c defaults conda
# source activate yxy
# chmod +rwx PCDARTS-cifar10
# sudo chmod 777 "/abhibha-volume/DARTS-cifar10" && echo "The file is now writable"
# chmod 777 PCDARTS-cifar10 
# if [ ! -w "/abhibha-volume/PCDARTS-cifar10" ]
# then
#   sudo chmod u+w "/abhibha-volume/PCDARTS-cifar10" && echo "The file is now writable"
# else
#   echo "The file is already writable"
# fi
# python /abhibha-volume/PCDARTS-cifar10/arch_search.py
CUDA_LAUNCH_BLOCKING=1 python /abhibha-volume/PCDARTS-cifar10/train.py 
# python /abhibha-volume/PCDARTS-cifar10/test_model.py 
# python visualize.py 'arch_name'