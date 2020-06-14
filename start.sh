#!/bin/bash
#PBS -A plgmwnetrzak2020a
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --time 1:00:00

export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 

SHAPE=124 \
DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/124x124_OCT2017 \
RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/124x124_OCT2017 \
EPOCHS=2 BATCH_SIZE=16 \
python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/main.py