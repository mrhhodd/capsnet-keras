#!/bin/bash
#PBS -A plgmwnetrzak2020a
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --time 1:00:00

export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda:$LD_LIBRARY_PATH 

SHAPE=124 \
DATA_DIR=/people/plgmwnetrzak/data/124x124_OCT2017 \
RESULTS_BASE_DIR=/people/plgmwnetrzak/result/124x124_OCT2017 \
EPOCHS=2 BATCH_SIZE=16 \
python /people/plgmwnetrzak/capsnet_keras/main.py