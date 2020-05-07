#!/bin/bash
#PBS -A plgmwnetrzak2020a
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time 1:00:00

DATA_DIR=/people/plgmwnetrzak/data/124x124_OCT2017_BAL \
RESULTS_BASE_DIR=/people/plgmwnetrzak/result/124x124_OCT2017_BAL \
EPOCHS=2 BATCH_SIZE=16 \
python /people/plgmwnetrzak/capsnet_keras/main.py