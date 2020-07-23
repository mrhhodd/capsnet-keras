#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J plgcapsnet_job
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcapsnet
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:4


# export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
# export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 
# module add plgrid/libs/tensorflow-gpu/2.2.0-python-3.8 

echo "${SPLIT} ${I}" 
# SHAPE=128 \
# DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/OCT2017_128x128_SBB/${split}/${i} \
# EPOCHS=30 \
# VALIDATION_SPLIT=${split} \
# BATCH_SIZE=12 \
# ROUTINGS=2 \
# LR=0.04 \
# LR_DECAY=0.97 \
# USE_LR_DECAY=True \
# RR=0.0000002 \
# A=96 B=12 C=16 D=16 \
# RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/finals/ \
# python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/run.py


