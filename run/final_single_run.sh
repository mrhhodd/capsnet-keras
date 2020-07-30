#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J plgcapsnet_job
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=8
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcapsnet
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:8

export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 
module add plgrid/libs/tensorflow-gpu/2.2.0-python-3.8 

SHAPE=128 \
DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/OCT2017_128x128_final/${1} \
EPOCHS=100 \
VALIDATION_SPLIT=${2} \
BATCH_SIZE=32 \
ROUTINGS=3 \
LR=0.04 \
LR_DECAY=0.96 \
RR=0.0000002 \
A=96 B=12 C=16 D=16 \
RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/finals/${2}/${1} \
python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/run.py


