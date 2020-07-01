#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J plgcapsnet_job
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=24
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=1GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcapsnet
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:6


export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 
module add plgrid/libs/tensorflow-gpu/2.2.0-python-3.8 


for RR in 0.00002 0.000002 0.0000004 0.0000002 0.0000001 0.00000008 0.00000004 0.00000002
do
    SHAPE=128 \
    DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/OCT2017_128x128_SBB/5 \
    RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/OCT2017_128x128_SBB/FIND_RR_$RR/ \
    EPOCHS=20 \
    BATCH_SIZE=96 \
    ROUTINGS=3 \
    LR=0.04 \
    LR_DECAY=0.95 \
    RR=$RR \
    python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/main.py
done