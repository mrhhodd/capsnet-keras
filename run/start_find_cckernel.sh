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


export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 
module add plgrid/libs/tensorflow-gpu/2.2.0-python-3.8 

for kernel in 3 5 7 9
do
    SHAPE=128 \
    DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/OCT2017_128x128_SBB/2 \
    EPOCHS=20 \
    BATCH_SIZE=16 \
    ROUTINGS=2 \
    LR=0.04 \
    LR_DECAY=0.97 \
    RR=0.0000002 \
    A=64, B=8,C=16,D=16, cckernel=${kernel},
    RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/OCT2017_128x128_SBB/FIND_ROUTING___${EPOCHS}EPOCHS_${ROUTINGS}ROUTINGS_${LR}LR_${LR_DECAY}LR_DECAY_${RR}RR_${A}A_${B}B_${C}C_${D}D_${cckernel}cckernel/ \
    python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/main.py
done
