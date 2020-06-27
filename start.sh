#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J plgcapsnet_job
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=24
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=04:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcapsnet
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:6


export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 
module add plgrid/libs/tensorflow-gpu/2.2.0-python-3.8 

for LR in 1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001
do
    SHAPE=128 \
    DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/OCT2017_128x128_SBB/2 \
    RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/OCT2017_128x128_SBB/LRTEST_LR/ \
    EPOCHS=20 BATCH_SIZE=96 ROUTINGS=3 LR=$LR LR_DECAY=0.96 \
    python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/main.py
done
