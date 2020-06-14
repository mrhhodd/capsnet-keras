#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J plgcapsnet_job
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=10GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcapsnet
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu:8


export LD_LIBRARY_PATH=/net/people/plgmwnetrzak/magisterka/cuda/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH=/net/people/plgmwnetrzak/magisterka/capsnet-keras:$PYTHONPATH 

SHAPE=124 \
DATA_DIR=/net/people/plgmwnetrzak/magisterka/data/124x124_OCT2017 \
RESULTS_BASE_DIR=/net/people/plgmwnetrzak/magisterka/result/124x124_OCT2017 \
EPOCHS=2 BATCH_SIZE=8 \
python3 /net/people/plgmwnetrzak/magisterka/capsnet-keras/main.py