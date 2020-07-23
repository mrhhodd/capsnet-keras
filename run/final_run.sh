#!/bin/bash -l
for split in 0.2 0.3 0.4; do
    for ((i=0;i<=9;i++)); do
        I=${i} SPLIT=${split} /net/people/plgmwnetrzak/magisterka/capsnet-keras/run/final_single_run.sh
    done
done
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


