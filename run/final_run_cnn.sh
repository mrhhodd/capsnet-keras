#!/bin/bash -l
for split in 0.2 0.3 0.4; do
    for ((i=0;i<=9;i++)); do
        sbatch /net/people/plgmwnetrzak/magisterka/capsnet-keras/run/final_single_run_cnn.sh ${i} ${split}
    done
done
