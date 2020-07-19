import sys
import os
from datetime import datetime
from pathlib import Path

from capsulenet import CapsNet, train
from data_generators import DataGen


SHAPE = int(os.getenv('SHAPE', 128))
INPUT_SHAPE = (SHAPE, SHAPE, 1)
DATA_DIR = os.getenv('DATA_DIR')
RESULTS_BASE_DIR = Path(os.getenv('RESULTS_BASE_DIR'))
EPOCHS = int(os.getenv('EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
ROUTINGS = int(os.getenv('ROUTINGS'))
LR = float(os.getenv('LR', 0.003))
LR_DECAY = float(os.getenv('LR_DECAY', 0.96))
RR = float(os.getenv('RR', 0.0000002))
REG_RATE = 0.000002
A = int(os.getenv('A', 64))
B = int(os.getenv('B', 8))
C = int(os.getenv('C', 16))
D = int(os.getenv('D', 16))
cckernel1 = int(os.getenv('cckernel1', 5))
cckernel2 = int(os.getenv('cckernel2', 5))

if __name__ == "__main__":
    cn = CapsNet(
        A=A,B=B,C=C,D=D,cckernel1=cckernel1,cckernel2=cckernel2,
        n_class=4, 
        input_shape=INPUT_SHAPE, 
        batch_size=BATCH_SIZE,
        routings=ROUTINGS,
        lr=LR,
        lr_decay=LR_DECAY,
        regularization_rate=RR
        )

    data_gen = DataGen(
        batch_size=BATCH_SIZE, 
        data_dir=DATA_DIR, 
        target_size=(SHAPE, SHAPE), 
        validation_split=0.2
        )
    cn_log_dir = RESULTS_BASE_DIR/cn.model.name/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(
        network=cn, 
        data_gen=data_gen, 
        save_dir=cn_log_dir, 
        epochs=EPOCHS
        )