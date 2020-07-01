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


if __name__ == "__main__":
    cn = CapsNet(
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
    cn_log_dir = RESULTS_BASE_DIR/cn.model.name/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_epochs_{EPOCHS}"
    train(
        network=cn, 
        data_gen=data_gen, 
        save_dir=cn_log_dir, 
        epochs=EPOCHS
        )