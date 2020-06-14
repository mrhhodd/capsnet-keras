
import sys
import os
from datetime import datetime
from pathlib import Path

# sys.path.insert(0,'capsnet_keras')

from capsulenet import CapsNet, train
from data_generators import DataGen


SHAPE = int(os.getenv('SHAPE', 128))
INPUT_SHAPE = (SHAPE, SHAPE, 1)
DATA_DIR = os.getenv('DATA_DIR')
RESULTS_BASE_DIR = os.getenv('RESULTS_BASE_DIR')
EPOCHS = os.getenv('EPOCHS')
BATCH_SIZE = os.getenv('BATCH_SIZE')


if __name__ == "__main__":
    cn = CapsNet(n_class=4, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)

    data_gen = DataGen(
        batch_size=BATCH_SIZE, 
        data_dir=DATA_DIR, 
        target_size=INPUT_SHAPE[:2], 
        validation_split=0.1
        )
    cn_log_dir = RESULTS_BASE_DIR/cn.model.name/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_epochs_{EPOCHS}"
    train(
        network=cn, 
        data_gen=data_gen, 
        save_dir=cn_log_dir, 
        epochs=EPOCHS
        )