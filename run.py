import os
from datetime import datetime
from pathlib import Path

from model import EmCapsNet
from data_generators import DataGen
from utils import train, test
from tensorflow.keras import callbacks


DATA_DIR = os.getenv('DATA_DIR') # Location where input data is stored
RESULTS_BASE_DIR = Path(os.getenv('RESULTS_BASE_DIR')) # Base location for output data 

MODEL_NAME = 'EM-CapsNet' 
SHAPE = int(os.getenv('SHAPE', 128))
EPOCHS = int(os.getenv('EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
ROUTINGS = int(os.getenv('ROUTINGS'))
LR = float(os.getenv('LR', 0.003))
LR_DECAY = float(os.getenv('LR_DECAY', 0.96))
RR = float(os.getenv('RR', 0.0000002))
A = int(os.getenv('A', 96))
B = int(os.getenv('B', 8))
C = int(os.getenv('C', 16))
D = int(os.getenv('D', 16))
USE_LR_DECAY = bool(os.getenv('USE_LR_DECAY', True))
WEIGHTS = os.getenv('WEIGHTS', '')
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', '0.2'))

if __name__ == "__main__":
    cn_log_dir = RESULTS_BASE_DIR/MODEL_NAME/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cn = EmCapsNet(
        name=MODEL_NAME,
        input_shape=(SHAPE, SHAPE, 1),
        batch_size=BATCH_SIZE,
        n_class=4,
        lr=LR,
        lr_decay=LR_DECAY,
        routings=ROUTINGS,
        regularization_rate=RR,
        A=A, B=B, C=C, D=D
    )

    data_gen = DataGen(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        target_size=(SHAPE, SHAPE),
        validation_split=VALIDATION_SPLIT
    )

    if WEIGHTS:
        print("Loading and evaluating model")
        cn.load_weights(str(WEIGHTS))
        test(
            model=cn.model,
            data_gen=data_gen,
            save_dir=cn_log_dir
        )
    else:
        print("Training model")

        cn_callbacks = [callbacks.CSVLogger(f"{cn_log_dir}/log.csv")]
        if USE_LR_DECAY:
            cn_callbacks.append(
                callbacks.LearningRateScheduler(
                    # schedule=lambda epoch, lr: lr * LR_DECAY ** min(20000.0, epoch))
                    schedule=lambda epoch, lr: lr * LR_DECAY)
            )

        train(
            model=cn.model,
            data_gen=data_gen,
            save_dir=cn_log_dir,
            epochs=EPOCHS,
            callbacks=cn_callbacks
        )