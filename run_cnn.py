import os
from datetime import datetime
from pathlib import Path

from data_generators import DataGen
from utils import train, test
from tensorflow.keras import callbacks


DATA_DIR = os.getenv('DATA_DIR')
RESULTS_BASE_DIR = Path(os.getenv('RESULTS_BASE_DIR'))

MODEL_NAME = 'LeNet-5'
SHAPE = int(os.getenv('SHAPE', 128))
EPOCHS = int(os.getenv('EPOCHS', 30))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', '0.2'))

if __name__ == "__main__":
    cn_log_dir = RESULTS_BASE_DIR/MODEL_NAME / \
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    lenet_model = Sequential(name=MODEL_NAME)
    lenet_model.add(layers.Conv2D(filters=64, kernel_size=(9, 9), strides=1, activation='relu', input_shape=(SHAPE,SHAPE,1))
    lenet_model.add(layers.AveragePooling2D())
    lenet_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    lenet_model.add(layers.AveragePooling2D())
    lenet_model.add(layers.Flatten())
    lenet_model.add(layers.Dense(units=16, activation='relu'))
    lenet_model.add(layers.Dense(units=16, activation='relu'))
    lenet_model.add(layers.Dense(units=CLASSES, activation = 'softmax'))
    lenet_model.compile(loss=losses.squared_hinge, optimizer=optimizers.Adam(),
                        metrics=[accuracy, specificity, sensitivity, f1_score])
    lenet_model.summary()

    data_gen = DataGen(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        target_size=(SHAPE, SHAPE),
        validation_split=VALIDATION_SPLIT
    )

    print("Training model")

    cn_callbacks = [
        callbacks.CSVLogger(f"{cn_log_dir}/log.csv"),
        callbacks.LearningRateScheduler(
            schedule=lambda epoch, lr: lr * LR_DECAY)
    ]

    train(
        model=lenet_model,
        data_gen=data_gen,
        save_dir=cn_log_dir,
        epochs=EPOCHS,
        callbacks=cn_callbacks
    )
