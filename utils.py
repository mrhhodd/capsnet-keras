from os import makedirs, getenv
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, initializers
from contextlib import redirect_stdout


def train(model, data_gen, save_dir, epochs, callbacks):
    makedirs(save_dir, exist_ok=True)
    model.fit(
        data_gen.training_generator,
        epochs=epochs,
        validation_data=data_gen.validation_generator,
        callbacks=callbacks
    )
    model.save_weights(f"{save_dir}/trained_model.h5")
    _log_results(model, data_gen, save_dir)


def test(model, data_gen, save_dir):
    makedirs(save_dir, exist_ok=True)
    _log_results(model, data_gen, save_dir)


def _log_results(model, data_gen, save_dir):
    # evaluate model and save results to file
    with open(f"{save_dir}/evaluate.txt", "w") as f:
        with redirect_stdout(f):
            print(f"{model.name} result on a test set:")
            model.evaluate(data_gen.validation_generator)

    print(f"Logs available in:\n {save_dir}")


def matrix_initializer(shape, dtype):
    return initializers.Identity()(shape=shape[-2:], dtype=dtype) \
        + initializers.RandomUniform(minval=-0.03,
                                     maxval=0.03)(shape=shape, dtype=dtype)