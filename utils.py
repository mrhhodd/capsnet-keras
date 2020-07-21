from os import makedirs

def train(model, data_gen, save_dir, epochs=30, callbacks=[]):
    makedirs(save_dir, exist_ok=True)
    model.fit(
        data_gen.training_generator,
        epochs=epochs,
        validation_data=data_gen.validation_generator,
        callbacks=callbacks
    )


def test(model, data_gen, save_dir, callbacks=[]):
    makedirs(save_dir, exist_ok=True)
    model.test(
        data_gen.validation_generator,
        callbacks=callbacks
    )


def log_results(model, log_dir, data_gen):
    # save model summary to file
    with open(log_dir/"model_summary.txt", "w") as f:
        with redirect_stdout(f):
            model.summary()

    # evaluate model and save results to file
    with open(log_dir/"evaluate.txt", "w") as f:
        with redirect_stdout(f):
            print(f"{model.name} result on a test set:")
            model.evaluate(data_gen.validation_generator)

    # save weights to file
    model.save_weights(str(log_dir/"trained_model.h5"))

    print(f"All logs available in:\n {log_dir}")
