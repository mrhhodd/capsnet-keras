import os
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, regularizers, losses
from tensorflow.keras import backend as K
from layers import PrimaryCaps, ConvCaps, ClassCapsules
from metrics import specificity, sensitivity, f1_score
K.set_image_data_format('channels_last')

# TODO: Data generators have some issues - need to investigate this
# TODO: How to test? divide data into 10 data sets, then run K-fold validation for different validation splits?
# TODO: Do we need normalization in the m_step?
# TODO: Analyze the problems that Gritzman mentions!
# TODO: tests with data augmentation?

# TODO: Rethink the structure of my code
# TODO: move spread loss outisde
# TODO: tensorflow vs tensorflow.keras.backend??
# TODO: switch from direct tensorflow to keras.backend?
# TODO: ask about the logarithms
# TODO: docstrings?
# TODO: mypy
# TODO: all shapes in tuples
# TODO: all function arguments with names
# TODO: currently only for Heigth==Weight = worth expanding?
# TODO: add some exception for base class instantiation
# TODO: check for proper input shape
# TODO: pretty formating


class CapsNet():
    def __init__(self,
                 input_shape=[32, 32, 1],
                 batch_size=64,
                 lr=3e-3,
                 lr_decay=0.96,
                 n_class=4,
                 routings=3,
                 regularization_rate=0.0000002):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.n_class = n_class
        self.global_step = K.variable(value=0)
        self.routings = routings
        self.lr = lr
        self.lr_decay = lr_decay
        # "We use a weight decay loss with a small factor of .0000002 rather than the reconstruction loss.
        # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
        self.regularizer = regularizers.l2(regularization_rate)

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        #     self.model = self._create_model()
        self.model = self._create_model()

    def _create_model(self):
        # A = B = C = D = 32
        # smaller values for POCs
        A = 16
        B = 16
        C = 16
        D = 16
        inputs = layers.Input(shape=self.input_shape)
        conv = layers.Conv2D(
            # filters=A, kernel_size=5, strides=2,
            filters=A, kernel_size=9, strides=3,
            padding='same', activation='relu',
            name='conv1')(inputs)
        [pc_act, pc_pose] = PrimaryCaps(
            capsules=B, kernel_size=1, strides=1, padding='valid',
            name='primCaps')(conv)
        [cc1_act, cc1_pose] = ConvCaps(
            capsules=C, kernel_size=3, strides=2, padding='valid',
            routings=self.routings, weights_reg=self.regularizer,
            name='conv_caps_1')([pc_act, pc_pose])
        # [cc1a_act, cc1a_pose] = ConvCaps(
        #     capsules=C, kernel_size=3, strides=2, padding='valid',
        #     routings=self.routings, weights_reg=self.regularizer,
        #     name='conv_caps_1a')([cc1_act, cc1_pose])
        [cc2_act, cc2_pose] = ConvCaps(
            capsules=D, kernel_size=3, strides=1, padding='valid',
            routings=self.routings, weights_reg=self.regularizer,
            # name='conv_caps_2')([cc1a_act, cc1a_pose])
            name='conv_caps_2')([cc1_act, cc1_pose])
        [fc_act, fc_pose] = ClassCapsules(
            capsules=self.n_class, routings=self.routings, weights_reg=self.regularizer,
            name='class_caps')([cc2_act, cc2_pose])
        model = models.Model(inputs, fc_act, name='EM-CapsNet')

        model.compile(optimizer=optimizers.Adam(lr=self.lr),
                      loss=self.spread_loss,
                    #   loss=losses.SquaredHinge(reduction="auto", name="squared_hinge"),
                      metrics=['accuracy', specificity])#, sensitivity, f1_score])

        print(model.layers)

        model.summary()
        return model

    def spread_loss(self, y_true, y_pred):
        # "The margin that we set is:
        # margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 50000.0 - 4))
        # where step is the training step. We trained with batch size of 64."
        # https://openreview.net/forum?id=HJWLfGWRb

        m_min = 0.2
        m_delta = 0.79
        p = 50000.0 * 64.0 / self.batch_size
        # p = 10000.0 * 64.0 / self.batch_size
        margin = m_min + m_delta * \
            K.sigmoid(K.minimum(10.0, self.global_step / p - 4))
        a_i = K.reshape(tf.boolean_mask(y_pred, 1 - y_true), shape=(-1, self.n_class - 1))
        a_t = K.reshape(tf.boolean_mask(y_pred, y_true), shape=(-1, 1))
        loss = K.square(K.maximum(0., margin - (a_t - a_i)))
        self.global_step.assign(self.global_step + 1)
        return K.mean(K.sum(loss, axis=1, keepdims=True))


def train(network, data_gen, save_dir, epochs=30):
    os.makedirs(save_dir, exist_ok=True)
    network.model.fit(
        data_gen.training_generator,
        epochs=epochs,
        validation_data=data_gen.validation_generator,
        callbacks=[
            callbacks.CSVLogger(f"{save_dir}/log.csv"),
            # "We use an exponential decay with learning rate: 3e-3, decay_steps: 20000, decay rate: 0.96."
            # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
            callbacks.LearningRateScheduler(
                schedule=lambda epoch, lr: lr * network.lr_decay ** K.minimum(20000.0, epoch))
        ]
    )

    _log_results(network.model, save_dir, data_gen)


def _log_results(model, log_dir, data_gen):
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
