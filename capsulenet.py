"""
Capsule network implementation based on CapsNet paper https://arxiv.org/pdf/1710.09829.pdf
Code is borrowing from: https://github.com/XifengGuo/CapsNet-Keras and https://github.com/bojone/Capsule/
"""

import os
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, regularizers
from tensorflow.keras import backend as K
from layers import PrimaryCaps, ConvCaps, ClassCapsules
from metrics import specificity, sensitivity, f1_score
K.set_image_data_format('channels_last')

# done:
# TODO: Spread loss function
# TODO: learnign rate?
# TODO: optimizer? expontential decay?
# TODO: simple data normalization
# TODO: Regularizations?

# TODO: How to test? divide data into 10 data sets, then run K-fold validation for different validation splits?
# TODO: Do we need normalization in the m_step?
# TODO: Analyze the problems that Gritzman mentions!
# TODO: tests with data augmentation?

# TODO: Rethink the structure of my code
# TODO: spread loss - wtf the numbers?
# TODO: spread loss - what is the relation between this number and a batch size?
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
# TODO: preety formating

class CapsNet():
    def __init__(self,
                 input_shape=[32, 32, 1],
                 lr=3e-3,
                 lr_decay=0.96,
                 n_class=4,
                 routings=3):
        self.input_shape = input_shape
        self.n_class = n_class
        # self.global_step = 0
        self.global_step = K.variable(value=0)
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = self._create_model()

   
    def _create_model(self):
        # "We use a weight decay loss with a small factor of .0000002 rather than the reconstruction loss.
        # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
        # A = B = C = D = 32
        A = 8
        B = 8
        C = 8
        D = 8
        reg = regularizers.l2(0.0000002)
        inputs = layers.Input(shape=self.input_shape)
        conv = layers.Conv2D(filters=A, kernel_size=5, strides=2,
                             padding='same', activation='relu', name='conv1')(inputs)
        [pc_act, pc_pose] = PrimaryCaps(
            capsules=B, kernel_size=1, strides=1, padding='valid', name='primCaps')(conv)
        [cc1_act, cc1_pose] = ConvCaps(capsules=C, kernel_size=3, strides=2, padding='valid',
                                       routings=3, weights_reg=reg, name='conv_caps_1')([pc_act, pc_pose])
        [cc2_act, cc2_pose] = ConvCaps(capsules=D, kernel_size=3, strides=1, padding='valid',
                                       routings=3, weights_reg=reg, name='conv_caps_2')([cc1_act, cc1_pose])
        [fc_act, fc_pose] = ClassCapsules(
            capsules=self.n_class, routings=3, weights_reg=reg, name='class_caps')([cc2_act, cc2_pose])
        model = models.Model(inputs, fc_act, name='EM-CapsNet')

        model.compile(optimizer=optimizers.Adam(lr=self.lr),
                      loss=self.spread_loss,
                      metrics=['accuracy', specificity, sensitivity, f1_score])

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
        p = 50000.0 * 64.0
        margin = m_min + m_delta * K.sigmoid(K.minimum(10.0, self.global_step / p - 4))
        tf.print("################# STEP?")
        tf.print(self.global_step)
        tf.print(self.global_step + 1)
        tf.print("################# STEP!")
        a_i = tf.multiply(1 - y_true, y_pred)
        a_i = a_i[a_i != 0]
        a_t = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=1, keepdims=True)
        loss = K.square(K.maximum(0., margin - (a_t - a_i)))
        # self.global_step += 1
        # self.global_step.assign(self.global_step+1)
        return K.mean(K.sum(loss))

class changeAlpha(callbacks.Callback):
    def __init__(self, alpha):
        super(changeAlpha, self).__init__()
        self.alpha = alpha 

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.alpha, epoch)
             

def train(network, data_gen, save_dir, epochs=30):
    os.makedirs(save_dir, exist_ok=True)
    network.model.fit(
        data_gen.training_generator,
        epochs=epochs,
        validation_data=data_gen.validation_generator,
        callbacks=[
            callbacks.CSVLogger(f"{save_dir}/log.csv"),
            # We use an exponential decay with learning rate: 3e-3, decay_steps: 20000, decay rate: 0.96.""
            # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
            callbacks.LearningRateScheduler(
                schedule=lambda epoch, lr: lr * network.lr_decay ** K.minimum(20000.0, epoch)),
            changeAlpha(alpha=network.global_step)
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
            model.evaluate(data_gen.test_generator)

    # save weights to file
    model.save_weights(str(log_dir/"trained_model.h5"))

    print(f"All logs available in:\n {log_dir}")
