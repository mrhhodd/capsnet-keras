import os
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras import backend as K
from layers import PrimaryCaps, ConvCaps, ClassCaps
from metrics import accuracy, specificity, sensitivity, f1_score

K.set_image_data_format('channels_last')

class EmCapsNet():
    def __init__(self,
                 name,
                 input_shape,
                 batch_size,
                 n_class,
                 lr,
                 lr_decay,
                 routings,
                 regularization_rate,
                 A, B, C, D,
                 starting_step=0
                 ):
        self.model_name = name
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.n_class = n_class
        self.routings = routings
        self.lr = lr
        self.lr_decay = lr_decay
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.regularizer = regularizers.l2(regularization_rate)
        self.global_step = K.variable(value=starting_step)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self._create_model()

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def _create_model(self):
        inputs = layers.Input(shape=self.input_shape)
        conv = layers.Conv2D(
            filters=self.A, kernel_size=9, strides=2,
            padding='same', activation='relu',
            name='conv')(inputs)
        [pc_act, pc_pose] = PrimaryCaps(
            capsules=self.B, kernel_size=1, strides=1, padding='valid',
            name='prim_caps')(conv)
        [cc1_act, cc1_pose] = ConvCaps(
            capsules=self.C, kernel_size=3, strides=2, padding='valid',
            routings=self.routings, weights_reg=self.regularizer,
            name='conv_caps_1')([pc_act, pc_pose])
        [cc2_act, cc2_pose] = ConvCaps(
            capsules=self.D, kernel_size=3, strides=1, padding='valid',
            routings=self.routings, weights_reg=self.regularizer,
            name='conv_caps_2')([cc1_act, cc1_pose])
        [fc_act, fc_pose] = ClassCaps(
            capsules=self.n_class, routings=self.routings, weights_reg=self.regularizer,
            name='class_caps')([cc2_act, cc2_pose])

        model = models.Model(inputs, fc_act, name=self.model_name)
        model.compile(optimizer=optimizers.Adam(lr=self.lr),
                      loss=self._spread_loss,
                      metrics=[accuracy, specificity, sensitivity, f1_score])

        model.summary()
        return model

    def _spread_loss(self, y_true, y_pred):
        m_min = 0.2
        m_delta = 0.79
        p = 12000 / self.batch_size
        margin = m_min + m_delta * \
            K.sigmoid(K.minimum(10.0, self.global_step / p - 4))
        a_i = K.reshape(tf.boolean_mask(y_pred, 1 - y_true),
                        shape=(-1, self.n_class - 1))
        a_t = K.reshape(tf.boolean_mask(y_pred, y_true), shape=(-1, 1))
        loss = K.square(K.maximum(0., margin - (a_t - a_i)))
        self.global_step.assign(self.global_step + 1)
        return K.mean(K.sum(loss, axis=1, keepdims=True))
