"""
Capsule network implementation based on CapsNet paper https://arxiv.org/pdf/1710.09829.pdf
Code is borrowing from: https://github.com/XifengGuo/CapsNet-Keras and https://github.com/bojone/Capsule/
"""

import os
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras import backend as K
from layers import PrimaryCaps, CapsuleLayer, Length
K.set_image_data_format('channels_last')


class CapsNet():
    def __init__(self,
                 input_shape=[32, 32, 1],
                 n_class=4,
                 routings=3,
                 lr=0.001,
                 lr_decay=0.9):
        self.input_shape = input_shape
        self.n_class = n_class
        self.routings = routings
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = self._create_model()

    def _create_model(self):
        """
        Create model of the capsule network.
        Model structure is a bit different than in the Hinton's paper - bigger images are generating much more parameters.
        Changes include:
            increased stride in the conv layer 1=>3
            increased stride in the primary caps layer 2=>3
        """
        model = models.Sequential(name='CapsNet')
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Conv2D(filters=256, kernel_size=9, strides=3,
                                padding='valid', activation='relu', name='conv1'))
        model.add(PrimaryCaps(dim_capsule=8, capsules=32, kernel_size=9,
                              strides=3, padding='valid', name='primary_caps'))
        model.add(CapsuleLayer(
            num_capsule=self.n_class, dim_capsule=16, routings=self.routings, name='caps1'))
        model.add(Length(name='outputs'))

        model.compile(optimizer=optimizers.Adam(lr=self.lr),
                      loss=self.margin_loss,
                      metrics=['accuracy'])

        model.summary()
        return model

    def margin_loss(self, y_true, y_pred):
        """
        Implemented as described in the paper
        """
        lambd = 0.5
        L = y_true * (K.maximum(0., 0.9 - y_pred)) + \
            lambd * (1 - y_true) * (K.maximum(0., y_pred - 0.1))
        return K.sum(K.sum(L, 1))


def train(network, data_gen, save_dir, epochs=30):
    network.model.fit(data_gen.training_generator,
              epochs=epochs,
              #   validation_data=data_gen.validation_generator,
              validation_data=data_gen.test_generator,
              callbacks=[
                  callbacks.CSVLogger(f"{save_dir}/log.csv"),
                  callbacks.LearningRateScheduler(
                      schedule=lambda epoch: network.lr * network.lr_decay ** epoch)
              ]
              )

    network.model.save_weights(f"{save_dir}/trained_model.h5")
    print(f"Logs available at {save_dir}")
