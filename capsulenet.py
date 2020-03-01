"""
Capsule network implementation based on CapsNet paper https://arxiv.org/pdf/1710.09829.pdf
Code is borrowing from: https://github.com/XifengGuo/CapsNet-Keras and https://github.com/bojone/Capsule/
"""

import os
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from layers import PrimaryCaps, CapsuleLayer, Length
from utils import plot_log
K.set_image_data_format('channels_last')


class CapsNet():
    def __init__(self, input_shape=[32, 32, 1], n_class=4, routings=3,
                 epochs=50, batch_size=100,
                 lr=0.001, lr_decay=0.9,
                 save_dir=None, data_dir=None):
        self.args = locals()
        self.data = {}
        self.class_map = {"NORMAL": 0, "CNV": 1, "DME": 2, "DRUSEN": 3}
        self.data_gen = ImageDataGenerator()

        self.model = self._create_model()
        os.makedirs(self.args['save_dir'], exist_ok=True)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def _create_model(self):
        """
        Create model of the capsule network.
        Model structure is a bit different than in the Hinton's paper - bigger images are generating much more parameters.
        Changes include:
            increased stride in the conv layer 1=>3
            increased stride in the primary caps layer 2=>3
        """
        model = models.Sequential(name='CapsNet')
        model.add(layers.Input(shape=self.args['input_shape']))
        model.add(layers.Conv2D(filters=256, kernel_size=9, strides=3, padding='valid', activation='relu', name='conv1'))
        model.add(PrimaryCaps(dim_capsule=8, capsules=32, kernel_size=9, strides=3, padding='valid', name='primary_caps'))
        model.add(CapsuleLayer(num_capsule=self.args['n_class'], dim_capsule=16, routings=self.args['routings'], name='caps1'))
        model.add(Length(name='outputs'))

        model.compile(optimizer=optimizers.Adam(lr=self.args["lr"]),
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

    def train(self):
        traninig_generator = self.data_gen.flow_from_directory(
            directory=os.path.join(self.args["data_dir"], "train"),
            target_size=tuple(self.args["input_shape"][:2]),
            color_mode='grayscale',
            batch_size=self.args["batch_size"],
            shuffle=True,
            seed=123)

        validation_generator = self.data_gen.flow_from_directory(
            # directory=os.path.join(self.args["data_dir"], "val"),
            directory=os.path.join(self.args["data_dir"], "test"),
            target_size=tuple(self.args["input_shape"][:2]),
            color_mode='grayscale',
            batch_size=self.args["batch_size"],
            shuffle=True)

        self.model.fit(traninig_generator,
                       epochs=self.args["epochs"],
                       validation_data=validation_generator,
                       callbacks=[
                           callbacks.CSVLogger(f"{self.args['save_dir']}/log.csv"),
                           callbacks.LearningRateScheduler(
                               schedule=lambda epoch: self.args["lr"] * (self.args["lr_decay"] ** epoch))
                       ]
                       )

        model_file = f"{self.args['save_dir']}/trained_model.h5"
        self.model.save_weights(model_file)
        print(f"Model saved to {model_file}")
        plot_log(f"{self.args['save_dir']}/log.csv", show=True)

    def test(self):
        test_generator = self.data_gen.flow_from_directory(
            directory=os.path.join(self.args["data_dir"], "test"),
            target_size=tuple(self.args["input_shape"][:2]),
            color_mode='grayscale',
            batch_size=self.args["batch_size"],
            shuffle=True)
        self.model.evaluate(test_generator)


if __name__ == "__main__":
    cn = CapsNet(epochs=1, batch_size=2, 
        save_dir="/home/hod/mag/results/OCT2017_preprocessed_128x128",
        data_dir="/home/hod/mag/data/OCT2017_preprocessed_128x128",
        input_shape=[128,128,1]
        )
    cn.train()
    # cn.test()