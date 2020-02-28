"""
Capsule network implementation loosely based on CapsNet.
Loosely Based on: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import numpy as np
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
                 save_dir=None, data_dir=None,
                 debug=False):
        self.args = locals()
        self.data = {}
        self.class_map = {"NORMAL": 0, "CNV": 1, "DME": 2, "DRUSEN": 3}
        self.data_gen = ImageDataGenerator()

        self.model = self._create_model()
        os.makedirs(self.args['save_dir'], exist_ok=True)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def _create_model(self):
        # x = layers.Input(shape=input_shape)  
        # conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv1')(x)
        # primarycaps = PrimaryCaps(dim_capsule=8, capsules=32, kernel_size=9, strides=2, padding='valid')(conv1)
        # caps1 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=self.args['routings'],
        #                          name='caps1')(primarycaps)        
        # out_caps = Length(name='outputs')(caps1)

        x = layers.Input(shape=self.args['input_shape'])  
        conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv1')(x)
        primarycaps = PrimaryCaps(dim_capsule=4, capsules=16, kernel_size=9, strides=2, padding='valid')(conv1)
        caps1 = CapsuleLayer(num_capsule=self.args['n_class'], dim_capsule=8, routings=self.args['routings'],
                            name='caps1')(primarycaps)        
        out_caps = Length(name='outputs')(caps1)
        model.summary()

        return model

    def margin_loss(self, y_true, y_pred):
        lamb, margin = 0.5, 0.1
        return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
            1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

    def train(self):

        self.model.compile(optimizer=optimizers.Adam(lr=self.args["lr"] ),
                    loss=[self.margin_loss],
                    loss_weights=[1.],
                    metrics={'outputs': 'accuracy'})

        traninig_generator = self.data_gen.flow_from_directory(
            directory=os.path.join(self.args["data_dir"], "train"), 
            target_size=tuple(self.args["input_shape"][:2]), 
            color_mode='grayscale', 
            batch_size=self.args["batch_size"], 
            shuffle=True,
            seed=123)

        validation_generator = self.data_gen.flow_from_directory(
            directory=os.path.join(self.args["data_dir"], "val"), 
            target_size=tuple(self.args["input_shape"][:2]), 
            color_mode='grayscale', 
            batch_size=self.args["batch_size"], 
            shuffle=True) 

        self.model.fit(traninig_generator,
                       epochs=self.args["epochs"], 
                       validation_data=validation_generator,
                       callbacks=[
                            callbacks.CSVLogger(self.args["save_dir"] + '/log.csv'),
                            callbacks.TensorBoard(log_dir=self.args["save_dir"] + '/tensorboard-logs', histogram_freq=int(self.args["debug"])),
                            callbacks.ModelCheckpoint(self.args["save_dir"] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                            save_best_only=True, save_weights_only=True, verbose=1),
                            callbacks.LearningRateScheduler(schedule=lambda epoch: self.args["lr"] * (self.args["lr_decay"]  ** epoch))
                            ]
                        )

        self.model.save_weights(self.args["save_dir"] + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.args["save_dir"])
        
        plot_log(self.args["save_dir"] + '/log.csv', show=True)

    def test(self):
        test_generator = self.data_gen.flow_from_directory(
            directory=os.path.join(self.args["data_dir"], "test"), 
            target_size=tuple(self.args["input_shape"][:2]), 
            color_mode='grayscale', 
            batch_size=self.args["batch_size"], 
            shuffle=True)
        # pred_y = self.model.predict(test_x, batch_size=self.args["batch_size"])
        self.model.evaluate(test_generator)
        # print('Test acc:', np.sum(np.argmax(pred_y, 1) == np.argmax(test_y, 1))/test_y.shape[0])


if __name__ == "__main__":
    cn = CapsNet(epochs=1, batch_size=10, 
        save_dir="/home/hod/mag/results/OCT2017_preprocessed_128x128",
        data_dir="/home/hod/mag/data/OCT2017_preprocessed_128x128",
        input_shape=[128,128,1]
        )
    cn.train()
    # cn.test()