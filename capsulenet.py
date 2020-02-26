"""
Capsule network implementation loosely based on CapsNet.
Loosely Based on: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
# from capsulelayers import CapsuleLayer, PrimaryCap, Length, CapsuleLayerNew
from capsulelayers import PrimaryCap
from layers import CapsuleLayer, Length
K.set_image_data_format('channels_last')


class CapsNet():
    def __init__(self, input_shape=[32, 32, 1], n_class=4, routings=3,
                 epochs=50, batch_size=100,
                 lr=0.001, lr_decay=0.9,
                 save_dir='/home/hod',
                 debug=False):
        self.args = locals()
        self.data = None
        self.class_map = {"NORMAL": 0, "CNV": 1, "DME": 2, "DRUSEN": 3}
        self.model = self._create_model()
        os.makedirs(self.args['save_dir'], exist_ok=True)
    
    def set_data(self, data):
        self.data = data

    def load_weights(self, weights):
        self.model.load_weights(weights)

        # define modelcapsnet
    def _create_model(self):
            input_shape = self.args['input_shape']
            n_class = self.args['n_class']
            x = layers.Input(shape=input_shape)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=3, padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

            # Layer 3: Capsule layer. Routing algorithm works here.
            digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=self.args['routings'],
                                     share_weights=False, name='digitcaps')(primarycaps)

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(digitcaps)

            return models.Model(x, out_caps) 

    def margin_loss(self, y_true, y_pred):
        lamb, margin = 0.5, 0.1
        return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
            1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

    def train(self):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
        :param args: arguments
        :return: The trained model
        """

        self.model.compile(optimizer=optimizers.Adam(lr=self.args["lr"] ),
                    loss=[self.margin_loss],
                    loss_weights=[1.],
                    metrics={'capsnet': 'accuracy'})

        self.model.fit(x=self._get_training_data(),
                       steps_per_epoch=np.ceil(len(self.data['train']) / self.args['batch_size']),
                       epochs=self.args["epochs"], 
                       validation_data=self._get_data(self.data["val"]), 
                       callbacks=[
                           callbacks.CSVLogger(self.args["save_dir"] + '/log.csv'),
                           callbacks.TensorBoard(log_dir=self.args["save_dir"] + '/tensorboard-logs', histogram_freq=int(self.args["debug"])),
                           callbacks.ModelCheckpoint(self.args["save_dir"] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                            save_best_only=True, save_weights_only=True, verbose=1),
                           callbacks.LearningRateScheduler(schedule=lambda epoch: self.args["lr"] * (self.args["lr_decay"]  ** epoch))
                        ])

        self.model.save_weights(self.args["save_dir"] + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.args["save_dir"])

    def _get_training_data(self):
        data_generator = self._training_example_generator()
        while True:
            temp_x = []
            temp_y = []
            while len(temp_x) < self.args["batch_size"]:
                (raw_x, raw_y) = next(data_generator)
                temp_x.append(raw_x)
                temp_y.append(raw_y)
            x = np.array(temp_x).reshape(-1, *self.args['input_shape']).astype('float32') / 255
            y = to_categorical(np.array(temp_y).astype('float32'), self.args['n_class'])
            yield (x, y)

    def _training_example_generator(self):
        while True:
            for file_path in self.data["train"]:
                img_arr = img_to_array(load_img(path=file_path, color_mode="grayscale"))
                class_no = self.class_map[os.path.basename(file_path).split("-")[0]]        
                yield (img_arr, class_no)

    def test(self):
        (x_test , y_test) = self._get_data(self.data["test"])
        y_pred = self.model.predict(x_test, batch_size=self.args["batch_size"])
        print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    def _get_data(self, data_set):
        temp_x = []
        temp_y = []
        for file_path in data_set:
            img_arr = img_to_array(load_img(path=file_path, color_mode="grayscale"))
            temp_x.append(img_arr)
            class_no = self.class_map[os.path.basename(file_path).split("-")[0]]
            temp_y.append(class_no)
        x = np.array(temp_x).reshape(-1, *self.args['input_shape']).astype('float32') / 255
        y = to_categorical(np.array(temp_y).astype('float32'), self.args['n_class'])
        return (x, y)

if __name__ == "__main__":
    from utils import load_data

    data = load_data("/home/hod/mag/data/OCT2017_preprocessed_128x128", balance_data=True)
    cn = CapsNet(epochs=10, batch_size=10, 
        save_dir="/home/hod/mag/results/OCT2017_preprocessed_128x128", 
        input_shape=[128,128,1]
        )
    cn.set_data(data)
    cn.train()

    # cn.test()