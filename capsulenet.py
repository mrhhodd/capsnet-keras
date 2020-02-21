"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
Based on: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import numpy as np
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length
K.set_image_data_format('channels_last')


class CapsNet():
    """
    cn = CapsNet(epochs=10, batch_size=100, 
        save_dir='/home/hod/64x64_oct2017_4k_examples_10epochs', 
        data_path="/home/hod/mag/64x64_oct2017_4k_examples", 
        input_shape=[64,64,1])
    cn.train() #or  cn.load_weights
    cn.test()

    """
    def __init__(self, epochs=50, batch_size=100,
            lr=0.001, lr_decay=0.9, lam_recon=0.392,
             routings=3, input_shape=[32,32,1], n_class=4,
             debug=False, weights=None, save_dir='/home/hod', data_path=''):
        self.args = locals()
        self.class_map = {"NORMAL":0, "CNV":1, "DME":2, "DRUSEN":3}
        self.data = self.load_data(data_path)
        # self.model = self._create_model()
        self.model = self._create_model_debug()
        if self.args["weights"]:
            self.load_weights()
        os.makedirs(self.args['save_dir'], exist_ok=True)

    def load_data(self, rootdir):
        data_sets = { "train": [],  "test": [],  "val": [] }
        for data_set_name, data_set in data_sets.items():
            files = []
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(rootdir, data_set_name)):
                files.extend([os.path.join(dirpath, filename) for filename in filenames])
            np.random.shuffle(files)
            data_set.extend(files)
        return data_sets

    def load_weights(self):
        self.model.load_weights(self.args["weights"])

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
                                    name='digitcaps')(primarycaps)

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(digitcaps)

            train_model = models.Model(x, out_caps)

            return train_model     

    def _create_model_debug(self):
            input_shape = self.args['input_shape']
            n_class = self.args['n_class']
            x = layers.Input(shape=input_shape)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = layers.Conv2D(filters=128, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

            # Layer 3: Capsule layer. Routing algorithm works here.
            caps_layer_1 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=self.args['routings'],
                                    name='caps_layer_1')(primarycaps)

            # Layer 3: Capsule layer. Routing algorithm works here.
            caps_layer_2 = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=self.args['routings'],
                                    name='caps_layer_2')(caps_layer_1)

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(caps_layer_2)

            train_model = models.Model(x, out_caps)

            return train_model 

    def margin_loss(self, y_true, y_pred):
        """
        Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
        """
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    def train(self):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
        :param args: arguments
        :return: The trained model
        """
        # callbacks
        log = callbacks.CSVLogger(self.args["save_dir"] + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=self.args["save_dir"] + '/tensorboard-logs',
                                batch_size=self.args["batch_size"] , histogram_freq=int(self.args["debug"]))
        checkpoint = callbacks.ModelCheckpoint(self.args["save_dir"] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                            save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.args["lr"] * (self.args["lr_decay"]  ** epoch))

        self.model.compile(optimizer=optimizers.Adam(lr=self.args["lr"] ),
                    loss=[self.margin_loss],
                    loss_weights=[1.],
                    metrics={'capsnet': 'accuracy'})

        self.model.fit_generator(generator=self._get_training_data(),
                            steps_per_epoch=np.ceil(len(self.data['train']) / self.args['batch_size']),
                            epochs=self.args["epochs"], 
                            validation_data=self._get_data(self.data["val"]), 
                            callbacks=[log, tb, checkpoint, lr_decay])

        self.model.save_weights(self.args["save_dir"] + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.args["save_dir"])

        from utils import plot_log
        plot_log(self.args["save_dir"] + '/log.csv', show=True)

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
                img_arr = img_to_array(load_img(path=file_path, grayscale=True))
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
            img_arr = img_to_array(load_img(path=file_path, grayscale=True))
            temp_x.append(img_arr)
            class_no = self.class_map[os.path.basename(file_path).split("-")[0]]
            temp_y.append(class_no)
        x = np.array(temp_x).reshape(-1, *self.args['input_shape']).astype('float32') / 255
        y = to_categorical(np.array(temp_y).astype('float32'), self.args['n_class'])
        return (x, y)