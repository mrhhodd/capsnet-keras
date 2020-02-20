"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import numpy as np
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
K.set_image_data_format('channels_last')


class CapsNet():
    """
    How this class works:
    1. run CapsNet to get a object 'cn'
    2. run cn.load_data()
    3. run cn.create()
        was this:
            model, eval_model, manipulate_model = create_model(input_shape=x_train.shape[1:],
                                                    n_class=len(np.unique(np.argmax(y_train, 1))),
                                                    routings=self.args.routings)
            model.summary()
    4. if weights are provided self.args.weights != None: model.load_weights(self.args.weights) # init the model weights with provided one, default are random
    5. (train) if not self.args.testing: train(model=model, data=((x_train, y_train), (x_test, y_test)), args=self.args)
    6. (test) else:  
        manipulate_latent(manipulate_model, (x_test, y_test), self.args) # why use this???
        test(model=eval_model, data=(x_test, y_test), args=self.args)

    so like this:
    cn = CapsNet()
    cn.load_data(data)
    (optional) cn.load_weights
    cn.train()
    cn.test()

    """
    def __init__(self, epochs=50, batch_size=100, lr=0.001, lr_decay=0.9, lam_recon=0.392,
             routings=3, input_shape=[32,32,1], n_class=4,
             debug=False, weights=None, save_dir='/home/hod', data_path=''):
        self.args = locals()
        self.load_data(data_path)
        self.model, self.eval_model = self._create_model()
        os.makedirs(self.args['save_dir'], exist_ok=True)

    def load_data(self, rootdir):
        class_map = {"NORMAL":0, "CNV":1, "DME":2, "DRUSEN":3}
        data_sets = { "train": {},  "test": {},  "val": {} }
        for data_set_name, data_set in data_sets.items():
            f = []
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(rootdir, data_set_name)):
                f.extend([os.path.join(dirpath, filename) for filename in filenames])
            np.random.shuffle(f)
            temp_x = []
            temp_y = []
            for file_path in f:
                img_arr = img_to_array(load_img(path=file_path, grayscale=True))
                temp_x.append(img_arr)
                class_no = class_map[os.path.basename(file_path).split("-")[0]]
                temp_y.append(class_no)
            data_set["x"] = np.array(temp_x).reshape(-1, *self.args['input_shape']).astype('float32') / 255
            data_set["y"] = to_categorical(np.array(temp_y).astype('float32'), self.args['n_class'])

        self.data = data_sets

        # define modelcapsnet
    def _create_model(self):
            """
            A Capsule Network
            :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                    `eval_model` can also be used for training.
            """
            input_shape = self.args['input_shape']
            n_class = self.args['n_class']
            x = layers.Input(shape=input_shape)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

            # Layer 3: Capsule layer. Routing algorithm works here.
            digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=self.args['routings'],
                                    name='digitcaps')(primarycaps)

            # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
            # If using tensorflow, this will not be necessary. :)
            out_caps = Length(name='capsnet')(digitcaps)

            # ##############################################################################################
            # # Decoder network.
            # y = layers.Input(shape=(n_class,))
            # masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
            # masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

            # # Shared Decoder model in training and prediction
            # decoder = models.Sequential(name='decoder')
            # decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
            # decoder.add(layers.Dense(1024, activation='relu'))
            # decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
            # decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

            # # Models for training and evaluation (prediction)
            # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
            # eval_model = models.Model(x, [out_caps, decoder(masked)])
            # ##############################################################################################
            train_model = models.Model(x, out_caps)
            eval_model = models.Model(x, out_caps)

            return train_model, eval_model      

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
        # unpacking the data
        x_train = self.data['train']['x']
        y_train = self.data['train']['y']
        x_val = self.data['val']['x']
        y_val = self.data['val']['y']
        x_test = self.data['test']['x']
        y_test = self.data['test']['y']

        # callbacks
        log = callbacks.CSVLogger(self.args["save_dir"] + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=self.args["save_dir"] + '/tensorboard-logs',
                                batch_size=self.args["batch_size"] , histogram_freq=int(self.args["debug"]))
        checkpoint = callbacks.ModelCheckpoint(self.args["save_dir"] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                            save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.args["lr"] * (self.args["lr_decay"]  ** epoch))

        # compile the model
        # self.model.compile(optimizer=optimizers.Adam(lr=self.args["lr"] ),
        #             loss=[self.margin_loss, 'mse'],
        #             loss_weights=[1., self.args["lam_recon"] ],
        #             metrics={'capsnet': 'accuracy'})
        self.model.compile(optimizer=optimizers.Adam(lr=self.args["lr"] ),
                    loss=[self.margin_loss],
                    loss_weights=[1.],
                    metrics={'capsnet': 'accuracy'})

        # Training without data augmentation:
        # self.model.fit([x_train, y_train], [y_train, x_train], batch_size=self.args["batch_size"] , epochs=self.args["epochs"] ,
        #     validation_data=[[x_val, y_val], [y_val, x_val]], callbacks=[log, tb, checkpoint, lr_decay])
        self.model.fit(x_train, y_train, batch_size=self.args["batch_size"] , epochs=self.args["epochs"] ,
            validation_data=[x_val, y_val], callbacks=[log, tb, checkpoint, lr_decay])

        self.model.save_weights(self.args["save_dir"] + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.args["save_dir"])

        from utils import plot_log
        plot_log(self.args["save_dir"] + '/log.csv', show=True)

    def test(self):
        x_test = self.data['test']['x']
        y_test = self.data['test']['y']
        # y_pred, x_recon = self.eval_model.predict(x_test, batch_size=self.args["batch_size"])
        y_pred = self.eval_model.predict(x_test, batch_size=self.args["batch_size"])
        print('-'*30 + 'Begin: test' + '-'*30)
        print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

        # img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
        # image = img * 255
        # Image.fromarray(image.astype(np.uint8)).save(self.args["save_dir"] + "/real_and_recon.png")
        # print()
        # print('Reconstructed images are saved to %s/real_and_recon.png' % self.args["save_dir"])
        # print('-' * 30 + 'End: test' + '-' * 30)
        # plt.imshow(plt.imread(self.args["save_dir"] + "/real_and_recon.png"))
        # plt.show()