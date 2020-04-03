"""
Capsule network implementation based on CapsNet paper https://arxiv.org/pdf/1710.09829.pdf
Code is borrowing from: https://github.com/XifengGuo/CapsNet-Keras and https://github.com/bojone/Capsule/
"""

import os
from contextlib import redirect_stdout
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras import backend as K
from layers import PrimaryCaps, ConvCaps, ClassCapsules
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
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = self._create_model()

    def _create_model(self):
        """
        Create model of the capsule network.
        Model structure is a bit different than in the Hinton's paper - bigger images are generating much more parameters.
        Changes include:
            increased stride in the conv layer 1=>3
            increased stride in the primary capsule layer 2=>3
            decreased number of filters in a conv layer 256 => 128
            halved the number of capsules in primary capsule layer, 32=>16
            decreased dimensions of capsule in the capsule layer 16=>12
        """
        inputs = layers.Input(shape=self.input_shape)
        # model.add(preprocessing_layer)
        conv = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
        [pc_act, pc_pose] = PrimaryCaps(capsules=32, kernel_size=1, strides=1, padding='valid', name='primCaps')(conv)
        [cc1_act, cc1_pose] = ConvCaps(capsules=32, kernel_size=3, strides=2, padding='valid', routings=3, name='conv_caps_1')([pc_act, pc_pose])
        [cc2_act, cc2_pose] = ConvCaps(capsules=32, kernel_size=3, strides=1, padding='valid', routings=3, name='conv_caps_2')([cc1_act, cc1_pose])
        [fc_act, fc_pose] = ClassCapsules(capsules=self.n_class, routings=3, name='class_caps')([cc2_act, cc2_pose])
        model = models.Model(inputs, fc_act, name='EM-CapsNet')

        model.compile(optimizer=optimizers.Adam(lr=0.001),
                        loss=self.spread_loss,
                        metrics=['accuracy'])
        
        print(model.layers)

        model.summary()
        return model

    def spread_loss(self, y_true, y_pred):
        print("spread loss", y_true.shape, y_pred.shape)
        return 1.0
        # global_step = tf.to_float(tf.compat.v1.train.get_or_create_global_step())
        # m_min = 0.2
        # m_delta = 0.79
        # margin = (m_min 
        #     + m_delta * tf.sigmoid(tf.minimum(10.0, global_step / 50000.0 - 4)))

        # num_class = 4

        # y = tf.one_hot(y, num_class, dtype=tf.float32)
        
        # # Get the score of the target class
        # # (64, 1, 5)
        # scores = tf.reshape(scores, shape=[batch_size, 1, num_class])
        # # (64, 5, 1)
        # y = tf.expand_dims(y, axis=2)
        # # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
        # at = tf.matmul(scores, y)
        
        # # Compute spread loss, paper eq (3)
        # loss = tf.square(tf.maximum(0., m - (at - scores)))
        
        # # Sum losses for all classes
        # # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
        # # e.g loss*[1 0 1 1 1]
        # loss = tf.matmul(loss, 1. - y)
        
        # # Compute mean
        # loss = tf.reduce_mean(loss)


def train(network, data_gen, save_dir, epochs=30):
    os.makedirs(save_dir, exist_ok=True)
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