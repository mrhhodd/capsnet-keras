from tensorflow.keras import initializers, layers
import tensorflow.keras.backend as K
import tensorflow as tf


def squash(x, axis=-1):
    """
    Activation squashing function as defined in the paper.
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * x / K.sqrt(s_squared_norm + K.epsilon())

# define our own softmax function instead of K.softmax -- why???
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


class Length(layers.Layer):
    """
    Replace vector with its length
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(Length, self).get_config()


class CapsuleLayer(layers.Layer):
    """
    Capsule layer implemented as in paper.
    inputs: shape=[None, in_num_capsule, in_dim_capsule]
    output: shape=[None, num_vectors]
    """

    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = squash

    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)
        self.W = self.add_weight(name='capsule_kernel',
                                 shape=(input_shape[1],
                                        input_shape[2],
                                        self.num_capsule * self.dim_capsule),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, u_vecs):
        u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        # shape = [None, num_capsule, input_num_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        return super(CapsuleLayer, self).get_config()


class PrimaryCaps(layers.Conv2D):
    """
    Primary Capsule layer implemented as in paper.
    inputs: [None, in_filter_height, in_filter_width, num_filters]
    output: shape=[None, num_capsule, dim_capsule]
    """
    def __init__(self, dim_capsule, capsules, **kwargs):
        super(PrimaryCaps, self).__init__(
            filters=dim_capsule*capsules, **kwargs)
        self.activation = squash
        self.dim_capsule = dim_capsule
        self.capsules = capsules

    def call(self, inputs):
        raw_outputs = super(PrimaryCaps, self).call(inputs)
        outputs = layers.Reshape(target_shape=self.compute_output_shape(
            raw_outputs.shape)[1:])(raw_outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        num_capsule = int(
            input_shape[1] * input_shape[2] * input_shape[3] / self.dim_capsule)
        return (None, num_capsule, self.dim_capsule)

    def get_config(self):
        return super(Length, self).get_config()
