import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import layers, activations


class PrimaryCaps(layers.Layer):
    def __init__(self, capsules, strides, padding, kernel_size, **kwargs):
        self.capsules = capsules
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        super(PrimaryCaps, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pose_weights = self.add_weight(name='pose',
                                            shape=(self.kernel_size,
                                                   self.kernel_size,
                                                   input_shape[-1],
                                                   self.capsules * 16),
                                            initializer='glorot_uniform',
                                            trainable=True)
        self.act_weights = self.add_weight(name='act',
                                           shape=(self.kernel_size,
                                                  self.kernel_size,
                                                  input_shape[-1],
                                                  self.capsules),
                                           initializer='glorot_uniform',
                                           trainable=True)

    def call(self, inputs):
        spatial_size = int(inputs.shape[1])

        pose = K.conv2d(inputs, self.pose_weights, strides=(1, 1), padding=self.padding, data_format='channels_last')
        pose = K.reshape(pose, shape=[spatial_size, spatial_size, self.capsules, 16])

        act = K.conv2d(inputs, self.act_weights, strides=(1, 1), padding=self.padding, data_format='channels_last')
        act = activations.sigmoid(act)
        act = K.reshape(act, shape=[spatial_size, spatial_size, self.capsules, 1])

        return act, pose

    def compute_output_shape(self, input_shape):
        spatial_shape = (None, input_shape[1], input_shape[2])
        return (spatial_shape + (self.capsules, 1)), (spatial_shape + (self.capsules, 16))


class BaseCaps(layers.Layer):
    def __init__(self, capsules, routings, **kwargs):
        # currently only for Heigth==Weight = worth expanding?
        self.capsules = capsules
        self.routing_method = em_routing
        self.routings = routings
        super(BaseCaps, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add check for the input shape!
        # Add check for valid output size?
        [in_act_shape, in_pose_shape] = input_shape
        print(self.name, "input_act shape", in_act_shape)
        print(self.name, "input_pose shape", in_pose_shape)
        self.spatial_size_in = in_act_shape[0]
        self.in_capsules = in_act_shape[2]
        self.transformation_weights = self.add_weight(name='transformation_weights',
                                                      shape=(1, self.kernel_size * self.kernel_size * self.in_capsules,
                                                             self.capsules,
                                                             4,
                                                             4),
                                                      initializer='glorot_uniform',
                                                      trainable=True)
        print(self.name, "transformation_weights shape", self.transformation_weights.shape)
        # this should be initialized differently??
        self.beta_v = self.add_weight(
                name='beta_v',
                shape=[1, 1, self.capsules, 1],
                initializer='glorot_uniform',
                trainable=True)
        self.beta_a = self.add_weight(
                name='beta_a',
                shape=[1, 1, self.capsules],
                initializer='glorot_uniform',
                trainable=True)
        print(self.name, "beta shapes:", self.beta_v.shape, self.beta_a.shape)


class ConvCaps(BaseCaps):
    def __init__(self, capsules, strides, padding, kernel_size, routings, **kwargs):
        # currently only for Heigth==Weight = worth expanding?
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        super(ConvCaps, self).__init__(capsules=capsules, routings=routings, **kwargs)

    def build(self, input_shape):
        super(ConvCaps, self).build(input_shape)
        self.spatial_size_out = int((self.spatial_size_in - self.kernel_size) / self.strides + 1)
        self.voting_map, self.child_parent_map = self._generate_voting_map(
                self.spatial_size_in,
                self.spatial_size_out,
                self.kernel_size)
        print(self.name, "voting_map shape", self.voting_map.shape)

    def _generate_voting_map(self, size_in, size_out, kernel_size):
        voting_map = np.zeros((size_out ** 2, size_in ** 2))
        for parent_id in range(0, size_out ** 2):
            for kernel_row in range(0, kernel_size):
                start = (kernel_row * size_in) + (parent_id % size_out) + (parent_id // size_out * size_in)
                voting_map[parent_id][start:start + kernel_size] = 1.0

        tmp = np.where(voting_map)
        child_parent_map = np.reshape(tmp[1], [size_out ** 2, -1])

        return voting_map, child_parent_map

    def call(self, inputs):
        [input_act, input_pose] = inputs
        print(self.name, "child_parent_map shape", self.child_parent_map.shape)
        input_pose = K.reshape(input_pose, [self.spatial_size_in ** 2, self.in_capsules, 4, 4])
        input_act = K.reshape(input_act, [self.spatial_size_in ** 2, self.in_capsules, 1])

        # input_pose_tiled = K.tile(input_pose_tiled, [self.spatial_size_out**2, 1, 1, 1, 1])
        print(self.name, "input_pose reshaped", input_pose.shape)
        input_pose_filtered = tf.gather(input_pose, self.child_parent_map, axis=0)
        input_act_filtered = tf.gather(input_act, self.child_parent_map, axis=0)
        print(self.name, "input_pose filtered", input_pose_filtered.shape)
        input_pose_tiled = K.reshape(input_pose_filtered, [self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 4, 4])
        input_act_tiled = K.reshape(input_act_filtered, [self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 1])
        input_pose_tiled = K.tile(input_pose_tiled, [1, 1, self.capsules, 1, 1])
        input_act_tiled = K.tile(input_act_tiled, [1, 1, self.capsules, 1])
        print(self.name, "input_pose_tiled", input_pose_tiled.shape)
        weights_tiled = K.tile(self.transformation_weights, [self.spatial_size_out ** 2, 1, 1, 1, 1])
        print(self.name, "weights tiled shape", weights_tiled.shape)
        # input_pose_filtered = tf.squeeze(input_pose_filtered)
        votes = tf.matmul(input_pose_tiled, weights_tiled)
        votes = K.reshape(votes, (self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, self.capsules, 16))
        print(self.name, "votes shape", votes.shape)
        out_act, out_pose = self.routing_method(input_act_tiled, votes, self.beta_a, self.beta_v, self.routings)
        out_act = K.reshape(out_act, [self.spatial_size_out, self.spatial_size_out, self.capsules, 1])
        out_pose = K.reshape(out_pose, [self.spatial_size_out, self.spatial_size_out, self.capsules, 4, 4])
        # print("OUT_SHAPE:", self.compute_output_shape(input.shape))
        return out_act, out_pose

    # def compute_output_shape(self, input_shape):
    #     spatial_shape = (input_shape[0][0], self.spatial_size_out, self.spatial_size_out)
    #     return ((spatial_shape + (self.capsules, 1)), (spatial_shape + (self.capsules, self.pose_size * self.pose_size)))


class ClassCapsules(BaseCaps):
    def __init__(self, **kwargs):
        self.kernel_size = 1
        super(ClassCapsules, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClassCapsules, self).build(input_shape)
        self.spatial_size_out = self.spatial_size_in

    def call(self, inputs):
        [input_act, input_pose] = inputs
        input_pose_tiled = K.reshape(input_pose, [self.spatial_size_out ** 2, self.in_capsules, 1, 4, 4])
        input_act_tiled = K.reshape(input_act, [self.spatial_size_out ** 2, self.in_capsules, 1, 1])
        input_pose_tiled = K.tile(input_pose_tiled, [1, 1, self.capsules, 1, 1])
        input_act_tiled = K.tile(input_act_tiled, [1, 1, self.capsules, 1])
        print(self.name, "input_pose_tiled", input_pose_tiled.shape)
        weights_tiled = K.tile(self.transformation_weights, [self.spatial_size_out ** 2, 1, 1, 1, 1])
        print(self.name, "weights tiled shape", weights_tiled.shape)
        votes = tf.matmul(input_pose_tiled, weights_tiled)
        print(self.name, "votes shape", votes.shape)
        votes = self._coord_addition(votes)
        votes = K.reshape(votes, (self.spatial_size_out ** 2, self.in_capsules, self.capsules, 16))
        print(self.name, "votes reshaped shape", votes.shape)
        out_act, out_pose = self.routing_method(input_act_tiled, votes, self.beta_a, self.beta_v, self.routings)
        out_act = K.reshape(out_act, [self.spatial_size_out, self.spatial_size_out, self.capsules, 1])
        out_pose = K.reshape(out_pose, [self.spatial_size_out, self.spatial_size_out, self.capsules, 4, 4])
        # tf.summary.histogram("activation_out", activation_out)
        # print("OUT_SHAPE:", self.compute_output_shape(input.shape))
        return out_act, out_pose

    def _coord_addition(self, votes):
        # Only considering quadratic images for now!
        votes = K.reshape(votes, (self.spatial_size_out, self.spatial_size_out, self.in_capsules, self.capsules, 16))

        offset_vals = (np.arange(self.spatial_size_out) + 0.5) / float(self.spatial_size_out)

        w_offset = np.zeros([self.spatial_size_out, 16])
        w_offset[:, 3] = offset_vals  # first val of the righmost column - as in papepr
        w_offset = np.reshape(w_offset, [1, self.spatial_size_out, 1, 1, 16])

        h_offset = np.zeros([self.spatial_size_out, 16])
        h_offset[:, 7] = offset_vals  # second val of the righmost column - as in papepr
        h_offset = np.reshape(h_offset, [self.spatial_size_out, 1, 1, 1, 16])

        # Combine w and h offsets using broadcasting
        offset = w_offset + h_offset

        # Convent from numpy to tensor
        offset = tf.constant(offset, dtype=tf.float32)

        votes = tf.add(votes, offset)

        return votes

    # def compute_output_shape(self, input_shape):
    #     spatial_shape = (input_shape[0][0], self.spatial_size_out, self.spatial_size_out)
    #     return ((spatial_shape + (self.capsules, 1)), (spatial_shape + (self.capsules, self.pose_size * self.pose_size)))


def em_routing(input_act, votes, beta_a, beta_v, routings):
    rr = tf.constant(1.0 / votes.shape[2], shape=votes.shape[:3] + [1])
    # print(self.name, "routing", "input_act shape", input_act.shape)
    # print(self.name, "routing", "R matrix shape", rr.shape)
    for i in range(0, routings):
        lambd = 0.01 * (1 - tf.pow(0.95, tf.cast(i, tf.float32)))
        output_act, means, variance = _routing_m_step(input_act, rr, votes, lambd, beta_a, beta_v)
        if i < routings - 1:
            rr = _routing_e_step(means, variance, output_act, votes)
    return output_act, means


def _routing_m_step(input_act, rr, votes, lambd, beta_a, beta_v):
    input_act_scaled = tf.multiply(rr, input_act)
    # print(self.name, "m_routing", "input_act_scaled shape", input_act_scaled.shape)
    rr_tiled = K.tile(rr, [1, 1, 1, 16])
    # print(self.name, "m_routing", "rr_tiled shape", rr_tiled.shape)
    rr_sum = tf.reduce_sum(rr_tiled, axis=1, keepdims=True)
    # print(self.name, "m_routing", "rr_sum shape", rr_sum.shape)
    means = tf.divide(
            tf.reduce_sum(rr_tiled * votes, axis=1, keepdims=True),
            rr_sum
    )
    # print(self.name, "m_routing", "means shape", means.shape)

    std_dev = tf.sqrt(
            tf.reduce_sum(rr_tiled * (votes - means), axis=1, keepdims=True) / rr_sum
    )
    # print(self.name, "m_routing", "std_dev shape", std_dev.shape)
    # check if other epsilons are needed
    costs = beta_v + tf.multiply(K.log(std_dev + K.epsilon()), rr_sum)
    # print(self.name, "m_routing", "costs shape", costs.shape)
    # could normalize the outputs here??
    out_act = K.sigmoid(lambd * (beta_a - tf.reduce_sum(costs, axis=-1)))
    out_act = K.reshape(out_act, out_act.shape + [1])
    # print(self.name, "m_routing", "out_act shape", out_act.shape)
    return out_act, means, std_dev


def _routing_e_step(means, std_dev, output_act, votes):
    # we are counting log of probabilities
    # we can discard PI beacuse i tak sie skroci
    prob_exp = - tf.reduce_sum(
            tf.square(votes - means) / (2 * tf.square(std_dev)), axis=-1, keepdims=True
    )
    prob_main = - tf.reduce_sum(
            K.log(std_dev + K.epsilon()), axis=-1, keepdims=True
    )
    prob = prob_main + prob_exp
    # print(self.name, "e_routing", "prob shape", prob.shape)
    zz = K.log(output_act + K.epsilon()) + prob  # zz?
    # print(self.name, "e_routing", "zz shape", zz.shape)
    rr = K.softmax(zz, axis=0)  # which axis should be used?
    # print(self.name, "e_routing", "rr shape", rr.shape)
    return rr
