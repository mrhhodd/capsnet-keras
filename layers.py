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

        pose = K.conv2d(inputs, self.pose_weights, strides=(
            1, 1), padding=self.padding, data_format='channels_last')
        pose = K.reshape(
            pose, shape=[spatial_size, spatial_size, self.capsules, 16])

        act = K.conv2d(inputs, self.act_weights, strides=(1, 1),
                       padding=self.padding, data_format='channels_last')
        act = activations.sigmoid(act)
        act = K.reshape(
            act, shape=[spatial_size, spatial_size, self.capsules, 1])

        return act, pose

    def compute_output_shape(self, input_shape):
        spatial_shape = (None, input_shape[1], input_shape[2])
        return (spatial_shape + (self.capsules, 1)), (spatial_shape + (self.capsules, 16))


class BaseCaps(layers.Layer):
    def __init__(self, capsules, routings, weights_reg, **kwargs):
        self.capsules = capsules
        self.routing_method = em_routing
        self.routings = routings
        self.weights_regularizer = weights_reg
        super(BaseCaps, self).__init__(**kwargs)

    def build(self, input_shape):
        # in_act_shape: [height, width, in_capsules, 1]
        # in_pose_shape: [height, width, in_capsules, 16]
        [in_act_shape, in_pose_shape] = input_shape

        self.spatial_size_in = in_act_shape[0]
        self.in_capsules = in_act_shape[2]

        # beta_v shape: [1, 1, capsules, 1]
        # beta_a shape: [1, 1, capsules]
        self.beta_v = self.add_weight(
            name='beta_v',
            shape=(1, 1, self.capsules, 1),
            initializer='glorot_uniform',
            regularizer=self.weights_regularizer,
            trainable=True)
        self.beta_a = self.add_weight(
            name='beta_a',
            shape=(1, 1, self.capsules),
            initializer='glorot_uniform',
            regularizer=self.weights_regularizer,
            trainable=True)

    def _generate_voting_map(self, size_in, size_out, kernel_size, stride):
        voting_map = np.zeros((size_out ** 2, size_in ** 2))
        parent_id = 0
        valid_in = range(0, size_in - kernel_size + 1, stride)
        assert size_out == len(valid_in), "SIZES DOESNT FIT"
        for row in valid_in:
            for col in valid_in:
                for kernel_row in range(0, kernel_size):
                    start = (row + kernel_row) * size_in + col
                    voting_map[parent_id][start:start + kernel_size] = 1.0
                parent_id += 1
        child_parent_indexes = np.where(voting_map)[1]
        child_parent_map = np.reshape(
            child_parent_indexes, [size_out ** 2, -1])

        return voting_map, child_parent_map


class ConvCaps(BaseCaps):
    def __init__(self, capsules, strides, padding, kernel_size, routings, **kwargs):
        # TODO: Allow for different height/width
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        super(ConvCaps, self).__init__(
            capsules=capsules, routings=routings, **kwargs)

    def build(self, input_shape):
        super(ConvCaps, self).build(input_shape)
        self.spatial_size_out = int(
            (self.spatial_size_in - self.kernel_size) / self.strides + 1)

        # transformation_weights shape: [1, 3*3*in_capsules, capsules, 4, 4]
        self.transformation_weights = self.add_weight(name='transformation_weights',
                                                      shape=(1,
                                                             self.kernel_size * self.kernel_size * self.in_capsules,
                                                             self.capsules,
                                                             4, 4),
                                                      initializer='glorot_uniform',
                                                      regularizer=self.weights_regularizer,
                                                      trainable=True)

        self.voting_map, self.child_parent_map = self._generate_voting_map(
            size_in=self.spatial_size_in,
            size_out=self.spatial_size_out,
            kernel_size=self.kernel_size,
            stride=self.strides)
        # voting map shape: [out_height*out_width, in_height*in_width]
        # child-parent map shape: [out_height*out_width, kernel_size^2]

    def call(self, inputs):
        [in_act, in_pose] = inputs

        # flatten 2D capsule array to 1D vector
        # in_act_shape: [height*width, in_capsules, 1]
        # in_pose_shape: [height*width, in_capsules, 4, 4]
        in_act = K.reshape(
            in_act, [self.spatial_size_in ** 2, self.in_capsules, 1])
        in_pose = K.reshape(
            in_pose, [self.spatial_size_in ** 2, self.in_capsules, 4, 4])

        # pick only child capsules in the receptive field of each parent capsule
        # in_act_filtered shape: [out_height*out_width, kernel_size^2, in_capsules, 1]
        # in_pose_filtered shape: [out_height*out_width, kernel_size^2, in_capsules, 4, 4]
        in_act_filtered = tf.gather(in_act, self.child_parent_map, axis=0)
        in_pose_filtered = tf.gather(in_pose, self.child_parent_map, axis=0)

        # reshape input - flatten all input capsules and add another dimension for parent capsules
        # replicate the data along the new dimension for each of the parent capsules
        # in_act_tiled shape: [out_height*out_width, in_capsules*kernel_size^2, out_capsules, 1]
        # in_pose_tiled shape: [out_height*out_width, in_capsules*kernel_size^2, out_capsules, 4, 4]
        in_act_tiled = K.reshape(in_act_filtered, [
            self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 1])
        in_pose_tiled = K.reshape(in_pose_filtered, [
            self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 4, 4])
        in_act_tiled = K.tile(in_act_tiled, [1, 1, self.capsules, 1])
        in_pose_tiled = K.tile(in_pose_tiled, [1, 1, self.capsules, 1, 1])

        # replicate the weights for each of the output spatial capsule
        # weights_tiled shape: [out_height*out_width, in_capsules*kernel_size^2, out_capsules, 4, 4]
        weights_tiled = K.tile(self.transformation_weights, [
                               self.spatial_size_out ** 2, 1, 1, 1, 1])

        # Compute all votes and reshape them for the routing purposes
        # votes shape: [out_height*out_width, in_capsules*kernel_size^2, out_capsules, 16]
        votes = tf.matmul(in_pose_tiled, weights_tiled)
        votes = K.reshape(votes, (self.spatial_size_out ** 2,
                                  self.kernel_size ** 2 * self.in_capsules, self.capsules, 16))

        # run routing to compute output activation and pose
        # out_act shape: [out_height, out_width, out_capsules, 1]
        # out_pose shape: [out_height, out_width, out_capsules, 4, 4]
        out_act, out_pose = self.routing_method(
            in_act_tiled, votes, self.beta_a, self.beta_v, self.routings)
        out_act = K.reshape(
            out_act, [self.spatial_size_out, self.spatial_size_out, self.capsules, 1])
        out_pose = K.reshape(
            out_pose, [self.spatial_size_out, self.spatial_size_out, self.capsules, 4, 4])
        return out_act, out_pose

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        spatial_shape = (batch_size, self.spatial_size_out,
                         self.spatial_size_out)
        return ((spatial_shape + (self.capsules, 1)), (spatial_shape + (self.capsules, 4, 4)))


class ClassCapsules(BaseCaps):
    # basically a fully connected layer?
    # def __init__(self, **kwargs):
    #     super(ClassCapsules, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClassCapsules, self).build(input_shape)
        # transformation_weights shape: [1, in_capsules, capsules, 4, 4]
        self.transformation_weights = self.add_weight(name='transformation_weights',
                                                      shape=(1,
                                                             self.in_capsules,
                                                             self.capsules,
                                                             4, 4),
                                                      initializer='glorot_uniform',
                                                      regularizer=self.weights_regularizer,

                                                      trainable=True)

        self.voting_map, self.child_parent_map = self._generate_voting_map(
            size_in=self.spatial_size_in,
            size_out=1,
            kernel_size=self.spatial_size_in,
            stride=1)

    def call(self, inputs):
        [in_act, in_pose] = inputs

        # flatten 2D capsule array to 1D vector
        # in_act_shape: [height*width, in_capsules, 1]
        # in_pose_shape: [height*width, in_capsules, 4, 4]
        in_act = K.reshape(
            in_act, [self.spatial_size_in ** 2, self.in_capsules, 1])
        in_pose = K.reshape(
            in_pose, [self.spatial_size_in ** 2, self.in_capsules, 4, 4])

        # pick only child capsules in the receptive field of each parent capsule
        # in_act_filtered shape: [out_height*out_width, kernel_size^2, in_capsules, 1]
        # in_pose_filtered shape: [out_height*out_width, kernel_size^2, in_capsules, 4, 4]
        in_act_filtered = tf.gather(in_act, self.child_parent_map, axis=0)
        in_pose_filtered = tf.gather(in_pose, self.child_parent_map, axis=0)
        # reshape input - flatten all input capsules and add another dimension for parent capsules
        # replicate the data along the new dimension for each of the parent capsules
        # in_act_tiled shape: [1, in_capsules*in_height*in_width, out_capsules, 1]
        # in_pose_tiled shape: [1, in_capsules*in_height*in_width, out_capsules, 4, 4]

        in_act_tiled = K.reshape(
            in_act_filtered, [1, self.spatial_size_in ** 2 * self.in_capsules, 1, 1])
        in_pose_tiled = K.reshape(
            in_pose_filtered, [1, self.spatial_size_in ** 2 * self.in_capsules, 1, 4, 4])
        in_act_tiled = K.tile(in_act_tiled, [1, 1, self.capsules, 1])
        in_pose_tiled = K.tile(in_pose_tiled, [1, 1, self.capsules, 1, 1])

        # replicate the weights for each of the input spatial capsule
        # weights_tiled shape: [1, in_capsules*in_height*in_width, out_capsules, 4, 4]
        weights_tiled = K.tile(self.transformation_weights, [
                               1, self.spatial_size_in ** 2, 1, 1, 1])

        # Compute all votes and add information about spatial position of each input capsule
        # Reshape the votes for the routing purposes
        # votes shape: [1, in_capsules*in_height*in_width, out_capsules, 16]
        votes = tf.matmul(in_pose_tiled, weights_tiled)
        votes = self._coord_addition(votes)
        votes = K.reshape(votes, (1, self.spatial_size_in **
                                  2 * self.in_capsules, self.capsules, 16))

        # run routing to compute output activation and pose
        # reshape output activation and pose
        # out_act shape: [out_capsules, 1]
        # out_pose shape: [out_capsules, 4, 4]
        out_act, out_pose = self.routing_method(
            in_act_tiled, votes, self.beta_a, self.beta_v, self.routings)
        out_act = K.reshape(out_act, [self.capsules])
        out_pose = K.reshape(out_pose, [self.capsules, 4, 4])

        return out_act, out_pose

    def _coord_addition(self, votes):
        size = self.spatial_size_in
        # restore the spatial shape of each capsule type
        # votes shape: [size, size, in_capsules, out_capsules, 16]
        votes = K.reshape(
            votes, (size, size, self.in_capsules, self.capsules, 16))

        # Create offset values for each capsule depending of its spatial location
        # e.g. for 5x5 we would have two matrices for coord addition:
        # for width:                 for height:
        # [0.1, 0.3, 0.5, 0.7, 0.9]  [0.1, 0.1, 0.1, 0.1, 0.1]
        # [0.1, 0.3, 0.5, 0.7, 0.9]  [0.3, 0.3, 0.3, 0.3, 0.3]
        # [0.1, 0.3, 0.5, 0.7, 0.9]  [0.5, 0.5, 0.5, 0.5, 0.5]
        # [0.1, 0.3, 0.5, 0.7, 0.9]  [0.7, 0.7, 0.7, 0.7, 0.7]
        # [0.1, 0.3, 0.5, 0.7, 0.9]  [0.9, 0.9, 0.9, 0.9, 0.9]

        offset_vals = (np.arange(size) + 0.5) / float(size)
        w_offset = np.zeros([size, 16])
        h_offset = np.zeros([size, 16])

        # first val of the righmost column for width
        w_offset[:, 3] = offset_vals
        w_offset = np.reshape(w_offset, [1, size, 1, 1, 16])

        # second val of the righmost column for height
        h_offset[:, 7] = offset_vals
        h_offset = np.reshape(h_offset, [size, 1, 1, 1, 16])

        # Combine width and height offsets using broadcasting
        # offset shape: [size, size, 1, 1, 16])
        offset = w_offset + h_offset

        votes = tf.add(votes, offset)

        return votes

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return ((batch_size, self.capsules), (batch_size, self.capsules, 4, 4))


def em_routing(in_act, votes, beta_a, beta_v, routings):
    # votes shape: [1, in_capsules*in_height*in_width, out_capsules, 16]
    # initialize R matrix with 1/out_capsules
    # rr shape: [1, in_capsules*in_height*in_width, out_capsules, 1]
    rr = tf.constant(1.0 / votes.shape[2], shape=votes.shape[:3] + [1])

    for i in range(0, routings):
        # lambda value based on comments in hintons review
        lambd = 0.01 * (1 - tf.pow(0.95, tf.cast(i, tf.float32)))

        # compute output activations, means and standard deviations
        # out_act shape
        # means shape
        # std_devs shape
        out_act, means, std_devs = _routing_m_step(
            in_act, rr, votes, lambd, beta_a, beta_v)

        # Skip the e_step for last iterations - no point in running it
        if i < routings - 1:
            # readjust the rr values for the next step
            rr = _routing_e_step(means, std_devs, out_act, votes)

    # return out_act and means for parent capsule poses
    return out_act, means


def _routing_m_step(in_act, rr, votes, lambd, beta_a, beta_v):
    # M_step 2 - scale the R matrix by their corresponding input activation values
    rr_scaled = tf.multiply(rr, in_act)

    # replicate it for each pose value
    # rr_tiled shape: [1, in_capsules*in_height*in_width, out_capsules,16]
    rr_tiled = K.tile(rr_scaled, [1, 1, 1, 16])

    # Compute the sum of all input capsules in rr matrix
    # rr_sum shape: [1, out_capsules, 16]
    rr_sum = tf.reduce_sum(rr_tiled, axis=1, keepdims=True)

    # M_step 3 - compute means for each parent capsule
    # means shape: [1, 1, out_capsules, 16]
    means = tf.reduce_sum(rr_tiled * votes, axis=1,
                          keepdims=True) / (rr_sum + K.epsilon())

    # M_step 4 - compute std_dev for each parent capsule
    # std_dev shape: [1, 1, out_capsules, 16]
    std_dev = tf.sqrt(
        tf.reduce_sum(rr_tiled * (votes - means), axis=1,
                      keepdims=True) / (rr_sum + K.epsilon())
    )

    # M_step 5 - compute costs for each parent capsule
    # beta_v shape: [1, 1, out_capsules, 1]
    # costs shape: [1, 1, out_capsules, 16]
    costs = beta_v + tf.multiply(K.log(std_dev + K.epsilon()), rr_sum)

    # M_step 6 - compute activation for each parent capsule
    # beta_a shape: [1, 1, out_capsules]
    # out_act shape: [out_height*out_width, 1, out_capsules, 1]
    # TODO: Do we need normalization here?
    out_act = K.sigmoid(lambd * (beta_a - tf.reduce_sum(costs, axis=-1)))
    out_act = K.reshape(out_act, out_act.shape + [1])

    return out_act, means, std_dev


def _routing_e_step(means, std_dev, output_act, votes):
    # E_step 2 - compute probabilities
    # we are using logarithms to simplify the calculations
    # prob shape: [1, 1, out_capsules, 1]
    prob_exp = - tf.reduce_sum(
        tf.square(votes - means) / (2 * tf.square(std_dev) + K.epsilon()),
        axis=-1, keepdims=True)
    prob_main = - tf.reduce_sum(
        K.log(std_dev + K.epsilon()),
        axis=-1, keepdims=True)
    prob = prob_main + prob_exp

    # E_step 3 - recompute the R matrix values
    # rr shape: [1, in_capsules*in_height*in_width, out_capsules, 1]
    zz = K.log(output_act + K.epsilon()) + prob
    rr = K.softmax(zz, axis=2)

    return rr
