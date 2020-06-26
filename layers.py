import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import layers, activations, initializers
import time

def paper_initializer(shape, dtype):
    """
        Recently we are using a new initialization method: 
        every 4x4 is initialized with I + noise of 0.03: (1 on the diag, random uniform noise in the range +/- 0.03 everywhere else). 
        This new method is more scale able and easier to train. 
    """
    # assert shape_dims >= 2
    assert shape[-1] == shape[-2], "last two value has to be an nxn matrix"
    return initializers.Identity()(shape=shape[-2:], dtype=dtype) + initializers.RandomUniform(minval=-0.03, maxval=0.03)(shape=shape, dtype=dtype)


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
        # tf.print("####")
        # tf.print(tf.reduce_max(inputs))
        # tf.print(tf.reduce_mean(inputs))
        # tf.print(tf.reduce_min(inputs))
        # tf.print("####")
        # # tf.print("in_act primary caps", inputs[0])
        # # tf.print("in_act primary caps", inputs[0])
        t0 = time.time()
        batch_size = tf.shape(inputs)[0]
        spatial_size = int(inputs.shape[1])

        pose = K.conv2d(inputs, self.pose_weights, strides=(1, 1),
                        padding=self.padding, data_format='channels_last')
        pose = K.reshape(
            pose, shape=(batch_size, spatial_size, spatial_size, self.capsules, 16))

        act = K.conv2d(inputs, self.act_weights, strides=(1, 1),
                       padding=self.padding, data_format='channels_last')
        # # tf.print("out_act2 primary caps", act[0])
        act = activations.sigmoid(act)
        # # tf.print("out_act1 primary caps", act[0])
        act = K.reshape(
            act, shape=(batch_size, spatial_size, spatial_size, self.capsules, 1))

        # # tf.print("out_act primary caps", act[0])

        # # print("\n TIME", self.name, tf.constant(time.time()-t0))
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
        # in_act_shape: [batch_size, height, width, in_capsules, 1]
        # in_pose_shape: [batch_size, height, width, in_capsules, 16]
        [in_act_shape, in_pose_shape] = input_shape

        self.spatial_size_in = in_act_shape[1]
        self.in_capsules = in_act_shape[3]

    def call(self, inputs):
        # beta_v shape: [capsules]
        # beta_a shape: [capsules]
        self.beta_v = self.add_weight(
            name='beta_v',
            shape=[self.capsules],
            initializer='glorot_uniform',
            # regularizer=self.weights_regularizer,
            regularizer=None,
            trainable=True
            )
        self.beta_a = self.add_weight(
            name='beta_a',
            shape=[self.capsules],
            # initializer='glorot_uniform',
            initializer=initializers.TruncatedNormal(mean=-2500.0, stddev=1250.0),
            # regularizer=self.weights_regularizer,
            trainable=True
            )

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
                                                      shape=(1, 1,
                                                             self.kernel_size * self.kernel_size * self.in_capsules,
                                                             self.capsules,
                                                             4, 4),
                                                      initializer=paper_initializer,
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
        super(ConvCaps, self).call(inputs)
        t0 = time.time()
        [in_act, in_pose] = inputs
        batch_size = tf.shape(in_act)[0]
        # # tf.print("in_act conv caps", in_act[0])

        # flatten 2D capsule array to 1D vector
        # in_act_shape: [batch_size, height*width, in_capsules, 1]
        # in_pose_shape: [batch_size, height*width, in_capsules, 4, 4]
        in_act = K.reshape(
            in_act, [batch_size, self.spatial_size_in ** 2, self.in_capsules, 1])
        # # tf.print("in_act 1 conv caps", in_act[0])
        in_pose = K.reshape(
            in_pose, [batch_size, self.spatial_size_in ** 2, self.in_capsules, 4, 4])

        # pick only child capsules in the receptive field of each parent capsule
        # in_act_filtered shape: [batch_size, out_height*out_width, kernel_size^2, in_capsules, 1]
        # in_pose_filtered shape: [batch_size, out_height*out_width, kernel_size^2, in_capsules, 4, 4]
        in_act_filtered = tf.gather(in_act, self.child_parent_map, axis=1)
        # # tf.print("in_act_filtered 1 conv caps", in_act_filtered[0])
        in_pose_filtered = tf.gather(in_pose, self.child_parent_map, axis=1)

        # reshape input - flatten all input capsules and add another dimension for parent capsules
        # replicate the data along the new dimension for each of the parent capsules
        # in_act_tiled shape: [batch_size, out_height*out_width, in_capsules*kernel_size^2, out_capsules, 1]
        # in_pose_tiled shape: [batch_size, out_height*out_width, in_capsules*kernel_size^2, out_capsules, 4, 4]
        in_act_tiled = K.reshape(in_act_filtered, [
            batch_size, self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 1])
        in_pose_tiled = K.reshape(in_pose_filtered, [
            batch_size, self.spatial_size_out ** 2, self.kernel_size ** 2 * self.in_capsules, 1, 4, 4])
        in_act_tiled = K.tile(in_act_tiled, [1, 1, 1, self.capsules, 1])
        in_pose_tiled = K.tile(in_pose_tiled, [1, 1, 1, self.capsules, 1, 1])
        # # tf.print("in_act_tiled 1 conv caps", in_act_tiled[0])

        # replicate the weights for each of the output spatial capsule
        # weights_tiled shape: [batch_size, out_height*out_width, in_capsules*kernel_size^2, out_capsules, 4, 4]
        weights_tiled = K.tile(self.transformation_weights, [
                               batch_size, self.spatial_size_out ** 2, 1, 1, 1, 1])
        # # tf.print("weights_tiled 1 conv caps", weights_tiled[0])

        # Compute all votes and reshape them for the routing purposes
        # votes shape: [batch_size, out_height*out_width, in_capsules*kernel_size^2, out_capsules, 16]
        votes = tf.matmul(in_pose_tiled, weights_tiled)
        votes = K.reshape(votes, (batch_size, self.spatial_size_out ** 2,
                                  self.kernel_size ** 2 * self.in_capsules, self.capsules, 16))
        # # tf.print("votes 1 conv caps", votes[0])

        # run routing to compute output activation and pose
        # out_act shape: [batch_size, out_height, out_width, out_capsules, 1]
        # out_pose shape: [batch_size, out_height, out_width, out_capsules, 4, 4]
        out_act, out_pose = self.routing_method(
            in_act_tiled, votes, self.beta_a, self.beta_v, self.routings)
        out_act = K.reshape(
            out_act, [batch_size, self.spatial_size_out, self.spatial_size_out, self.capsules, 1])
        out_pose = K.reshape(
            out_pose, [batch_size, self.spatial_size_out, self.spatial_size_out, self.capsules, 4, 4])
        # # tf.print("out_act conv caps", out_act[0])
        # # print("\n TIME", self.name, tf.constant(time.time()-t0))
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
                                                      shape=(1, 1,
                                                             self.in_capsules,
                                                             self.capsules,
                                                             4, 4),
                                                      initializer=paper_initializer,
                                                      regularizer=self.weights_regularizer,
                                                      trainable=True)

        self.voting_map, self.child_parent_map = self._generate_voting_map(
            size_in=self.spatial_size_in,
            size_out=1,
            kernel_size=self.spatial_size_in,
            stride=1)

    def call(self, inputs):
        super(ClassCapsules, self).call(inputs)
        t0=time.time()
        [in_act, in_pose] = inputs
        batch_size = tf.shape(in_act)[0]
        # # tf.print("IN_ACT capsule_caps", in_act[0])
        # flatten 2D capsule array to 1D vector
        # in_act_shape: [batch_size, height*width, in_capsules, 1]
        # in_pose_shape: [batch_size, height*width, in_capsules, 4, 4]
        in_act = K.reshape(
            in_act, [batch_size, self.spatial_size_in ** 2, self.in_capsules, 1])
        in_pose = K.reshape(
            in_pose, [batch_size, self.spatial_size_in ** 2, self.in_capsules, 4, 4])

        # pick only child capsules in the receptive field of each parent capsule
        # in_act_filtered shape: [batch_size, out_height*out_width, kernel_size^2, in_capsules, 1]
        # in_pose_filtered shape: [batch_size, out_height*out_width, kernel_size^2, in_capsules, 4, 4]
        in_act_filtered = tf.gather(in_act, self.child_parent_map, axis=1)
        in_pose_filtered = tf.gather(in_pose, self.child_parent_map, axis=1)
        # reshape input - flatten all input capsules and add another dimension for parent capsules
        # replicate the data along the new dimension for each of the parent capsules
        # in_act_tiled shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 1]
        # in_pose_tiled shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 4, 4]

        in_act_tiled = K.reshape(
            in_act_filtered, [batch_size, 1, self.spatial_size_in ** 2 * self.in_capsules, 1, 1])
        in_pose_tiled = K.reshape(
            in_pose_filtered, [batch_size, 1, self.spatial_size_in ** 2 * self.in_capsules, 1, 4, 4])
        in_act_tiled = K.tile(in_act_tiled, [1, 1, 1, self.capsules, 1])
        in_pose_tiled = K.tile(in_pose_tiled, [1, 1, 1, self.capsules, 1, 1])

        # replicate the weights for each of the input spatial capsule
        # weights_tiled shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 4, 4]
        weights_tiled = K.tile(self.transformation_weights, [
                               batch_size, 1, self.spatial_size_in ** 2, 1, 1, 1])

        # Compute all votes and add information about spatial position of each input capsule
        # Reshape the votes for the routing purposes
        # votes shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 16]
        votes = tf.matmul(in_pose_tiled, weights_tiled)
        votes = self._coord_addition(votes)
        votes = K.reshape(votes, (batch_size, 1, self.spatial_size_in **
                                  2 * self.in_capsules, self.capsules, 16))

        # run routing to compute output activation and pose
        # reshape output activation and pose
        # out_act shape: [batch_size, out_capsules, 1]
        # out_pose shape: [batch_size, out_capsules, 4, 4]
        # tf.print("IN_ACT return", in_act_tiled)
        out_act, out_pose = self.routing_method(
            in_act_tiled, votes, self.beta_a, self.beta_v, self.routings, log=True)
        out_act = K.reshape(out_act, [batch_size, self.capsules])
        out_pose = K.reshape(out_pose, [batch_size, self.capsules, 4, 4])
        # # print("\n TIME", self.name, tf.constant(time.time()-t0))
        return out_act, out_pose

    def _coord_addition(self, votes):
        size = self.spatial_size_in
        # restore the spatial shape of each capsule type
        # votes shape: [size, size, in_capsules, out_capsules, 16]
        votes = K.reshape(
            votes, (-1, size, size, self.in_capsules, self.capsules, 16))

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
        w_offset = np.reshape(w_offset, [1, 1, size, 1, 1, 16])

        # second val of the righmost column for height
        h_offset[:, 7] = offset_vals
        h_offset = np.reshape(h_offset, [1, size, 1, 1, 1, 16])

        # Combine width and height offsets using broadcasting
        # offset shape: [size, size, 1, 1, 16])
        offset = w_offset + h_offset

        votes = tf.add(votes, offset)

        return votes

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return ((batch_size, self.capsules), (batch_size, self.capsules, 4, 4))


def em_routing(in_act, votes, beta_a, beta_v, routings, log=False):
    t0=time.time()
    batch_size = tf.shape(votes)[0]
    out_capsules = tf.shape(votes)[3]
    # votes shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 16]
    # initialize R matrix with 1/out_capsules
    # rr shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 1]
    # TODO: rewrite this?
    rr = K.mean(K.ones_like(votes) / out_capsules, axis=-1, keepdims=True)

    # beta_v shape: [batch_size, 1, 1, out_capsules, 1]
    beta_v = K.reshape(beta_v, [1, 1, 1, out_capsules, 1])
    beta_v = K.tile(beta_v, (batch_size, 1, 1, 1, 1))

    # beta_a shape: [batch_size, 1, 1, out_capsules]
    beta_a = K.reshape(beta_a, [1, 1, 1, out_capsules])
    beta_a = K.tile(beta_a, (batch_size, 1, 1, 1))

    for i in range(routings):
        # lambda value based on comments in hintons review
        lambd = 0.01 * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))

        # compute output activations, means and standard deviations
        # out_act shape
        # means shape
        # std_devs shape
        t1=time.time()
        out_act, means, std_devs = _routing_m_step(
            in_act, rr, votes, lambd, beta_a, beta_v)
        # print("\n TIME", " m_step routing", time.time()-t1)

        # Skip the e_step for last iterations - no point in running it
        if i < routings - 1:
            # readjust the rr values for the next step
            t1=time.time()
            rr = _routing_e_step(means, std_devs, out_act, votes)
            # # print("\n TIME", " e_step routing", time.time()-t1)

    # return out_act and means for parent capsule poses
    # # print("\n TIME", "routing", tf.constant(time.time()-t0))
    # tf.print("##### end of routing")
    # tf.print(out_act[0][0][0])
    # tf.print("\nmean: ", tf.reduce_mean(out_act), " # max: ", tf.reduce_max(out_act), " # min: ", tf.reduce_min(out_act))
    return out_act, means


def _routing_m_step(in_act, rr, votes, lambd, beta_a, beta_v):
    # M_step 2 - scale the R matrix by their corresponding input activation values
    t1=time.time()

    rr_scaled = tf.multiply(rr, in_act)
    # tf.print("_routing_m_step in_act caps", in_act[0])
    # tf.print("_routing_m_step rr caps", rr[0])
    # print("\n TIME", "0 m_step routing", time.time()-t1);t1=time.time()
    # replicate it for each pose value
    # rr_tiled shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules,16]
    rr_tiled = K.tile(rr_scaled, [1, 1, 1, 1, 16])
    # print("\n TIME", "1 m_step routing", time.time()-t1);t1=time.time()

    # calculate normalization factor - so that beta values are always relevant
    child_caps = float(tf.shape(rr_tiled)[2])
    parent_caps = float(tf.shape(rr_tiled)[3])
    norm_factor = 100 * parent_caps / child_caps

    # Compute the sum of all input capsules in rr matrix
    # rr_sum shape: [batch_size, 1, 1, out_capsules, 16]
    rr_sum = tf.reduce_sum(rr_tiled, axis=2, keepdims=True)
    # print("\n TIME", "2 m_step routing", time.time()-t1);t1=time.time()

    # M_step 3 - compute means for each parent capsule
    # means shape: [batch_size, 1, 1, out_capsules, 16]
    means = tf.reduce_sum(tf.multiply(rr_tiled, votes), axis=2,
                          keepdims=True) / (rr_sum + K.epsilon())
    # print("\n TIME", "3 m_step routing", time.time()-t1);t1=time.time()

    # M_step 4 - compute std_dev for each parent capsule
    # std_dev shape: [batch_size, 1, 1, out_capsules, 16]
    std_dev =  tf.reduce_sum(tf.multiply(rr_tiled, tf.square(votes - means)), axis=2,
                      keepdims=True) / (rr_sum + K.epsilon())
    # std_dev = tf.sqrt(
    #     tf.reduce_sum(tf.multiply(rr_tiled, tf.square(votes - means)), axis=2,
    #                   keepdims=True) / (rr_sum + K.epsilon())
    # )
    # print("\n TIME", "4 m_step routing", time.time()-t1);t1=time.time()

    # M_step 5 - compute costs for each parent capsule
    # beta_v shape: [batch_size, 1, 1, 1, out_capsules, 1]
    # costs shape: [batch_size, 1, 1, out_capsules, 16]
    costs = beta_v + tf.multiply(K.log(std_dev + K.epsilon()), rr_sum * norm_factor)
    # print("\n TIME", "5 m_step routing", time.time()-t1);t1=time.time()

    # M_step 6 - compute activation for each parent capsule
    # beta_a shape: [batch_size, 1, 1, out_capsules]
    # out_act shape: [batch_size, out_height*out_width, 1, out_capsules, 1]
    # TODO: Do we need normalization here?
    # tf.print("_routing_m_step costs ", costs[0])
    # tf.print("#MROUTINAG")
    # tf.print(costs[0][0][0])
    out_act = K.sigmoid(lambd * (beta_a - tf.reduce_sum(costs, axis=-1)))
    # tf.print(out_act[0][0][0])
    # tf.print("_routing_m_step out_act ", out_act[0])
    # tf.print("_routing_m_step beta_a ", beta_a[0])
    # print("\n TIME", "6 m_step routing", time.time()-t1);t1=time.time()

    out_act = K.expand_dims(out_act, -1)
    # tf.print("_routing_m_step out_act 2", out_act[0])
    # # tf.print("_routing_m_step means", means[0])
    # # tf.print("_routing_m_step std_dev", std_dev[0])
    # print("\n TIME", "7 m_step routing", time.time()-t1);t1=time.time()

    return out_act, means, std_dev


def _routing_e_step(means, std_dev, out_act, votes):
    # E_step 2 - compute probabilities
    # we are using logarithms to simplify the calculations
    # prob shape: [batch_size, 1, 1, out_capsules, 1]
    prob_exp = - tf.reduce_sum(
        tf.square(votes - means) / (2 * tf.square(std_dev) + K.epsilon()),
        axis=-1, keepdims=True)
    prob_main = - tf.reduce_sum(
        K.log(std_dev + K.epsilon()),
        axis=-1, keepdims=True)
    prob = prob_main + prob_exp

    # E_step 3 - recompute the R matrix values
    # rr shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 1]
    zz = K.log(out_act + K.epsilon()) + prob
    # tf.print("#####")
    # tf.print("zz shape", tf.shape(zz))
    # tf.print("zz", zz[0][0][0])
    # # tf.print("BEFORE SOFTMAX:", zz)
    rr = K.softmax(zz, axis=3)
    # # tf.print("AFTER SOFTMAX:", zz)
    # tf.print("#####")
    # tf.print("rr shape",tf.shape(rr))
    # tf.print("rr", rr[0][0][0])
    return rr
