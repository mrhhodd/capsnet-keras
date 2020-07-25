import tensorflow as tf
from tensorflow.keras import backend as K


def em_routing(in_act, votes, beta_a, beta_v, routings, log=False):
    batch_size = tf.shape(votes)[0]
    out_capsules = tf.shape(votes)[3]
    # votes shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 16]
    # initialize R matrix with 1/out_capsules
    # rr shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules, 1]
    rr = K.mean(K.ones_like(votes) / tf.cast(out_capsules,
                                             tf.float32), axis=-1, keepdims=True)

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
        out_act, means, std_devs = _routing_m_step(
            in_act, rr, votes, lambd, beta_a, beta_v)

        # Skip the e_step for last iterations - no point in running it
        if i < routings - 1:
            # readjust the rr values for the next step
            rr = _routing_e_step(means, std_devs, out_act, votes)

    # return out_act and means for parent capsule poses
    tf.print("\n min:", tf.reduce_min(out_act), " mean:", tf.reduce_mean(out_act), " max:", tf.reduce_max(out_act))
    return out_act, means


def _routing_m_step(in_act, rr, votes, lambd, beta_a, beta_v):
    # M_step 2 - scale the R matrix by their corresponding input activation values
    rr_scaled = tf.multiply(rr, in_act)
    # replicate it for each pose value
    # rr_tiled shape: [batch_size, 1, in_capsules*in_height*in_width, out_capsules,16]
    rr_tiled = K.tile(rr_scaled, [1, 1, 1, 1, 16])

    # calculate normalization factor - so that beta values are always relevant
    child_caps = float(tf.shape(rr_tiled)[2])
    parent_caps = float(tf.shape(rr_tiled)[3])
    norm_factor = 100 * parent_caps / child_caps

    # Compute the sum of all input capsules in rr matrix
    # rr_sum shape: [batch_size, 1, 1, out_capsules, 16]
    rr_sum = tf.reduce_sum(rr_tiled, axis=2, keepdims=True)

    # M_step 3 - compute means for each parent capsule
    # means shape: [batch_size, 1, 1, out_capsules, 16]
    means = tf.reduce_sum(tf.multiply(rr_tiled, votes), axis=2,
                          keepdims=True) / (rr_sum + K.epsilon())

    # M_step 4 - compute std_dev for each parent capsule
    # std_dev shape: [batch_size, 1, 1, out_capsules, 16]
    # sqrt causing some NaNs? use variance instead
    std_dev = tf.reduce_sum(tf.multiply(rr_tiled, tf.square(votes - means)), axis=2,
                            keepdims=True) / (rr_sum + K.epsilon())
    # std_dev = tf.sqrt(
    #     tf.reduce_sum(tf.multiply(rr_tiled, tf.square(votes - means)), axis=2,
    #                   keepdims=True) / (rr_sum + K.epsilon())
    # )

    # M_step 5 - compute costs for each parent capsule
    # beta_v shape: [batch_size, 1, 1, 1, out_capsules, 1]
    # costs shape: [batch_size, 1, 1, out_capsules, 16]
    # costs = beta_v + tf.multiply(0.5 * K.log(std_dev + K.epsilon()), rr_sum * norm_factor)
    costs = tf.multiply(beta_v + 0.5 * K.log(std_dev +
                                             K.epsilon()), rr_sum * norm_factor)

    # M_step 6 - compute activation for each parent capsule
    # beta_a shape: [batch_size, 1, 1, out_capsules]
    # out_act shape: [batch_size, out_height*out_width, 1, out_capsules, 1]
    out_act = K.sigmoid(lambd * (beta_a - tf.reduce_sum(costs, axis=-1)))

    out_act = K.expand_dims(out_act, -1)

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
    rr = K.softmax(zz, axis=3)
    return rr
