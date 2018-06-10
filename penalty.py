# encoding=utf-8
# Project: transfer_cws
# Author: xingjunjie
# Create Time: 04/12/2017 9:35 PM on PyCharm

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

"""
Maximum Mean Discrepancy
"""


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
            gaussians in the kernel.
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def MMD(x, y, kernel=gaussian_kernel_matrix):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def _lengths_to_masks(lengths, max_length):
    """Creates a binary matrix that can be used to mask away padding.
    Args:
      lengths: A vector of integers representing lengths.
      max_length: An integer indicating the maximum length. All values in
        lengths should be less than max_length.
    Returns:
      masks: Masks that can be used to get rid of padding.
    """
    tiled_ranges = array_ops.tile(
        array_ops.expand_dims(math_ops.range(max_length), 0),
        [array_ops.shape(lengths)[0], 1])
    lengths = array_ops.expand_dims(lengths, 1)
    masks = math_ops.to_float(
        math_ops.to_int64(tiled_ranges) < math_ops.to_int64(lengths))
    return masks


def _de_pad(word_embedding, sequence_length):
    batch_size = array_ops.shape(word_embedding)[0]
    max_length = array_ops.shape(word_embedding)[1]
    embedding_size = array_ops.shape(word_embedding)[2]

    mask = tf.cast(_lengths_to_masks(sequence_length, max_length), tf.int32)

    index = tf.reshape(tf.range(1, batch_size * max_length + 1, 1), [batch_size, max_length])
    index = index * mask
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.reshape(tf.greater(index, zero), [-1])
    indices = tf.reshape(tf.where(where), [-1])

    return tf.gather(tf.reshape(word_embedding, [-1, embedding_size]), indices)


"""
KL-Divergence
"""


def kl(dista, distb):
    """

    :param dista: [ndim]
    :param diatb: [ndim]
    :return: KL(dista, distb)
    """
    # temp = tf.where(tf.equal(distb, tf.zeros_like(distb)), tf.zeros_like(distb) + 1e-10, dista / distb)
    # temp = tf.log(temp)
    return tf.reduce_sum(dista * tf.log(dista / distb))


def MKL(x, y):
    """
    Compute the KL-Divergence between x and y sampled from p and q respectively
    :param x: [n_samples_x, n_dim]
    :param y: [n_samples_y, n_dim]
    :return: KL-Divergence loss
    """
    mean_x = tf.reduce_mean(x, axis=0)
    mean_y = tf.reduce_mean(y, axis=0)
    # dist_x = tf.where(tf.reduce_sum(mean_x) > 0, mean_x / tf.reduce_sum(mean_x), tf.zeros_like(mean_x))
    # dist_y = tf.where(tf.reduce_sum(mean_y) > 0, mean_y / tf.reduce_sum(mean_y), tf.zeros_like(mean_y))
    dist_x = mean_x / tf.reduce_sum(mean_x)
    dist_y = mean_y / tf.reduce_sum(mean_y)

    return tf.clip_by_value(kl(dist_x, dist_y) + kl(dist_y, dist_x), 0, 10)


"""
Central Moment Discrepancy
"""


def norm(x, y):
    """
    Calculate the Euclidean Distance
    :param x:
    :param y:
    :return:
    """
    return tf.sqrt(tf.reduce_sum((x - y) ** 2))


def scm(x, y, k):
    """
    Calculate K order distance
    :param x:
    :param y:
    :param k:
    :return:
    """
    cx = tf.reduce_mean(x ** k, axis=0)
    cy = tf.reduce_mean(y ** k, axis=0)
    return norm(cx, cy)


def CMD(x, y, n_moments):
    """
    Calculate Central Moment Discrepancy
    :param x: [n_samples_x, n_dim]
    :param y: [n_samples_x, n_dim]
    :param n_moments: Number of moments
    :return:
    """
    mean_x = tf.reduce_mean(x, axis=0)
    mean_y = tf.reduce_mean(y, axis=0)
    diff_x = x - mean_x
    diff_y = y - mean_y

    dm = norm(mean_x, mean_y)
    cmd = dm
    for i in range(n_moments - 1):
        cmd += scm(diff_x, diff_y, i + 2)
    return cmd
