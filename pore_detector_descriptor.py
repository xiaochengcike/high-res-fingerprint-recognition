from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf


class Net:
  def __init__(self, images, window_size, layers=7, training=True,
               reuse=False):
    # kernel size is fixed across entire net
    kernel_size = [3, 3]

    # input of first layer is 'images'
    layer = tf.layers.batch_normalization(
        images, reuse=reuse, training=training, name='batch_norm0')

    # conv layers
    for i in range(1, 1 + layers):
      layer = tf.layers.conv2d(
          layer,
          filters=2**((i + 1) // 2 + 3),
          kernel_size=kernel_size,
          activation=tf.nn.relu,
          use_bias=False,
          padding='same',
          name='conv{}'.format(i),
          reuse=reuse)
      layer = tf.layers.batch_normalization(
          layer, reuse=reuse, training=training, name='batch_norm{}'.format(i))
      layer = tf.layers.max_pooling2d(
          layer, pool_size=kernel_size, strides=1, padding='valid')

    # detection conv layer
    det_layer = tf.layers.conv2d(
        layer,
        kernel_size=kernel_size,
        filters=1,
        activation=None,
        name='det_conv',
        use_bias=False,
        padding='valid',
        reuse=reuse)
    det_layer = tf.layers.batch_normalization(
        det_layer, training=training, name='det_batch_norm', reuse=reuse)

    # detection logits
    self.logits = tf.identity(det_layer, name='logits')

    # build detection op
    self.dets = tf.nn.sigmoid(self.logits, name='detections')

    # description conv layer
    descs_layer = tf.layers.conv2d(
        layer,
        filters=128,
        kernel_size=kernel_size,
        activation=tf.nn.relu,
        use_bias=False,
        name='desc_conv',
        reuse=reuse)
    descs_layer = tf.layers.batch_normalization(
        descs_layer, training=training, name='descs_batch_norm', reuse=reuse)

    # descriptors
    descs_layer = tf.reshape(descs_layer, (-1, 128))
    self.descs = tf.nn.l2_normalize(descs_layer, axis=1, name='descriptors')

  def build_detection_loss(self, labels):
    # reshape labels to be compatible with logits
    labels = tf.reshape(labels, tf.shape(self.logits))

    # cross entropy loss
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=self.logits, name='xentropy')
    self.det_loss = tf.reduce_mean(xentropy, name='xentropy_mean')

    return self.det_loss

  def build_detection_train(self, learning_rate):
    global_step = tf.Variable(1, name='det_global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_rate=0.96,
        decay_steps=2000,
        staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.det_train = optimizer.minimize(
          self.det_loss, global_step=global_step)

    return self.det_train

  def build_description_loss(self, labels):
    # make labels' shape compatible with triplet loss
    labels = tf.reshape(labels, (-1, ))

    self.desc_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels, self.descs)

    return self.desc_loss

  def build_description_train(self, learning_rate):
    global_step = tf.Variable(1, name='desc_global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.desc_train = optimizer.minimize(
          self.desc_loss, global_step=global_step)

    return self.desc_train

  def build_description_validation(self, labels, thresholds):
    # flatten labels
    labels = tf.reshape(labels, (-1, ))

    # recognition labels matrix with (i, j) = label[i] == label[j]
    labels_matrix = tf.tile(
        tf.expand_dims(labels, -1),
        tf.shape(tf.expand_dims(labels, 0)),
        name='labels_matrix')
    rec_labels = tf.equal(
        labels_matrix, tf.transpose(labels_matrix), name='recognition_labels')
    rec_labels = tf.cast(rec_labels, tf.int32)

    # distances matrix
    r_desc = tf.reduce_sum(self.descs * self.descs, axis=1)
    distances = tf.expand_dims(r_desc, -1) - \
        2 * tf.matmul(self.descs, self.descs, transpose_b=True) + \
        r_desc

    # mask for extraction of strict upper triangular band of matrices
    ones = tf.ones_like(distances)
    mask = tf.cast(ones - tf.matrix_band_part(ones, -1, 0), dtype=tf.bool)

    # make thresholds and distances shapes broadcast compatible
    thresholds = tf.reshape(thresholds, (-1, 1, 1))
    distances = tf.expand_dims(distances, 0)

    # recognition predictions over thresholds
    rec_preds = tf.less(distances, thresholds, name='recognition_predictions')
    rec_preds = tf.cast(rec_preds, tf.int32)

    # statistics over thresholds
    # false positives
    false_pos_mat = tf.cast(tf.greater(rec_preds, rec_labels), tf.int32)
    false_pos = tf.reduce_sum(
        tf.boolean_mask(false_pos_mat, mask, axis=1), axis=1)

    # false negatives
    false_neg_mat = tf.cast(tf.less(rec_preds, rec_labels), tf.int32)
    false_neg = tf.reduce_sum(
        tf.boolean_mask(false_neg_mat, mask, axis=1), axis=1)

    # true positives
    pos_mat = tf.cast(tf.greater_equal(rec_preds, 1), tf.int32)
    pos = tf.reduce_sum(tf.boolean_mask(pos_mat, mask, axis=1), axis=1)
    true_pos = pos - false_pos

    # true negatives
    neg_mat = tf.cast(tf.less_equal(rec_preds, 0), tf.int32)
    neg = tf.reduce_sum(tf.boolean_mask(neg_mat, mask, axis=1), axis=1)
    true_neg = neg - false_neg

    self.desc_val = [true_pos, true_neg, false_pos, false_neg]

    return self.desc_val
