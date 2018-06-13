from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf


class Net:
  def __init__(self, images, layers=7, training=True, reuse=False):
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
          name='conv{}'.format(i),
          reuse=reuse)
      layer = tf.layers.batch_normalization(
          layer, reuse=reuse, training=training, name='batch_norm{}'.format(i))

    # final conv layer
    layer = tf.layers.conv2d(
        layer,
        kernel_size=kernel_size,
        filters=1,
        activation=None,
        name='conv{}'.format(layers + 1),
        use_bias=False,
        reuse=reuse)
    layer = tf.layers.batch_normalization(
        layer,
        training=training,
        name='batch_norm{}'.format(layers + 1),
        reuse=reuse)

    self.logits = tf.identity(layer, name='logits')

    # build prediction op
    self.preds = tf.nn.sigmoid(self.logits)

  def build_loss(self, labels):
    # reshape labels to be compatible with logits
    labels = tf.reshape(labels, tf.shape(self.logits))

    # cross entropy loss
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=self.logits, name='xentropy')
    self.loss = tf.reduce_mean(xentropy, name='xentropy_mean')

    return self.loss

  def build_train(self, learning_rate):
    global_step = tf.Variable(1, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_rate=0.96,
        decay_steps=2000,
        staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train = optimizer.minimize(self.loss, global_step=global_step)

    return self.train
