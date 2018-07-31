import tensorflow as tf


class Net:
  def __init__(self, inputs, reuse=False, training=True, scope='description'):
    self.loss = None
    self.train = None
    self.validation = None

    # capture scope
    self.scope = scope

    with tf.variable_scope(scope, reuse=reuse):
      # reduction convolutions
      net = inputs
      filters_list = [64, 128, 128, 128]
      activations = [tf.nn.relu for _ in range(3)] + [None]
      i = 1
      for filters, activation in zip(filters_list, activations):
        # ith conv layer
        net = tf.layers.conv2d(
            net,
            filters=filters,
            kernel_size=5,
            strides=1,
            padding='valid',
            activation=activation,
            use_bias=False,
            name='conv_{}'.format(i),
            reuse=reuse)

        # ith batch norm
        net = tf.layers.batch_normalization(
            net, training=training, name='batchnorm_{}'.format(i), reuse=reuse)

        # ith max pooling
        net = tf.layers.max_pooling2d(
            net,
            pool_size=5,
            strides=1,
            padding='valid',
            name='maxpool_{}'.format(i))

        i += 1

      # descriptors
      self.spatial_descriptors = tf.nn.l2_normalize(
          net, axis=-1, name='spatial_descriptors')
      self.descriptors = tf.reshape(
          self.spatial_descriptors, [-1, 128], name='descriptors')

  def build_loss(self, labels):
    with tf.variable_scope(self.scope, reuse=True):
      with tf.name_scope('loss'):
        # make labels' shape compatible with triplet loss
        labels = tf.reshape(labels, (-1, ))

        self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            labels, self.descriptors)

    return self.loss

  def build_train(self, learning_rate):
    with tf.variable_scope(self.scope, reuse=True):
      with tf.name_scope('train'):
        global_step = tf.Variable(1, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          self.train = optimizer.minimize(self.loss, global_step=global_step)

    return self.train
