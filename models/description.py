import tensorflow as tf


class Net:
  def __init__(self,
               inputs,
               dropout_rate=None,
               reuse=tf.AUTO_REUSE,
               training=True,
               scope='description'):
    self.loss = None
    self.train = None
    self.validation = None

    # capture scope
    self.scope = scope

    with tf.variable_scope(scope, reuse=reuse):
      # conv layers
      net = inputs
      filters_ls = [32, 32, 64, 64, 128, 128]
      strides_ls = [1, 1, 2, 1, 2, 1]
      i = 1
      for filters, strides in zip(filters_ls, strides_ls):
        net = tf.layers.conv2d(
            net,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding='same',
            activation=tf.nn.relu,
            use_bias=False,
            name='conv_{}'.format(i),
            reuse=reuse)
        net = tf.layers.batch_normalization(
            net, training=training, name='batchnorm_{}'.format(i), reuse=reuse)

        i += 1

      # dropout
      if dropout_rate is not None:
        net = tf.layers.dropout(net, rate=dropout_rate, training=training)

      # last conv layer
      net = tf.layers.conv2d(
          net,
          filters=128,
          kernel_size=8,
          strides=1,
          padding='valid',
          activation=None,
          use_bias=False,
          name='conv_{}'.format(i),
          reuse=reuse)
      net = tf.layers.batch_normalization(
          net, training=training, name='batchnorm_{}'.format(i), reuse=reuse)

      # descriptors
      spatial_descriptors = tf.nn.l2_normalize(
          net, axis=-1, name='spatial_descriptors')
      self.descriptors = tf.reshape(
          spatial_descriptors, [-1, 128], name='descriptors')

  def build_loss(self, labels, decay_weight=None):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      with tf.name_scope('loss'):
        # triplet loss
        labels = tf.reshape(labels, (-1, ))
        self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            labels, self.descriptors)

        # weight decay
        if decay_weight is not None:
          weight_decay = 0
          for var in tf.trainable_variables(self.scope):
            if 'kernel' in var.name:
              weight_decay += tf.nn.l2_loss(var)
          self.loss += decay_weight * weight_decay

    return self.loss

  def build_train(self, learning_rate):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      with tf.name_scope('train'):
        global_step = tf.Variable(1, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          self.train = optimizer.minimize(self.loss, global_step=global_step)

    return self.train
