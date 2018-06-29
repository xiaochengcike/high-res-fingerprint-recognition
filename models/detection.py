import tensorflow as tf


class Net:
  def __init__(self, inputs, reuse=False, training=True, scope='detection'):
    with tf.variable_scope(scope):
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
            kernel_size=3,
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
            net, pool_size=3, strides=1, name='maxpool_{}'.format(i))

        i += 1

      # logits is mean of descriptors
      net = tf.reduce_mean(net, axis=-1, keepdims=True)
      self.logits = tf.identity(net, name='logits')

      # build prediction op
      self.predictions = tf.nn.sigmoid(self.logits)

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
