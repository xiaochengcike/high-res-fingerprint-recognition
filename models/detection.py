import tensorflow as tf


class Net:
  '''
  Pore detection model.

  Net.predictions is the model's output op. It has shape
  [batch_size, N - 16, M - 16, 1] for an input with shape
  [batch_size, N, M, 1]. Net.loss and Net.train are
  respectively the model's loss op and training step op,
  if built with Net.build_loss and Net.build_train.
  '''

  def __init__(self,
               inputs,
               dropout_rate=None,
               reuse=tf.AUTO_REUSE,
               training=True,
               scope='detection'):
    '''
    Args:
      inputs: input placeholder of shape [None, None, None, 1].
      dropout_rate: if None, applies no dropout. Otherwise,
        dropout rate of second to last layer is dropout_rate.
      reuse: whether to reuse net variables in scope.
      training: whether net is training. Required for dropout
        and batch normalization.
      scope: model's variable scope.
    '''
    with tf.variable_scope(scope, reuse=reuse):
      # reduction convolutions
      net = inputs
      filters_ls = [32, 64, 128]
      i = 1
      for filters in filters_ls:
        # ith conv layer
        net = tf.layers.conv2d(
            net,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation=tf.nn.relu,
            use_bias=False,
            name='conv_{}'.format(i),
            reuse=reuse)
        net = tf.layers.batch_normalization(
            net, training=training, name='batchnorm_{}'.format(i), reuse=reuse)
        net = tf.layers.max_pooling2d(
            net, pool_size=3, strides=1, name='maxpool_{}'.format(i))

        i += 1

      # dropout
      if dropout_rate is not None:
        net = tf.layers.dropout(net, rate=dropout_rate, training=training)

      # logits
      net = tf.layers.conv2d(
          net,
          filters=1,
          kernel_size=5,
          strides=1,
          padding='valid',
          activation=None,
          use_bias=False,
          name='conv_{}'.format(i),
          reuse=reuse)
      net = tf.layers.batch_normalization(
          net, training=training, name='batchnorm_{}'.format(i), reuse=reuse)
      self.logits = tf.identity(net, name='logits')

      # build prediction op
      self.predictions = tf.nn.sigmoid(self.logits)

  def build_loss(self, labels):
    '''
    Builds the model's loss node op. The loss is the
    cross-entropy between the model's predictions and
    the labels.

    Args:
      labels: labels placeholder of shape [batch_size, 1].

    Returns:
      the loss node op.
    '''
    # reshape labels to be compatible with logits
    labels = tf.reshape(labels, tf.shape(self.logits))

    # cross entropy loss
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=self.logits, name='xentropy')
    self.loss = tf.reduce_mean(xentropy, name='xentropy_mean')

    return self.loss

  def build_train(self, learning_rate):
    '''
    Builds the model's training step op. It minimizes the
    model's loss with Stochastic Gradient Descent (SGD) with
    an exponentially decayed learning rate and updates the means
    and variances of batch normalization.

    Args:
      learning_rate: initial SGD learning rate.
    '''
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
