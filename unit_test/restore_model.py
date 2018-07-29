import tensorflow as tf
import numpy as np

import utils
from models import description
from models import detection


def restore_description():
  # create network graph
  inputs, _ = utils.placeholder_inputs()
  net = description.Net(inputs)

  # save random weights and keep them
  # in program's memory for comparison
  vars_ = []
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    # assign random values to variables
    # and save those values for comparison
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # create random values for variable
      var_val = np.random.random(var.shape)

      # save for later comparison
      vars_.append(var_val)

      # assign it to tf var
      assign = tf.assign(var, var_val)
      sess.run(assign)

    # save initialized model
    saver.save(sess, '/tmp/description/model.ckpt', global_step=0)

  # create new session to restore saved weights
  with tf.Session() as sess:
    # make new initialization of weights
    sess.run(tf.global_variables_initializer())

    # assert weights are different
    i = 0
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # get new var val
      var_val = sess.run(var)

      # compare with old one
      assert not np.isclose(np.sum(np.abs(var_val - vars_[i])), 0)

      i += 1

    # restore model
    utils.restore_model(sess, '/tmp/description')

    # check if weights are equal
    i = 0
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # get new var val
      var_val = sess.run(var)

      # compare with old one
      if ~np.any(np.isclose(var_val, vars_[i])):
        print(np.isclose(var_val, vars_[i]))
        print('Failed to load variable "{}"'.format(var.name))
        return False

      i += 1

  return True


def restore_detection():
  # create network graph
  inputs, _ = utils.placeholder_inputs()
  net = detection.Net(inputs)

  # save random weights and keep them
  # in program's memory for comparison
  vars_ = []
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    # assign random values to variables
    # and save those values for comparison
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # create random values for variable
      var_val = np.random.random(var.shape)

      # save for later comparison
      vars_.append(var_val)

      # assign it to tf var
      assign = tf.assign(var, var_val)
      sess.run(assign)

    # save initialized model
    saver.save(sess, '/tmp/detection/model.ckpt', global_step=0)

  # create new session to restore saved weights
  with tf.Session() as sess:
    # make new initialization of weights
    sess.run(tf.global_variables_initializer())

    # assert weights are different
    i = 0
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # get new var val
      var_val = sess.run(var)

      # compare with old one
      assert not np.isclose(np.sum(np.abs(var_val - vars_[i])), 0)

      i += 1

    # restore model
    utils.restore_model(sess, '/tmp/detection')

    # check if weights are equal
    i = 0
    for var in sorted(tf.global_variables(), key=lambda x: x.name):
      # get new var val
      var_val = sess.run(var)

      # compare with old one
      if ~np.any(np.isclose(var_val, vars_[i])):
        print(np.isclose(var_val, vars_[i]))
        print('Failed to load variable "{}"'.format(var.name))
        return False

      i += 1

  return True


if __name__ == '__main__':
  assert restore_description()
  print('[OK - Description Model Restoration]')

  tf.reset_default_graph()

  assert restore_detection()
  print('[OK - Detection Model Restoration]')
