from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import numpy as np
import os
import argparse

import descriptor_detector
import polyu
import utils
import validate

FLAGS = None


def train(dataset, log_dir):
  # other directories paths
  train_dir = os.path.join(log_dir, 'train')
  plot_dir = os.path.join(log_dir, 'plot')

  with tf.Graph().as_default():
    # gets placeholders for patches and labels
    patches_pl, labels_pl = utils.placeholder_inputs()
    thresholds_pl = tf.placeholder(tf.float32, [None], name='thrs')

    # build net graph
    net = descriptor_detector.Net(patches_pl)

    # build training related ops
    net.build_description_loss(labels_pl)
    net.build_description_train(FLAGS.learning_rate)

    # builds validation graph
    val_net = descriptor_detector.Net(patches_pl, training=False, reuse=True)
    val_net.build_description_validation(labels_pl, thresholds_pl)

    # add summary to plot loss and eer
    eer_pl = tf.placeholder(tf.float32, shape=())
    loss_pl = tf.placeholder(tf.float32, shape=())
    eer_summary_op = tf.summary.scalar('eer', eer_pl)
    loss_summary_op = tf.summary.scalar('loss', loss_pl)

    # early stopping vars
    best_eer = 1
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(tf.global_variables_initializer())

      # train loop
      for step in range(1, FLAGS.steps + 1):
        feed_dict = utils.fill_feed_dict(dataset.train, patches_pl, labels_pl,
                                         FLAGS.batch_size)
        loss_value, _ = sess.run(
            [net.desc_loss, net.desc_train], feed_dict=feed_dict)

        # write loss summary every 100 steps
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))

          # summarize loss
          loss_summary = sess.run(
              loss_summary_op, feed_dict={loss_pl: loss_value})
          summary_writer.add_summary(loss_summary, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          print('Validation:')
          eer = validate.recognition_eer(
              patches_pl, labels_pl, thresholds_pl, dataset.val, FLAGS.thr_res,
              val_net.desc_val, sess, FLAGS.batch_size)
          print('EER = {}'.format(eer))

          # early stopping
          if eer < best_eer:
            # update best statistics
            best_eer = eer

            saver.save(
                sess, os.path.join(train_dir, 'model.ckpt'), global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write eer to summary
          eer_summary = sess.run(eer_summary_op, feed_dict={eer_pl: eer})
          summary_writer.add_summary(eer_summary, global_step=step)

  print('Finished')
  print('best EER = {}'.format(best_eer))


def load_description_dataset(dataset_path):
  print('Loading description dataset...')
  dataset = polyu.description.Dataset(dataset_path)
  print('Loaded.')

  return dataset


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir, FLAGS.batch_size,
                              FLAGS.learning_rate)

  # load dataset
  dataset = load_description_dataset(FLAGS.dataset_path)

  # train
  train(dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path', required=True, type=str, help='Path to dataset.')
  parser.add_argument(
      '--learning_rate', type=float, default=1e-1, help='Learning rate.')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='Early stopping tolerance.')
  parser.add_argument(
      '--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--thr_res',
      type=float,
      default=0.01,
      help='Threshold resolution of ROC curve.')

  FLAGS = parser.parse_args()

  main()
