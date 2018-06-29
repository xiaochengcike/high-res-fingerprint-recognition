from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse

from models import description
from models import detection
import polyu
import utils
import validate

FLAGS = None


def train(desc_dataset, det_dataset, log_dir):
  # other directories paths
  train_dir = os.path.join(log_dir, 'train')

  with tf.Graph().as_default():
    # gets placeholders for patches and labels
    patches_pl, labels_pl = utils.placeholder_inputs()

    # build description net graph
    desc_net = description.Net(patches_pl, scope='description-detection')

    # build description training related ops
    desc_net.build_loss(labels_pl)
    desc_net.build_train(FLAGS.desc_learning_rate)

    # build description validation net graph
    desc_val_net = description.Net(
        patches_pl, training=False, reuse=True, scope='description-detection')

    # build detection net graph
    det_net = detection.Net(
        patches_pl, reuse=True, scope='description-detection')
    det_net.build_loss(labels_pl)
    det_net.build_train(FLAGS.det_learning_rate)

    # build detection validation net graph
    det_val_net = detection.Net(
        patches_pl, training=False, reuse=True, scope='description-detection')

    # add summaries to plot scores and losses
    f_score_pl = tf.placeholder(tf.float32, shape=(), name='f_score_pl')
    tdr_pl = tf.placeholder(tf.float32, shape=(), name='tdr_pl')
    fdr_pl = tf.placeholder(tf.float32, shape=(), name='fdr_pl')
    rank_pl = tf.placeholder(tf.float32, shape=(), name='rank_pl')
    desc_loss_pl = tf.placeholder(tf.float32, shape=(), name='desc_loss_pl')
    det_loss_pl = tf.placeholder(tf.float32, shape=(), name='det_loss_pl')
    scores_summary_op = tf.summary.merge([
        tf.summary.scalar('f_score', f_score_pl),
        tf.summary.scalar('tdr', tdr_pl),
        tf.summary.scalar('fdr', fdr_pl),
        tf.summary.scalar('rank-n', rank_pl)
    ])
    losses_summary_op = tf.summary.merge([
        tf.summary.scalar('desc_loss', desc_loss_pl),
        tf.summary.scalar('det_loss', det_loss_pl)
    ])

    # early stopping vars
    best_rank = 0
    best_f_score = 0
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(tf.global_variables_initializer())

      # train loop
      for step in range(1, FLAGS.steps + 1):
        # description training step
        feed_dict = utils.fill_feed_dict(desc_dataset.train, patches_pl,
                                         labels_pl, FLAGS.desc_batch_size)
        desc_loss_value, _ = sess.run(
            [desc_net.loss, desc_net.train], feed_dict=feed_dict)

        # detection training step
        feed_dict = utils.fill_feed_dict(det_dataset.train, patches_pl,
                                         labels_pl, FLAGS.det_batch_size)
        det_loss_value, _ = sess.run(
            [det_net.loss, det_net.train], feed_dict=feed_dict)

        # write loss summary periodically
        if step % 100 == 0:
          print('Step {}: desc_loss = {}, det_loss = {}'.format(
              step, desc_loss_value, desc_loss_value))

          # summarize losses
          losses_summary = sess.run(
              losses_summary_op,
              feed_dict={
                  desc_loss_pl: desc_loss_value,
                  det_loss_pl: det_loss_value
              })
          summary_writer.add_summary(losses_summary, step)

        # evaluate the model periodically
        if step % 100 == 0:
          # evaluate description
          print('Validation:')
          rank = validate.description.dataset_rank_n(
              patches_pl, sess, desc_val_net.descriptors, desc_dataset.val,
              FLAGS.desc_batch_size, FLAGS.sample_size)
          print('Rank-1 = {}'.format(rank))

          # evaluate detection
          _, _, f_score, fdr, tdr, _ = validate.detection.by_patches(
              sess, det_val_net.predictions, FLAGS.det_batch_size, patches_pl,
              labels_pl, det_dataset.val)
          print('TDR = {}'.format(tdr))
          print('FDR = {}'.format(fdr))
          print('F score = {}'.format(f_score))

          # early stopping
          if rank > best_rank and f_score > best_f_score:
            # update best statistics
            best_rank = rank

            saver.save(
                sess, os.path.join(train_dir, 'model.ckpt'), global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write scores to summary
          scores_summary = sess.run(
              scores_summary_op,
              feed_dict={
                  rank_pl: rank,
                  f_score_pl: f_score,
                  fdr_pl: fdr,
                  tdr_pl: tdr
              })
          summary_writer.add_summary(scores_summary, global_step=step)

  print('Finished')
  print('best Rank-1 = {}'.format(best_rank))


def load_description_dataset(dataset_path):
  print('Loading description dataset...')
  dataset = polyu.description.Dataset(dataset_path)
  print('Loaded.')

  return dataset


def load_detection_dataset(dataset_path, patch_size):
  print('Loading PolyU-HRF dataset...')
  polyu_path = os.path.join(dataset_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.detection.Dataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=patch_size)
  print('Loaded.')

  return dataset


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir, FLAGS.desc_batch_size,
                              FLAGS.desc_learning_rate)

  # load datasets
  desc_dataset = load_description_dataset(FLAGS.desc_dataset_path)
  det_dataset = load_detection_dataset(FLAGS.det_dataset_path,
                                       FLAGS.patch_size)

  # train
  train(desc_dataset, det_dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--desc_dataset_path',
      required=True,
      type=str,
      help='Path to description dataset.')
  parser.add_argument(
      '--det_dataset_path',
      required=True,
      type=str,
      help='Path to detection dataset.')
  parser.add_argument(
      '--desc_learning_rate',
      type=float,
      default=1e-1,
      help='Description learning rate.')
  parser.add_argument(
      '--det_learning_rate',
      type=float,
      default=1e-1,
      help='Detection learning rate.')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='Early stopping tolerance.')
  parser.add_argument(
      '--det_batch_size', type=int, default=256, help='Detection batch size.')
  parser.add_argument(
      '--desc_batch_size',
      type=int,
      default=256,
      help='Description batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='Pore patch size.')
  parser.add_argument(
      '--sample_size',
      type=int,
      default=425,
      help='Sample size to retrieve from in rank-N validation.')

  FLAGS = parser.parse_args()

  main()
