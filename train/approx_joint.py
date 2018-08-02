import os
import argparse
import numpy as np
import tensorflow as tf

from models import description
from models import detection
import polyu
import utils
import validate

FLAGS = None


def train(desc_dataset, det_dataset, log_dir):
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
                                         labels_pl, FLAGS.desc_batch_size,
                                         FLAGS.augment_desc)
        desc_loss_value, _ = sess.run(
            [desc_net.loss, desc_net.train], feed_dict=feed_dict)

        # detection training step
        feed_dict = utils.fill_feed_dict(det_dataset.train, patches_pl,
                                         labels_pl, FLAGS.det_batch_size,
                                         FLAGS.augment_det)
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
        if step % 1000 == 0:
          # evaluate description
          print('Validation:')
          rank = validate.description.dataset_rank_n(
              patches_pl, sess, desc_val_net.descriptors, desc_dataset.val,
              FLAGS.desc_batch_size, FLAGS.sample_size)
          print('Rank-1 = {}'.format(rank))

          # evaluate detection
          f_score, fdr, tdr = validate.detection.by_patches(
              sess, det_val_net.predictions, FLAGS.det_batch_size, patches_pl,
              labels_pl, det_dataset.val)
          print('TDR = {}'.format(tdr))
          print('FDR = {}'.format(fdr))
          print('F score = {}'.format(f_score))

          # early stopping
          if rank > best_rank and f_score > best_f_score:
            # update best statistics
            best_rank = rank
            best_f_score = f_score

            saver.save(
                sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)
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
  print('best F score = {}'.format(best_f_score))


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir, FLAGS.desc_batch_size,
                              FLAGS.desc_learning_rate, FLAGS.det_batch_size,
                              FLAGS.det_learning_rate)

  # load datasets
  print('Loading description dataset...')
  desc_dataset = polyu.description.Dataset(FLAGS.desc_dataset_path)
  print('Loaded.')

  print('Loading PolyU-HRF dataset...')
  det_dataset_path = os.path.join(FLAGS.det_dataset_path, 'GroundTruth',
                                  'PoreGroundTruth')
  det_dataset = polyu.detection.Dataset(
      os.path.join(det_dataset_path, 'PoreGroundTruthSampleimage'),
      os.path.join(det_dataset_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=FLAGS.patch_size)
  print('Loaded.')

  # train
  train(desc_dataset, det_dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--desc_dataset_path',
      required=True,
      type=str,
      help='path to description dataset')
  parser.add_argument(
      '--det_dataset_path',
      required=True,
      type=str,
      help='path to detection dataset')
  parser.add_argument(
      '--desc_learning_rate',
      type=float,
      default=1e-1,
      help='description learning rate')
  parser.add_argument(
      '--det_learning_rate',
      type=float,
      default=1e-1,
      help='detection learning rate')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='logging directory')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='early stopping tolerance')
  parser.add_argument(
      '--det_batch_size', type=int, default=256, help='detection batch size')
  parser.add_argument(
      '--desc_batch_size',
      type=int,
      default=256,
      help='description batch size')
  parser.add_argument(
      '--steps', type=int, default=100000, help='maximum training steps')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--sample_size',
      type=int,
      default=425,
      help='sample size to retrieve from in rank-N validation')
  parser.add_argument(
      '--augment_desc',
      action='store_true',
      help='use this flag to perform description dataset augmentation')
  parser.add_argument(
      '--augment_det',
      action='store_true',
      help='use this flag to perform detection dataset augmentation')
  parser.add_argument('--seed', type=int, help='random seed')

  FLAGS = parser.parse_args()

  # set random seeds
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  main()
