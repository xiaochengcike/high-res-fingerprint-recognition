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


def train(det_dataset, desc_dataset, log_dir):
  # other directories paths
  train_dir = os.path.join(log_dir, 'train')
  plot_dir = os.path.join(log_dir, 'plot')

  with tf.Graph().as_default():
    # gets placeholders for patches and labels
    patches_pl, labels_pl = utils.placeholder_inputs()
    thresholds_pl = tf.placeholder(tf.float32, [None])

    # build net graph
    net = descriptor_detector.Net(patches_pl)

    # build detection training related ops
    net.build_detection_loss(labels_pl)
    net.build_detection_train(FLAGS.det_lr)

    # build description training related ops
    net.build_description_loss(labels_pl)
    net.build_description_train(FLAGS.desc_lr)

    # builds validation graph
    val_net = descriptor_detector.Net(patches_pl, training=False, reuse=True)
    val_net.build_description_validation(labels_pl, thresholds_pl)

    # add summary to plot losses, eer, f score, tdr and fdr
    f_score_pl = tf.placeholder(tf.float32, shape=())
    eer_pl = tf.placeholder(tf.float32, shape=())
    tdr_pl = tf.placeholder(tf.float32, shape=())
    fdr_pl = tf.placeholder(tf.float32, shape=())
    det_loss_pl = tf.placeholder(tf.float32, shape=())
    desc_loss_pl = tf.placeholder(tf.float32, shape=())
    score_summaries_ops = [
        tf.summary.scalar('f_score', f_score_pl),
        tf.summary.scalar('tdr', tdr_pl),
        tf.summary.scalar('fdr', fdr_pl),
        tf.summary.scalar('eer', eer_pl)
    ]
    det_loss_summary_op = tf.summary.scalar('det_loss', det_loss_pl)
    desc_loss_summary_op = tf.summary.scalar('desc_loss', desc_loss_pl)

    # resources to tensorboard plots
    plot_buf_pl = tf.placeholder(tf.string)
    plot_png = tf.image.decode_png(plot_buf_pl)
    expanded_plot_png = tf.expand_dims(plot_png, 0)
    plot_summary_op = tf.summary.image('plot', expanded_plot_png)

    # early stopping vars
    best_f_score = 0
    best_eer = 1
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(tf.global_variables_initializer())

      # joint train loop
      for step in range(1, FLAGS.steps + 1):
        # detection train step
        feed_dict = utils.fill_detection_feed_dict(
            det_dataset.train, patches_pl, labels_pl, FLAGS.det_batch_size)
        det_loss_value, _ = sess.run(
            [net.det_loss, net.det_train], feed_dict=feed_dict)

        # description train step
        feed_dict = utils.fill_description_feed_dict(
            desc_dataset.train, patches_pl, labels_pl, FLAGS.classes_by_batch)
        desc_loss_value, _ = sess.run(
            [net.desc_loss, net.desc_train], feed_dict=feed_dict)

        # write loss summary every 100 steps
        if step % 100 == 0:
          print('Step {}: det_loss = {}, desc_loss = {}'.format(
              step, det_loss_value, desc_loss_value))

          # summarize detection loss
          det_loss_summary = sess.run(
              det_loss_summary_op, feed_dict={det_loss_pl: det_loss_value})
          summary_writer.add_summary(det_loss_summary, step)

          # summarize description loss
          desc_loss_summary = sess.run(
              desc_loss_summary_op, feed_dict={desc_loss_pl: desc_loss_value})
          summary_writer.add_summary(desc_loss_summary, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          # detection validation
          print('Validation:')
          tdrs, fdrs, f_score, fdr, tdr, det_thr = validate.detection_by_patches(
              sess, val_net.dets, FLAGS.det_batch_size, patches_pl, labels_pl,
              det_dataset.val)
          print(
              'Detection:',
              '\tTDR = {}'.format(tdr),
              '\tFDR = {}'.format(fdr),
              '\tF score = {}'.format(f_score),
              sep='\n')

          # description validation
          eer = validate.report_recognition_eer(
              patches_pl, labels_pl, thresholds_pl, desc_dataset.val,
              FLAGS.thr_res, val_net.desc_val, sess, desc_dataset.val.n_labels,
              FLAGS.val_steps)
          print('Description:', '\tEER = {}'.format(eer), sep='\n')

          # early stopping
          if f_score > best_f_score and eer < best_eer:
            # update best statistics
            best_f_score = f_score
            best_eer = eer

            saver.save(
                sess,
                os.path.join(train_dir, 'model-{}.ckpt'.format(det_thr)),
                global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write f score, tdr and fdr to summary
          score_summaries = sess.run(
              score_summaries_ops,
              feed_dict={
                  f_score_pl: f_score,
                  tdr_pl: tdr,
                  fdr_pl: fdr,
                  eer_pl: eer
              })
          for score_summary in score_summaries:
            summary_writer.add_summary(score_summary, global_step=step)

          # plot recall vs precision
          buf = utils.plot_precision_recall(tdrs, fdrs,
                                            os.path.join(
                                                plot_dir,
                                                '{}.png'.format(step)))

          # TODO: plot recognition roc?

          # write plot to summary
          plot_summary = sess.run(
              plot_summary_op, feed_dict={plot_buf_pl: buf.getvalue()})
          summary_writer.add_summary(plot_summary, global_step=step)

  print('Finished')
  print('best F score = {}'.format(best_f_score))
  print('best EER = {}'.format(best_eer))


def load_description_dataset(dataset_path, patch_size):
  import pickle
  print('Loading precomputed descriptor dataset...')
  with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
  assert dataset.patch_size == patch_size, 'Incompatible patch sizes: {} and {}'.format(
      dataset.patch_size, patch_size)
  print('Loaded.')

  return dataset


def load_detection_dataset(polyu_dir_path, patch_size):
  print('Loading PolyU-HRF PoreGroundTruth dataset...')
  polyu_dir_path = os.path.join(polyu_dir_path, 'GroundTruth',
                                'PoreGroundTruth')
  dataset = polyu.detection.Dataset(
      os.path.join(polyu_dir_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_dir_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=patch_size)
  print('Loaded.')

  return dataset


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir, FLAGS.det_batch_size,
                              FLAGS.det_lr)

  # load datasets
  det_dataset = load_detection_dataset(FLAGS.polyu_dir_path, FLAGS.patch_size)
  desc_dataset = load_description_dataset(FLAGS.rec_dataset_path,
                                          FLAGS.patch_size)

  # train
  train(det_dataset, desc_dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='Path to PolyU-HRF dataset.')
  parser.add_argument(
      '--rec_dataset_path',
      required=True,
      type=str,
      help='Path to precomputed descriptor dataset.')
  parser.add_argument(
      '--desc_lr', type=float, default=1e-1, help='Description learning rate.')
  parser.add_argument(
      '--det_lr', type=float, default=1e-1, help='Detection learning rate.')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='Early stopping tolerance.')
  parser.add_argument(
      '--det_batch_size', type=int, default=256, help='Detection batch size.')
  parser.add_argument(
      '--classes_by_batch',
      type=int,
      default=28,
      help='Number of classes in description batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='Pore patch size.')
  parser.add_argument(
      '--thr_res',
      type=float,
      default=0.01,
      help='Threshold resolution of ROC curve')
  parser.add_argument(
      '--val_steps',
      type=int,
      default=500,
      help='Number of sampled batches when validating')

  FLAGS = parser.parse_args()

  main()
