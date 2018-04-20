from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse
import numpy as np

import pore_detector_descriptor
import polyu
import util
import validation

FLAGS = None


def train(det_dataset, desc_dataset, log_dir):
  # other directories paths
  train_dir = os.path.join(log_dir, 'train')
  plot_dir = os.path.join(log_dir, 'plot')

  with tf.Graph().as_default():
    # gets placeholders for windows and labels
    windows_pl, labels_pl = util.window_placeholder_inputs()
    thresholds_pl = tf.placeholder(tf.float32, [None])

    # build net graph
    net = pore_detector_descriptor.Net(windows_pl, FLAGS.window_size)

    # build detection training related ops
    net.build_detection_loss(labels_pl)
    net.build_detection_train(FLAGS.det_lr)

    # build description training related ops
    net.build_description_loss(labels_pl)
    net.build_description_train(FLAGS.desc_lr)

    # builds validation graph
    val_net = pore_detector_descriptor.Net(
        windows_pl, FLAGS.window_size, training=False, reuse=True)
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
        feed_dict = util.fill_detection_feed_dict(
            det_dataset.train, windows_pl, labels_pl, FLAGS.det_batch_sz)
        det_loss_value, _ = sess.run(
            [net.det_loss, net.det_train], feed_dict=feed_dict)

        # description train step
        feed_dict = util.fill_description_feed_dict(
            desc_dataset.train, windows_pl, labels_pl, FLAGS.desc_batch_sz)
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
          tdrs, fdrs, f_score, fdr, tdr, det_thr = validation.detection_by_windows(
              sess, val_net.dets, FLAGS.det_batch_sz, windows_pl, labels_pl,
              det_dataset.val)
          print(
              'Detection:',
              '\tTDR = {}'.format(tdr),
              '\tFDR = {}'.format(fdr),
              '\tF score = {}'.format(f_score),
              sep='\n')

          # description validation
          eer, desc_thr = validation.report_recognition_eer(
              windows_pl, labels_pl, thresholds_pl, desc_dataset.val,
              FLAGS.thr_res, val_net.desc_val, sess, FLAGS.window_size,
              FLAGS.desc_batch_sz,
              (FLAGS.desc_batch_sz % desc_dataset.val.n_labels),
              FLAGS.val_steps)
          print('Description:', '\tEER = {}'.format(eer))

          # early stopping
          if f_score > best_f_score and eer < best_eer:
            # update best statistics
            best_f_score = f_score
            best_eer = eer

            saver.save(
                sess,
                os.path.join(train_dir, 'model-{}.ckpt'.format(thr)),
                global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write f score, tdr and fdr to summary
          score_summaries = sess.run(
              score_summaries_op,
              feed_dict={
                  f_score_pl: f_score,
                  tdr_pl: tdr,
                  fdr_pl: fdr,
                  eer_pl: eer
              })
          for score_summary in score_summaries:
            summary_writer.add_summary(score_summary, global_step=step)

          # plot recall vs precision
          buf = util.plot_precision_recall(tdrs, fdrs,
                                           os.path.join(
                                               plot_dir,
                                               '{}.png'.format(step)))

          # plot roc?

          # write plot to summary
          plot_summary = sess.run(
              plot_summary_op, feed_dict={plot_buf_pl: buf.getvalue()})
          summary_writer.add_summary(plot_summary, global_step=step)

  print('Finished')
  print('best F score = {}'.format(best_f_score))
  print('best EER = {}'.format(best_eer))


def load_description_dataset(polyu_dir_path):
  print('Loading PolyU-HRF DBI-Training dataset...')
  polyu_path = os.path.join(polyu_dir_path, 'DBI', 'Training')
  dataset = polyu.RecognitionDataset(polyu_path)
  print('Loaded.')

  return dataset


def load_detection_dataset(polyu_path):
  print('Loading PolyU-HRF PoreGroundTruth dataset...')
  polyu_path = os.path.join(polyu_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.DetectionDataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      window_size=FLAGS.window_size)
  print('Loaded.')

  return dataset


def main():
  # create folders to save train resources
  log_dir = util.create_dirs(FLAGS.log_dir, FLAGS.det_batch_sz, FLAGS.det_lr)

  # load datasets
  det_dataset = load_detection_dataset(FLAGS.polyu_dir)
  desc_dataset = load_description_dataset(FLAGS.polyu_dir)

  # train
  train(det_dataset, desc_dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir', required=True, type=str, help='Path to PolyU-HRF dataset')
  parser.add_argument(
      '--desc_lr', type=float, default=1e-1, help='Description learning rate.')
  parser.add_argument(
      '--det_lr', type=float, default=1e-1, help='Detection learning rate.')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=40, help='Early stopping tolerance.')
  parser.add_argument(
      '--det_batch_sz', type=int, default=256, help='Detection batch size.')
  parser.add_argument(
      '--desc_batch_sz', type=int, default=256, help='Description batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--window_size', type=int, default=17, help='Pore window size.')
  parser.add_argument(
      '--thr_res',
      type=float,
      default=0.01,
      help='Threshold resolution of ROC curve')
  FLAGS, unparsed = parser.parse_known_args()

  main()
