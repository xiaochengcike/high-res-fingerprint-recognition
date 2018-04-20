from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse
import numpy as np

import pore_sliding_window_detector
import polyu
import util
import validation


def train(dataset, learning_rate, batch_size, max_steps, tolerance, log_dir,
          train_dir, plot_dir):
  with tf.Graph().as_default():
    # gets placeholders for windows and labels
    windows_pl, labels_pl = util.window_placeholder_inputs()

    # build train related ops
    pore_det = pore_sliding_window_detector.PoreDetector(
        windows_pl, dataset.train.window_size)
    pore_det.build_loss(labels_pl)
    pore_det.build_train(learning_rate)

    # builds validation inference graph
    val_pores = pore_sliding_window_detector.PoreDetector(
        windows_pl, dataset.train.window_size, training=False, reuse=True)

    # add summary to plot loss, f score, tdr and fdr
    f_score_pl = tf.placeholder(tf.float32, shape=())
    tdr_pl = tf.placeholder(tf.float32, shape=())
    fdr_pl = tf.placeholder(tf.float32, shape=())
    f_score_summary_op = tf.summary.scalar('f_score', f_score_pl)
    tdr_summary_op = tf.summary.scalar('tdr', tdr_pl)
    fdr_summary_op = tf.summary.scalar('fdr', fdr_pl)
    loss_summary_op = tf.summary.scalar('loss', pore_det.loss)

    # resources to tensorboard plots
    plot_buf_pl = tf.placeholder(tf.string)
    plot_png = tf.image.decode_png(plot_buf_pl)
    expanded_plot_png = tf.expand_dims(plot_png, 0)
    plot_summary_op = tf.summary.image('plot', expanded_plot_png)

    # add variable initialization to graph
    init = tf.global_variables_initializer()

    # early stopping vars
    best_f_score = 0
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(init)

      for step in range(1, max_steps + 1):
        feed_dict = util.fill_window_feed_dict(dataset.train, windows_pl,
                                               labels_pl, batch_size)

        _, loss_value = sess.run(
            [pore_det.train, pore_det.loss], feed_dict=feed_dict)

        # write loss summary every 100 steps
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))
          summary_str = sess.run(loss_summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          print('Evaluation:')
          tdrs, fdrs, f_score, fdr, tdr, thr = validation.by_windows(
              sess, val_pores.preds, batch_size, windows_pl, labels_pl,
              dataset.val)
          print(
              '\tTDR = {}'.format(tdr),
              '\tFDR = {}'.format(fdr),
              '\tF score = {}'.format(f_score),
              sep='\n')

          # early stopping
          if f_score > best_f_score:
            best_f_score = f_score
            saver.save(
                sess,
                os.path.join(train_dir, 'model-{}.ckpt'.format(thr)),
                global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= tolerance:
              print('Training stopped early')
              break

          # write f score, tdr and fdr to summary
          score_summaries = sess.run(
              [f_score_summary_op, tdr_summary_op, fdr_summary_op],
              feed_dict={f_score_pl: f_score,
                         tdr_pl: tdr,
                         fdr_pl: fdr})
          for score_summary in score_summaries:
            summary_writer.add_summary(score_summary, global_step=step)

          # plot recall vs precision
          buf = util.plot(tdrs, fdrs,
                          os.path.join(plot_dir, '{}.png'.format(step)))

          # write plot to summary
          plot_summary = sess.run(
              plot_summary_op, feed_dict={plot_buf_pl: buf.getvalue()})
          summary_writer.add_summary(plot_summary, global_step=step)


def main(log_dir_path, polyu_path, window_size, label_size, label_mode,
         max_steps, learning_rate, batch_size, tolerance):
  # create folders to save train resources
  log_dir, train_dir, plot_dir = util.create_dirs(
      log_dir_path, batch_size, learning_rate, label_mode, label_size)

  # load polyu dataset
  print('Loading PolyU-HRF dataset...')
  polyu_path = os.path.join(polyu_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.DetectionDataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      window_size=window_size,
      label_mode=label_mode,
      label_size=label_size)
  print('Loaded.')

  # train
  train(dataset, learning_rate, batch_size, max_steps, tolerance, log_dir,
        train_dir, plot_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir', required=True, type=str, help='Path to PolyU-HRF dataset')
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
      '--window_size', type=int, default=17, help='Pore window size.')
  parser.add_argument(
      '--label_size', type=int, default=3, help='Pore window size.')
  parser.add_argument(
      '--label_mode', type=str, default='hard_bb', help='Pore window size.')
  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS.log_dir, FLAGS.polyu_dir, FLAGS.window_size, FLAGS.label_size,
       FLAGS.label_mode, FLAGS.steps, FLAGS.learning_rate, FLAGS.batch_size,
       FLAGS.tolerance)
