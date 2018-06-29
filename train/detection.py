from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse

from models import detection
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

    # build train related ops
    net = detection.Net(patches_pl)
    net.build_loss(labels_pl)
    net.build_train(FLAGS.learning_rate)

    # builds validation inference graph
    val_net = detection.Net(patches_pl, training=False, reuse=True)

    # add summary to plot loss, f score, tdr and fdr
    f_score_pl = tf.placeholder(tf.float32, shape=())
    tdr_pl = tf.placeholder(tf.float32, shape=())
    fdr_pl = tf.placeholder(tf.float32, shape=())
    f_score_summary_op = tf.summary.scalar('f_score', f_score_pl)
    tdr_summary_op = tf.summary.scalar('tdr', tdr_pl)
    fdr_summary_op = tf.summary.scalar('fdr', fdr_pl)
    loss_summary_op = tf.summary.scalar('loss', net.loss)

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

      for step in range(1, FLAGS.steps + 1):
        feed_dict = utils.fill_feed_dict(dataset.train, patches_pl, labels_pl,
                                         FLAGS.batch_size)

        _, loss_value = sess.run([net.train, net.loss], feed_dict=feed_dict)

        # write loss summary periodically
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))
          summary_str = sess.run(loss_summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          print('Evaluation:')
          tdrs, fdrs, f_score, fdr, tdr, thr = validate.detection.by_patches(
              sess, val_net.predictions, FLAGS.batch_size, patches_pl,
              labels_pl, dataset.val)
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
            if faults >= FLAGS.tolerance:
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
          buf = utils.plot_precision_recall(tdrs, fdrs,
                                            os.path.join(
                                                plot_dir,
                                                '{}.png'.format(step)))

          # write plot to summary
          plot_summary = sess.run(
              plot_summary_op, feed_dict={plot_buf_pl: buf.getvalue()})
          summary_writer.add_summary(plot_summary, global_step=step)


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir_path, FLAGS.batch_size,
                              FLAGS.learning_rate)

  # load polyu dataset
  print('Loading PolyU-HRF dataset...')
  polyu_path = os.path.join(FLAGS.polyu_dir_path, 'GroundTruth',
                            'PoreGroundTruth')
  dataset = polyu.detection.Dataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=FLAGS.patch_size,
      label_mode=FLAGS.label_mode,
      label_size=FLAGS.label_size)
  print('Loaded.')

  # train
  train(dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='Path to PolyU-HRF dataset')
  parser.add_argument(
      '--learning_rate', type=float, default=1e-1, help='Learning rate.')
  parser.add_argument(
      '--log_dir_path', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='Early stopping tolerance.')
  parser.add_argument(
      '--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='Pore patch size.')
  parser.add_argument(
      '--label_size', type=int, default=3, help='Pore label size.')
  parser.add_argument(
      '--label_mode', type=str, default='hard_bb', help='Pore patch size.')
  FLAGS = parser.parse_args()

  main()
