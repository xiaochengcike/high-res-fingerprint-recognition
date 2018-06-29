import tensorflow as tf
import os
import argparse

from models import detection
import polyu
import utils
import validate

FLAGS = None


def train(dataset, log_dir):
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
    scores_summary_op = tf.summary.merge([
        tf.summary.scalar('f_score', f_score_pl),
        tf.summary.scalar('tdr', tdr_pl),
        tf.summary.scalar('fdr', fdr_pl)
    ])
    loss_summary_op = tf.summary.scalar('loss', net.loss)

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
          f_score, fdr, tdr = validate.detection.by_patches(
              sess, val_net.predictions, FLAGS.batch_size, patches_pl,
              labels_pl, dataset.val)
          print('TDR = {}'.format(tdr))
          print('FDR = {}'.format(fdr))
          print('F score = {}'.format(f_score))

          # early stopping
          if f_score > best_f_score:
            best_f_score = f_score
            saver.save(
                sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write f score, tdr and fdr to summary
          scores_summary = sess.run(
              scores_summary_op,
              feed_dict={f_score_pl: f_score,
                         tdr_pl: tdr,
                         fdr_pl: fdr})
          summary_writer.add_summary(scores_summary, global_step=step)

  print('Finished')
  print('best F score = {}'.format(best_f_score))


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
      help='path to PolyU-HRF dataset')
  parser.add_argument(
      '--learning_rate', type=float, default=1e-1, help='learning rate')
  parser.add_argument(
      '--log_dir_path', type=str, default='log', help='logging directory')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='early stopping tolerance')
  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument(
      '--steps', type=int, default=100000, help='maximum training steps')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--label_size', type=int, default=3, help='pore label size')
  parser.add_argument(
      '--label_mode', type=str, default='hard_bb', help='pore patch size')
  FLAGS = parser.parse_args()

  main()
