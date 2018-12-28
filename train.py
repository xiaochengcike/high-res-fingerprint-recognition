import os
import argparse
import numpy as np
import tensorflow as tf

from models import description
import polyu
import utils
import validate

FLAGS = None


def train(dataset, log_dir):
  with tf.Graph().as_default():
    # gets placeholders for images and labels
    images_pl, labels_pl = utils.placeholder_inputs()

    # build net graph
    net = description.Net(images_pl, FLAGS.dropout)

    # build training related ops
    net.build_loss(labels_pl, FLAGS.weight_decay)
    net.build_train(FLAGS.learning_rate)

    # builds validation graph
    val_net = description.Net(images_pl, training=False, reuse=True)

    # add summary to plot loss and rank
    eer_pl = tf.placeholder(tf.float32, shape=(), name='eer_pl')
    loss_pl = tf.placeholder(tf.float32, shape=(), name='loss_pl')
    eer_summary_op = tf.summary.scalar('eer', eer_pl)
    loss_summary_op = tf.summary.scalar('loss', loss_pl)

    # early stopping vars
    best_eer = 1
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # initialize summary and variables
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      # 'compute_descriptors' function for validation
      def compute_descriptors(img, pts):
        return utils.trained_descriptors(
            img,
            pts,
            patch_size=dataset.train.images_shape[1],
            session=sess,
            imgs_pl=images_pl,
            descs_op=val_net.descriptors)

      # train loop
      for step in range(1, FLAGS.steps + 1):
        # fill feed dict
        feed_dict = utils.fill_feed_dict(dataset.train, images_pl, labels_pl,
                                         FLAGS.batch_size, FLAGS.augment)
        # train step
        loss_value, _ = sess.run([net.loss, net.train], feed_dict=feed_dict)

        # write loss summary periodically
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))

          # summarize loss
          loss_summary = sess.run(
              loss_summary_op, feed_dict={loss_pl: loss_value})
          summary_writer.add_summary(loss_summary, step)

        # evaluate model periodically
        if step % 500 == 0 and dataset.val is not None:
          print('Validation:')
          eer = validate.matching.validation_eer(dataset.val,
                                                 compute_descriptors)
          print('EER = {}'.format(eer))

          # summarize eer
          eer_summary = sess.run(eer_summary_op, feed_dict={eer_pl: eer})
          summary_writer.add_summary(eer_summary, global_step=step)

          # early stopping
          if eer < best_eer:
            # update early stopping vars
            best_eer = eer
            faults = 0
            saver.save(
                sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

      # if no validation set, save model when training completes
      if dataset.val is None:
        saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

  print('Finished')
  print('best EER = {}'.format(best_eer))


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir_path, FLAGS.batch_size,
                              FLAGS.learning_rate)

  # load dataset
  print('Loading description dataset...')
  dataset = polyu.description.Dataset(FLAGS.dataset_path)
  print('Loaded')

  # train
  train(dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path', required=True, type=str, help='path to dataset')
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
      '--augment',
      action='store_true',
      help='use this flag to perform dataset augmentation')
  parser.add_argument(
      '--dropout', type=float, help='dropout rate in last convolutional layer')
  parser.add_argument('--weight_decay', type=float, help='weight decay lambda')
  parser.add_argument('--seed', type=int, help='random seed')

  FLAGS = parser.parse_args()

  # set random seeds
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  main()
