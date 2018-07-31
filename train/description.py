import tensorflow as tf
import os
import argparse

from models import description
import polyu
import utils
import validate

FLAGS = None


def train(dataset, log_dir):
  with tf.Graph().as_default():
    # gets placeholders for patches and labels
    patches_pl, labels_pl = utils.placeholder_inputs()

    # build net graph
    net = description.Net(patches_pl, FLAGS.dropout_rate)

    # build training related ops
    net.build_loss(labels_pl, FLAGS.weight_decay)
    net.build_train(FLAGS.learning_rate)

    # builds validation graph
    val_net = description.Net(patches_pl, training=False, reuse=True)

    # add summary to plot loss and rank
    rank_pl = tf.placeholder(tf.float32, shape=(), name='rank_pl')
    loss_pl = tf.placeholder(tf.float32, shape=(), name='loss_pl')
    rank_summary_op = tf.summary.scalar('rank', rank_pl)
    loss_summary_op = tf.summary.scalar('loss', loss_pl)

    # early stopping vars
    best_rank = 0
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(tf.global_variables_initializer())

      # train loop
      for step in range(1, FLAGS.steps + 1):
        feed_dict = utils.fill_feed_dict(dataset.train, patches_pl, labels_pl,
                                         FLAGS.batch_size, FLAGS.augment)
        loss_value, _ = sess.run([net.loss, net.train], feed_dict=feed_dict)

        # write loss summary periodically
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))

          # summarize loss
          loss_summary = sess.run(
              loss_summary_op, feed_dict={loss_pl: loss_value})
          summary_writer.add_summary(loss_summary, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          print('Validation:')
          rank = validate.description.dataset_rank_1(
              patches_pl, sess, val_net.descriptors, dataset.val,
              FLAGS.batch_size, FLAGS.sample_size)
          print('Rank-1 = {}'.format(rank))

          # early stopping
          if rank > best_rank:
            # update best statistics
            best_rank = rank

            saver.save(
                sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= FLAGS.tolerance:
              print('Training stopped early')
              break

          # write rank to summary
          rank_summary = sess.run(rank_summary_op, feed_dict={rank_pl: rank})
          summary_writer.add_summary(rank_summary, global_step=step)

  print('Finished')
  print('best Rank-1 = {}'.format(best_rank))


def main():
  # create folders to save train resources
  log_dir = utils.create_dirs(FLAGS.log_dir, FLAGS.batch_size,
                              FLAGS.learning_rate)

  # load dataset
  print('Loading description dataset...')
  dataset = polyu.description.Dataset(FLAGS.dataset_path)
  print('Loaded.')

  # train
  train(dataset, log_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path', required=True, type=str, help='path to dataset')
  parser.add_argument(
      '--learning_rate', type=float, default=1e-1, help='learning rate')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='logging directory')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='early stopping tolerance')
  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument(
      '--steps', type=int, default=100000, help='maximum training steps')
  parser.add_argument(
      '--sample_size',
      type=int,
      default=425,
      help='sample size to retrieve from in rank-N validation')
  parser.add_argument(
      '--augment',
      action='store_true',
      help='use this flag to perform dataset augmentation')
  parser.add_argument(
      '--dropout_rate',
      type=float,
      help='dropout rate in last convolutional layer')
  parser.add_argument('--weight_decay', type=float, help='weight decay lambda')

  FLAGS = parser.parse_args()

  main()
