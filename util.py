from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io


def to_windows(img, window_size):
  windows = []
  for i in range(img.shape[0] - window_size):
    for j in range(img.shape[1] - window_size):
      windows.append(img[i:i + window_size, j:j + window_size])

  return windows


def window_placeholder_inputs():
  windows = tf.placeholder(tf.float32, [None, None, None, 1])
  labels = tf.placeholder(tf.float32, [None, 1])
  return windows, labels


def fill_window_feed_dict(dataset, windows_pl, labels_pl, batch_size):
  windows_feed, labels_feed = dataset.next_batch(batch_size)
  feed_dict = {
      windows_pl:
      windows_feed.reshape([-1, dataset.window_size, dataset.window_size, 1]),
      labels_pl:
      labels_feed.reshape([-1, 1])
  }
  return feed_dict


def create_dirs(log_dir_path, batch_size, learning_rate):
  import datetime
  import os
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  log_dir = os.path.join(log_dir_path, 'bs_{}_lr_{:.0e}_t_{}'.format(
      batch_size, learning_rate, timestamp))
  plot_dir = os.path.join(log_dir, 'plot')
  train_dir = os.path.join(log_dir, 'train')
  tf.gfile.MakeDirs(train_dir)
  tf.gfile.MakeDirs(log_dir)
  tf.gfile.MakeDirs(plot_dir)

  return log_dir, train_dir, plot_dir


def plot(tdr, fdr, path):
  plt.plot(tdr, 1 - fdr, 'g-')
  plt.xlabel('recall')
  plt.ylabel('precision')
  plt.axis([0, 1, 0, 1])
  plt.grid()

  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  plt.savefig(path)
  plt.clf()

  return buf
