from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io


def to_windows(img, window_size):
  windows = []
  for i in range(img.shape[0] - window_size + 1):
    for j in range(img.shape[1] - window_size + 1):
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


def nms(centers, probs, window_size, thr):
  area = window_size * window_size
  half_window_size = window_size // 2

  xs, ys = np.transpose(centers)
  x1 = xs - half_window_size
  x2 = xs + half_window_size
  y1 = ys - half_window_size
  y2 = ys + half_window_size

  order = np.argsort(probs)[::-1]

  dets = []
  while len(order) > 0:
    i = order[0]
    dets.append(centers[i])
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (2 * area - inter)

    inds = np.where(ovr <= thr)[0]
    order = order[inds + 1]

  return dets


def project_and_find_correspondences(pores, dets, dist_thr, proj_shape=None):
  # compute projection shape, if not given
  if proj_shape is None:
    ys = dets.T[0]
    xs = dets.T[1]
    proj_shape = (np.max(ys) + 1, np.max(xs) + 1)

  # project detections
  projection = np.zeros(proj_shape, dtype=np.int32)
  for i, (y, x) in enumerate(dets):
    projection[y, x] = i + 1

  # find correspondences
  pore_corrs = np.full(len(pores), -1, dtype=np.int32)
  pore_dcorrs = np.full(len(pores), dist_thr)
  det_corrs = np.full(len(dets), -1, dtype=np.int32)
  det_dcorrs = np.full(len(dets), dist_thr)
  for pore_ind, pore in enumerate(pores):
    # all detections within l2 'dist_thr' distance from
    # 'pore' are within l1 'dist_thr' distance from it
    pore_i, pore_j = pore
    window = projection[pore_i - dist_thr:pore_i + dist_thr + 1,
                        pore_j - dist_thr:pore_j + dist_thr + 1]

    for det, det_ind in np.ndenumerate(window):
      # check whether 'det' has a detection
      if det_ind != 0:
        det_ind -= 1

        # compute pore-detection distance
        dist = np.linalg.norm(det)

        # update pore-detection correspondence
        if dist < pore_dcorrs[pore_ind]:
          pore_dcorrs[pore_ind] = dist
          pore_corrs[pore_ind] = det_ind

        # update detection-pore correspondence
        if dist < det_dcorrs[det_ind]:
          det_dcorrs[det_ind] = dist
          det_corrs[det_ind] = pore_ind

  return pore_corrs, det_corrs


def matmul_corr_finding(pores, dets):
  # memory efficient implementation based on Yaroslav Bulatov's answer in
  # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  # compute pair-wise distances
  D = np.sum(pores * pores, axis=1).reshape(-1, 1) - \
      2 * np.dot(pores, dets.T) + np.sum(dets * dets, axis=1)

  # get pore-detection correspondences
  pore_corrs = np.argmin(D, axis=1)

  # get detection-pore correspondences
  det_corrs = np.argmin(D, axis=0)

  return pore_corrs, det_corrs


def restore_model(model_dir):
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    raise IOError('No model found in {}.'.format(ckpt_dir))
