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
import os
import scipy.misc


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


def create_dirs(log_dir_path, batch_size, learning_rate, label_mode,
                label_size):
  import datetime
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  log_dir = os.path.join(
      log_dir_path, 'm-{}_sz-{}_bs-{}_lr-{:.0e}_t-{}'.format(
          label_mode, label_size, batch_size, learning_rate, timestamp))
  plot_dir = os.path.join(log_dir, 'plot')
  train_dir = os.path.join(log_dir, 'train')
  tf.gfile.MakeDirs(train_dir)
  tf.gfile.MakeDirs(log_dir)
  tf.gfile.MakeDirs(plot_dir)

  return log_dir


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
  det_probs = []
  while len(order) > 0:
    i = order[0]
    order = order[1:]
    dets.append(centers[i])
    det_probs.append(probs[i])

    xx1 = np.maximum(x1[i], x1[order])
    yy1 = np.maximum(y1[i], y1[order])
    xx2 = np.minimum(x2[i], x2[order])
    yy2 = np.minimum(y2[i], y2[order])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (2 * area - inter)

    inds = np.where(ovr <= thr)[0]
    order = order[inds]

  return np.array(dets), np.array(det_probs)


def project_and_find_correspondences(pores,
                                     dets,
                                     dist_thr=np.inf,
                                     proj_shape=None,
                                     mode='window'):
  # compute projection shape, if not given
  if proj_shape is None:
    ys = np.concatenate([dets.T[0], pores.T[0]])
    xs = np.concatenate([dets.T[1], pores.T[1]])
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
  if mode == 'window':
    for pore_ind, (pore_i, pore_j) in enumerate(pores):
      # all detections within l2 'dist_thr' distance from
      # 'pore' are within l1 'dist_thr' distance from it
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
  else:
    for pore_ind, pore in enumerate(pores):
      # enqueue 'pore'
      queue = [pore]

      # do not revisit pore
      projection[pore[0], pore[1]] = -1

      # bfs over possible correspondences
      while len(queue) > 0:
        # pop front of queue
        coords = queue[0]
        queue = queue[1:]
        dist = np.linalg.norm(coords - pore)

        # only consider correspondences better than current
        if dist <= pore_dcorrs[pore_ind]:
          # enqueue valid neighbors
          for d in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            ngh = coords + d
            if 0 <= ngh[0] < proj_shape[0] and \
                0 <= ngh[1] < proj_shape[1] and \
                projection[ngh[0], ngh[1]] >= 0:
              queue.append(ngh)

          # check whether 'det' has a detection
          det_ind = projection[coords[0], coords[1]] - 1
          if det_ind > -1:
            # update pore-detection correspondence
            if dist < pore_dcorrs[pore_ind]:
              pore_dcorrs[pore_ind] = dist
              pore_corrs[pore_ind] = det_ind

            # update detection-pore correspondence
            if dist < det_dcorrs[det_ind]:
              det_dcorrs[det_ind] = dist
              det_corrs[det_ind] = pore_ind

  return pore_corrs, pore_dcorrs, det_corrs, det_dcorrs


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


def restore_model(sess, model_dir):
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restored.')
  else:
    raise IOError('No model found in {}.'.format(model_dir))


def _load_image(image_path):
  '''
  Loads the image in 'image_path' as a single channel np float32 array in range [0, 1].

  Args:
    image_path: Path to the image being loaded.

  Returns:
    The loaded image as a single channel np float32 array in range [0, 1].
  '''
  return np.asarray(scipy.misc.imread(image_path, mode='F'),
                    np.float32) / 255.0


def load_images(folder_path):
  '''
  Loads all images in formats 'jpg', 'png' and 'bmp' in folder 'folder_path'.

  Args:
    folder_path: Path to folder for which images are going to be loaded.

  Returns:
    images: List of all loaded images.
  '''
  images = []
  for image_path in sorted(os.listdir(folder_path)):
    if image_path.endswith(('.jpg', '.png', '.bmp')):
      images.append(_load_image(os.path.join(folder_path, image_path)))

  return images
