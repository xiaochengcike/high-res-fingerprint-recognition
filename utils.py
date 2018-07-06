import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import os
import cv2


def to_patches(img, patch_size):
  patches = []
  for i in range(img.shape[0] - patch_size + 1):
    for j in range(img.shape[1] - patch_size + 1):
      patches.append(img[i:i + patch_size, j:j + patch_size])

  return patches


def placeholder_inputs():
  images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
  labels = tf.placeholder(tf.float32, [None, 1], name='labels')
  return images, labels


def fill_feed_dict(dataset, patches_pl, labels_pl, batch_size, augment=False):
  patches_feed, labels_feed = dataset.next_batch(batch_size)

  if augment:
    patches_feed = _transform_mini_batch(patches_feed)

  feed_dict = {
      patches_pl: np.expand_dims(patches_feed, axis=-1),
      labels_pl: np.expand_dims(labels_feed, axis=-1)
  }

  return feed_dict


def _transform_mini_batch(sample):
  # contrast and brightness variations
  contrast = np.random.normal(loc=1, scale=0.05, size=(sample.shape[0], 1, 1))
  brightness = np.random.normal(
      loc=0, scale=0.05, size=(sample.shape[0], 1, 1))
  sample = contrast * sample + brightness

  # translation and rotation
  transformed = []
  for point in sample:
    # random translation
    dx = np.random.normal(loc=0, scale=1)
    dy = np.random.normal(loc=0, scale=1)
    A = np.array([[1, 0, dx], [0, 1, dy]])

    # random rotation
    theta = np.random.normal(loc=0, scale=7.5)
    center = (point.shape[1] // 2, point.shape[0] // 2)
    B = cv2.getRotationMatrix2D(center, theta, 1)

    # transform patch
    point = cv2.warpAffine(point, A, point.shape[::-1])
    point = cv2.warpAffine(point, B, point.shape[::-1])

    # add to batch patches
    transformed.append(point)

  return np.array(transformed)


def create_dirs(log_dir_path,
                batch_size,
                learning_rate,
                batch_size2=None,
                learning_rate2=None):
  import datetime
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  if batch_size2 is None or learning_rate2 is None:
    # individual training
    log_dir = os.path.join(log_dir_path, 'bs-{}_lr-{:.0e}_t-{}'.format(
        batch_size, learning_rate, timestamp))
  else:
    # approximate joint training
    log_dir = os.path.join(log_dir_path, 'bs-{}x{}_lr-{:.0e}x{}_t-{}'.format(
        batch_size, batch_size2, learning_rate, learning_rate2, timestamp))

  tf.gfile.MakeDirs(log_dir)

  return log_dir


def plot_precision_recall(tdr, fdr, path):
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


def nms(centers, probs, patch_size, thr):
  area = patch_size * patch_size
  half_patch_size = patch_size // 2

  xs, ys = np.transpose(centers)
  x1 = xs - half_patch_size
  x2 = xs + half_patch_size
  y1 = ys - half_patch_size
  y2 = ys + half_patch_size

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
                                     mode='patch'):
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
  if mode == 'patch':
    for pore_ind, (pore_i, pore_j) in enumerate(pores):
      # all detections within l2 'dist_thr' distance from
      # 'pore' are within l1 'dist_thr' distance from it
      patch = projection[pore_i - dist_thr:pore_i + dist_thr + 1,
                         pore_j - dist_thr:pore_j + dist_thr + 1]

      for det, det_ind in np.ndenumerate(patch):
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
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    raise IOError('No model found in {}.'.format(model_dir))


def load_image(image_path):
  image = cv2.imread(image_path, 0)
  image = np.array(image, dtype=np.float32) / 255
  return image


def load_images(folder_path):
  images = []
  for image_path in sorted(os.listdir(folder_path)):
    if image_path.endswith(('.jpg', '.png', '.bmp')):
      images.append(load_image(os.path.join(folder_path, image_path)))

  return images


def load_images_with_names(images_dir):
  images = load_images(images_dir)
  image_names = [
      path.split('.')[0] for path in sorted(os.listdir(images_dir))
      if path.endswith(('.jpg', '.bmp', '.png'))
  ]

  return images, image_names


def save_dets_txt(dets, filename):
  with open(filename, 'w') as f:
    for coord in dets:
      print(coord[0] + 1, coord[1] + 1, file=f)


def load_dets_txt(pts_path):
  pts = []
  with open(pts_path, 'r') as f:
    for line in f:
      row, col = [int(t) for t in line.split()]
      pts.append((row - 1, col - 1))

  return pts


def draw_matches(img1, pts1, img2, pts2, pairs):
  matches = []
  for pair in pairs:
    matches.append(
        cv2.DMatch(
            _distance=pair[2], _queryIdx=pair[0], _trainIdx=pair[1],
            _imgIdx=0))

  pts1 = list(np.asarray(pts1)[:, [1, 0]])
  pts2 = list(np.asarray(pts2)[:, [1, 0]])
  matched = cv2.drawMatches(img1,
                            cv2.KeyPoint.convert(pts1), img2,
                            cv2.KeyPoint.convert(pts2), matches[:10], None)

  return matched


def bilinear_interpolation(x, y, f):
  x1 = int(x)
  y1 = int(y)
  x2 = x1 + 1
  y2 = y1 + 1

  fq = [[f[x1, y1], f[x1, y2]], [f[x2, y1], f[x1, y2]]]
  lhs = [[x2 - x, x - x1]]
  rhs = [y2 - y, y - y1]

  return np.dot(np.dot(lhs, fq), rhs)


def sift_descriptors(img, pts, scale=4):
  # convert float image to np uint8
  if img.dtype == np.float32:
    img = np.array(255 * img, dtype=np.uint8)

  # improve image quality with median blur and clahe
  img = cv2.medianBlur(img, ksize=3)
  clahe = cv2.createCLAHE(clipLimit=3)
  img = clahe.apply(img)

  # convert points to cv2.keypoints
  pts = list(np.asarray(pts)[:, [1, 0]])
  kpts = cv2.KeyPoint.convert(pts, size=scale)

  # extract sift descriptors
  sift = cv2.xfeatures2d.SIFT_create()
  _, descs = sift.compute(img, kpts)

  return descs


def find_correspondences(descs1,
                         descs2,
                         pts1=None,
                         pts2=None,
                         euclidean_weight=0,
                         transf=None,
                         thr=None):
  # compute descriptors' pairwise distances
  sqr1 = np.sum(descs1 * descs1, axis=1, keepdims=True)
  sqr2 = np.sum(descs2 * descs2, axis=1)
  D = sqr1 - 2 * np.matmul(descs1, descs2.T) + sqr2

  # add points' euclidean distance
  if euclidean_weight != 0:
    assert transf is not None
    assert pts1 is not None
    assert pts2 is not None

    # assure pts are np array
    pts1 = transf(np.array(pts1))
    pts2 = np.array(pts2)

    # compute points' pairwise distances
    sqr1 = np.sum(pts1 * pts1, axis=1, keepdims=True)
    sqr2 = np.sum(pts2 * pts2, axis=1)
    euclidean_D = sqr1 - 2 * np.matmul(pts1, pts2.T) + sqr2

    # add to overral keypoints distance
    D += euclidean_weight * euclidean_D

  # find bidirectional corresponding points
  pairs = []
  if thr is None:
    # find the best correspondence of each element
    # in 'descs2' to an element in 'descs1'
    corrs2 = np.argmin(D, axis=0)

    # find the best correspondence of each element
    # in 'descs1' to an element in 'descs2'
    corrs1 = np.argmin(D, axis=1)

    # keep only bidirectional correspondences
    for i, j in enumerate(corrs2):
      if corrs1[j] == i:
        pairs.append((j, i, D[j, i]))
  else:
    # find the 2 best correspondences of each
    # element in 'descs2' to an element in 'descs1'
    corrs2 = np.argpartition(D.T, [0, 1])[:, :2]

    # find the 2 best correspondences of each
    # element in 'descs1' to an element in 'descs2'
    corrs1 = np.argpartition(D, [0, 1])[:, :2]

    # find bidirectional corresponding points
    # with second best correspondence 'thr'
    # worse than best one
    for i, (j, _) in enumerate(corrs2):
      if corrs1[j, 0] == i:
        # discard close best second correspondences
        if D[j, i] < D[corrs2[i, 1], i] * thr:
          if D[j, i] < D[j, corrs1[j, 1]] * thr:
            pairs.append((j, i, D[j, i]))

  return pairs


def load_images_with_labels(folder_path):
  images = []
  labels = []
  for image_path in sorted(os.listdir(folder_path)):
    if image_path.endswith(('.jpg', '.png', '.bmp')):
      images.append(load_image(os.path.join(folder_path, image_path)))
      labels.append(retrieve_label_from_image_path(image_path))

  return images, labels


def retrieve_label_from_image_path(image_path):
  return int(image_path.split('_')[0])


def eer(pos, neg):
  # sort comparisons arrays for efficiency
  pos = sorted(pos, reverse=True)
  neg = sorted(neg, reverse=True)

  # iterate to find equal error rate
  far = old_far = 0
  frr = old_frr = 0
  j = 0
  for i, neg_score in enumerate(neg):
    # find correspondent positive score
    while j < len(pos) and pos[j] > neg_score:
      j += 1

    # keep old metrics for approximation
    old_far = far
    old_frr = frr

    # compute new metrics
    far = i / len(neg)
    frr = 1 - j / len(pos)

    # if crossing happened, eer is found
    if far >= frr:
      break

  # if crossing is precisely found, return it
  # otherwise, approximate it though linear
  # interpolation and mean of curves
  if far == frr:
    return far
  else:
    return (old_far + far + frr + old_frr) / 4
