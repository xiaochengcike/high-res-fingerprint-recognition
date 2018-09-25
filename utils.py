import os
import tensorflow as tf
import numpy as np
import cv2


def placeholder_inputs():
  images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
  labels = tf.placeholder(tf.float32, [None, 1], name='labels')
  return images, labels


def fill_feed_dict(dataset, patches_pl, labels_pl, batch_size, augment=False):
  '''
  Creates a tf feed_dict containing patches and corresponding labels from a
  mini-batch of size batch_size sampled from dataset, possibly transforming it
  for online dataset augmentation.

  Args:
    dataset: dataset object satisfying polyu._Dataset.next_batch signature.
    patches_pl: tf placeholder for patches.
    labels_pl: tf placeholder for labels.
    batch_size: size of mini-batch to be sampled.
    augment: whether or not this mini-batch should be transformed.

  Returns:
    tf feed_dict containing possibly augmented patches and corresponding labels.
  '''
  patches_feed, labels_feed = dataset.next_batch(batch_size)

  if augment:
    patches_feed = _transform_mini_batch(patches_feed)

  feed_dict = {
      patches_pl: np.expand_dims(patches_feed, axis=-1),
      labels_pl: np.expand_dims(labels_feed, axis=-1)
  }

  return feed_dict


def _transform_mini_batch(sample):
  '''
  Transforms an image sample with contrast and brightness variations,
  translations and rotations. These transformations are sampled from a
  multivariate normal distribution with zero covariance or, equivalently,
  from individual normal distributions. Rotations and translations are
  done as if each image of sample is zero-padded. Rotations are also done
  by linearly interpolating the image.

  Args:
    sample: array of images to be transformed.

  Returns:
    randomly transformed sample.
  '''
  # contrast and brightness variations
  contrast = np.random.normal(loc=1, scale=0.05, size=(sample.shape[0], 1, 1))
  brightness = np.random.normal(
      loc=0, scale=0.05, size=(sample.shape[0], 1, 1))
  sample = contrast * sample + brightness

  # translation and rotation
  transformed = []
  for image in sample:
    # random translation
    dx = np.random.normal(loc=0, scale=1)
    dy = np.random.normal(loc=0, scale=1)
    A = np.array([[1, 0, dx], [0, 1, dy]])

    # random rotation
    theta = np.random.normal(loc=0, scale=7.5)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    B = cv2.getRotationMatrix2D(center, theta, 1)

    # transform image
    image = cv2.warpAffine(image, A, image.shape[::-1])
    image = cv2.warpAffine(image, B, image.shape[::-1], flags=cv2.INTER_LINEAR)

    # add to batch images
    transformed.append(image)

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
    log_dir = os.path.join(
        log_dir_path, 'bs-{}_lr-{:.0e}_t-{}'.format(batch_size, learning_rate,
                                                    timestamp))
  else:
    # approximate joint training
    log_dir = os.path.join(
        log_dir_path, 'bs-{}x{}_lr-{:.0e}x{}_t-{}'.format(
            batch_size, batch_size2, learning_rate, learning_rate2, timestamp))

  tf.gfile.MakeDirs(log_dir)

  return log_dir


def nms(centers, probs, bb_size, thr):
  '''
  Converts each center in centers into center centered bb_size x bb_size
  bounding boxes and applies Non-Maximum-Suppression to them.

  Args:
    centers: centers of detection bounding boxes.
    probs: probabilities for each of the detections.
    bb_size: bounding box size.
    thr: NMS discarding intersection threshold.

  Returns:
    dets: np.array of NMS filtered detections.
    det_probs: np.array of corresponding detection probabilities.
  '''
  area = bb_size * bb_size
  half_bb_size = bb_size // 2

  xs, ys = np.transpose(centers)
  x1 = xs - half_bb_size
  x2 = xs + half_bb_size
  y1 = ys - half_bb_size
  y2 = ys + half_bb_size

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


def pairwise_distances(x1, x2):
  # memory efficient implementation based on Yaroslav Bulatov's answer in
  # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  sqr1 = np.sum(x1 * x1, axis=1, keepdims=True)
  sqr2 = np.sum(x2 * x2, axis=1)
  D = sqr1 - 2 * np.matmul(x1, x2.T) + sqr2

  return D


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


def bilinear_interpolation(x, y, f):
  x1 = int(x)
  y1 = int(y)
  x2 = x1 + 1
  y2 = y1 + 1

  fq = [[f[x1, y1], f[x1, y2]], [f[x2, y1], f[x1, y2]]]
  lhs = [[x2 - x, x - x1]]
  rhs = [y2 - y, y - y1]

  return np.dot(np.dot(lhs, fq), rhs)


def sift_descriptors(img, pts, scale=4, normalize=True):
  '''
  Computes SIFT descriptors in pts keypoints of the image img.
  If normalize is True, image is normalized with median blur and
  CLAHE.

  Args:
    img: image from which descriptors will be computed.
    pts: [N, 2] coordinates of N keypoints in img from which
      descriptors will be computed.
    scale: SIFT scale.
    normalize: whether to normalize image with median blur and
      CLAHE prior to descriptor computation.

  Returns:
    [N, 128] np.array of N SIFT descriptors.
  '''
  # empty detections set
  if len(pts) == 0:
    return []

  # convert float image to np uint8
  if img.dtype == np.float32:
    img = np.array(255 * img, dtype=np.uint8)

  # improve image quality with median blur and clahe
  if normalize:
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
  '''
  Finds bidirectional correspondences between descs1 descriptors and
  descs2 descriptors. If thr is provided, discards correspondences
  that fail a distance ratio check with threshold thr. If pts1, pts2,
  and transf are give, the metric considered when finding correspondences
  is
    d(i, j) = ||descs1(j) - descs2(j)||^2 + euclidean_weight *
      * ||transf(pts1(i)) - pts2(j)||^2

  Args:
    descs1: [N, M] np.array of N descriptors of dimension M each.
    descs2: [N, M] np.array of N descriptors of dimension M each.
    pts1: [N, 2] np.array of N coordinates for each descriptor in descs1.
    pts2: [N, 2] np.array of N coordinates for each descriptor in descs2.
    euclidean_weight: weight given to spatial constraint in comparison
      metric.
    transf: alignment transformation that aligns pts1 to pts2.
    thr: distance ratio check threshold.

  Returns:
    list of correspondence tuples (j, i, d) in which index j of
      descs2 corresponds with i of descs1 with distance d.
  '''
  # compute descriptors' pairwise distances
  D = pairwise_distances(descs1, descs2)

  # add points' euclidean distance
  if euclidean_weight != 0:
    assert transf is not None
    assert pts1 is not None
    assert pts2 is not None

    # assure pts are np array
    pts1 = transf(np.array(pts1))
    pts2 = np.array(pts2)

    # compute points' pairwise distances
    euclidean_D = pairwise_distances(pts1, pts2)

    # add to overral keypoints distance
    D += euclidean_weight * euclidean_D

  # find bidirectional corresponding points
  pairs = []
  if thr is None or len(descs1) == 1 or len(descs2) == 1:
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
  '''
  Computes the Equal Error Rate of given comparison scores.
  If FAR and FRR crossing is not exact, lineary interpolate for EER.

  Args:
    pos: scores of genuine comparisons.
    neg: scores of impostor comparisons.

  Returns:
    EER of comparisons.
  '''
  # compute roc curve
  fars, frrs = roc(pos, neg)

  # iterate to find equal error rate
  old_far = None
  old_frr = None
  for far, frr in zip(fars, frrs):
    # if crossing happened, eer is found
    if far >= frr:
      break
    else:
      old_far = far
      old_frr = frr

  # if crossing is precisely found, return it
  # otherwise, approximate it though linear
  # interpolation and mean of curves
  if far == frr:
    return far
  else:
    return (old_far + far + frr + old_frr) / 4


def roc(pos, neg):
  '''
  Computes Receiver Operating Characteristic curve for given comparison scores.

  Args:
    pos: scores of genuine comparisons.
    neg: scores of impostor comparisons.

  Returns:
    fars: False Acceptance Rates (FARs) over all possible thresholds.
    frrs: False Rejection Rates (FRRs) over all possible thresholds.
  '''
  # sort comparisons arrays for efficiency
  pos = sorted(pos, reverse=True)
  neg = sorted(neg, reverse=True)

  # get all scores
  scores = list(pos) + list(neg)
  scores = np.unique(scores)

  # iterate to compute statistsics
  fars = [0.0]
  frrs = [1.0]
  pos_cursor = 0
  neg_cursor = 0
  for score in reversed(scores):
    # find correspondent positive score
    while pos_cursor < len(pos) and pos[pos_cursor] > score:
      pos_cursor += 1

    # find correspondent negative score
    while neg_cursor < len(neg) and neg[neg_cursor] > score:
      neg_cursor += 1

    # compute metrics for score
    far = neg_cursor / len(neg)
    frr = 1 - pos_cursor / len(pos)

    # add to overall statisics
    fars.append(far)
    frrs.append(frr)

  # add last step
  fars.append(1.0)
  frrs.append(0.0)

  return fars, frrs


def retrieval_rank(probe_instance, probe_label, instances, labels):
  # compute distance of 'probe_instance' to
  # every instance in 'instances'
  dists = np.sum((instances - probe_instance)**2, axis=1)

  # sort labels according to instances distances
  matches = np.argsort(dists)
  labels = labels[matches]

  # find index of first instance with label 'probe_label'
  index = np.argwhere(labels == probe_label)[0, 0]

  # compute retrieval rank
  labels_up_to_index = np.unique(labels[:index + 1])
  rank = len(labels_up_to_index)

  return rank


def rank_n(instances, labels, sample_size):
  # get unique labels
  unique_labels = np.unique(labels)

  # initialize ranks
  ranks = np.zeros_like(unique_labels, dtype=np.int32)

  # sort examples by labels
  inds = np.argsort(labels)
  instances = instances[inds]
  labels = labels[inds]

  # compute rank following protocol in belongie et al.
  examples = list(zip(instances, labels))
  for i, (probe, probe_label) in enumerate(examples):
    for target, target_label in examples[i + 1:]:
      if probe_label != target_label:
        break
      else:
        # mix examples of other labels
        other_labels_inds = np.argwhere(labels != probe_label)
        other_labels_inds = np.squeeze(other_labels_inds)
        inds_to_pick = np.random.choice(
            other_labels_inds, sample_size - 1, replace=False)
        instances_to_mix = instances[inds_to_pick]
        labels_to_mix = labels[inds_to_pick]

        # make set for retrieval
        target = np.expand_dims(target, axis=0)
        instance_set = np.concatenate([instances_to_mix, target], axis=0)
        target_label = np.expand_dims(target_label, axis=0)
        label_set = np.concatenate([labels_to_mix, target_label], axis=0)

        # compute retrieval rank for probe
        rank = retrieval_rank(probe, probe_label, instance_set, label_set)

        # update ranks, indexed from 0
        ranks[rank - 1] += 1

  # rank is cumulative
  ranks = np.cumsum(ranks)

  # normalize rank to [0, 1] range
  ranks = ranks / ranks[-1]

  return ranks


def trained_descriptors(img, pts, patch_size, session, imgs_pl, descs_op):
  '''
  Computes descriptors according to descs_op tf tensor operation for
  pts keypoints and patch size of patch_size in the given image.
  If patch_size is even, the last row and last column of the patch
  centered on each keypoint is discarded before descriptor computation.

  Args:
    img: image in which keypoints were detected.
    pts: [N, 2] keypoint coordinates for which descriptors
      should be computed.
    patch_size: patch size for descriptor.
    session: tf session.
    imgs_pl: image input placeholder for descs_op.
    descs_op: tf tensor op that describes images in imgs_pl.

  Returns:
    [N, M] np.array of N descriptors of descs_op of dimension M.
  '''
  # adjust for odd patch sizes
  odd = 1 if patch_size % 2 != 0 else 0

  # get patch locations at 'pts'
  half = patch_size // 2
  patches = []
  for pt in pts:
    if half <= pt[0] < img.shape[0] - half - odd:
      if half <= pt[1] < img.shape[1] - half - odd:
        patch = img[pt[0] - half:pt[0] + half + odd, pt[1] - half:pt[1] +
                    half + odd]
        patches.append(patch)

  # empty detections set
  if len(patches) == 0:
    return []

  # describe patches
  feed_dict = {imgs_pl: np.reshape(patches, np.shape(patches) + (1, ))}
  descs = session.run(descs_op, feed_dict=feed_dict)

  return descs


def compute_orientation(img):
  '''
  Computes fingerprint ridge orientation for every spatial location using
  the method proposed by Hong et al. (Fingerprint Image Enhancement:
  Algorithm and Performance Evaluation, 1998).

  Args:
    img: fingerprint image for which orientation will be computed.

  Returns:
    matrix with ridge orientation per spatial location.
  '''
  # compute gradients
  dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
  dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

  # compute noisy orientation
  nu_x = np.zeros_like(img, dtype=np.float32)
  nu_y = np.zeros_like(img, dtype=np.float32)
  for i in range(8, img.shape[0] - 8):
    for j in range(8, img.shape[1] - 8):
      sub_dx = np.reshape(dx[i - 8:i + 9, j - 8:j + 9], -1)
      sub_dy = np.reshape(dy[i - 8:i + 9, j - 8:j + 9], -1)

      nu_x[i, j] = 2 * np.dot(sub_dx, sub_dy)
      nu_y[i, j] = np.dot(sub_dx + sub_dy, sub_dx - sub_dy)

  # refine orientation
  phi_x = cv2.GaussianBlur(nu_x, (5, 5), 0)
  phi_y = cv2.GaussianBlur(nu_y, (5, 5), 0)
  orientation = np.arctan2(phi_x, phi_y) / 2

  return orientation


def dp_descriptors(img, pts, patch_size):
  '''
  Computes Direct Pore (DP) descriptors (Direct Pore Matching for
  Fingerprint Recognition, 2009) for pts keypoints and patch
  size of patch_size in the given image. If patch_size is even,
  the last row and last column of the patch centered on each keypoint
  is discarded before descriptor computation.

  Args:
    img: image in which keypoints were detected.
    pts: [N, 2] keypoint coordinates for which DP descriptors
      should be computed.
    patch_size: patch size for DP descriptor.

  Returns:
    [N, M] np.array of N DP descriptors of dimension M.
  '''
  # adjust for odd patch sizes
  odd = 1 if patch_size % 2 != 0 else 0

  # compute image's orientation per pixel
  orientation = compute_orientation(img)

  # gaussian blur image
  img = cv2.GaussianBlur(img, (3, 3), 0)

  # get patch locations at 'pts'
  half = patch_size // 2
  center = (half, half)
  descs = []
  for pt in pts:
    if half <= pt[0] < img.shape[0] - half - odd:
      if half <= pt[1] < img.shape[1] - half - odd:
        # extract patch at 'pt'
        patch = img[pt[0] - half:pt[0] + half + odd, pt[1] - half:pt[1] +
                    half + odd]

        # normalize orientation
        theta = orientation[pt[0], pt[1]] - np.pi / 2
        rot_mat = cv2.getRotationMatrix2D(center, 180 * theta / np.pi, 1)
        patch = cv2.warpAffine(
            patch, rot_mat, patch.shape[::-1], flags=cv2.INTER_LINEAR)

        # circular mask
        for i in range(patch_size):
          for j in range(patch_size):
            if np.hypot(i - half, j - half) > half:
              patch[i, j] = 0

        # reshape and normalize
        patch = np.reshape(patch, -1)
        patch -= np.mean(patch)
        patch = patch / np.linalg.norm(patch)

        descs.append(patch)

  return np.array(descs)
