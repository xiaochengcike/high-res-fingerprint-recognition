from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os
import numpy as np

import util


class _DetectionDataset:
  def __init__(self,
               images,
               labels,
               shuffle_behavior,
               incomplete_batches,
               window_size=None,
               one_hot=False,
               label_mode=None,
               label_size=None):
    self._images = np.array(images, dtype=np.float32)
    self._labels = np.array(labels, dtype=np.float32)
    self.window_size = window_size
    self._shuffle_behavior = shuffle_behavior
    self._one_hot = one_hot
    self._incomplete_batches = incomplete_batches

    # window batch pointers
    self._index_in_epoch = 0
    self._epochs_completed = 0

    # image batch pointers
    self._image_index_in_epoch = 0
    self._image_epochs_completed = 0

    self.num_images = len(self._images)
    self._image_rows = self._images[0].shape[0]
    self._image_cols = self._images[0].shape[1]
    if window_size is not None:
      self.num_samples = self.num_images * (self._image_rows - window_size) * (
          self._image_cols - window_size)

      # create labels to be sampled by windows
      if label_mode == 'hard_bb':
        self._window_labels = np.zeros_like(self._labels, dtype=np.float32)

        # draw 'label_size' bounding box around pores
        for k, i, j in np.ndindex(self._window_labels.shape):
          i_ = max(i - label_size, 0)
          j_ = max(j - label_size, 0)
          self._window_labels[k, i, j] = np.max(
              self._labels[k, i_:i + 1 + label_size, j_:j + 1 + label_size])

      elif label_mode == 'hard_l1' or label_mode == 'hard_l2':
        self._window_labels = np.zeros_like(self._labels, dtype=np.float32)

        # define norm to use
        norm_l = int(label_mode[-1])

        # enqueue pores with origins
        queue = []
        for pore in np.argwhere(self._labels >= 0.5):
          self._window_labels[pore[0], pore[1], pore[2]] = self._labels[pore[
              0], pore[1], pore[2]]
          queue.append((pore, pore))

        # bfs to draw l'norm_l' ball around pores
        while queue:
          # pop front
          coords, anchor = queue[0]
          queue = queue[1:]

          # propagate pore anchor label
          k, i, j = anchor
          val = self._window_labels[k, i, j]

          # enqueue valid neighbors
          for d in [(0, 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1)]:
            ngh = coords + d
            _, i, j = ngh
            if 0 <= i < self._window_labels[k].shape[0] and \
                0 <= j < self._window_labels[k].shape[1] and \
                self._window_labels[k, i, j] == 0 and \
                np.linalg.norm(ngh - anchor, norm_l) <= label_size:
              self._window_labels[k, i, j] = val
              queue.append((ngh, anchor))

  def next_batch(self, batch_size, shuffle=None, incomplete=None):
    '''
    Sample next batch, of size 'batch_size', of image windows.

    Args:
      batch_size: Size of batch to be sampled.
      shuffle: Overrides dataset split shuffle behavior, ie if data should be shuffled.
      incomplete: Overrides dataset incomplete batch behavior, ie if when completing an epoch, a batch should be provided with less samples than predicted so to not get samples from different epochs.

    Returns:
      The sampled window batch and corresponding labels as np arrays.
    '''
    # determine shuffle and incomplete behaviors
    if shuffle is None:
      shuffle = self._shuffle_behavior

    if incomplete is None:
      incomplete = self._incomplete_batches

    start = self._index_in_epoch

    # shuffle for first epoch
    if self._epochs_completed == 0 and start == 0:
      self._perm = np.arange(self.num_samples)
      if shuffle:
        np.random.shuffle(self._perm)

    # go to next epoch
    if start + batch_size >= self.num_samples:
      # finished epoch
      self._epochs_completed += 1

      # get the rest of samples in this epoch
      rest_num_samples = self.num_samples - start
      images_rest_part, labels_rest_part = self._to_windows(self._perm[start:])

      # shuffle the data
      if shuffle:
        self._perm = np.arange(self.num_samples)
        np.random.shuffle(self._perm)

      # return incomplete batch
      if incomplete:
        self._index_in_epoch = 0
        return images_rest_part, labels_rest_part

      # start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_samples
      end = self._index_in_epoch

      # retrive samples in the new epoch
      images_new_part, labels_new_part = self._to_windows(
          self._perm[start:end])

      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._to_windows(self._perm[start:end])

  def next_image_batch(self, batch_size, shuffle=None, incomplete=None):
    '''
    Sample next batch, of size 'batch_size', of images.

    Args:
      batch_size: Size of batch to be sampled.
      shuffle: Overrides dataset split shuffle behavior, ie if data should be shuffled.
      incomplete: Overrides dataset incomplete batch behavior, ie if when completing an epoch, a batch should be provided with less images than predicted so to not get images from different epochs.

    Returns:
      The sampled image batch and corresponding labels as np arrays.
    '''
    # determine shuffle and incomplete behaviors
    if shuffle is None:
      shuffle = self._shuffle_behavior

    if incomplete is None:
      incomplete = self._incomplete_batches

    start = self._image_index_in_epoch

    # shuffle for first epoch
    if self._image_epochs_completed == 0 and start == 0:
      self._image_perm = np.arange(self.num_images)
      if shuffle:
        np.random.shuffle(self._image_perm)

    # go to next epoch
    if start + batch_size >= self.num_images:
      # finished epoch
      self._image_epochs_completed += 1

      # get the rest of images in this epoch
      rest_num_images = self.num_images - start
      images_rest_part, labels_rest_part = self._images[start:], self._labels[
          start:]

      # shuffle the data
      if shuffle:
        self._image_perm = np.arange(self.num_images)
        np.random.shuffle(self._image_perm)

      # return incomplete batch
      if incomplete:
        self._image_index_in_epoch = 0
        return images_rest_part, labels_rest_part

      # start next epoch
      start = 0
      self._image_index_in_epoch = batch_size - rest_num_images
      end = self._image_index_in_epoch

      # retrive images in the new epoch
      images_new_part, labels_new_part = self._images[start:end], self._labels[
          start:end]

      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._image_index_in_epoch += batch_size
      end = self._image_index_in_epoch
      return self._images[start:end], self._labels[start:end]

  def _to_windows(self, indices):
    '''
    Retrieves image windows, on demand, corresponding to given indices.
    A window of size WINDOW_SIZE is centered in the i-th row of the j-th column of the k-th image (of dimensions ROWS x COLS) according to the given index I:
      k = I / ((ROWS - WINDOW_SIZE) * (COLS - WINDOW_SIZE))
      i = I / (COLS - WINDOW_SIZE) - k * (ROWS - WINDOW_SIZE)
      j = I mod (COLS - SIZE)

      Args:
        indices: Indices for which windows will be produced.

      Returns:
        windows: Windows corresponding to the given indices.
        labels: Labels of the returned windows.

    '''
    # windows (samples) and labels
    size = self.window_size
    windows = np.empty([indices.shape[0], size, size], np.float32)
    if self._one_hot:
      labels = np.empty([indices.shape[0], 2], np.float32)
    else:
      labels = np.empty([indices.shape[0]], np.float32)

    for index in range(indices.shape[0]):
      image_index = indices[index]

      # retrieve image number, row and column from index
      k = image_index // ((self._image_rows - size) *
                          (self._image_cols - size))
      i = image_index // (self._image_cols - size) - k * (
          self._image_rows - size)
      j = image_index % (self._image_cols - size)

      # generate window
      windows[index] = self._images[k, i:i + size, j:j + size]

      # generate corresponding label
      center = size // 2
      if self._one_hot:
        labels[index, 0] = self._window_labels[k, i + center, j + center]
        labels[index, 1] = 1 - labels[index, 0]
      else:
        labels[index] = self._window_labels[k, i + center, j + center]

    return windows, labels


class DetectionDataset:
  def __init__(self,
               images_folder_path,
               labels_folder_path,
               split,
               window_size=None,
               label_mode='hard_bb',
               label_size=3,
               should_shuffle=True,
               one_hot=False):
    self._images = util.load_images(images_folder_path)
    self._labels = self._load_labels(labels_folder_path)

    # splits loaded according to given 'split'
    if split[0] > 0:
      self.train = _DetectionDataset(
          self._images[:split[0]],
          self._labels[:split[0]],
          shuffle_behavior=should_shuffle,
          incomplete_batches=False,
          window_size=window_size,
          one_hot=one_hot,
          label_mode=label_mode,
          label_size=label_size)
    else:
      self.train = None

    if split[1] > 0:
      self.val = _DetectionDataset(
          self._images[split[0]:split[0] + split[1]],
          self._labels[split[0]:split[0] + split[1]],
          shuffle_behavior=False,
          incomplete_batches=True,
          window_size=window_size,
          one_hot=one_hot,
          label_mode=label_mode,
          label_size=label_size)
    else:
      self.val = None

    if split[2] > 0:
      self.test = _DetectionDataset(
          self._images[split[0] + split[1]:split[0] + split[1] + split[2]],
          self._labels[split[0] + split[1]:split[0] + split[1] + split[2]],
          shuffle_behavior=False,
          incomplete_batches=True,
          window_size=window_size,
          one_hot=one_hot,
          label_mode=label_mode,
          label_size=label_size)
    else:
      self.test = None

  def _load_labels(self, folder_path):
    labels = []
    for img_index, label_path in enumerate(sorted(os.listdir(folder_path))):
      if label_path.endswith('.txt'):
        labels.append(
            self._load_txt_label(
                os.path.join(folder_path, label_path), img_index))

    return labels

  def _load_txt_label(self, label_path, img_index):
    label = np.zeros(self._images[img_index].shape, np.float32)
    with open(label_path, 'r') as f:
      for line in f:
        row, col = [int(j) for j in line.split()]
        label[row - 1, col - 1] = 1

    return label


class _RecognitionDataset:
  def __init__(self, images_by_labels, labels, should_shuffle,
               balanced_batches, incomplete_batches):
    self.n_labels = len(labels)
    self._shuffle = should_shuffle
    self._balance = balanced_batches
    self._incomplete = incomplete_batches
    self.images_shape = images_by_labels[0][0].shape

    if self._balance:
      # images must be separated by labels
      self._images = images_by_labels
      self._labels = labels

      # count images
      self.n_images = 0
      for images in self._images:
        self.n_images += len(images)

      # shuffle for first epoch
      if self._shuffle:
        for i in range(self.n_labels):
          self._images[i] = np.random.permutation(self._images[i])

      # initialize inner class pointers
      self._epochs_completed = np.zeros_like(labels)
      self._index_in_epoch = np.zeros_like(labels)
    else:
      # images can be flattened
      self._images = np.reshape(images_by_labels, (-1, self.images_shape[0],
                                                   self.images_shape[1]))
      self._labels = np.repeat(labels, len(self._images) // len(labels))
      self.n_images = len(self._images)

      # shuffle for first epoch
      if self._shuffle:
        perm = np.random.permutation(self.n_images)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # initialize dataset pointers
      self._epochs_completed = 0
      self._index_in_epoch = 0

  def next_batch(self, batch_size):
    if self._balance:
      # determine class batch sizes for almost equal split over classes
      class_batch_size = batch_size // self.n_labels
      remainder_index = batch_size % self.n_labels

      # randomly select over-represented classes
      perm = np.random.permutation(self.n_labels)

      # retrieve batch
      batch_images = []
      batch_labels = []
      for i in range(self.n_labels):
        # fix for almost equal split over classes
        complement = 0
        if i < remainder_index:
          complement += 1

        # check if finished sampling batch
        if class_batch_size + complement == 0:
          break
        else:
          # retrieve batch portion relative to class 'perm[i]'
          class_images, class_labels = self._next_class_batch(
              perm[i], class_batch_size + complement)

          # update batch
          batch_images.extend(class_images)
          batch_labels.extend(class_labels)

      return batch_images, batch_labels
    else:
      if self._index_in_epoch + batch_size >= self.n_images:
        # finished epoch
        self._epochs_completed += 1

        # get remainder of observations in this epoch
        start = self._index_in_epoch
        rest_num_images = self.n_images - start
        images_rest_part = self._images[start:]
        labels_rest_part = self._labels[start:]

        # shuffle the data
        if self._shuffle:
          perm = np.random.permutation(self.n_images)
          self._images = self._images[perm]
          self._labels = self._labels[perm]

        # return incomplete batch
        if self._incomplete:
          self._index_in_epoch = 0
          return images_rest_part, labels_rest_part

        # start next epoch
        self._index_in_epoch = batch_size - rest_num_images

        # retrieve observations in new epoch
        images_new_part = self._images[0:self._index_in_epoch]
        labels_new_part = self._labels[0:self._index_in_epoch]

        return np.concatenate(
            [images_rest_part, images_new_part], axis=0), np.concatenate(
                [labels_rest_part, labels_new_part], axis=0)
      else:
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        return self._images[start:self._index_in_epoch], self._labels[
            start:self._index_in_epoch]

  def _next_class_batch(self, index, batch_size):
    if self._index_in_epoch[index] + batch_size >= len(self._images[index]):
      # finished epoch
      self._epochs_completed[index] += 1

      # get remainder of images in this epoch
      start = self._index_in_epoch[index]
      rest_num_images = len(self._images[index]) - start
      images_rest_part = self._images[index][start:]

      # shuffle the data
      if self._shuffle:
        self._images[index] = np.random.permutation(self._images[index])

      # start next epoch
      self._index_in_epoch[index] = batch_size - rest_num_images

      # retrieve images in new epoch
      images_new_part = self._images[index][0:self._index_in_epoch[index]]

      # image batch
      images = np.concatenate([images_rest_part, images_new_part], axis=0)
    else:
      start = self._index_in_epoch[index]
      self._index_in_epoch[index] += batch_size
      images = self._images[index][start:self._index_in_epoch[index]]

    # produce labels
    labels = np.repeat(self._labels[index], len(images))

    return images, labels


class RecognitionDataset:
  def __init__(self,
               images_folder_path,
               val_split=True,
               should_shuffle=True,
               balanced_batches=True):
    images, labels = self._load_images_with_labels(images_folder_path)
    images_by_labels, labels = self._group_images_by_labels(images, labels)

    # create separate validation set
    if val_split:
      # randomly pick subjects comprising 20% of
      # whole dataset for validation set
      val_images_by_labels = []
      val_labels = []
      val_size = 0
      perm = np.random.permutation(len(images_by_labels))
      i = 0
      while val_size < 0.2 * len(images):
        val_size += len(images_by_labels[perm[i]])
        val_images_by_labels.append(images_by_labels[perm[i]])
        val_labels.append(labels[perm[i]])
        i += 1
      self.val = _RecognitionDataset(
          val_images_by_labels,
          val_labels,
          should_shuffle=should_shuffle,
          balanced_batches=balanced_batches,
          incomplete_batches=True)

      # remainder of images for training set
      train_images_by_labels = []
      train_labels = []
      while i < len(perm):
        train_images_by_labels.append(images_by_labels[perm[i]])
        train_labels.append(labels[perm[i]])
        i += 1
      self.train = _RecognitionDataset(train_images_by_labels, train_labels,
                                       should_shuffle, balanced_batches, False)
    else:
      self.train = _RecognitionDataset(images_by_labels, labels,
                                       should_shuffle, balanced_batches, False)

  def _load_images_with_labels(self, folder_path):
    images = []
    labels = []
    for image_path in sorted(os.listdir(folder_path)):
      if image_path.endswith(('.jpg', '.png', '.bmp')):
        images.append(util.load_image(os.path.join(folder_path, image_path)))
        labels.append(self._retrieve_label_from_image_path(image_path))

    return images, labels

  def _retrieve_label_from_image_path(self, image_path):
    return int(image_path.split('_')[0])

  def _group_images_by_labels(self, images, labels):
    # convert to np array
    images = np.array(images)
    labels = np.array(labels)

    grouped_images = []
    all_labels = np.unique(labels)
    for label in all_labels:
      indices = np.where(labels == label)
      grouped_images.append(images[indices])
    return grouped_images, all_labels
