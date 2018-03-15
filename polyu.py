from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os
import numpy as np
import scipy.misc


class Dataset:
  def __init__(self, images, labels, window_size, shuffle_behavior, one_hot,
               incomplete_batches, label_mode, label_size):
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
        labels[index, 0] = self._window_labels[k, i + size // 2, j + size // 2]
        labels[index, 1] = 1 - labels[index, 0]
      else:
        labels[index] = self._window_labels[k, i + size // 2, j + size // 2]

    return windows, labels


class PolyUDataset:
  def __init__(self,
               images_folder_path,
               labels_folder_path,
               split,
               window_size=None,
               label_mode='hard_l2',
               label_size=3,
               should_shuffle=True,
               one_hot=False):
    self._images = self._load_images(images_folder_path)
    self._labels = self._load_labels(labels_folder_path)

    # splits loaded according to given 'split'
    self.train = Dataset(
        self._images[:split[0]],
        self._labels[:split[0]],
        window_size,
        shuffle_behavior=should_shuffle,
        one_hot=one_hot,
        incomplete_batches=False,
        label_mode=label_mode,
        label_size=label_size)
    self.val = Dataset(
        self._images[split[0]:split[0] + split[1]],
        self._labels[split[0]:split[0] + split[1]],
        window_size,
        shuffle_behavior=False,
        one_hot=one_hot,
        incomplete_batches=True,
        label_mode=label_mode,
        label_size=label_size)
    self.test = Dataset(
        self._images[split[0] + split[1]:split[0] + split[1] + split[2]],
        self._labels[split[0] + split[1]:split[0] + split[1] + split[2]],
        window_size,
        shuffle_behavior=False,
        one_hot=one_hot,
        incomplete_batches=True,
        label_mode=label_mode,
        label_size=label_size)

  def _load_images(self, folder_path):
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
        images.append(self._load_image(os.path.join(folder_path, image_path)))

    return images

  def _load_labels(self, folder_path):
    '''
    Loads corresponding pore labels in '.txt' files for the loaded images.
    The label file's path must be sorted, when considering all paths in the given folder, to the same position of its image correspondent.

    Args:
      folder_path: Path to folder containing the labels.

    Returns:
      labels: List of image labels as np float32 arrays, with 1s indicating pores.
    '''
    labels = []
    for img_index, label_path in enumerate(sorted(os.listdir(folder_path))):
      if label_path.endswith('.txt'):
        labels.append(
            self._load_txt_label(
                os.path.join(folder_path, label_path), img_index))

    return labels

  def _load_txt_label(self, label_path, img_index):
    '''
    Load a label '.txt' file. It must have a single pore per line, with format '[row] [col]' and indices starting from 1.

    Args:
      label_path: Full path to the label '.txt' file.
      img_index: Corresponding image index in 'self._images'.

    Returns:
      label: Label np float32 array containing 1s for pore centers.
    '''
    label = np.zeros(self._images[img_index].shape, np.float32)
    with open(label_path, 'r') as f:
      for line in f:
        row, col = [int(j) for j in line.split()]
        label[row - 1, col - 1] = 1

    return label

  def _load_image(self, image_path):
    '''
    Loads the image in 'image_path' as a single channel np float32 array in range [0, 1].

    Args:
      image_path: Path to the image being loaded.

    Returns:
      The loaded image as a single channel np float32 array in range [0, 1].
    '''
    return np.asarray(scipy.misc.imread(image_path, mode='F'),
                      np.float32) / 255.0
