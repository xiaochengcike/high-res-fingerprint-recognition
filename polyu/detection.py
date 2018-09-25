import os
import numpy as np

import utils


class _Dataset:
  def __init__(self,
               images,
               labels,
               shuffle_behavior,
               incomplete_batches,
               patch_size=None,
               one_hot=False,
               label_mode=None,
               label_size=None):
    self._images = np.array(images, dtype=np.float32)
    self._labels = np.array(labels, dtype=np.float32)
    self.patch_size = patch_size
    self._shuffle_behavior = shuffle_behavior
    self._one_hot = one_hot
    self._incomplete_batches = incomplete_batches

    # patch batch pointers
    self._index_in_epoch = 0
    self._epochs_completed = 0

    # image batch pointers
    self._image_index_in_epoch = 0
    self._image_epochs_completed = 0

    self.num_images = len(self._images)
    self._image_rows = self._images[0].shape[0]
    self._image_cols = self._images[0].shape[1]
    if patch_size is not None:
      self.num_samples = self.num_images * (self._image_rows - patch_size) * (
          self._image_cols - patch_size)

      # create labels to be sampled by patches
      if label_mode == 'hard_bb':
        self._patch_labels = np.zeros_like(self._labels, dtype=np.float32)

        # draw 'label_size' bounding box around pores
        for k, i, j in np.ndindex(self._patch_labels.shape):
          i_ = max(i - label_size, 0)
          j_ = max(j - label_size, 0)
          self._patch_labels[k, i, j] = np.max(
              self._labels[k, i_:i + 1 + label_size, j_:j + 1 + label_size])

      elif label_mode == 'hard_l1' or label_mode == 'hard_l2':
        self._patch_labels = np.zeros_like(self._labels, dtype=np.float32)

        # define norm to use
        norm_l = int(label_mode[-1])

        # enqueue pores with origins
        queue = []
        for pore in np.argwhere(self._labels >= 0.5):
          self._patch_labels[pore[0], pore[1], pore[2]] = self._labels[pore[
              0], pore[1], pore[2]]
          queue.append((pore, pore))

        # bfs to draw l'norm_l' ball around pores
        while queue:
          # pop front
          coords, anchor = queue[0]
          queue = queue[1:]

          # propagate pore anchor label
          k, i, j = anchor
          val = self._patch_labels[k, i, j]

          # enqueue valid neighbors
          for d in [(0, 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1)]:
            ngh = coords + d
            _, i, j = ngh
            if 0 <= i < self._patch_labels[k].shape[0] and \
                0 <= j < self._patch_labels[k].shape[1] and \
                self._patch_labels[k, i, j] == 0 and \
                np.linalg.norm(ngh - anchor, norm_l) <= label_size:
              self._patch_labels[k, i, j] = val
              queue.append((ngh, anchor))

  def next_batch(self, batch_size, shuffle=None, incomplete=None):
    '''
    Sample next batch, of size 'batch_size', of image patches.

    Args:
      batch_size: Size of batch to be sampled.
      shuffle: Overrides dataset split shuffle behavior, ie if data should be shuffled.
      incomplete: Overrides dataset incomplete batch behavior, ie if when completing an epoch, a batch should be provided with less samples than predicted so to not get samples from different epochs.

    Returns:
      The sampled patch batch and corresponding labels as np arrays.
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
      images_rest_part, labels_rest_part = self._to_patches(self._perm[start:])

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
      images_new_part, labels_new_part = self._to_patches(
          self._perm[start:end])

      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._to_patches(self._perm[start:end])

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

  def _to_patches(self, indices):
    '''
    Retrieves image patches, on demand, corresponding to given indices.
    A patch of size WINDOW_SIZE is centered in the i-th row of the j-th column of the k-th image (of dimensions ROWS x COLS) according to the given index I:
      k = I / ((ROWS - WINDOW_SIZE) * (COLS - WINDOW_SIZE))
      i = I / (COLS - WINDOW_SIZE) - k * (ROWS - WINDOW_SIZE)
      j = I mod (COLS - SIZE)

      Args:
        indices: Indices for which patches will be produced.

      Returns:
        patches: Windows corresponding to the given indices.
        labels: Labels of the returned patches.

    '''
    # patches (samples) and labels
    size = self.patch_size
    patches = np.empty([indices.shape[0], size, size], np.float32)
    if self._one_hot:
      labels = np.empty([indices.shape[0], 2], np.float32)
    else:
      labels = np.empty([indices.shape[0]], np.float32)

    for index in range(indices.shape[0]):
      image_index = indices[index]

      # retrieve image number, row and column from index
      k = image_index // (
          (self._image_rows - size) * (self._image_cols - size))
      i = image_index // (self._image_cols - size) - k * (
          self._image_rows - size)
      j = image_index % (self._image_cols - size)

      # generate patch
      patches[index] = self._images[k, i:i + size, j:j + size]

      # generate corresponding label
      center = size // 2
      if self._one_hot:
        labels[index, 0] = self._patch_labels[k, i + center, j + center]
        labels[index, 1] = 1 - labels[index, 0]
      else:
        labels[index] = self._patch_labels[k, i + center, j + center]

    return patches, labels


class Dataset:
  '''
  PolyU-HRF detection dataset handler. Contains a _Dataset for
  training and, depending on how it splits the entire dataset,
  another one for validation and another for testing. It
  converts ground truth coordinates into region labels according
  to label_mode and label_size.
  '''

  def __init__(self,
               images_folder_path,
               labels_folder_path,
               split,
               patch_size=None,
               label_mode='hard_bb',
               label_size=3,
               should_shuffle=True,
               one_hot=False):
    '''
    Args:
      images_folder_path: path from which to load images.
      labels_folder_path: path from which to label txt files.
        Must have name correspondence with images in
        images_folder_path.
      split: tuple determining how to split the detection dataset.
        split[0] gives the number of training images, split[1]
        the number of validation images, and split[2] the number
        of test images. Images are split sequentially into these
        sets, i.e. the training set takes the first split[0] images,
        the validation set takes the next split[1] images etc.
      patch_size: if not None, allows the sampling of patches,
        instead of images, from the dataset. Patches are then
        of size patch_size by patch_size.
      label_mode: mode of converting detection coordinates to
        labels. Can be either 'hard_bb', 'hard_l2' or 'hard_l1'.
        'hard_bb' draws a bounding box of size 'label_size' around
        each detection. 'hard_lX' draws an LX ball of radius
        'label_size' around each detection.
      label_size: see above, in label_mode description, for
        meaning.
      should_shuffle: whether the training set should be shuffled
        between batches.
      one_hot: whether the labels should be provided as one hot
        vectors or with integer values.
    '''
    self._images = utils.load_images(images_folder_path)
    self._labels = self._load_labels(labels_folder_path)

    # splits loaded according to given 'split'
    if split[0] > 0:
      self.train = _Dataset(
          self._images[:split[0]],
          self._labels[:split[0]],
          shuffle_behavior=should_shuffle,
          incomplete_batches=False,
          patch_size=patch_size,
          one_hot=one_hot,
          label_mode=label_mode,
          label_size=label_size)
    else:
      self.train = None

    if split[1] > 0:
      self.val = _Dataset(
          self._images[split[0]:split[0] + split[1]],
          self._labels[split[0]:split[0] + split[1]],
          shuffle_behavior=False,
          incomplete_batches=True,
          patch_size=patch_size,
          one_hot=one_hot,
          label_mode=label_mode,
          label_size=label_size)
    else:
      self.val = None

    if split[2] > 0:
      self.test = _Dataset(
          self._images[split[0] + split[1]:split[0] + split[1] + split[2]],
          self._labels[split[0] + split[1]:split[0] + split[1] + split[2]],
          shuffle_behavior=False,
          incomplete_batches=True,
          patch_size=patch_size,
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
