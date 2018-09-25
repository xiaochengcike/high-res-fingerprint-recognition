import os
import numpy as np

import utils


class TrainingSet:
  def __init__(self, images_by_labels, labels, should_shuffle,
               balanced_batches, incomplete_batches):
    self.n_labels = len(labels)
    self._shuffle = should_shuffle
    self._balance = balanced_batches
    self._incomplete = incomplete_batches
    self.images_shape = images_by_labels[0][0].shape

    if self._balance:
      # images must be separated by labels
      self._images = np.array(images_by_labels)
      self._labels = np.array(labels)

      # count images
      self.n_images = 0
      for images in self._images:
        self.n_images += len(images)

      # images per label
      self._imgs_per_label = self.n_images // self.n_labels

      # shuffle for first epoch
      if self._shuffle:
        perm = np.random.permutation(self.n_labels)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # initialize data pointers
      self.epochs = 0
      self._index = 0
    else:
      # images can be flattened
      self._images = np.reshape(
          images_by_labels, (-1, self.images_shape[0], self.images_shape[1]))
      self._labels = np.repeat(labels, len(self._images) // len(labels))
      self.n_images = len(self._images)

      # shuffle for first epoch
      if self._shuffle:
        perm = np.random.permutation(self.n_images)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # initialize dataset pointers
      self.epochs = 0
      self._index = 0

  def next_batch(self, batch_size):
    # adjust for balanced batch sampling
    if not self._balance:
      end = self.n_images
    else:
      batch_size = batch_size // self._imgs_per_label
      full_batch_size = batch_size * self._imgs_per_label
      end = self.n_labels

    if self._index + batch_size >= end:
      # finished epoch
      self.epochs += 1

      # get remainder of examples in this epoch
      start = self._index
      images_rest_part = self._images[start:]
      labels_rest_part = self._labels[start:]
      rest_num_images = end - start

      # shuffle the data
      if self._shuffle:
        perm = np.random.permutation(end)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # handle incomplete batches
      if self._incomplete:
        # return incomplete batch
        self._index = 0
        batch_images = images_rest_part
        batch_labels = labels_rest_part
      else:
        # start next epoch
        self._index = batch_size - rest_num_images

        # retrieve observations in new epoch
        images_new_part = self._images[0:self._index]
        labels_new_part = self._labels[0:self._index]

        batch_images = np.concatenate(
            [images_rest_part, images_new_part], axis=0)
        batch_labels = np.concatenate(
            [labels_rest_part, labels_new_part], axis=0)
    else:
      start = self._index
      self._index += batch_size
      batch_images = self._images[start:self._index]
      batch_labels = self._labels[start:self._index]

    # reshape balanced batches
    if self._balance:
      batch_images = np.reshape(batch_images,
                                (full_batch_size, ) + batch_images.shape[2:])
      batch_labels = np.repeat(batch_labels,
                               len(batch_images) // len(batch_labels))

    return batch_images, batch_labels


class ValidationSet:
  def __init__(self, images, detections, labels):
    self._images = np.array(images)
    self._detections = np.array(detections)
    self._labels = np.array(labels)

  def __getitem__(self, val):
    return self._images[val], self._detections[val], self._labels[val]


class Dataset:
  def __init__(self, path, should_shuffle=True, balanced_batches=True):
    # split paths
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    # load training set
    images, labels = utils.load_images_with_labels(train_path)
    images_by_labels, labels = self._group_images_by_labels(images, labels)
    self.train = TrainingSet(
        images_by_labels,
        labels,
        should_shuffle=should_shuffle,
        balanced_batches=balanced_batches,
        incomplete_batches=False)

    # load validation set, if any
    self.val = None
    if os.path.exists(val_path):
      images, detections, labels = self._load_validation_data(val_path)
      self.val = ValidationSet(images, detections, labels)

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

  def _load_validation_data(self, val_path):
    # load images with respective names
    images, names = utils.load_images_with_names(val_path)

    # convert 'names' to validation 'labels'
    # each 'name' in 'names' is 'subject-id_session-id_register-rd'
    labels = [tuple(map(int, name.split('_'))) for name in names]

    # load detections, aligned with images and labels
    paths = map(lambda name: os.path.join(val_path, name), names)
    detections = [utils.load_dets_txt(path + '.txt') for path in paths]

    return images, detections, labels
