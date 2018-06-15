from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os
import numpy as np

import utils


class _Dataset:
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


class Dataset:
  def __init__(self,
               images_folder_path,
               val_split=True,
               should_shuffle=True,
               balanced_batches=True):
    images, labels = utils.load_images_with_labels(images_folder_path)
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
      self.val = _Dataset(
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
      self.train = _Dataset(train_images_by_labels, train_labels,
                            should_shuffle, balanced_batches, False)
    else:
      self.train = _Dataset(images_by_labels, labels, should_shuffle,
                            balanced_batches, False)

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
