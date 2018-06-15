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
      self.epochs = 0
      self._index = 0

  def next_batch(self, batch_size):
    # adjust for balanced batch sampling
    if not self._balance:
      end = self.n_images
    else:
      full_batch_size = batch_size
      batch_size = batch_size // self._imgs_per_label
      end = self.n_labels

    if self._index + batch_size >= self.n_images:
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

      # return incomplete batch
      if self._incomplete:
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
          balanced_batches=False,
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
