from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os
import numpy as np

import utils
import aligned_images


class _Dataset:
  def __init__(self, images_by_labels, pts_by_labels, patch_size,
               should_shuffle):
    self.n_labels = len(images_by_labels)
    self._shuffle = should_shuffle
    self.images_shape = images_by_labels[0][0].shape

    # create aligned images handler for each class
    self._classes = []
    for imgs, pts in zip(images_by_labels, pts_by_labels):
      self._classes.append(aligned_images.Handler(imgs, pts, patch_size))

    # shuffle for first epoch
    if self._shuffle:
      for i in range(self.n_labels):
        self._class_indices[i] = np.random.permutation(len(self._classes[i]))

    # initialize inner class pointers
    self._epochs_completed = np.zeros(self.n_labels)
    self._index_in_epoch = np.zeros(self.n_labels)

  def next_batch(self, batch_size):
    assert batch_size <= self.n_labels, 'Batch size must be at most number of labels'

    # randomly select labels in batch
    perm = np.random.permutation(self.n_labels)[:batch_size]

    # retrieve batch
    batch_patches = []
    batch_labels = []
    for i in perm:
      # retrieve examples of class 'i'
      class_patches, class_labels = self._next_class_batch(i)

      # update batch
      batch_patches.extend(class_patches)
      batch_labels.extend(class_labels)

    return batch_patches, batch_labels

  def _next_class_batch(self, label):
    # retrieve patches from class
    patches = self._classes[label][self._index_in_epoch[label]]

    # update class index
    if self._index_in_epoch[label] + 1 < len(self._class[label]):
      self._class_indices[label] += 1
    else:
      # finished epoch
      self._epochs_completed[label] += 1

      # shuffle the data
      if self._shuffle:
        self._class_indices[label] = np.random.permutation(
            len(self._classes[label]))

      # return class index to 0
      self._class_indices[label] = 0

    # produce labels
    labels = np.repeat(label, len(patches))

    return patches, labels


class Dataset:
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

  def _load_images_with_labels(self, folder_path):
    images = []
    labels = []
    for image_path in sorted(os.listdir(folder_path)):
      if image_path.endswith(('.jpg', '.png', '.bmp')):
        images.append(utils.load_image(os.path.join(folder_path, image_path)))
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
