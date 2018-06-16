from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import argparse

import polyu
from utils import extract_sift_descriptors as sift
import validate

FLAGS = None


def load_description_dataset(dataset_path):
  dataset = polyu.description.Dataset(
      dataset_path,
      val_split=False,
      should_shuffle=False,
      balanced_batches=False)

  return dataset.train


def main():
  print('Loading description dataset...')
  dataset = load_description_dataset(FLAGS.desc_dataset_path)
  print('Done.')

  print('Unpacking dataset...')
  imgs = []
  labels = []
  prev_epoch = dataset.epochs
  while prev_epoch == dataset.epochs:
    (img, *_), (label, *_) = dataset.next_batch(1)
    imgs.append(img)
    labels.append(label)
  print('Done.')

  print('Extracting sift descriptors...')
  pts = [[np.array(img.shape) // 2] for img in imgs]
  descs = [sift(img, pt) for img, pt in zip(imgs, pts)]
  print('Done.')

  # perform 'repeats' random splits
  ranks = []
  descs = np.array(descs)
  descs = np.squeeze(descs)
  labels = np.array(labels)
  for i in range(FLAGS.repeats):
    print('Split {}:'.format(i + 1))

    # compute rank-n for current split
    split_ranks = validate.rank_n(descs, labels, FLAGS.sample_size)
    rank_1 = split_ranks[0]
    print('Rank-1 = {}'.format(rank_1))

    # add only rank-1
    ranks.append(rank_1)

  # compute mean rank-1
  print('Mean Rank-1 = {}'.format(np.mean(ranks)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--desc_dataset_path',
      required=True,
      type=str,
      help='Path to description dataset.')
  parser.add_argument(
      '--sample_size',
      default=100,
      type=int,
      help='Number of instances to mix in retrieval task.')
  parser.add_argument(
      '--repeats',
      default=5,
      type=int,
      help='Number of random splits to average validation from.')

  FLAGS = parser.parse_args()

  main()
