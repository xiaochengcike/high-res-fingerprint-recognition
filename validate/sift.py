import argparse
import itertools
import numpy as np

import utils
import polyu

FLAGS = None


def main():
  print('Loading description dataset...')
  dataset = polyu.description.Dataset(
      FLAGS.dataset_path, should_shuffle=False).val
  print('Done.')

  print('Unpacking dataset and extracting sift descriptors...')
  descs = []
  labels = []
  prev_epoch = dataset.epochs
  while prev_epoch == dataset.epochs:
    (img, *_), (label, *_) = dataset.next_batch(1)
    pt = [np.array(img.shape) // 2]
    descs.append(utils.sift_descriptors(img, pt, normalize=False))
    labels.append(label)
  print('Done.')

  if FLAGS.mode == 'rank-n':
    # perform 'repeats' random splits
    ranks = []
    descs = np.squeeze(descs)
    labels = np.array(labels)
    for i in range(FLAGS.repeats):
      print('Split {}:'.format(i + 1))

      # compute rank-n for current split
      split_ranks = utils.rank_n(descs, labels, FLAGS.sample_size)
      rank_1 = split_ranks[0]
      print('Rank-1 = {}'.format(rank_1))

      # add only rank-1
      ranks.append(rank_1)

    # compute mean rank-1
    print('Mean Rank-1 = {}'.format(np.mean(ranks)))
  elif FLAGS.mode == 'eer':
    # perform 'repeats' samples of size 'sample_size'
    descs = np.squeeze(descs)
    labels = np.array(labels)
    pos = []
    neg = []
    for _ in range(FLAGS.repeats):
      # sample randomly
      sample = np.random.choice(len(descs), FLAGS.sample_size, replace=False)
      sample_descs = descs[sample]
      sample_labels = labels[sample]
      sample_examples = zip(sample_descs, sample_labels)

      # compare pair-wise
      for (desc1, label1), (desc2, label2) in itertools.combinations(
          sample_examples, 2):
        dist = -np.sum((desc1 - desc2)**2)
        if label1 == label2:
          pos.append(dist)
        else:
          neg.append(dist)

    # compute equal error rate
    print('EER = {}'.format(utils.eer(pos, neg)))
  else:
    raise ValueError(
        'Validation mode "{}" not available. Possible are "eer" and "rank-n".')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path',
      required=True,
      type=str,
      help='Path to description dataset.')
  parser.add_argument(
      '--sample_size',
      default=425,
      type=int,
      help='Number of instances to mix in retrieval task.')
  parser.add_argument(
      '--repeats',
      default=5,
      type=int,
      help='Number of random splits to average validation from.')
  parser.add_argument(
      '--mode',
      default='rank-n',
      type=str,
      help='Mode of validation. Possible are "eer" and "rank-n".')

  FLAGS = parser.parse_args()

  main()
