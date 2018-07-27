import numpy as np
from itertools import combinations

import utils


def _dataset_descriptors(patches_pl, session, descs_op, dataset, batch_size):
  # extracting descriptors for entire dataset
  descs = []
  labels = []
  prev_epoch = dataset.epochs
  while prev_epoch == dataset.epochs:
    # sample next batch
    patches, batch_labels = dataset.next_batch(batch_size)
    feed_dict = {patches_pl: np.expand_dims(patches, axis=-1)}

    # describe batch
    batch_descs = session.run(descs_op, feed_dict=feed_dict)

    # add to overall
    descs.extend(batch_descs)
    labels.extend(batch_labels)

  # convert to np array and remove extra dims
  descs = np.squeeze(descs)
  labels = np.squeeze(labels)

  return descs, labels


def dataset_eer(patches_pl, session, descs_op, dataset, batch_size):
  # extracting descriptors for entire dataset
  descs, labels = _dataset_descriptors(patches_pl, session, descs_op, dataset,
                                       batch_size)

  # get pairwise comparisons
  examples = zip(descs, labels)
  pos = []
  neg = []
  for (desc1, label1), (desc2, label2) in combinations(examples, 2):
    dist = -np.sum((desc1 - desc2)**2)
    if label1 == label2:
      pos.append(dist)
    else:
      neg.append(dist)

  # compute eer
  eer = utils.eer(pos, neg)

  return eer


def dataset_rank_1(patches_pl, session, descs_op, dataset, batch_size,
                   sample_size):
  # extracting descriptors for entire dataset
  descs, labels = _dataset_descriptors(patches_pl, session, descs_op, dataset,
                                       batch_size)

  # compute ranks
  ranks = utils.rank_n(descs, labels, sample_size)

  return ranks[0]
