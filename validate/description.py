from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np

import utils


def statistics_by_thresholds(patches_pl, labels_pl, thresholds_pl, dataset,
                             thresholds, statistics_op, session, batch_size):
  # initialize statistics
  true_pos = np.zeros_like(thresholds, np.int32)
  true_neg = np.zeros_like(thresholds, np.int32)
  false_pos = np.zeros_like(thresholds, np.int32)
  false_neg = np.zeros_like(thresholds, np.int32)

  # validate in entire dataset, as specified by user
  prev_epoch_n = dataset.epochs
  while prev_epoch_n == dataset.epochs:
    # sample mini-batch
    feed_dict = utils.fill_feed_dict(dataset, patches_pl, labels_pl,
                                     batch_size)
    feed_dict[thresholds_pl] = thresholds

    # evaluate on mini-batch
    batch_true_pos, batch_true_neg, batch_false_pos, batch_false_neg = session.run(
        statistics_op, feed_dict=feed_dict)

    # update running statistics
    true_pos += batch_true_pos
    true_neg += batch_true_neg
    false_pos += batch_false_pos
    false_neg += batch_false_neg

  return true_pos, true_neg, false_pos, false_neg


def recognition_eer(patches_pl, labels_pl, thresholds_pl, dataset,
                    threshold_resolution, statistics_op, session, batch_size):
  true_pos, true_neg, false_pos, false_neg = statistics_by_thresholds(
      patches_pl, labels_pl, thresholds_pl, dataset,
      np.arange(0, 2 + threshold_resolution, threshold_resolution),
      statistics_op, session, batch_size)

  # compute recall and specificity
  eps = 1e-12
  recall = true_pos / (true_pos + false_neg + eps)
  specificity = false_pos / (true_neg + false_pos + eps)

  # compute equal error rate
  for i in range(0, len(recall)):
    if recall[i] >= 1.0 - specificity[i]:
      return 1 - (recall[i] + 1.0 - specificity[i]) / 2

  return 0


def retrieval_rank(probe_instance, probe_label, instances, labels):
  # compute distance of 'probe_instance' to
  # every instance in 'instances'
  dists = np.sum((instances - probe_instance)**2, axis=1)

  # sort labels according to instances distances
  matches = np.argsort(dists)
  labels = labels[matches]

  # find index of last instance of label 'probe_label'
  last_ind = np.argwhere(labels == probe_label)[-1, 0]

  # compute retrieval rank
  labels_up_to_last_ind = np.unique(labels[:last_ind])
  rank = len(labels_up_to_last_ind)

  return rank


def rank_n(instances, labels, sample_size):
  # initialize ranks
  ranks = np.zeros_like(labels, dtype=np.int32)

  # sort examples by labels
  inds = np.argsort(labels)
  instances = instances[inds]
  labels = labels[inds]

  # compute rank following protocol in belongie et al.
  examples = list(zip(instances, labels))
  for i, (probe, probe_label) in enumerate(examples):
    for target, target_label in examples[i + 1:]:
      if probe_label != target_label:
        break
      else:
        # mix examples of other labels
        other_labels_inds = np.argwhere(labels != probe_label)
        other_labels_inds = np.squeeze(other_labels_inds)
        inds_to_pick = np.random.choice(
            other_labels_inds, sample_size - 1, replace=False)
        instances_to_mix = instances[inds_to_pick]
        labels_to_mix = labels[inds_to_pick]

        # make set for retrieval
        target = np.expand_dims(target, axis=0)
        instance_set = np.concatenate([instances_to_mix, target], axis=0)
        target_label = np.expand_dims(target_label, axis=0)
        label_set = np.concatenate([labels_to_mix, target_label], axis=0)

        # compute retrieval rank for probe
        rank = retrieval_rank(probe, probe_label, instance_set, label_set)

        # update ranks, indexed from 0
        ranks[rank] += 1

  # rank is cumulative
  ranks = np.cumsum(ranks)

  return ranks / ranks[-1]


def dataset_rank_n(patches_pl, sess, descs_op, dataset, batch_size,
                   sample_size):
  # extracting descriptors for entire dataset
  descs = []
  labels = []
  prev_epoch = dataset.epochs
  while prev_epoch == dataset.epochs:
    # sample next batch
    patches, batch_labels = dataset.next_batch(batch_size)
    feed_dict = {patches_pl: np.expand_dims(patches, axis=-1)}

    # describe batch
    batch_descs = sess.run(descs_op, feed_dict=feed_dict)

    # add to overall
    descs.extend(batch_descs)
    labels.extend(batch_labels)

  # convert to np array and remove extra dims
  descs = np.squeeze(descs)
  labels = np.squeeze(labels)

  # compute ranks
  ranks = rank_n(descs, labels, sample_size)

  return ranks[0]
