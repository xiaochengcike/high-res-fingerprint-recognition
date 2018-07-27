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
      np.arange(0, 4 + threshold_resolution, threshold_resolution),
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
  ranks = utils.rank_n(descs, labels, sample_size)

  return ranks[0]
