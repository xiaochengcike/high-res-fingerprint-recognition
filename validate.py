from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import utils


def detection_by_patches(sess, preds, batch_size, patches_pl, labels_pl,
                         dataset):
  # initialize dataset statistics
  true_preds = []
  false_preds = []
  total = 0

  steps_per_epoch = (dataset.num_samples + batch_size - 1) // batch_size
  for _ in range(steps_per_epoch):
    feed_dict = utils.fill_feed_dict(dataset, patches_pl, labels_pl,
                                     batch_size)

    # evaluate batch
    batch_preds = sess.run(preds, feed_dict=feed_dict)
    batch_labels = feed_dict[labels_pl]
    batch_total = np.sum(batch_labels)

    # update dataset statistics
    total += batch_total
    if batch_total > 0:
      true_preds.extend(batch_preds[batch_labels == 1].flatten())
    if batch_total < batch_labels.shape[0]:
      false_preds.extend(batch_preds[batch_labels == 0].flatten())

  # sort for efficient computation of tdr/fdr over thresholds
  true_preds.sort()
  true_preds.reverse()
  false_preds.sort()
  false_preds.reverse()

  # compute tdr/fdr score over thresholds
  tdrs = []
  fdrs = []
  best_thr = 0
  best_f_score = 0
  best_fdr = None
  best_tdr = None

  true_pointer = 0
  false_pointer = 0

  eps = 1e-5
  thrs = np.arange(1.01, -0.01, -0.01)
  for thr in thrs:
    # compute true positives
    while true_pointer < len(true_preds) and true_preds[true_pointer] >= thr:
      true_pointer += 1

    # compute false positives
    while false_pointer < len(
        false_preds) and false_preds[false_pointer] >= thr:
      false_pointer += 1

    # compute tdr and fdr
    tdr = true_pointer / (total + eps)
    fdr = false_pointer / (true_pointer + false_pointer + eps)
    tdrs.append(tdr)
    fdrs.append(fdr)

    # compute and update f score
    f_score = 2 * (tdr * (1 - fdr)) / (tdr + 1 - fdr)
    if f_score > best_f_score:
      best_tdr = tdr
      best_fdr = fdr
      best_f_score = f_score
      best_thr = thr

  return np.array(tdrs, np.float32), np.array(
      fdrs, np.float32), best_f_score, best_fdr, best_tdr, best_thr


def detection_by_images(sess, pred_op, patches_pl, dataset):
  patch_size = dataset.patch_size
  half_patch_size = patch_size // 2
  preds = []
  pores = []
  print('Predicting pores...')
  for _ in range(dataset.num_images):
    # get next image and corresponding image label
    img, label = dataset.next_image_batch(1)
    img = img[0]
    label = label[0]

    # predict for each image
    pred = sess.run(
        pred_op,
        feed_dict={patches_pl: np.reshape(img, (-1, ) + img.shape + (1, ))})

    # put predictions in image format
    pred = np.array(pred).reshape(img.shape[0] - patch_size + 1,
                                  img.shape[1] - patch_size + 1)

    # add borders lost in convolution
    pred = np.pad(pred, ((half_patch_size, half_patch_size),
                         (half_patch_size, half_patch_size)), 'constant')

    # add image prediction to predictions
    preds.append(pred)

    # turn pore label image into list of pore coordinates
    pores.append(np.argwhere(label))
  print('Done.')

  # validate over thresholds
  inter_thrs = np.arange(0.7, 0, -0.1)
  prob_thrs = np.arange(0.9, 0, -0.1)
  best_f_score = 0
  best_tdr = None
  best_fdr = None
  best_inter_thr = None
  best_prob_thr = None

  # put inference in nms proper format
  for prob_thr in prob_thrs:
    coords = []
    probs = []
    for i in range(dataset.num_images):
      img_preds = preds[i]
      pick = img_preds > prob_thr
      coords.append(np.argwhere(pick))
      probs.append(img_preds[pick])

    for inter_thr in inter_thrs:
      # filter detections with nms
      dets = []
      for i in range(dataset.num_images):
        det, _ = utils.nms(coords[i], probs[i], 7, inter_thr)
        dets.append(det)

      # find correspondences between detections and pores
      total_pores = 0
      total_dets = 0
      true_dets = 0
      for i in range(dataset.num_images):
        # update totals
        total_pores += len(pores[i])
        total_dets += len(dets[i])

        # coincidences in pore-detection and detection-pore correspondences are true detections
        pore_corrs, det_corrs = utils.matmul_corr_finding(pores[i], dets[i])
        for pore_ind, pore_corr in enumerate(pore_corrs):
          if det_corrs[pore_corr] == pore_ind:
            true_dets += 1

      # compute tdr, fdr and f score
      eps = 1e-5
      tdr = true_dets / (total_pores + eps)
      fdr = (total_dets - true_dets) / (total_dets + eps)
      f_score = 2 * (tdr * (1 - fdr)) / (tdr + (1 - fdr))

      # update best parameters
      if f_score > best_f_score:
        best_f_score = f_score
        best_tdr = tdr
        best_fdr = fdr
        best_inter_thr = inter_thr
        best_prob_thr = prob_thr

  return best_f_score, best_tdr, best_fdr, best_inter_thr, best_prob_thr


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
  total = 0

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
        ranks[rank - 1] += 1

  return ranks / np.sum(ranks)
