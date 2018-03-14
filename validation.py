from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import util


def by_windows(sess, preds, batch_size, windows_pl, labels_pl, dataset):
  # initialize dataset statistics
  true_preds = []
  false_preds = []
  total = 0

  steps_per_epoch = (dataset.num_samples + batch_size - 1) // batch_size
  for _ in range(steps_per_epoch):
    feed_dict = util.fill_window_feed_dict(dataset, windows_pl, labels_pl,
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


def by_images(sess, pred_op, batch_size, windows_pl, dataset):
  window_size = dataset.window_size
  half_window_size = window_size // 2
  preds = []
  pores = []
  for _ in range(dataset.num_images):
    # get next image and corresponding image label
    img, label = dataset.next_image_batch(1)
    img = img[0]
    label = label[0]

    # convert 'img' to windows
    windows = np.array(util.to_windows(img, window_size))

    # predict for each window
    pred = []
    for i in range((len(windows) + batch_size - 1) // batch_size):
      # sample batch
      batch_windows = windows[batch_size * i:batch_size * (i + 1)].reshape(
          [-1, window_size, window_size, 1])

      # predict for batch
      batch_preds = sess.run(pred_op, feed_dict={windows_pl: batch_windows})

      # update image pred
      pred.extend(batch_preds)

    # put predictions in image format
    pred = np.array(pred).reshape(img.shape[0] - window_size + 1,
                                  img.shape[1] - window_size + 1)

    # add borders lost in convolution
    pred = np.pad(pred, ((half_window_size, half_window_size),
                         (half_window_size, half_window_size)), 'constant')

    # add image prediction to predictions
    preds.append(pred)

    # turn pore label image into list of pore coordinates
    pores.append(np.array(np.where(label > 0)).T)

  # validate over thresholds
  thrs = np.arange(0, 1.1, 0.1)
  nms_inter_thrs = np.arange(.7, -.1, -.2)
  dist_thrs = np.arange(2, 10)

  best_f_score = 0
  best_tdr = None
  best_fdr = None
  best_prob_thr = None
  best_nms_inter_thr = None
  best_ngh_dist_thr = None

  for prob_thr in thrs:
    # filter detections by probability threshold
    selected = []
    probs = []
    for i in range(dataset.num_images):
      img_preds = preds[i]
      pick = img_preds >= prob_thr
      selected.append(np.array(np.where(pick)).T)
      probs.append(img_preds[pick])

    for nms_inter_thr in nms_inter_thrs:
      # filter detections with nms
      dets = []
      for i in range(dataset.num_images):
        dets.append(util.nms(selected[i], probs[i], 7, nms_inter_thr))

      # find correspondences between detections and pores
      for ngh_dist_thr in dist_thrs:
        true_dets = 0
        false_dets = 0
        total = 0

        for i in range(dataset.num_images):
          # update total number of pores
          total += len(pores[i])

          # coincidences in pore-detection and detection-pore correspondences are true detections
          pore_corrs, det_corrs = util.project_and_find_correspondences(
              pores[i], dets[i], ngh_dist_thr, preds[i].shape)
          for det_ind, det_corr in enumerate(det_corrs):
            # safe to not check if 'det_corr == -1' because if
            # 'pore_corrs[-1] == det_ind', then
            # 'dist(pores[-1], det) < dist_thr' and 'det_corr != -1'
            if pore_corrs[det_corr] == det_ind:
              true_dets += 1
            else:
              false_dets += 1

        # compute tdr, fdr and f score
        eps = 1e-5
        tdr = true_dets / (total + eps)
        fdr = false_dets / (true_dets + false_dets + eps)
        f_score = 2 * (tdr * (1 - fdr)) / (tdr + (1 - fdr))

        # update best parameters
        if f_score > best_f_score:
          best_f_score = f_score
          best_tdr = tdr
          best_fdr = fdr
          best_prob_thr = prob_thr
          best_nms_inter_thr = nms_inter_thr
          best_ngh_dist_thr = ngh_dist_thr

  return best_f_score, best_tdr, best_fdr, best_prob_thr, best_nms_inter_thr, best_ngh_dist_thr