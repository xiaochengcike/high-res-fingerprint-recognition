from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse
import numpy as np

import pore_window_detector
import polyu
import util


def image_validation(sess, pred_op, batch_size, windows_pl, dataset):
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
    pred = np.pad(pred, ((half_window_size, half_window_size),
                         (half_window_size, half_window_size)), 'constant')

    # add image prediction to predictions
    preds.append(pred)

    # turn pore label image into list of pore coordinates
    pores.append(np.array(np.where(label > 0)).T)

  # validate over thresholds
  thrs = np.arange(0, 1.05, 0.05)
  nms_inter_thrs = np.arange(0.3, 0.9, 0.2)
  dist_thrs = np.arange(2, 18, 2)

  best_f_score = 0
  best_tdr = None
  best_fdr = None
  best_prob_thr = None
  best_nms_inter_thr = None
  best_nms_dist_thr = None
  best_ngh_dist_thr = None

  for prob_thr in thrs:
    # filter detections by probability threshold
    selected = []
    probs = []
    for i in range(dataset.num_images):
      img_preds = preds[i]
      pick = img_preds > prob_thr
      selected.append(np.array(np.where(pick)).T)
      probs.append(img_preds[pick])

    for nms_inter_thr in nms_inter_thrs:
      for nms_dist_thr in dist_thrs:
        # filter detections with nms
        dets = []
        for i in range(dataset.num_images):
          dets.append(
              util.nms(selected[i], probs[i], nms_dist_thr, nms_inter_thr))

        # find correspondences between detections and pores
        for ngh_dist_thr in dist_thrs:
          true_dets = 0
          false_dets = 0
          total = 0

          for i in range(dataset.num_images):
            # update total number of pores
            total += len(pores[i])

            # find pore-detection and detection-pore correspondences
            pore_corrs = np.full(len(pores[i]), -1, dtype=np.int32)
            pore_dcorrs = np.full(len(pores[i]), np.inf)
            det_corrs = np.full(len(dets[i]), -1, dtype=np.int32)
            det_dcorrs = np.full(len(dets[i]), np.inf)
            for pore_ind, pore in enumerate(pores[i]):
              for det_ind, det in enumerate(dets[i]):
                # pore-detection distance
                dist = np.linalg.norm(pore - det)

                # update pore-detection correspondence
                if dist < ngh_dist_thr and dist < pore_dcorrs[pore_ind]:
                  pore_dcorrs[pore_ind] = dist
                  pore_corrs[pore_ind] = det_ind

                # update detection-pore correspondence
                if dist < ngh_dist_thr and dist < det_dcorrs[det_ind]:
                  det_dcorrs[det_ind] = dist
                  det_corrs[det_ind] = pore_ind

            # coincidences in pore-detection and detection-pore correspondences are true detections
            for det_ind, det_corr in enumerate(det_corrs):
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
            best_nms_dist_thr = nms_dist_thr
            best_ngh_dist_thr = ngh_dist_thr

  return best_f_score, best_tdr, best_fdr, best_prob_thr, best_nms_inter_thr, best_nms_dist_thr, best_ngh_dist_thr


def window_validation(sess, preds, batch_size, windows_pl, labels_pl, dataset):
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


def train(dataset, learning_rate, batch_size, max_steps, tolerance, log_dir,
          train_dir, plot_dir):
  with tf.Graph().as_default():
    # gets placeholders for windows and labels
    windows_pl, labels_pl = util.window_placeholder_inputs()

    # builds inference graph
    pore_det = pore_window_detector.PoreDetector(windows_pl,
                                                 dataset.train.window_size)

    # build train related ops
    pore_det.build_loss(labels_pl)
    pore_det.build_train(learning_rate)

    # add summary to plot loss, f score, tdr and fdr
    f_score_pl = tf.placeholder(tf.float32, shape=())
    tdr_pl = tf.placeholder(tf.float32, shape=())
    fdr_pl = tf.placeholder(tf.float32, shape=())
    f_score_summary_op = tf.summary.scalar('f_score', f_score_pl)
    tdr_summary_op = tf.summary.scalar('tdr', tdr_pl)
    fdr_summary_op = tf.summary.scalar('fdr', fdr_pl)
    loss_summary_op = tf.summary.scalar('loss', pore_det.loss)

    # add summaries to plot image f score, tdr and fdr
    image_f_score_pl = tf.placeholder(tf.float32, shape=())
    image_tdr_pl = tf.placeholder(tf.float32, shape=())
    image_fdr_pl = tf.placeholder(tf.float32, shape=())
    image_f_score_summary_op = tf.summary.scalar('image_f_score',
                                                 image_f_score_pl)
    image_tdr_summary_op = tf.summary.scalar('image_image_tdr', image_tdr_pl)
    image_fdr_summary_op = tf.summary.scalar('image_image_fdr', image_fdr_pl)

    # resources to tensorboard plots
    plot_buf_pl = tf.placeholder(tf.string)
    plot_png = tf.image.decode_png(plot_buf_pl)
    expanded_plot_png = tf.expand_dims(plot_png, 0)
    plot_summary_op = tf.summary.image('plot', expanded_plot_png)

    # add variable initialization to graph
    init = tf.global_variables_initializer()

    # early stopping vars
    best_f_score = 0
    faults = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

      sess.run(init)

      for step in range(1, max_steps + 1):
        feed_dict = util.fill_window_feed_dict(dataset.train, windows_pl,
                                               labels_pl, batch_size)

        _, loss_value = sess.run(
            [pore_det.train, pore_det.loss], feed_dict=feed_dict)

        # write loss summary every 100 steps
        if step % 100 == 0:
          print('Step {}: loss = {}'.format(step, loss_value))
          summary_str = sess.run(loss_summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # evaluate the model periodically
        if step % 1000 == 0:
          tdrs, fdrs, f_score, fdr, tdr, thr = window_validation(
              sess, pore_det.preds, batch_size, windows_pl, labels_pl,
              dataset.val)
          print(
              'Evaluation:',
              '\tTDR = {}'.format(tdr),
              '\tFDR = {}'.format(fdr),
              '\tF score = {}'.format(f_score),
              sep='\n')

          image_f_score, image_tdr, image_fdr, _ = image_validation(
              sess, pore_det.preds, batch_size, windows_pl, dataset.val)
          print(
              'Whole image evaluation:',
              '\tTDR = {}'.format(image_tdr),
              '\tFDR = {}'.format(image_fdr),
              '\tF score = {}'.format(image_f_score),
              sep='\n')

          # early stopping
          if f_score > best_f_score:
            best_f_score = f_score
            saver.save(
                sess,
                os.path.join(train_dir, 'model-{}.ckpt'.format(thr)),
                global_step=step)
            faults = 0
          else:
            faults += 1
            if faults >= tolerance:
              print('Training stopped early')
              break

          # write f score, tdr and fdr to summary
          score_summaries = sess.run(
              [f_score_summary_op, tdr_summary_op, fdr_summary_op],
              feed_dict={f_score_pl: f_score,
                         tdr_pl: tdr,
                         fdr_pl: fdr})
          for score_summary in score_summaries:
            summary_writer.add_summary(score_summary, global_step=step)

          # write image f score, tdr and fdr to summary
          score_summaries = sess.run(
              [
                  image_f_score_summary_op, image_tdr_summary_op,
                  image_fdr_summary_op
              ],
              feed_dict={
                  image_f_score_pl: image_f_score,
                  image_tdr_pl: image_tdr,
                  image_fdr_pl: image_fdr
              })
          for score_summary in score_summaries:
            summary_writer.add_summary(score_summary, global_step=step)

          # plot recall vs precision
          buf = util.plot(tdrs, fdrs,
                          os.path.join(plot_dir, '{}.png'.format(step)))

          # write plot to summary
          plot_summary = sess.run(
              plot_summary_op, feed_dict={plot_buf_pl: buf.getvalue()})
          summary_writer.add_summary(plot_summary, global_step=step)


def main(log_dir_path, polyu_path, window_size, max_steps, learning_rate,
         batch_size, tolerance):
  # create folders to save train resources
  log_dir, train_dir, plot_dir = util.create_dirs(log_dir_path, batch_size,
                                                  learning_rate)

  # load polyu dataset
  polyu_path = os.path.join(polyu_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.PolyUDataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      window_size=window_size)

  # train
  train(dataset, learning_rate, batch_size, max_steps, tolerance, log_dir,
        train_dir, plot_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir', required=True, type=str, help='Path to PolyU-HRF dataset')
  parser.add_argument(
      '--learning_rate', type=float, default=1e-4, help='Learning rate.')
  parser.add_argument(
      '--log_dir', type=str, default='log', help='Logging directory.')
  parser.add_argument(
      '--tolerance', type=int, default=5, help='Early stopping tolerance.')
  parser.add_argument(
      '--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument(
      '--steps', type=int, default=100000, help='Maximum training steps.')
  parser.add_argument(
      '--window_size', type=int, default=17, help='Pore window size.')
  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS.log_dir, FLAGS.polyu_dir, FLAGS.window_size, FLAGS.steps,
       FLAGS.learning_rate, FLAGS.batch_size, FLAGS.tolerance)
