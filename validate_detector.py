from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse

import detector
import validate
import polyu
import utils


def main(model_dir, polyu_path, patch_size, batch_size):
  # load polyu dataset
  print('Loading PolyU-HRF dataset...')
  polyu_path = os.path.join(polyu_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.detection.dataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=patch_size)
  print('Loaded.')

  with tf.Graph().as_default():
    # gets placeholders for patches and labels
    patches_pl, labels_pl = utils.placeholder_inputs()

    # builds inference graph
    net = detector.Net(patches_pl, dataset.train.patch_size, training=False)

    with tf.Session() as sess:
      utils.restore_model(sess, model_dir)

      image_f_score, image_tdr, image_fdr, inter_thr, prob_thr = validate.detection_by_images(
          sess, net.preds, patches_pl, dataset.val)
      print(
          'Whole image evaluation:',
          '\tTDR = {}'.format(image_tdr),
          '\tFDR = {}'.format(image_fdr),
          '\tF score = {}'.format(image_f_score),
          '\tinter_thr = {}'.format(inter_thr),
          '\tprob_thr = {}'.format(prob_thr),
          sep='\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir', required=True, type=str, help='Path to PolyU-HRF dataset')
  parser.add_argument(
      '--model_dir', type=str, required=True, help='Logging directory.')
  parser.add_argument(
      '--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='Pore patch size.')
  FLAGS = parser.parse_args()

  main(FLAGS.model_dir, FLAGS.polyu_dir, FLAGS.patch_size, FLAGS.batch_size)
