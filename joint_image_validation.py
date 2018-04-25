from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import os
import argparse

import pore_detector_descriptor
import validation
import polyu
import util


def load_detection_dataset(polyu_path):
  polyu_path = os.path.join(polyu_path, 'GroundTruth', 'PoreGroundTruth')
  dataset = polyu.DetectionDataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      window_size=FLAGS.window_size)

  return dataset


def main(model_dir, polyu_path, window_size, batch_size):
  print('Loading PolyU-HRF dataset...')
  dataset = load_detection_dataset(polyu_path)
  print('Done.')

  with tf.Graph().as_default():
    # gets placeholders for windows and labels
    windows_pl, labels_pl = util.placeholder_inputs()

    # builds inference graph
    print('Building graph...')
    net = pore_detector_descriptor.Net(
        windows_pl, dataset.train.window_size, training=False)
    print('Done.')

    with tf.Session() as sess:
      print('Restoring model in {}...'.format(FLAGS.model_dir))
      util.restore_model(sess, model_dir)
      print('Done.')

      image_f_score, image_tdr, image_fdr, inter_thr = validation.detection_by_images(
          sess, net.dets, windows_pl, dataset.val)
      print(
          'Whole image evaluation:',
          '\tTDR = {}'.format(image_tdr),
          '\tFDR = {}'.format(image_fdr),
          '\tF score = {}'.format(image_f_score),
          '\tinter_thr = {}'.format(inter_thr),
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
      '--window_size', type=int, default=17, help='Pore window size.')
  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS.model_dir, FLAGS.polyu_dir, FLAGS.window_size, FLAGS.batch_size)
