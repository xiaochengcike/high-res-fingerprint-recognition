import tensorflow as tf
import argparse
import os

import utils
from models import detection

FLAGS = None


def batch_detect(load_path, save_path, detect_fn):
  '''
  Detects pores in all images in directory load_path
  using detect_fn and saves corresponding detections
  in save_path.

  Args:
    load_path: path to load image from.
    save_path: path to save detections. Will be created
      if non existent.
    detect_fn: function that receives an image and
      returns an array of shape [N, 2] of detections.
  '''
  # load images from 'load_path'
  images, names = utils.load_images_with_names(load_path)

  # create 'save_path' directory tree
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # detect in each image and save it
  for image, name in zip(images, names):
    # detect pores
    detections = detect_fn(image)

    # save results
    filename = os.path.join(save_path, '{}.txt'.format(name))
    utils.save_dets_txt(detections, filename)


def main():
  half_patch_size = FLAGS.patch_size // 2

  with tf.Graph().as_default():
    image_pl, _ = utils.placeholder_inputs()

    print('Building graph...')
    net = detection.Net(image_pl, training=False)
    print('Done')

    with tf.Session() as sess:
      print('Restoring model in {}...'.format(FLAGS.model_dir_path))
      utils.restore_model(sess, FLAGS.model_dir_path)
      print('Done')

      # capture arguments in lambda function
      def detect_pores(image):
        return utils.detect_pores(image, image_pl, net.predictions,
                                  half_patch_size, FLAGS.prob_thr,
                                  FLAGS.inter_thr, sess)

      # batch detect in dbi training
      print('Detecting pores in PolyU-HRF DBI Training images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Training')
      batch_detect(load_path, save_path, detect_pores)
      print('Done')

      # batch detect in dbi test
      print('Detecting pores in PolyU-HRF DBI Test images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Test')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Test')
      batch_detect(load_path, save_path, detect_pores)
      print('Done')

      # batch detect in dbii
      print('Detecting pores in PolyU-HRF DBII images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBII')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBII')
      batch_detect(load_path, save_path, detect_pores)
      print('Done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='path to PolyU-HRF dataset')
  parser.add_argument(
      '--model_dir_path',
      type=str,
      required=True,
      help='path from which to restore trained model')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--results_dir_path',
      type=str,
      default='result',
      help='path to folder in which results should be saved')
  parser.add_argument(
      '--prob_thr',
      type=float,
      default=0.9,
      help='probability threshold to filter detections')
  parser.add_argument(
      '--inter_thr',
      type=float,
      default=0.1,
      help='nms intersection threshold')

  FLAGS = parser.parse_args()

  main()
