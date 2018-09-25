import tensorflow as tf
import numpy as np
import argparse
import os

import utils
from models import detection

FLAGS = None


def extract_pores(image, image_pl, predictions, half_patch_size, prob_thr,
                  inter_thr, sess):
  '''
  Extracts pores from an image, inferring from predictions and post-processing
  with thresholding and NMS.

  Args:
    image: image in which to detect pores.
    image_pl: tf placeholder holding net's image input.
    predictions: tf tensor op of net's output.
    half_patch_size: half the detection patch size. used for padding the
      predictions to the input's original dimensions.
    prob_thr: probability threshold.
    inter_thr: NMS intersection threshold.
    sess: tf session

  Returns: 
    detections for image in shape [N, 2]
  '''
  # predict probability of pores
  pred = sess.run(
      predictions,
      feed_dict={image_pl: np.reshape(image, (1, ) + image.shape + (1, ))})

  # add borders lost in convolution
  pred = np.reshape(pred, pred.shape[1:-1])
  pred = np.pad(pred, ((half_patch_size, half_patch_size),
                       (half_patch_size, half_patch_size)), 'constant')

  # convert into coordinates
  pick = pred > prob_thr
  coords = np.argwhere(pick)
  probs = pred[pick]

  # filter detections with nms
  dets, _ = utils.nms(coords, probs, 7, inter_thr)

  return dets


def batch_extract(load_path, save_path, extract_fn):
  '''
  Extracts pores from all images in directory load_path
  using extract_fn and saves corresponding detections
  in save_path.

  Args:
    load_path: path to load image from.
    save_path: path to save detections. will be created
      if non existent.
    extract_fn: function that receives an image and
      returns an array of shape [N, 2] of detections.
  '''
  # load images from 'load_path'
  images, names = utils.load_images_with_names(load_path)

  # create 'save_path' directory tree
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # extract from each image and save it
  for image, name in zip(images, names):
    # extract pores
    detections = extract_fn(image)

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
      extract_fn = lambda image: extract_pores(image, image_pl, net.predictions, half_patch_size, FLAGS.prob_thr, FLAGS.inter_thr, sess)

      # batch extract from dbi training
      print('Extracting pores from PolyU-HRF DBI Training images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Training')
      batch_extract(load_path, save_path, extract_fn)
      print('Done')

      # batch extract from dbi test
      print('Extracting pores from PolyU-HRF DBI Test images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Test')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBI', 'Test')
      batch_extract(load_path, save_path, extract_fn)
      print('Done')

      # batch extract from dbii
      print('Extracting pores from PolyU-HRF DBII images...')
      load_path = os.path.join(FLAGS.polyu_dir_path, 'DBII')
      save_path = os.path.join(FLAGS.results_dir_path, 'DBII')
      batch_extract(load_path, save_path, extract_fn)
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
