from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import tensorflow as tf
import numpy as np
import argparse
import os

import utils
import detector

FLAGS = None


def main():
  print('Loading images...')
  images, image_names = utils.load_images_with_names(FLAGS.imgs_dir)
  print('Done.')

  half_patch_size = FLAGS.patch_size // 2

  with tf.Graph().as_default():
    images_pl, _ = utils.placeholder_inputs()

    print('Building graph...')
    net = detector.Net(images_pl, FLAGS.patch_size, training=False)
    print('Done.')

    with tf.Session() as sess:
      print('Restoring model in {}...'.format(FLAGS.model_dir))
      utils.restore_model(sess, FLAGS.model_dir)
      print('Done.')

      # extract pores for each image
      for i, img in enumerate(images):
        print('Extracting pores in image {}...'.format(image_names[i]))
        # predict probability of pores
        pred = sess.run(
            net.preds,
            feed_dict={images_pl: np.reshape(img, (1, ) + img.shape + (1, ))})

        # add borders lost in convolution
        pred = np.reshape(pred, pred.shape[1:-1])
        pred = np.pad(pred, ((half_patch_size, half_patch_size),
                             (half_patch_size, half_patch_size)), 'constant')

        # convert into coordinates
        pick = pred > 0.9
        coords = np.argwhere(pick)
        probs = pred[pick]

        # filter detections with nms
        dets, _ = utils.nms(coords, probs, 7, 0.1)

        # save results
        filename = os.path.join(FLAGS.results_dir,
                                '{}.txt'.format(image_names[i]))
        utils.save_dets_txt(dets, filename)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--imgs_dir', required=True, type=str, help='Path to images directory')
  parser.add_argument(
      '--model_dir', type=str, required=True, help='Logging directory.')
  parser.add_argument(
      '--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='Pore patch size.')
  parser.add_argument(
      '--results_dir',
      type=str,
      default='result',
      help='Path to folder in which results should be saved.')
  FLAGS = parser.parse_args()

  main()
