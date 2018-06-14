from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import argparse
import os
import cv2

import utils
from polyu import aligned_images

if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--imgs_dir_path',
      required=True,
      type=str,
      help='Path to images (with PolyU-HRF name format) dataset.')
  parser.add_argument(
      '--pts_dir_path',
      type=str,
      required=True,
      help='Path to PolyU-HRF DBI Training dataset keypoints detections.')
  parser.add_argument(
      '--patch_size',
      type=int,
      default=17,
      help='Image patch size for descriptor.')
  parser.add_argument(
      '--result_dir_path',
      type=str,
      required=True,
      help='Path to save description dataset.')
  FLAGS = parser.parse_args()

  # load detections
  print('Loading detections...')
  pts = []
  for pts_path in sorted(os.listdir(FLAGS.pts_dir_path)):
    if pts_path.endswith('.txt'):
      pts.append(
          utils.load_dets_txt(os.path.join(FLAGS.pts_dir_path, pts_path)))
  print('Done.')

  # load images with respective labels
  print('Loading images...')
  imgs, labels = utils.load_images_with_labels(FLAGS.imgs_dir_path)
  print('Done.')

  # convert to np array
  imgs = np.array(imgs)
  pts = np.array(pts)
  labels = np.array(labels)

  # group (imgs, pts) by labels
  print('Grouping (images, detections) by labels...')
  grouped_imgs = []
  grouped_pts = []
  unique_labels = np.unique(labels)
  for label in unique_labels:
    indices = np.where(labels == label)
    grouped_imgs.append(imgs[indices])
    grouped_pts.append(pts[indices])
  print('Done.')

  # extract all patches from all images
  # patch index is a unique patch identifier
  patch_index = 1
  for imgs, pts, label in zip(grouped_imgs, grouped_pts, unique_labels):
    print('Aligning images of subject {}...'.format(label))
    handler = aligned_images.Handler(imgs, pts, FLAGS.patch_size)
    print('Done.')

    print("Extracting patches of subject's {} images...".format(label))
    for patches in handler:
      for i, patch in enumerate(patches):
        patch_path = os.path.join(FLAGS.result_dir_path, '{}_{}.png'.format(
            patch_index, i + 1))
        cv2.imwrite(patch_path, 255 * patch)
      patch_index += 1
    print('Done.')
