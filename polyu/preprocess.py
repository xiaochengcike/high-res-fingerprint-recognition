import numpy as np
import argparse
import os
import cv2

import utils
from polyu import aligned_images

FLAGS = None

if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='path to PolyU-HRF dataset')
  parser.add_argument(
      '--pts_dir_path',
      type=str,
      required=True,
      help='path to PolyU-HRF DBI Training dataset keypoints detections')
  parser.add_argument(
      '--patch_size',
      type=int,
      default=17,
      help='image patch size for descriptor')
  parser.add_argument(
      '--result_dir_path',
      type=str,
      required=True,
      help='path to save description dataset')
  parser.add_argument(
      '--flip',
      action='store_true',
      help=
      'use this flag to also generate flipped patches, doubling the description dataset size'
  )
  parser.add_argument(
      '--for_sift',
      action='store_true',
      help='use this flag to generate patches fit for validate.sift use')
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
  imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
  imgs, labels = utils.load_images_with_labels(imgs_dir_path)
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
    handler = aligned_images.Handler(imgs, pts, FLAGS.patch_size, FLAGS.flip,
                                     FLAGS.for_sift)
    print('Done.')

    print("Extracting patches of subject's {} images...".format(label))
    for patches in handler:
      for i, patch in enumerate(patches):
        patch_path = os.path.join(FLAGS.result_dir_path, '{}_{}.png'.format(
            patch_index, i + 1))
        cv2.imwrite(patch_path, 255 * patch)
      patch_index += 1
    print('Done.')
