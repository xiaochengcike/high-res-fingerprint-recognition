from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import argparse
import os
import cv2

import utils

if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--imgs_dir_path',
      required=True,
      type=str,
      help='Path to PolyU-HRF dataset.')
  parser.add_argument(
      '--pts_dir_path',
      type=str,
      required=True,
      help='Path to PolyU-HRF DBI Training dataset keypoints detections.')
  parser.add_argument(
      '--results_path',
      type=str,
      default='alignment_matching.txt',
      help='Path to results file.')
  parser.add_argument(
      '--mode',
      type=str,
      default='sift',
      help='Mode to match images. Can be "alignment" or "sift".')
  FLAGS, _ = parser.parse_known_args()

  # find out which matching mode was specified
  if FLAGS.mode == 'sift':
    from matching.sift_bidirectional import match
  else:
    from matching.alignment import match

  # make dir path be full DBI Training path
  imgs_dir_path = os.path.join(FLAGS.imgs_dir_path, 'DBI', 'Training')

  subject_ids = [
      6, 9, 11, 13, 16, 18, 34, 41, 42, 47, 62, 67, 118, 186, 187, 188, 196,
      198, 202, 207, 223, 225, 226, 228, 242, 271, 272, 278, 287, 293, 297,
      307, 311, 321, 323
  ]
  register_ids = [1, 2, 3]
  session_ids = [1, 2]

  # load imgs, pts and make index correspondence
  print('Loading images...')
  id2index_dict = {}
  imgs = []
  pts = []
  index = 0
  for subject_id in subject_ids:
    for session_id in session_ids:
      for register_id in register_ids:
        instance = '{}_{}_{}'.format(subject_id, session_id, register_id)

        # load img
        img_path = os.path.join(imgs_dir_path, '{}.jpg'.format(instance))
        img = cv2.imread(img_path, 0)
        imgs.append(img)

        # load pts
        pts_path = os.path.join(FLAGS.pts_dir_path, '{}.txt'.format(instance))
        pts.append(utils.load_dets_txt(pts_path))

        # make id2index correspondence
        id2index_dict[(subject_id, session_id, register_id)] = index
        index += 1

  id2index = lambda x: id2index_dict[tuple(x)]
  print('Done.')

  print('Matching...')
  with open(FLAGS.results_path, 'w') as f:
    # same subject comparisons
    for subject_id in subject_ids:
      for register_id1 in register_ids:
        index1 = id2index((subject_id, 1, register_id1))
        img1 = imgs[index1]
        pts1 = pts[index1]
        for register_id2 in register_ids:
          index2 = id2index((subject_id, 2, register_id2))
          img2 = imgs[index2]
          pts2 = pts[index2]
          print('{}_{}_{} x {}_{}_{}'.format(subject_id, 1, register_id1,
                                             subject_id, 2, register_id2))
          print(1, match(img1, pts1, img2, pts2), file=f)

    # different subject comparisons
    for subject_id1 in subject_ids:
      for subject_id2 in subject_ids:
        if subject_id1 != subject_id2:
          index1 = id2index((subject_id1, 1, 1))
          index2 = id2index((subject_id2, 2, 1))

          img1 = imgs[index1]
          pts1 = pts[index1]

          img2 = imgs[index2]
          pts2 = pts[index2]

          print('{}_{}_{} x {}_{}_{}'.format(subject_id1, 1, 1, subject_id2, 2,
                                             1))
          print(0, match(img1, pts1, img2, pts2), file=f)
