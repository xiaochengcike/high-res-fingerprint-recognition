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
      '--polyu_dir_path',
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
      default='matching_results.txt',
      help='Path to results file.')
  parser.add_argument(
      '--descriptors',
      type=str,
      default='sift',
      help='Which descriptors to use. Can be "sift" or "trained"')
  parser.add_argument(
      '--mode',
      type=str,
      default='basic',
      help='Mode to match images. Can be "basic" or "spatial".')
  parser.add_argument(
      '--thr', type=float, help='Second correspondence elimination threshold.')
  parser.add_argument(
      '--model_dir_path', type=str, help='Trained model directory path.')
  parser.add_argument(
      '--patch_size', type=int, required=True, help='Pore patch size.')
  FLAGS = parser.parse_args()

  # parse descriptor and adjust accordingly
  if FLAGS.descriptors == 'sift':
    compute_descriptors = utils.extract_sift_descriptors
  else:
    assert FLAGS.model_dir_path is not None, \
        'Trained model path is required when using trained descriptor'
    import tensorflow as tf
    import numpy as np

    import descriptor_detector

    img_pl, _ = utils.placeholder_inputs()
    pts_pl = tf.placeholder(tf.int32, shape=[None, 2])
    net = descriptor_detector.Net(img_pl, training=False)
    sess = tf.Session()

    print('Restoring model in {}...'.format(FLAGS.model_dir_path))
    utils.restore_model(sess, FLAGS.model_dir_path)
    print('Done.')

    trained_descs = tf.gather_nd(tf.squeeze(net.spatial_descs), pts_pl)
    compute_descriptors = lambda img, pts: sess.run(trained_descs,
        feed_dict={
          img_pl: np.reshape(img, (1,) + img.shape + (1,)),
          pts_pl: np.array(pts) - FLAGS.patch_size // 2
          })

  # parse matching mode and adjust accordingly
  if FLAGS.mode == 'basic':
    from matching import basic as match
  else:
    from matching import spatial as match

  # make dir path be full DBI Training path
  imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')

  subject_ids = [
      6, 9, 11, 13, 16, 18, 34, 41, 42, 47, 62, 67, 118, 186, 187, 188, 196,
      198, 202, 207, 223, 225, 226, 228, 242, 271, 272, 278, 287, 293, 297,
      307, 311, 321, 323
  ]
  register_ids = [1, 2, 3]
  session_ids = [1, 2]

  # load images, points, compute descriptors and make indices correspondences
  print('Loading images and detections, and computing descriptors...')
  id2index_dict = {}
  all_descs = []
  all_pts = []
  index = 0
  for subject_id in subject_ids:
    for session_id in session_ids:
      for register_id in register_ids:
        instance = '{}_{}_{}'.format(subject_id, session_id, register_id)

        # load image
        img_path = os.path.join(imgs_dir_path, '{}.jpg'.format(instance))
        img = cv2.imread(img_path, 0)

        # load detections
        pts_path = os.path.join(FLAGS.pts_dir_path, '{}.txt'.format(instance))
        pts = utils.load_dets_txt(pts_path)
        all_pts.append(pts)

        # compute image descriptors
        descs = compute_descriptors(img, pts)
        all_descs.append(descs)

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
        descs1 = all_descs[index1]
        pts1 = all_pts[index1]
        for register_id2 in register_ids:
          index2 = id2index((subject_id, 2, register_id2))
          descs2 = all_descs[index2]
          pts2 = all_pts[index2]
          print('{}_{}_{} x {}_{}_{}'.format(subject_id, 1, register_id1,
                                             subject_id, 2, register_id2))
          print(1, match(descs1, descs2, pts1, pts2, thr=FLAGS.thr), file=f)

    # different subject comparisons
    for subject_id1 in subject_ids:
      for subject_id2 in subject_ids:
        if subject_id1 != subject_id2:
          index1 = id2index((subject_id1, 1, 1))
          index2 = id2index((subject_id2, 2, 1))

          descs1 = all_descs[index1]
          descs2 = all_descs[index2]
          pts1 = all_pts[index1]
          pts2 = all_pts[index2]

          print('{}_{}_{} x {}_{}_{}'.format(subject_id1, 1, 1, subject_id2, 2,
                                             1))
          print(0, match(descs1, descs2, pts1, pts2, thr=FLAGS.thr), file=f)
