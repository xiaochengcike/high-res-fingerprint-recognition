import os
import argparse
import numpy as np
import tensorflow as tf

import utils
import matching

FLAGS = None


def validation_eer(images_pl, session, descs_op, dataset, patch_size):
  # describe patches with detections and get
  # subject and register ids from labels
  all_descs = []
  all_pts = []
  subject_ids = set()
  register_ids = set()
  id2index_dict = {}
  index = 0
  for img, pts, label in dataset:
    # add image detections to all detections
    all_pts.append(pts)

    # compute descriptors
    descs = utils.trained_descriptors(img, pts, patch_size, session, images_pl,
                                      descs_op)
    all_descs.append(descs)

    # add ids to all ids
    subject_ids.add(label[0])
    register_ids.add(label[2])

    # make 'id2index' correspondence
    id2index_dict[tuple(label)] = index
    index += 1

  # convert dict into function
  id2index = lambda x: id2index_dict[tuple(x)]

  # convert sets into lists
  subject_ids = list(subject_ids)
  register_ids = list(register_ids)

  # match and compute eer
  pos, neg = polyu_match(
      all_descs,
      all_pts,
      subject_ids,
      register_ids,
      id2index,
      matching.basic,
      thr=0.7)
  eer = utils.eer(pos, neg)

  return eer


def load_dataset(imgs_dir_path,
                 pts_dir_path,
                 subject_ids,
                 session_ids,
                 register_ids,
                 compute_descriptors,
                 patch_size=None):
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
        img = utils.load_image(img_path)

        # load detections
        pts_path = os.path.join(pts_dir_path, '{}.txt'.format(instance))
        pts = utils.load_dets_txt(pts_path)

        # filter detections at non valid border
        if patch_size is not None:
          half = patch_size // 2
          pts_ = []
          for pt in pts:
            if half <= pt[0] < img.shape[0] - half:
              if half <= pt[1] < img.shape[1] - half:
                pts_.append(pt)
          pts = pts_

        all_pts.append(pts)

        # compute image descriptors
        descs = compute_descriptors(img, pts)
        all_descs.append(descs)

        # make id2index correspondence
        id2index_dict[(subject_id, session_id, register_id)] = index
        index += 1

  # turn id2index into conversion function
  id2index = lambda x: id2index_dict[tuple(x)]

  return all_descs, all_pts, id2index


def polyu_match(all_descs,
                all_pts,
                subject_ids,
                register_ids,
                id2index,
                match,
                thr=None):
  pos = []
  neg = []

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
        pos.append(match(descs1, descs2, pts1, pts2, thr=thr))

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

        neg.append(match(descs1, descs2, pts1, pts2, thr=thr))

  return pos, neg


def main():
  # parse descriptor and adjust accordingly
  compute_descriptors = None
  if FLAGS.descriptors == 'sift':
    compute_descriptors = utils.sift_descriptors
  else:
    if FLAGS.model_dir_path is None:
      raise TypeError(
          'Trained model path is required when using trained descriptor')
    if FLAGS.patch_size is None:
      raise TypeError('Patch size is required when using trained descriptor')

    # create net graph and restore saved model
    from models import description

    img_pl, _ = utils.placeholder_inputs()
    net = description.Net(img_pl, training=False)
    sess = tf.Session()
    print('Restoring model in {}...'.format(FLAGS.model_dir_path))
    utils.restore_model(sess, FLAGS.model_dir_path)
    print('Done')

    compute_descriptors = lambda img, pts: utils.trained_descriptors(img, pts, FLAGS.patch_size, sess, img_pl, net.descriptors)

  # parse matching mode and adjust accordingly
  if FLAGS.mode == 'basic':
    match = matching.basic
  else:
    match = matching.spatial

  # make dir path be full appropriate dir path
  imgs_dir_path = None
  pts_dir_path = None
  subject_ids = None
  register_ids = None
  session_ids = None
  if FLAGS.fold == 'DBI-train':
    # adjust paths for appropriate fold
    imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
    pts_dir_path = os.path.join(FLAGS.pts_dir_path, 'DBI', 'Training')

    # adjust ids for appropriate fold
    subject_ids = [
        6, 9, 11, 13, 16, 18, 34, 41, 42, 47, 62, 67, 118, 186, 187, 188, 196,
        198, 202, 207, 223, 225, 226, 228, 242, 271, 272, 278, 287, 293, 297,
        307, 311, 321, 323
    ]
    register_ids = [1, 2, 3]
    session_ids = [1, 2]
  else:
    # adjust paths for appropriate fold
    if FLAGS.fold == 'DBI-test':
      imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Test')
      pts_dir_path = os.path.join(FLAGS.pts_dir_path, 'DBI', 'Test')
    else:
      imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBII')
      pts_dir_path = os.path.join(FLAGS.pts_dir_path, 'DBII')

    # adjust ids for appropriate fold
    subject_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95, 96, 97, 98, 99, 100, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
        144, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168
    ]
    register_ids = [1, 2, 3, 4, 5]
    session_ids = [1, 2]

  # load images, points, compute descriptors and make indices correspondences
  print('Loading images and detections, and computing descriptors...')
  all_descs, all_pts, id2index = load_dataset(
      imgs_dir_path, pts_dir_path, subject_ids, session_ids, register_ids,
      compute_descriptors)
  print('Done')

  print('Matching...')
  pos, neg = polyu_match(
      all_descs,
      all_pts,
      subject_ids,
      register_ids,
      id2index,
      match,
      thr=FLAGS.thr)
  print('Done')

  # print equal error rate
  print('EER = {}'.format(utils.eer(pos, neg)))

  # save results to file
  if FLAGS.results_path is not None:
    print('Saving results to file {}...'.format(FLAGS.results_path))
    with open(FLAGS.results_path, 'w') as f:
      # save same subject scores
      for score in pos:
        print(1, score, file=f)

      # save different subject scores
      for score in neg:
        print(0, score, file=f)
    print('Done')


if __name__ == '__main__':
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
      help='path to chosen dataset keypoints detections')
  parser.add_argument('--results_path', type=str, help='path to results file')
  parser.add_argument(
      '--descriptors',
      type=str,
      default='sift',
      help='which descriptors to use. Can be "sift" or "trained"')
  parser.add_argument(
      '--mode',
      type=str,
      default='basic',
      help='mode to match images. Can be "basic" or "spatial"')
  parser.add_argument(
      '--thr', type=float, help='second correspondence elimination threshold')
  parser.add_argument(
      '--model_dir_path', type=str, help='trained model directory path')
  parser.add_argument('--patch_size', type=int, help='pore patch size')
  parser.add_argument(
      '--fold',
      type=str,
      default='DBI-train',
      help=
      'choose what fold of polyu to use. Can be "DBI-train", "DBI-test" and "DBII"'
  )
  parser.add_argument('--seed', type=int, help='random seed')

  FLAGS = parser.parse_args()

  # set random seeds
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  main()
