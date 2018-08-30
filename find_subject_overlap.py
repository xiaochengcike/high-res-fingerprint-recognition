import os
import argparse

import utils
import matching

FLAGS = None


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


def top_k_candidates(query_descs, query_pts, gallery_descs, gallery_pts,
                     subject_ids, session_ids, register_ids, id2index, match,
                     k):
  subjects_and_scores = []

  # same subject comparisons
  for subject_id in subject_ids:
    subject_score = 0
    for session_id in session_ids:
      for register_id in register_ids:
        # retrieve example data
        index = id2index((subject_id, session_id, register_id))
        descs = gallery_descs[index]
        pts = gallery_pts[index]

        # update subject score
        subject_score += match(query_descs, descs, query_pts, pts, thr=0.7)

    # keep only 'k' most similar subjects
    subjects_and_scores.append((subject_score, subject_id))
    if len(subjects_and_scores) > k:
      subjects_and_scores = sorted(subjects_and_scores)[1:]

  _, subjects = zip(*subjects_and_scores)
  subjects = list(reversed(subjects))

  return subjects


def main():
  compute_descriptors = utils.sift_descriptors
  match = matching.spatial

  # load query set
  imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
  pts_dir_path = os.path.join(FLAGS.pts_dir_path, 'DBI', 'Training')

  # adjust ids for appropriate fold
  query_ids = [
      6, 9, 11, 13, 16, 18, 34, 41, 42, 47, 62, 67, 118, 186, 187, 188, 196,
      198, 202, 207, 223, 225, 226, 228, 242, 271, 272, 278, 287, 293, 297,
      307, 311, 321, 323
  ]
  register_ids = [1]
  session_ids = [1]

  print('Loading query set...')
  query_descs, query_pts, _ = load_dataset(imgs_dir_path, pts_dir_path,
                                           query_ids, session_ids,
                                           register_ids, compute_descriptors)
  print('Done')

  # load gallery set
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
      93, 94, 95, 96, 97, 98, 99, 100, 105, 106, 107, 108, 109, 110, 111, 112,
      113, 114, 115, 116, 117, 118, 119, 120, 125, 126, 127, 128, 129, 130,
      131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
      157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168
  ]
  register_ids = [1, 2, 3, 4, 5]
  session_ids = [1, 2]

  # load images, points, compute descriptors and make indices correspondences
  print('Loading gallery set...')
  gallery_descs, gallery_pts, id2index = load_dataset(
      imgs_dir_path, pts_dir_path, subject_ids, session_ids, register_ids,
      compute_descriptors)
  print('Done')

  # find best gallery ids for query ids
  print('Matching...')
  candidate_ids = []
  for descs, pts in zip(query_descs, query_pts):
    point_candidate_ids = top_k_candidates(
        descs, pts, gallery_descs, gallery_pts, subject_ids, session_ids,
        register_ids, id2index, match, FLAGS.k)
    candidate_ids.append(point_candidate_ids)
  print('Done')

  # save candidate subject ids to file
  if FLAGS.results_path is not None:
    print('Saving to file {}...'.format(FLAGS.results_path))

    # create directory tree, if non-existing
    dirname = os.path.dirname(FLAGS.results_path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    with open(FLAGS.results_path, 'w') as f:
      for point_candidate_ids in candidate_ids:
        print(*point_candidate_ids, file=f)

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
  parser.add_argument(
      '--results_path', required=True, type=str, help='path to results file')
  parser.add_argument(
      '--fold',
      type=str,
      default='DBI-test',
      help='choose what fold of polyu to use. Can be "DBI-test" or "DBII"')
  parser.add_argument(
      '--k',
      type=int,
      default=5,
      help='number of best subject id candidates to keep for each register')

  FLAGS = parser.parse_args()

  main()
