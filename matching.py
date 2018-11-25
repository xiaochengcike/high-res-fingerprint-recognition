import numpy as np
import itertools

import utils


def basic(descs1, descs2, pts1=None, pts2=None, thr=None):
  '''
  Finds bidirectional correspondences between descriptors
  descs1 and descs2. If thr is provided, discards correspondences
  that fail a distance ratio check with threshold thr; in this
  case, returns correspondences satisfying SIFT's criterion.

  Args:
    descs1: [N, M] array of N descriptors of dimension M each.
    descs2: [N, M] array of N descriptors of dimension M each.
    pts1: sentinel argument for matching function signature
      standardization.
    pts2: sentinel argument for matching function signature
      standardization.
    thr: distance ratio check threshold.

  Returns:
    number of found bidirectional correspondences. If thr is not
      None, number of bidirectional correspondences that satisfy
      a distance ratio check.
  '''
  if len(descs1) == 0 or len(descs2) == 0:
    return 0

  return len(utils.find_correspondences(descs1, descs2, thr=thr))


def spatial(descs1, descs2, pts1, pts2, thr=None):
  '''
  Computes the matching score proposed by Pamplona Segundo &
  Lemes (Pore-based ridge reconstruction for fingerprint
  recognition, 2015) using bidirectional correspondences
  between descriptors descs1 and descs2.
  If thr is provided, correspondences that fail a distance
  ratio check with threshold thr are discarded.

  Args:
    descs1: [N, M] array of N descriptors of dimension M each.
    descs2: [N, M] array of N descriptors of dimension M each.
    pts1: [N, 2] array of coordinates from which each descriptor
      of descs1 was computed.
    pts1: [N, 2] array of coordinates from which each descriptor
      of descs2 was computed.
    thr: distance ratio check threshold.

  Returns:
    matching score between descs1 and descs2.
  '''
  if len(descs1) == 0 or len(descs2) == 0:
    return 0

  pairs = utils.find_correspondences(descs1, descs2, thr=thr)

  pts1 = np.array(pts1)
  pts2 = np.array(pts2)
  score = 0
  for pair1, pair2 in itertools.combinations(pairs, 2):
    d1 = np.linalg.norm(pts1[pair1[0]] - pts1[pair2[0]])
    d2 = np.linalg.norm(pts2[pair1[1]] - pts2[pair2[1]])
    score += 1 / (1 + abs(d1 - d2))

  return score
