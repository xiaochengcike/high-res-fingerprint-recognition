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
