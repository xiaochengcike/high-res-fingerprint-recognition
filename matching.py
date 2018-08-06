import numpy as np
import itertools

import utils


def basic(descs1, descs2, pts1=None, pts2=None, thr=None):
  if len(descs1) == 0 or len(descs2) == 0:
    return 0

  return len(utils.find_correspondences(descs1, descs2, thr=thr))


def spatial(descs1, descs2, pts1, pts2, thr=None):
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
