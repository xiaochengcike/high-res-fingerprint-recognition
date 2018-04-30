from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np


def align(L, R, weights=None, scale=True):
  '''
  Use Horn's closed form absolute orientation
  method with orthogonal matrices to align a set
  of points L to a set of points R. L and R are
  sets in which each row is a point and the ith
  point of L corresponds to the ith point of R.

  Args:
    L: set of points to align, a point per row.
    R: set of points to align to, a point per row.
    weights: an array of weights in [0, 1] for
      each correspondence. If 'None',
      correspondences are unweighted.
    scale: whether should also solve the scale. If
      'False', s = 1.

  Returns:
    A, b: a transformation such that
        L @ A + b ~ R
      and the squared residues are as low as
      possible. See the paper (Horn et al., 1988)
      for details.
  '''
  L = np.asarray(L)
  R = np.asarray(R)
  if weights is None:
    weights = 1
  else:
    weights = np.expand_dims(weights, axis=-1)

  # compute points' centroids
  L_centroid = np.mean(weights * L, axis=0)
  R_centroid = np.mean(weights * R, axis=0)

  # translate points to make centroids origin
  L_ = L - L_centroid
  R_ = R - R_centroid

  # find scale 's'
  if scale:
    Sr = np.sum(np.reshape(weights, -1) * np.sum(R_ * R_, axis=1), axis=0)
    Sl = np.sum(np.reshape(weights, -1) * np.sum(L_ * L_, axis=1), axis=0)
    s = np.sqrt(Sr / Sl)
  else:
    s = 1

  # find rotation 'A'
  M = np.dot(R_.T, weights * L_)
  MTM = np.dot(M.T, M)
  w, u = np.linalg.eigh(MTM)
  u = u.T

  # solve even if M is not full rank
  rank = M.shape[0] - np.sum(np.isclose(w, 0))
  if rank < M.shape[0] - 1:
    raise Exception(
        'rank of M must be at least shape(M)[0] - 1 for unique solution')
  elif rank == M.shape[0] - 1:
    # get non zero eigenvalues and eigenvectors
    mask = np.logical_not(np.isclose(w, 0))
    u = u[mask]
    w = w[mask]

    # compute S pseudoinverse
    lhs = np.expand_dims(u / np.sqrt(w), axis=-1)
    rhs = np.expand_dims(u, axis=1)
    S_pseudoinv = np.sum(np.matmul(lhs, rhs), axis=0)

    # compute MTM svd
    u, _, v = np.linalg.svd(MTM)

    # compute tomasi's fix
    lhs = np.expand_dims(u.T[-1], 0)
    rhs = np.expand_dims(v.T[-1], 0).T
    tfix = np.dot(lhs, rhs)

    # find whether should sum or subtract
    A_base = np.dot(M, S_pseudoinv)
    A = A_base + tfix
    if np.linalg.det(A) < 0:
      A = A_base - tfix
  else:
    lhs = np.expand_dims(u / np.sqrt(w), axis=-1)
    rhs = np.expand_dims(u, axis=1)
    S_inv = np.sum(np.matmul(lhs, rhs), axis=0)
    A = np.dot(M, S_inv)

  # include scale in A
  A *= s

  # find translation 'b'
  b = R_centroid - np.dot(A, L_centroid)

  return A, b


if __name__ == '__main__':
  import sys
  import cv2

  import util
  import pamplona_lemes_matching as matching

  img1_path, pts1_path, img2_path, pts2_path = sys.argv[1:]

  # load images
  img1 = cv2.imread(img1_path, 0)
  img2 = cv2.imread(img2_path, 0)

  # load detection points
  pts1 = util.load_dets_txt(pts1_path)
  pts2 = util.load_dets_txt(pts2_path)

  pairs = matching.find_correspondences(img1, pts1, img2, pts2)
  max_dist = np.max(np.asarray(pairs)[:, 2])
  L = []
  R = []
  w = []
  weighted = False
  for pair in pairs:
    L.append(pts1[pair[0]])
    R.append(pts2[pair[1]])
    w.append((max_dist - pair[2]) / max_dist)
  if not weighted:
    w = None
  A, b = align(L, R, weights=w)

  # generate aligned images
  aligned1 = np.zeros_like(img1, dtype=img1.dtype)
  for ref_row in range(img1.shape[0]):
    for ref_col in range(img1.shape[1]):
      t_row, t_col = np.dot(A, np.array([ref_row, ref_col])) + b
      if 0 <= t_row + 0.5 < img1.shape[0] and 0 <= t_col + 0.5 < img1.shape[1]:
        aligned1[int(t_row + 0.5), int(t_col + 0.5)] = img1[ref_row, ref_col]

  aligned2 = np.zeros_like(img2, dtype=img2.dtype)
  aligned2[np.nonzero(aligned1)] = img2[np.nonzero(aligned1)]

  matched = util.draw_matches(img1, pts1, img2, pts2, pairs)
  cv2.imshow('correspondences', matched)
  cv2.imshow('aligned1', aligned1)
  cv2.imshow('aligned2', aligned2)
  cv2.waitKey(0)
