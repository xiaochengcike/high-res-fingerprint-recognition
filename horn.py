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
        s * (L @ A) + b ~ R,
      making the sum of squares of errors the
      least possible. See the paper (Horn et al.,
      1988) for details.
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
    w = np.expand_dims(w, 1)
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
    w = np.expand_dims(w, 1)
    lhs = np.expand_dims(u / np.sqrt(w), axis=-1)
    rhs = np.expand_dims(u, axis=1)
    S_inv = np.sum(np.matmul(lhs, rhs), axis=0)
    A = np.dot(M, S_inv)

  # find translation 'b'
  b = R_centroid - s * np.dot(A, L_centroid)

  return A, b, s


def bilinear_interpolation(x, y, f):
  x1 = int(x)
  y1 = int(y)
  x2 = x1 + 1
  y2 = y1 + 1

  fq = [[f[x1, y1], f[x1, y2]], [f[x2, y1], f[x1, y2]]]
  lhs = [[x2 - x, x - x1]]
  rhs = [y2 - y, y - y1]

  return np.dot(np.dot(lhs, fq), rhs)


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
  A, b, s = align(L, R, weights=w)

  # generate aligned images
  aligned = np.zeros_like(img1, dtype=img1.dtype)
  for ref_row in range(img1.shape[0]):
    for ref_col in range(img1.shape[1]):
      t_row, t_col = np.dot(A.T, (np.array([ref_row, ref_col]) - b) / s)
      if 0 <= t_row < img1.shape[0] - 1 and 0 <= t_col < img1.shape[1] - 1:
        aligned[ref_row, ref_col] = bilinear_interpolation(t_row, t_col, img1)

  diff = np.stack([img2, img2, aligned], axis=-1)
  cv2.imshow('img', img2)
  cv2.imshow('aligned', aligned)
  cv2.imshow('diff', diff)
  cv2.waitKey(0)
  cv2.imwrite('aligned.png', aligned)
  cv2.imwrite('diff.png', diff)
  cv2.imwrite('img.png', img2)
