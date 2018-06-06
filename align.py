from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np

import sift_bidirectional_matching as matching


def _inside(img, pt):
  return 0 <= pt[0] < img.shape[0] and 0 <= pt[1] < img.shape[1]


def _transf(pt, A, s, b):
  return s * np.dot(pt, A.T) + b


def _inv_transf(pt, A, s, b):
  return np.dot(pt - b, A) / s


def _horn(L, R, weights=None, scale=True):
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
    A, b, s: a transformation such that
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
    lhs = np.expand_dims(u[-1], axis=-1)
    rhs = np.expand_dims(v[-1], axis=0)
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


def iterative(img1,
              pts1,
              img2,
              pts2,
              euclidean_lambda=500,
              weighted=False,
              max_iter=10):
  '''
  Iteratively align image 'img1' to 'img2', using
  Horn's absolute orientation method to minimize
  the mean squared error between keypoints'
  correspondences in sets 'pts1' and 'pts2'.
  Correspondences between keypoints are found
  with the following metric:

    d(u, v) = ||SIFT(u) - SIFT(v)||^2 +
        + (\lambda * ||u - v||^2) / MSE

  where '\lambda' is a user specified weight and
  'MSE' is the mean squared error from the
  previous alignment. For the first iteration,
  MSE = Inf.

  Args:
    img1: np array with image to align.
    pts1: np array with img1 keypoints, one
      keypoint coordinate per row.
    img2: np array with image to align to.
    pts2: same as pts1, but for img2.
    euclidean_lambda: \lambda in above equation.
    weighted: whether should consider the
      correspondence confidence - computed as
      the reciprocal of its distance - in
      Horn's method.
    max_iter: Maximum number of iterations.

  Returns:
    A, b, s: the found alignment. For further
      information, read _horn() documentation.
  '''
  # initialize before first alignment
  mse = np.inf
  euclidean_weight = -1
  A = np.identity(2)
  s = 1
  b = np.array([0, 0])

  # iteratively align
  for _ in range(max_iter):
    # convergence criterion
    if np.isclose(mse * euclidean_weight, euclidean_lambda):
      break

    # compute weight of correspondences' euclidean distance
    euclidean_weight = euclidean_lambda / mse

    # find correspondences
    pairs = matching.find_correspondences(
        img1,
        pts1,
        img2,
        pts2,
        scale=8.0,
        euclidean_weight=euclidean_weight,
        transf=lambda x: _transf(x, A, s, b))

    # end alignment if no further correspondences are found
    if len(pairs) <= 1:
      break

    # make correspondence aligned array
    if weighted:
      max_dist = np.max(np.asarray(pairs)[:, 2])
      w = []
      L = []
      R = []
      for pair in pairs:
        L.append(pts1[pair[0]])
        R.append(pts2[pair[1]])
        w.append((max_dist - pair[2]) / max_dist)
    else:
      w = None
      L = []
      R = []
      for pair in pairs:
        L.append(pts1[pair[0]])
        R.append(pts2[pair[1]])

    # find alignment transformation
    A, b, s = _horn(L, R, weights=w)

    # compute alignment mse
    L = np.array(L)
    R = np.array(R)
    error = R - (s * np.dot(L, A.T) + b)
    dists = np.sum(error * error, axis=1)
    mse = np.mean(dists)

    # filter points out of images overlap
    pts1 = filter(lambda pt: _inside(img2, _transf(pt, A, s, b)), pts1)
    pts1 = list(pts1)

    pts2 = filter(lambda pt: _inside(img1, _inv_transf(pt, A, s, b)), pts2)
    pts2 = list(pts2)

  return A, b, s


if __name__ == '__main__':
  import sys
  import cv2
  import utils

  if len(sys.argv) < 5:
    raise Exception(
        'Expected 4 arguments: <img1_path> <pts1_path> <img2_path> <pts2_path>, found {}'.
        format(len(sys.argv) - 1))

  img1_path, pts1_path, img2_path, pts2_path = sys.argv[1:]

  # load images
  img1 = cv2.imread(img1_path, 0)
  img2 = cv2.imread(img2_path, 0)

  # load detection points
  pts1 = utils.load_dets_txt(pts1_path)
  pts2 = utils.load_dets_txt(pts2_path)

  A, b, s = iterative(img1, pts1, img2, pts2)

  # generate aligned images
  aligned = np.zeros_like(img1, dtype=img1.dtype)
  for ref_row in range(img1.shape[0]):
    for ref_col in range(img1.shape[1]):
      t_row, t_col = np.dot(A.T, (np.array([ref_row, ref_col]) - b) / s)
      if 0 <= t_row < img1.shape[0] - 1 and 0 <= t_col < img1.shape[1] - 1:
        aligned[ref_row, ref_col] = utils.bilinear_interpolation(
            t_row, t_col, img1)

  # display current alignment
  diff = np.stack([img2, img2, aligned], axis=-1)
  cv2.imshow('prealignment', img1)
  cv2.imshow('target', img2)
  cv2.imshow('aligned', aligned)
  cv2.imshow('diff', diff)
  cv2.waitKey(0)
