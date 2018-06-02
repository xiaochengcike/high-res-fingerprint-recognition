from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import cv2

import utils


def extract_descriptors(img, pts, scale):
  # improve image quality with median blur and clahe
  img = cv2.medianBlur(img, ksize=3)
  clahe = cv2.createCLAHE(clipLimit=3)
  img = clahe.apply(img)

  # convert points to cv2.keypoints
  pts = list(np.asarray(pts)[:, [1, 0]])
  kpts = cv2.KeyPoint.convert(pts, size=scale)

  # extract sift descriptors
  sift = cv2.xfeatures2d.SIFT_create()
  _, descs = sift.compute(img, kpts)

  return descs


def find_correspondences(img1,
                         pts1,
                         img2,
                         pts2,
                         scale=4.0,
                         thr=0.8,
                         euclidean_weight=0,
                         transf=None,
                         fast=True):
  # extract descriptors from both images
  descs1 = extract_descriptors(img1, pts1, scale)
  descs2 = extract_descriptors(img2, pts2, scale)

  # match descriptors
  if fast and euclidean_weight == 0:
    # match descs2 to descs1
    flann1 = cv2.flann_Index(
        descs1, params=dict(algorithm=0, trees=4), distType=1)
    indices1, dists1 = flann1.knnSearch(descs2, knn=2)

    # match descs1 to descs2
    flann2 = cv2.flann_Index(
        descs2, params=dict(algorithm=0, trees=4), distType=1)
    indices2, dists2 = flann2.knnSearch(descs1, knn=2)
  else:
    # compute descriptors' pairwise distances
    sqr1 = np.sum(descs1 * descs1, axis=1, keepdims=True)
    sqr2 = np.sum(descs2 * descs2, axis=1)
    D = sqr1 - 2 * np.matmul(descs1, descs2.T) + sqr2

    # add points' euclidean distance
    if euclidean_weight != 0 and transf is not None:
      # assure pts are np array
      pts1 = transf(np.array(pts1))
      pts2 = np.array(pts2)

      # compute points' pairwise distances
      sqr1 = np.sum(pts1 * pts1, axis=1, keepdims=True)
      sqr2 = np.sum(pts2 * pts2, axis=1)
      euclidean_D = sqr1 - 2 * np.matmul(pts1, pts2.T) + sqr2

      # add to overral keypoints distance
      D += euclidean_weight * euclidean_D

    # match descs2 to descs1
    indices1 = np.argsort(D.T)[:, :2]
    dists1 = np.sort(D.T)[:, :2]

    # match descs1 to descs2
    indices2 = np.argsort(D)[:, :2]
    dists2 = np.sort(D)[:, :2]

  # keep bidirectional corresponding points
  pairs = []
  len_descs1 = len(descs1)
  len_descs2 = len(descs2)
  for i in range(len_descs2):
    j = indices1[i, 0]
    if -1 < j < len_descs1 and indices2[j, 0] == i and \
      -1 < indices1[i, 1] < len_descs1 and -1 < indices2[j, 1] < len_descs2 \
        and dists1[i, 0] / dists1[i, 1] < thr and dists2[j, 0] / dists2[j, 1] < thr:
      pairs.append((j, i, dists1[i, 0]))

  return pairs


def matching(img1,
             pts1,
             img2,
             pts2,
             scale=4.0,
             thr=0.8,
             mode='basic',
             fast=True):
  pairs = find_correspondences(img1, pts1, img2, pts2, scale, thr, fast)

  if mode == 'basic':
    return len(pairs)


def main(img1_path, pts1_path, img2_path, pts2_path):
  # load images
  img1 = cv2.imread(img1_path, 0)
  img2 = cv2.imread(img2_path, 0)

  # load detection points
  pts1 = utils.load_dets_txt(pts1_path)
  pts2 = utils.load_dets_txt(pts2_path)

  print(matching(img1, pts1, img2, pts2))


if __name__ == '__main__':
  import sys

  main(*sys.argv[1:])
