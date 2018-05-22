from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
import cv2

import align
from pamplona_lemes_matching import extract_descriptors
import util


def match(img1, pts1, img2, pts2, min_overlap=20):
  try:
    A, b, s = align.iterative(img1, pts1, img2, pts2)
  except:
    return 0

  # find keypoints' correspondences
  pts1 = np.array(pts1)
  pts1_in_img2 = s * np.dot(pts1, A.T) + b
  pts2 = np.array(pts2)
  pts2_in_img1 = np.dot(pts2 - b, A) / s

  # filter keypoints outside target image
  is_inside = lambda pt, img: 0 <= pt[0] < img.shape[0] and 0 <= pt[1] < img.shape[1]

  tmp_pts1 = []
  tmp_pts1_in_img2 = []
  for i, pt in enumerate(pts1_in_img2):
    if is_inside(pt, img2):
      tmp_pts1.append(pts1[i])
      tmp_pts1_in_img2.append(pt)
  pts1 = tmp_pts1
  pts1_in_img2 = tmp_pts1_in_img2

  tmp_pts2 = []
  tmp_pts2_in_img1 = []
  for i, pt in enumerate(pts2_in_img1):
    if is_inside(pt, img1):
      tmp_pts2.append(pts2[i])
      tmp_pts2_in_img1.append(pt)
  pts2 = tmp_pts2
  pts2_in_img1 = tmp_pts2_in_img1

  # discard images with small overlap
  if len(pts1) < min_overlap or len(pts2) < min_overlap:
    return 0

  # extract corresponding descriptors
  descs1 = extract_descriptors(img1, pts1, scale=8.0)
  descs1_in_img2 = extract_descriptors(img2, pts1_in_img2, scale=8.0)
  descs2 = extract_descriptors(img2, pts2, scale=8.0)
  descs2_in_img1 = extract_descriptors(img1, pts2_in_img1, scale=8.0)

  # compute distances
  dists1 = np.mean((descs1 - descs1_in_img2)**2)
  dists2 = np.mean((descs2 - descs2_in_img1)**2)

  return 1 / (dists1 + dists2)


def main(img1_path, pts1_path, img2_path, pts2_path):
  # load images
  img1 = cv2.imread(img1_path, 0)
  img2 = cv2.imread(img2_path, 0)

  # load detection points
  pts1 = util.load_dets_txt(pts1_path)
  pts2 = util.load_dets_txt(pts2_path)

  print(match(img1, pts1, img2, pts2))


if __name__ == '__main__':
  import sys

  main(*sys.argv[1:])
