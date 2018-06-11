from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np

import align


class _Transf:
  def __init__(self, s, A, b):
    self._s = s
    self._A = A
    self._b = b

  def __call__(self, x):
    return self._s * np.dot(x, self._A.T) + self._b


def _find_alignments(all_imgs, all_pts):
  # find alignment between first and every other image
  img1 = all_imgs[0]
  transfs = []
  for i, img2 in enumerate(all_imgs[1:]):
    # find minimum mse alignment iteratively
    A, b, s = align.iterative(img1, all_pts[0], img2, all_pts[i + 1])

    # create mapping from img1 coordinates to img2 coordinates
    transfs.append(_Transf(s, A, b))

  return transfs


def _find_overlap(all_imgs, transfs):
  # find pixels present in all images
  img1 = all_imgs[0]
  overlap = np.ones(img1.shape, dtype=np.bool)
  for i, img2 in enumerate(all_imgs[1:]):
    # find pixels in both img1 and img2
    aligned = np.zeros_like(img1, dtype=np.bool)
    for ndindex in np.ndindex(img1.shape):
      row, col = transfs[i](ndindex)
      row = int(np.round(row))
      col = int(np.round(col))
      if 0 <= row < img2.shape[0]:
        if 0 <= col < img2.shape[1]:
          aligned[ndindex[0], ndindex[1]] = True

    # update overall overlap
    overlap = np.logical_and(overlap, aligned)

  return overlap


def _make_patch_mask(mask, patch_size):
  # make mask True only where a patch
  # can be taken in every image

  # for efficiency, this discards some
  # valid indices, those outside the
  # required circle but inside the
  # bounding box
  new_mask = np.zeros_like(mask, dtype=np.bool)
  half = patch_size // 2
  for i in range(mask.shape[0] - patch_size + 1):
    for j in range(mask.shape[1] - patch_size + 1):
      new_mask[i, j] = True
      for di in range(0, 3):
        for dj in range(0, 3):
          if not mask[i + di * half, j + dj * half]:
            new_mask[i, j] = False
            break
        if not new_mask[i, j]:
          break

  return new_mask


class Handler:
  def __init__(self, all_imgs, all_pts, patch_size):
    self._imgs = all_imgs
    self._patch_size = patch_size
    self._half = patch_size // 2

    self._transfs = _find_alignments(self._imgs, all_pts)

    # find valid area of overlap
    overlap = _find_overlap(self._imgs, self._transfs)
    self._mask = _make_patch_mask(overlap, self._patch_size)

    # get valid indices and store them for access
    self._inds = np.argwhere(self._mask)

  def __getitem__(self, val):
    if isinstance(val, slice):
      samples = []
      for i, j in self._inds[val]:
        # add first image patch
        sample = [
            self._imgs[0][i:i + self._patch_size, j:j + self._patch_size]
        ]

        # add remaining image patches
        for k, img in enumerate(self._imgs[1:]):
          # find transformed coordinates of patch
          ti, tj = self._transfs[k]((i + self._half, j + self._half))

          # convert them to int
          ti = int(np.round(ti))
          tj = int(np.round(tj))

          # add to index overall
          sample.append(img[ti - self._half:ti + self._half + 1,
                            tj - self._half:tj + self._half + 1])

        # add to overall
        samples.append(sample)

      return np.array(samples)
    else:
      # retrieve coordinates for given index
      i, j = self._inds[val]

      # add first image patch
      samples = [self._imgs[0][i:i + self._patch_size, j:j + self._patch_size]]

      # add remaining image patches
      for k, img in enumerate(self._imgs[1:]):
        # find transformed coordinates of patch
        ti, tj = self._transfs[k]((i + self._half, j + self._half))

        # convert them to int
        ti = int(np.round(ti))
        tj = int(np.round(tj))

        # add to overall
        samples.append(img[ti - self._half:ti + self._half + 1,
                           tj - self._half:tj + self._half + 1])

      return samples

  def __len__(self):
    return len(self._inds)


if __name__ == '__main__':
  import utils
  import cv2
  import sys

  imgs_path = '../datasets/polyu_hrf/DBI/Training/' + sys.argv[1]
  pts_path = 'src/Pores_Dahia/' + sys.argv[1]

  imgs_ = []
  pts_ = []
  for i in range(1, 3):
    for j in range(1, 4):
      img_path = '{}_{}_{}.jpg'.format(imgs_path, i, j)
      img_ = cv2.imread(img_path, 0)
      imgs_.append(img_)

      img_pts_path = '{}_{}_{}.txt'.format(pts_path, i, j)
      img_pts = utils.load_dets_txt(img_pts_path)
      pts_.append(img_pts)

  d = Handler(imgs_, pts_, 17)
  for sample in d:
    for i, patch in enumerate(sample):
      cv2.imshow('patch' + str(i), patch)
    cv2.waitKey(0)
  mask = 255 * np.array(d._mask, dtype=np.uint8)
  cv2.imshow('mask', mask)
  cv2.waitKey(0)
