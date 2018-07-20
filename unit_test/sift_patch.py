import numpy as np
import cv2

import utils


def sift_patch(img_path, pts_path, scale=4):
  # load image
  img = cv2.imread(img_path, 0)

  # load pts
  pts = utils.load_dets_txt(pts_path)

  # find closes pore to center
  center = np.array(img.shape) / 2
  sqr_dist = np.sum((pts - center)**2, axis=1)
  closest_ind = np.argmin(sqr_dist)
  pt = pts[closest_ind][::-1]

  # improve image quality with median blur and clahe
  img = cv2.medianBlur(img, ksize=3)
  clahe = cv2.createCLAHE(clipLimit=3)
  img = clahe.apply(img)

  # extract original descriptor
  kpt = cv2.KeyPoint.convert([pt], size=scale)
  sift = cv2.xfeatures2d.SIFT_create()
  _, original = sift.compute(img, kpt)

  # test patch sizes
  for i in range(3, 113):
    # extract patch centered in 'pt'
    patch = img[pt[1] - i:pt[0] + i + 1, pt[0] - i:pt[0] + i + 1]

    # extract patch keypoint
    patch_kpt = cv2.KeyPoint.convert([(i, i)], size=scale)
    _, patched = sift.compute(patch, patch_kpt)

    if np.isclose(np.linalg.norm(original - patched), 0):
      return i

  return -1


if __name__ == '__main__':
  import sys
  patch_size = sift_patch(sys.argv[1], sys.argv[2])
  assert patch_size > 0
  print('[OK - Sift Patch ({})]'.format(2 * patch_size + 1))
