import numpy as np
import os

import utils


def validate(pores_by_image, detections_by_image):
  # find correspondences between detections and pores
  total_pores = 0
  total_dets = 0
  true_dets = 0
  for i, pores in enumerate(pores_by_image):
    dets = detections_by_image[i]

    # update totals
    total_pores += len(pores)
    total_dets += len(dets)
    true_dets += len(utils.find_correspondences(pores, dets))

  # compute tdr, fdr and f score
  eps = 1e-12
  tdr = true_dets / (total_pores + eps)
  fdr = (total_dets - true_dets) / (total_dets + eps)
  f_score = 2 * (tdr * (1 - fdr)) / (tdr + 1 - fdr)

  print('TDR = {}'.format(tdr))
  print('FDR = {}'.format(fdr))
  print('F score = {}'.format(f_score))


def load_keypoints_from_txt(txt_folder_path):
  keypoints_by_image = []
  for txt_path in sorted(os.listdir(txt_folder_path)):
    if txt_path.endswith('.txt'):
      keypoints = []
      with open(os.path.join(txt_folder_path, txt_path)) as f:
        for line in f:
          x, y = [int(j) for j in line.split()]
          keypoints.append((x, y))
      keypoints_by_image.append(np.array(keypoints))

  return keypoints_by_image


def main(pores_txt_folder_path, dets_txt_folder_path):
  pores = load_keypoints_from_txt(pores_txt_folder_path)
  dets = load_keypoints_from_txt(dets_txt_folder_path)
  validate(pores, dets)


if __name__ == '__main__':
  import sys
  if len(sys.argv) < 3:
    print('Insufficient arguments')

  main(sys.argv[1], sys.argv[2])
