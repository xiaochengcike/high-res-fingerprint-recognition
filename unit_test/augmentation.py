import cv2

import utils
import polyu


def minibatch_transformation(dataset):
  patches_pl, labels_pl = utils.placeholder_inputs()
  feed_dict = utils.fill_feed_dict(
      dataset.train, patches_pl, labels_pl, 36, augment=True)

  for patch in feed_dict[patches_pl]:
    cv2.imshow('patch', patch)
    cv2.waitKey(0)


if __name__ == '__main__':
  import sys

  print('Loading dataset...')
  dataset = polyu.description.Dataset(sys.argv[1])
  print('Done')

  minibatch_transformation(dataset)
