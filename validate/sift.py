import argparse

import polyu
import utils
from validate.matching import validation_eer

FLAGS = None


def main():
  print('Loading dataset...')
  dataset = polyu.description.Dataset(FLAGS.dataset_path).val
  print('Done')

  # compute eer
  print('EER = {}'.format(validation_eer(dataset, utils.sift_descriptors)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path', required=True, type=str, help='path to dataset')
  FLAGS = parser.parse_args()

  main()
