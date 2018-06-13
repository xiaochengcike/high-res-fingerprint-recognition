from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import argparse
import os
import pickle

import polyu.recognition

if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='Path to PolyU-HRF dataset.')
  parser.add_argument(
      '--pts_dir_path',
      type=str,
      required=True,
      help='Path to PolyU-HRF DBI Training dataset keypoints detections.')
  parser.add_argument(
      '--patch_size',
      type=int,
      default=17,
      help='Image patch size for descriptor.')
  parser.add_argument(
      '--result_path',
      type=str,
      default='rec_dataset.pkl',
      help='Path to save precomputed recognition dataset.')
  FLAGS, _ = parser.parse_known_args()

  print('Creating dataset object instance...')
  imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
  dataset = polyu.recognition.Dataset(imgs_dir_path, FLAGS.pts_dir_path,
                                      FLAGS.patch_size)
  print('Done.')

  print('Saving dataset to path "{}"...'.format(FLAGS.result_path))
  with open(FLAGS.result_path, 'wb') as output:
    pickle.dump(dataset, output, -1)
  print('Done.')
