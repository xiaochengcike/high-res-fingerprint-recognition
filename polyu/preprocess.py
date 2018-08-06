import argparse
import os
import numpy as np
import tensorflow as tf
import cv2

import utils
from polyu import aligned_images

FLAGS = None


def load_detections(path):
  pts = []
  for pts_path in sorted(os.listdir(path)):
    if pts_path.endswith('.txt'):
      pts.append(utils.load_dets_txt(os.path.join(path, pts_path)))

  return pts


def group_by_label(imgs, pts, labels):
  grouped_imgs = []
  grouped_pts = []
  unique_labels = np.unique(labels)
  for label in unique_labels:
    indices = np.where(labels == label)
    grouped_imgs.append(imgs[indices])
    grouped_pts.append(pts[indices])
  labels = unique_labels

  return grouped_imgs, grouped_pts, labels


def create_dirs(path, should_create_val):
  # create train path
  train_path = os.path.join(path, 'train')
  if not os.path.exists(train_path):
    os.makedirs(train_path)

  # create val path
  val_path = os.path.join(path, 'val')
  if not os.path.exists(val_path) and should_create_val:
    os.makedirs(val_path)

  return train_path, val_path


def train_val_split(imgs, pts, labels, names, total_imgs, split):
  # build training set
  perm = np.random.permutation(len(imgs))
  train_imgs = []
  train_pts = []
  train_labels = []
  train_size = 0
  i = 0
  while train_size < split * total_imgs:
    # add 'perm[i]'th-element to train set
    train_size += len(imgs[perm[i]])
    train_imgs.append(imgs[perm[i]])
    train_pts.append(pts[perm[i]])
    train_labels.append(labels[perm[i]])
    i += 1

  # build validation set
  val_imgs = []
  val_pts = []
  val_labels = []
  val_names = []
  for j in perm[i:]:
    val_imgs.append(imgs[j])
    val_pts.append(pts[j])
    val_labels.append(labels[j])
    val_names.append(names[j])

  # assert that both sets do not have
  # overlap in subjects identities
  assert not set(train_labels).intersection(val_labels)

  train = (train_imgs, train_pts)
  val = (val_imgs, val_pts, val_names)

  return train, val


def save_patches(grouped_imgs, grouped_pts, path, patch_size, flip, for_sift):
  # 'patch_index' is a unique patch identifier
  patch_index = 1
  for imgs, pts in zip(grouped_imgs, grouped_pts):
    # align images
    handler = aligned_images.Handler(imgs, pts, patch_size, flip, for_sift)

    # extract patches
    for patches in handler:
      for i, patch in enumerate(patches):
        patch_path = os.path.join(path, '{}_{}.png'.format(patch_index, i + 1))
        cv2.imwrite(patch_path, 255 * patch)
      patch_index += 1


def save_dataset(grouped_imgs, grouped_pts, grouped_names, path):
  # save images with name "subject_session_identifier"
  for imgs, all_pts, names in zip(grouped_imgs, grouped_pts, grouped_names):
    for img, pts, name in zip(imgs, all_pts, names):
      # save image
      img_path = os.path.join(path, name + '.png')
      cv2.imwrite(img_path, 255 * img)

      # save detections
      pts_path = os.path.join(path, name + '.txt')
      utils.save_dets_txt(pts, pts_path)


def main():
  # load detections
  print('Loading detections...')
  pts = load_detections(FLAGS.pts_dir_path)
  print('Done')

  # load images with names and retrieve labels
  print('Loading images...')
  imgs_dir_path = os.path.join(FLAGS.polyu_dir_path, 'DBI', 'Training')
  imgs, names = utils.load_images_with_names(imgs_dir_path)
  name2label = utils.retrieve_label_from_image_path
  labels = [name2label(name) for name in names]
  print('Done')

  # convert to np array
  imgs = np.array(imgs)
  pts = np.array(pts)
  labels = np.array(labels)

  # group (imgs, pts, names) by label
  print('Grouping (images, detections) by label...')
  total_imgs = len(imgs)
  imgs, pts, labels = group_by_label(imgs, pts, labels)
  names = [[name for name in names if name2label(name) == label]
           for label in labels]
  print('Done')

  # create 'train' & 'val' folders
  print('Creating directory tree...')
  should_create_val = FLAGS.split < 1
  train_path, val_path = create_dirs(FLAGS.result_dir_path, should_create_val)
  print('Done')

  # split dataset into train/val
  print('Splitting dataset...')
  train, val = train_val_split(imgs, pts, labels, names, total_imgs,
                               FLAGS.split)
  train_imgs, train_pts = train
  val_imgs, val_pts, val_names = val
  print('Done')

  # extract and save all patches from train images
  print('Creating training set patches...')
  save_patches(train_imgs, train_pts, train_path, FLAGS.patch_size, FLAGS.flip,
               FLAGS.for_sift)
  print('Done')

  # save validation images
  print('Saving validation images...')
  save_dataset(val_imgs, val_pts, val_names, val_path)
  print('Done')


if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='path to PolyU-HRF dataset')
  parser.add_argument(
      '--pts_dir_path',
      type=str,
      required=True,
      help='path to PolyU-HRF DBI Training dataset keypoints detections')
  parser.add_argument(
      '--patch_size',
      type=int,
      required=True,
      help='image patch size for descriptor')
  parser.add_argument(
      '--result_dir_path',
      type=str,
      required=True,
      help='path to save description dataset')
  parser.add_argument(
      '--flip',
      action='store_true',
      help=
      'use this flag to also generate flipped patches, doubling the description dataset size'
  )
  parser.add_argument(
      '--split',
      default='0.5',
      type=float,
      help='floating point percentage of training set in train/val split')
  parser.add_argument(
      '--for_sift',
      action='store_true',
      help='use this flag to generate patches fit for validate.sift use')
  parser.add_argument('--seed', type=int, help='random seed')

  FLAGS = parser.parse_args()

  # set random seeds
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  main()
