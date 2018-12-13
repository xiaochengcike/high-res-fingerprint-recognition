import tensorflow as tf

import utils
import matching
from models import detection, description

FLAGS = None


def detect_pores(imgs):
  with tf.Graph().as_default():
    # placeholder for image
    image_pl, _ = utils.placeholder_inputs()

    # build detection net
    print('Building detection net graph...')
    det_net = detection.Net(image_pl, training=False)
    print('Done')

    with tf.Session() as sess:
      print('Restoring detection model in {}...'.format(FLAGS.det_model_dir))
      utils.restore_model(sess, FLAGS.det_model_dir)
      print('Done')

      # capture detection arguments in function
      def single_detect_pores(image):
        return utils.detect_pores(
            image, image_pl, det_net.predictions, FLAGS.det_patch_size // 2,
            FLAGS.det_prob_thr, FLAGS.nms_inter_thr, sess)

      # detect pores
      dets = [single_detect_pores(img) for img in imgs]

  return dets


def describe_detections(imgs, dets):
  with tf.Graph().as_default():
    # placeholder for image
    image_pl, _ = utils.placeholder_inputs()

    # build description net
    print('Building description net graph...')
    desc_net = description.Net(image_pl, training=False)
    print('Done')

    with tf.Session() as sess:
      print('Restoring description model in {}...'.format(
          FLAGS.desc_model_dir))
      utils.restore_model(sess, FLAGS.desc_model_dir)
      print('Done')

      # capture description arguments in function
      def compute_descriptors(image, dets):
        return utils.trained_descriptors(image, dets, FLAGS.desc_patch_size,
                                         sess, image_pl, desc_net.descriptors)

      # compute descriptors
      descs = [
          compute_descriptors(img, img_dets)
          for img, img_dets in zip(imgs, dets)
      ]

  return descs


def main():
  # load images
  imgs = [utils.load_image(path) for path in FLAGS.img_paths]

  dets = detect_pores(imgs)

  tf.reset_default_graph()

  descs = describe_detections(imgs, dets)

  score = matching.basic(descs[0], descs[1], dets[0], dets[1], thr=0.7)
  print('similarity score = {}'.format(score))
  if score > FLAGS.score_thr:
    print('genuine pair')
  else:
    print('impostor pair')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--img_paths',
      required=True,
      type=str,
      nargs=2,
      help='path to images to be recognized')
  parser.add_argument(
      '--det_model_dir',
      required=True,
      type=str,
      help='path to pore detection trained model')
  parser.add_argument(
      '--desc_model_dir',
      required=True,
      type=str,
      help='path to pore description trained model')
  parser.add_argument(
      '--score_thr',
      default=3,
      type=int,
      help='score threshold to determine if pair is genuine or impostor')
  parser.add_argument(
      '--det_patch_size', default=17, type=int, help='detection patch size')
  parser.add_argument(
      '--det_prob_thr',
      default=0.9,
      type=float,
      help='probability threshold for discarding detections')
  parser.add_argument(
      '--nms_inter_thr',
      default=0.1,
      type=float,
      help='NMS area intersection threshold')
  parser.add_argument(
      '--desc_patch_size',
      default=32,
      type=int,
      help='patch size around each detected keypoint to describe')

  FLAGS = parser.parse_args()

  main()
