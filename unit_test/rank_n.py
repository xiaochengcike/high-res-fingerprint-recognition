import numpy as np
import matplotlib.pyplot as plt

import utils


def random_rank(size):
  # random data should give 1/n_labels
  # rank-1 and linear curve
  instances = np.random.random((size, 128))
  labels = np.random.randint(100, size=size)

  # get ranks and plot
  ranks = utils.rank_n(instances, labels, 100)
  print(ranks[0])
  plt.plot(ranks)
  plt.show()


def discernible_rank(size):
  # generate balanced labels
  labels = np.r_[:4]
  labels = np.repeat(labels, size // 4)

  # generate instances
  instances = []
  cov = np.diag(np.repeat(0.1, 2))
  for label in labels:
    # create mean of normal
    mean = np.zeros(2, dtype=np.float32)
    mean[label // 2] += (-1)**(1 + (label % 2))

    # create instance
    instance = np.random.multivariate_normal(mean, cov)
    instances.append(instance)
  instances = np.array(instances)

  # get ranks and plot
  ranks = utils.rank_n(instances, labels, 100)
  print(ranks[0])
  plt.plot(ranks)
  plt.show()


def perfect_rank(size):
  # generate balanced labels
  labels = np.r_[:4]
  labels = np.repeat(labels, size // 4)

  # generate instances
  instances = []
  for label in labels:
    # create instance
    instance = np.zeros(2, dtype=np.float32)
    instance[label // 2] += (-1)**(1 + (label % 2))
    instances.append(instance)
  instances = np.array(instances)

  # get ranks and plot
  ranks = utils.rank_n(instances, labels, 100)
  print(ranks[0])
  plt.plot(ranks)
  plt.show()


if __name__ == '__main__':
  size = 1000
  random_rank(size)
  discernible_rank(size)
  perfect_rank(size)
