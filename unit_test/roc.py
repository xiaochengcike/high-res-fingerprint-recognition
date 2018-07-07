import numpy as np
import matplotlib.pyplot as plt

import utils


def random_roc():
  # random data should give 0.5 eer
  # and random roc curve
  pos = np.random.random(1000)
  neg = np.random.random(1000)

  # compare eer versions
  print(utils.eer_(pos, neg))
  print(utils.eer(pos, neg))

  # plot curve
  fars, frrs = utils.roc(pos, neg)
  plt.plot(fars, frrs)
  plt.show()


def separable_roc():
  # separable data should give low
  # eer and convex roc curve
  pos = np.random.normal(1, 0.5, 1000)
  neg = np.random.normal(0, 0.5, 1000)

  # compare eer versions
  print(utils.eer_(pos, neg))
  print(utils.eer(pos, neg))

  # plot curve
  fars, frrs = utils.roc(pos, neg)
  plt.plot(fars, frrs)
  plt.show()


if __name__ == '__main__':
  random_roc()
  separable_roc()
