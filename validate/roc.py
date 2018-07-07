import matplotlib.pyplot as plt

import utils

if __name__ == '__main__':
  import sys

  # read comparisons file
  pos = []
  neg = []
  with open(sys.argv[1], 'r') as f:
    for line in f:
      t, score = line.split()
      score = float(score)
      if int(t) == 1:
        pos.append(score)
      else:
        neg.append(score)

  # compute roc
  fars, frrs = utils.roc(pos, neg)

  # plot roc
  plt.plot(fars, frrs)
  plt.xlabel('FAR')
  plt.ylabel('FRR')
  plt.axis([0, 1, 0, 1])
  plt.grid()
  plt.show()
