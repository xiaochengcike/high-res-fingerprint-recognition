import matplotlib.pyplot as plt

import utils

if __name__ == '__main__':
  import sys

  for path in sys.argv[1:]:
    # read comparisons file
    pos = []
    neg = []
    with open(path, 'r') as f:
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
    plt.plot(fars, frrs, label=path)

  plt.legend(loc='upper right')
  plt.xlabel('FAR')
  plt.ylabel('FRR')
  plt.axis([0, 0.1, 0, 0.1])
  plt.grid()
  plt.show()
