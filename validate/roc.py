import argparse
import matplotlib.pyplot as plt

import utils

if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--files',
      required=True,
      type=str,
      nargs='+',
      help='files containing results to plot')
  parser.add_argument(
      '--xrange',
      default=[0, 0.1],
      type=int,
      nargs=2,
      help='range to plot x axis')
  parser.add_argument(
      '--yrange',
      default=[0, 0.1],
      type=int,
      nargs=2,
      help='range to plot y axis')

  flags = parser.parse_args()

  for path in flags.files:
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
  plt.axis(flags.xrange + flags.yrange)
  plt.grid()
  plt.show()
