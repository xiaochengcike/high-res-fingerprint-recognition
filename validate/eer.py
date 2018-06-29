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

  print(utils.eer(pos, neg))
