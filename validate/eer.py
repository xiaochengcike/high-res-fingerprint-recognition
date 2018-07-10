import utils

if __name__ == '__main__':
  import sys

  # read comparisons file
  for path in sys.argv[1:]:
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

    print(path, utils.eer(pos, neg))
