import numpy as np

import utils


def _close(D1, D2):
  return np.any(np.isclose(D1, D2))


def _naive_pairwise_distances(x1, x2):
  D = []
  for x in x1:
    row = []
    for y in x2:
      dist = np.sum((x - y)**2)
      row.append(dist)
    D.append(row)
  return np.array(D)


def random_vectors():
  # random vectors
  x1 = np.random.random((100, 32))
  x2 = np.random.random((100, 32))

  # compute pairwise distances
  D1 = utils.pairwise_distances(x1, x2)
  D2 = _naive_pairwise_distances(x1, x2)

  # check shape
  if D1.shape != D2.shape:
    return False

  # compare naive approach with employed
  if not _close(D1, D2):
    return False

  # check if it is non-negative
  if not _close(D1, np.abs(D1)):
    return False

  return True


def orthogonal_vectors():
  # get set of random vectors
  x1 = np.random.random((100, 32))
  norm_x1 = [x / np.linalg.norm(x) for x in x1]

  # get set of vectors orthogonal to x1
  x2 = np.random.random((100, 32))
  x2 = [x - np.dot(x, y) * y for x, y in zip(x2, norm_x1)]

  # compute pairwise distances
  x1 = np.array(x1)
  x2 = np.array(x2)
  D1 = utils.pairwise_distances(x1, x2)
  D2 = _naive_pairwise_distances(x1, x2)

  # check shape
  if D1.shape != D2.shape:
    return False

  # compare naive approach with employed
  if not _close(D1, D2):
    return False

  # check if it is non-negative
  if not _close(D1, np.abs(D1)):
    return False

  # check if distance is close to sum of squares of magnitudes
  for i, x in enumerate(x1):
    y = x2[i]
    dist = np.sum(x**2) + np.sum(y**2)
    if not _close(dist, D1[i, i]):
      return False

  return True


def unitary_vectors():
  # get set of random unitary vectors
  x1 = np.random.random((100, 32))
  x1 = [x / np.linalg.norm(x) for x in x1]

  # get origin vector
  x2 = np.zeros((1, 32))

  # compute pairwise distances
  x1 = np.array(x1)
  x2 = np.array(x2)
  D1 = utils.pairwise_distances(x1, x2)
  D2 = _naive_pairwise_distances(x1, x2)

  # check shape
  if D1.shape != D2.shape:
    return False

  # compare naive approach with employed
  if not _close(D1, D2):
    return False

  # check if it is non-negative
  if not _close(D1, np.abs(D1)):
    return False

  # check if distance is close to 1
  for row in D1:
    for d in row:
      if not _close(d, 1):
        return False

  return True


def same_vectors():
  # single set of random vectors
  x1 = np.random.random((100, 32))

  # compute pairwise distances
  D1 = utils.pairwise_distances(x1, x1)
  D2 = _naive_pairwise_distances(x1, x1)

  # check shape
  if D1.shape != D2.shape:
    return False

  # compare naive approach with employed
  if not _close(D1, D2):
    return False

  # check if it is non-negative
  if not _close(D1, np.abs(D1)):
    return False

  # check if main diagonal is zero
  for d in np.diag(D1):
    if not _close(d, 0):
      return False

  # check if it is symmetrical
  for i, row in enumerate(D1):
    for j, d in enumerate(row):
      if not _close(d, D1[j, i]):
        return False

  return True


if __name__ == '__main__':
  assert random_vectors()
  print('[OK - Random Vectors]')

  assert orthogonal_vectors()
  print('[OK - Orthogonal Vectors]')

  assert unitary_vectors()
  print('[OK - Unitary Vectors]')

  assert same_vectors()
  print('[OK - Same Vectors]')
