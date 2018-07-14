import numpy as np

import utils


def random_recovery():
  # get random instances
  instances1 = np.random.random((100, 32))
  instances2 = np.random.random((100, 32))

  # find correspondences
  pairs = utils.find_correspondences(instances1, instances2)

  # check uniqueness
  seen_indices1 = set()
  seen_indices2 = set()
  for i, j, _ in pairs:
    if i in seen_indices1 or j in seen_indices2:
      return False

    seen_indices1.add(i)
    seen_indices2.add(j)

  return True


def perfect_recovery():
  # every instance has a single perfect match
  instances = []
  for i in range(32):
    instance = np.zeros(32, dtype=np.float32)
    instance[i] = 1
    instances.append(instance)

  # find correspondences
  instances = np.array(instances)
  pairs = utils.find_correspondences(instances, instances)

  # check correctness
  for i, j, d in pairs:
    if i != j or d != 0:
      return False

  return True


def zero_recovery():
  # every instance in one side has the same match
  # in the other side, leading to a single pair
  instances1 = []
  for i in range(32):
    instance = np.zeros(32, dtype=np.float32)
    instance[i] = 1
    instances1.append(instance)

  instances2 = [instance + 1 for instance in instances1]
  instances2.append(np.zeros(32, dtype=np.float32))

  # find correspondences
  instances1 = np.array(instances1)
  instances2 = np.array(instances2)
  pairs = utils.find_correspondences(instances1, instances2)

  if len(pairs) != 1:
    return False
  if pairs[0][1] != len(instances2) - 1:
    return False
  if pairs[0][2] != 1:
    return False

  return True


if __name__ == '__main__':
  assert random_recovery()
  print('[OK - Random Recovery]')

  assert perfect_recovery()
  print('[OK - Perfect Recovery]')

  assert zero_recovery()
  print('[OK - Zero Recovery]')
