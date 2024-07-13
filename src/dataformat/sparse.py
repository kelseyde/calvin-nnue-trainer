import numpy as np


def to_sparse(arr):
    # Get the indices of the 1s in the array
    indices = np.where(arr == 1)[0]
    return indices


def to_dense(indices, length=768):
    # Create an array of zeros
    arr = np.zeros(length, dtype=int)
    # Set the indices to 1
    arr[indices] = 1
    return arr
