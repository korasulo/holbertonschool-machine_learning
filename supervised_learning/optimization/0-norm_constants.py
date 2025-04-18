import numpy as np


def normalization_constants(X):
    return np.mean(X, axis=0), np.std(X, axis=0)
