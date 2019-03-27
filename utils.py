import numpy as np


def one_hot(n, k):
    """

    :param n: Index of the 1.
    :param k: Length of the one hot
    :return: A one-hot np array.
    """
    v = np.zeros(k)
    v[n] = 1
    return v
