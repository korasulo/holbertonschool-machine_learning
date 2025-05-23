#!/usr/bin/env python3
"""
    Sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
        calculates the sensitivity for each class in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), matrix confusion

        :return: ndarray, shape(classes,) containing sensitivity of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize sensitivity
    sensitivity_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        # sum along the row
        total_positives = np.sum(confusion[i, :])

        sensitivity_matrix[i] = true_positive / total_positives

    return sensitivity_matrix
