#!/usr/bin/env python3
"""
    Class Neuron
"""

import numpy as np


class NeuralNetwork:
    """
        Class NeuralNetwork : define a neural
        network performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))  # zhvillon 1 vektor me 0
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1/(1 + np.exp(-Z1))

        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1/(1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the neural model
        """
        m = Y.shape[1]   # get nr of examples/images
        log_loss = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                        * np.log(1.0000001 - A))
        return log_loss

    def evaluate(self, X, Y):
        """
        evaluate the neuron's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        # percaktimi klases nfund nqs me 1 do jete po dhe me
        # 0 do jete jo, nqs mshum se 0.5 esht 1
        result = np.where(A2 >= 0.5, 1, 0)
        # result do jet predictioni 1 ose 0 ndersa cost numri cost
        return result, cost
