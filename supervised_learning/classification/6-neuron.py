#!/usr/bin/env python3
"""
    Class Neuron
"""

import numpy as np


class Neuron:
    """
        Class Neuron : define single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter method for the private instance attribute
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the private instance attribute
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the private instance attribute
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-Z))
        return self.__A

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
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        # percaktimi klases nfund nqs me 1 do jete po dhe me
        # 0 do jete jo, nqs mshum se 0.5 esht 1
        result = np.where(A >= 0.5, 1, 0)
        # result do jet predictioni 1 ose 0 ndersa cost numri cost
        return result, cost

    # gabimi total shperndaje tek te gjithe faktoret
    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        grad_w = (1 / m) * np.matmul((A - Y), X.T)  # formulat
        grad_b = (1 / m) * np.sum(A - Y)

        # pesha e re = pesha qe une kisha - ndryshimi,
        # ( vetem se ndryshimi tani do merret me rezerve alpha)
        self.__W = self.__W - alpha * grad_w
        self.__b = self.__b - alpha * grad_b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        """
        for i in range(iterations):

            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            result, cost = self.evaluate(X, Y)

            return result, cost
