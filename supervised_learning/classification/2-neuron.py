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
        Z=np.matmul(self.__W, X)+ self.__b
        self.__A=1/(1+np.exp(-Z))
        return self.__A