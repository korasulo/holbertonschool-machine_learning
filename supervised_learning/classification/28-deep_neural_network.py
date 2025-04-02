#!/usr/bin/env python3
'''
Module that defines a deep neural network
performing binary classification
'''
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork():
    '''Class that defines a deep neural network'''
    def __init__(self, nx, layers, activation='sig'):
        '''Class constructur'''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for x in range(self.L):
            if layers[x] < 1 or type(layers[x]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(x + 1)] = np.zeros((layers[x], 1))
            if x == 0:
                He_et_0 = np.random.randn(layers[x], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(x + 1)] = He_et_0
            if x > 0:
                He_et_1 = np.random.randn(layers[x], layers[x - 1])
                He_et_2 = np.sqrt(2 / layers[x - 1])
                He_et_3 = He_et_1 * He_et_2
                self.__weights["W" + str(x + 1)] = He_et_3

    @property
    def L(self):
        '''L getter function'''
        return self.__L

    @property
    def weights(self):
        '''weight getter function'''
        return self.__weights

    @property
    def cache(self):
        '''cache getter function'''
        return self.__cache

    @property
    def activation(self):
        '''activation getter function'''
        return self.__activation

    def forward_prop(self, X):
        '''
        Calculates the forward propagation
        of the neural network
        '''
        m = self.__L
        self.__cache['A0'] = X
        for i in range(m):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            a = 'A' + str(i + 1)
            a_0 = 'A' + str(i)
            out_Z = np.matmul(self.__weights[w],
                              self.__cache[a_0]) + self.weights[b]
            if i == self.__L - 1:
                e = np.exp(out_Z)
                out_A = np.exp(out_Z) / np.sum(e, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    out_A = 1 / (1 + np.exp(-out_Z))
                if self.__activation == 'tanh':
                    out_A = ((np.exp(out_Z) - np.exp(-out_Z))
                             / (np.exp(out_Z) + np.exp(-out_Z)))
            self.__cache[a] = out_A
        return out_A, self.__cache

    def cost(self, Y, A):
        '''
        Calculates the cost of the model
        using logistic regression
        '''
        m = - (1 / Y.shape[1])
        uno = Y * np.log(A)
        dos = np.sum(uno)
        C = m * dos
        return C

    def evaluate(self, X, Y):
        '''
        Evaluates the neural networks predictions
        '''
        a, _ = self.forward_prop(X)
        P = np.where(a == np.amax(a, axis=0), 1, 0)
        return P, self.cost(Y, a)

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''
        Calculates one pass of gradient
        descent on the neural network
        '''
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            a = 'A' + str(i + 1)
            a_0 = 'A' + str(i)
            A = cache[a]
            A_0 = cache[a_0]
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[w]
            else:
                if self.__activation == 'sig':
                    da = A * (1 - A)
                if self.__activation == 'tanh':
                    da = 1 - (A * A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[w]
            dw = np.matmul(A_0, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[w] = self.__weights[w] - alpha * dw.T
            self.__weights[b] = self.__weights[b] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        '''
        Trains the deep neural network
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError(
                'iterations must be a positive integer'
                )
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        iteration = []
        c = []
        for x in range(iterations + 1):
            self.forward_prop(X)
            a, cost = self.evaluate(X, Y)
            if x % step == 0:
                iteration.append(x)
                c.append(cost)
                if verbose:
                    print('Cost after {} iterations: {}'.format(x, cost))
            if x < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(iteration, c, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        '''
        Saves the instance object to a file in pickle format
        '''
        import pickle
        if filename[-4:] != '.pkl':
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        '''
        Loads a pickled DeepNeuralNetwork object
        '''
        import pickle
        try:
            with open(filename, 'rb') as f:
                fOpen = pickle.load(f)
            return fOpen
        except FileNotFoundError:
            return None
