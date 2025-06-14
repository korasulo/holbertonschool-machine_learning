#!/usr/bin/env python3
'''
Module that creates a TensorFlow 2.x layer with L2 regularization
'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''
    Creates a Dense layer with L2 regularization.

    Parameters
    ----------
    prev : tf.Tensor
        Output of the previous layer.
    n : int
        Number of nodes in the layer.
    activation : callable
        Activation function to use (e.g., tf.nn.relu, tf.nn.sigmoid).
    lambtha : float
        L2 regularization parameter.

    Returns
    -------
    tf.Tensor
        Output of the new layer.
    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    l2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=l2
    )(prev)
    return layer
