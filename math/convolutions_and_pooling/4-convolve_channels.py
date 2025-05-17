#!/usr/bin/env python3
'''
Modulus that has a function that performs valid convolution grayscale images
'''
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    '''
    Function that performs a valid convolution grayscale images:
        images: np.ndarray. images to be convoluted
        kernel. np.ndarray. filter to be used
        padding. tuple with paddin height and weight or same or valid
        stride. tuple, steps at the filter is moving
    '''
    m, h, w, c = images.shape
    kh, kw,  _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    elif padding == 'valid':
        ph, pw = (0, 0)

    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding

    padded_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw),
                                 (0, 0)), 'constant')

    ch = int(((h + 2 * ph - kh) / sh) + 1)
    cw = int(((w + 2 * pw - kw) / sw) + 1)
    conv_dim = (m, ch, cw)
    conv = np.zeros(conv_dim)

    for i in range(conv_dim[1]):
        for j in range(conv_dim[2]):
            image_slice = padded_img[:,
                                     i * sh:i * sh + kh,
                                     j * sw:j * sw + kw]
            conv[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))
    return conv
