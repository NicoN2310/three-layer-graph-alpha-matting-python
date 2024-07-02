import numpy as np
import scipy.ndimage as ndimage


def filter_image_3d(I):
    h = fspecial_gauss(9, 0.5)

    conv_r = ndimage.correlate(I[:, :, 0], h, mode="nearest")
    conv_g = ndimage.correlate(I[:, :, 1], h, mode="nearest")
    conv_b = ndimage.correlate(I[:, :, 2], h, mode="nearest")

    return np.stack((conv_r, conv_g, conv_b), axis=-1)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()
