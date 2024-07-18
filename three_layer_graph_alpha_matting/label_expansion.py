import numpy as np
import scipy.ndimage as ndimage
from numba import njit


def label_expansion(img, cdata):
    I = img.astype(np.float64)
    trimap = cdata.copy().astype(np.float64)

    grayI = _rgb2gray(I)

    J = _window_stdev(grayI.astype(np.float64), 7)
    J = _window_min_stride_tricks(J, 4)

    # Calculate Geodesic Distance
    M = (trimap < 255).astype(np.float64)
    DF = get_geodesic_distance(I, J, M)

    M = (trimap > 0).astype(np.float64)
    DB = get_geodesic_distance(I, J, M)

    # Calculate trimap output
    MF = cdata == 255
    MB = cdata == 0

    for k in range(1, 101):
        SE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        MF = ndimage.binary_dilation(MF, structure=SE)
        if np.any(MF & MB):
            break

    threshold = min((k - 1) / 2, 20)

    EF = DF < threshold
    trimap[EF] = 255

    EB = DB < threshold
    trimap[EB] = 0

    return trimap.astype(np.uint8)


def _rgb2gray(rgb):
    weights = np.array([0.29894, 0.58704, 0.11402])
    result = np.dot(rgb[..., :3], weights)
    result = _custom_round(result)
    result = result.astype(np.uint8)
    return result

def _custom_round(x):
    rounded = np.round(x)
    temp = rounded + ((x - rounded) == 0.5) * (rounded % 2)
    return temp


def _window_stdev(X, window_size):
    c1 = ndimage.uniform_filter(X, window_size, mode='reflect')
    c2 = ndimage.uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1) * np.sqrt(window_size**2 / (window_size**2 - 1))

def _window_min_stride_tricks(image, r, mode="constant", constant_values=0):
    # O(h w r) algorithmic complexity
    k = 2 * r + 1
    image_padded_x = np.pad(image, ((0, 0), (r, r)), mode=mode, constant_values=constant_values)
    windows_x = np.lib.stride_tricks.sliding_window_view(image_padded_x, (1, k))
    image_min_x = windows_x.min(axis=(2, 3))
    image_padded_y = np.pad(image_min_x, ((r, r), (0, 0)), mode=mode, constant_values=constant_values)
    windows_y = np.lib.stride_tricks.sliding_window_view(image_padded_y, (k, 1))
    image_min = windows_y.min(axis=(2, 3))
    return image_min


def get_geodesic_distance(I, J, M):
    m, n = J.shape
    M = M.flatten(order="F")
    v = 10_000
    D = M * v

    I = I.copy().flatten(order="F")
    D = D.copy().flatten(order="F")
    J = J.copy().flatten(order="F")

    return raster_scan(I, D, J, m, n).reshape(m, n, order="F")


@njit
def raster_scan(I, D, J, m, n):
    shape = D.shape
    Ak = np.array(
        [
            [-1, -1, -1, 0, -1, 1, 0, -1],
            [1, 1, 1, 0, 1, -1, 0, 1],
            [1, -1, 0, -1, -1, -1, 1, 0],
            [-1, 1, 0, 1, 1, 1, -1, 0],
        ]
    )

    ak = Ak[0]
    for j in range(n):
        for i in range(m):
            D[i + j * m] = find_min_d(I, D, J, m, n, ak, i, j)

    ak = Ak[1]
    for j in range(n - 1, -1, -1):
        for i in range(m - 1, -1, -1):
            D[i + j * m] = find_min_d(I, D, J, m, n, ak, i, j)

    ak = Ak[2]
    for i in range(m):
        for j in range(n - 1, -1, -1):
            D[i + j * m] = find_min_d(I, D, J, m, n, ak, i, j)

    ak = Ak[3]
    for i in range(m - 1, -1, -1):
        for j in range(n):
            D[i + j * m] = find_min_d(I, D, J, m, n, ak, i, j)

    return D.reshape(shape)


@njit
def find_min_d(I, D, J, m, n, ak, i, j):
    N = m * n
    min_d = D[i + j * m]

    for k in range(1, 5):
        x = j + ak[2 * k - 2]
        y = i + ak[2 * k - 1]

        if y < 0 or x < 0 or y >= m or x >= n:
            continue
        else:
            gradient = (
                (I[i + j * m] - I[y + x * m]) ** 2
                + (I[i + j * m + N] - I[y + x * m + N]) ** 2
                + (I[i + j * m + 2 * N] - I[y + x * m + 2 * N]) ** 2
            )
            gamma = 1 / (J[i + j * m] + 0.01)
            temp = (
                D[y + x * m]
                + ((ak[2 * k - 2] ** 2) + (ak[2 * k - 1] ** 2) + (gamma**2) * gradient)
                ** 0.5
            )

            if temp < min_d:
                min_d = temp

    return min_d
