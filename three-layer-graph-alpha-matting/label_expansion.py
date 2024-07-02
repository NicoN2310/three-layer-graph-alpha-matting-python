import numpy as np
import scipy.ndimage as ndimage
from numba import njit


def label_expansion(img, cdata):
    I = img.astype(np.float64)
    trimap = cdata.copy().astype(np.float64)

    # Convert img to grayscale and filter it using convolution
    weights = np.array([0.29894, 0.58704, 0.11402])
    weights /= weights.sum()

    grayI = np.frompyfunc(_custom_round, 1, 1)(
        np.round(np.dot(I[..., :3], weights), 4)
    ).astype(np.uint8)

    J = ndimage.generic_filter(grayI.astype(np.float64), _std_filter, size=(7, 7))
    J = ndimage.generic_filter(J, np.min, size=(9, 9), mode="constant", cval=0)

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


def _custom_round(x):
    rounded = np.round(x)
    temp = rounded + ((x - rounded) == 0.5) * (rounded % 2)
    return temp


def _std_filter(x):
    return np.std(x, ddof=1)


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
