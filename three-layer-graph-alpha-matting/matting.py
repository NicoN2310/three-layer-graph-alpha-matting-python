import numpy as np
import scipy.sparse as sparse

from label_expansion import label_expansion
from three_graph import get_three_graph
from color_line_laplace import get_color_line_laplace
from utils import *


def calculate_matte(
    image,
    trimap,
    K0=12,
    K1=12,
    K2=7,
    K3=3,
    ratio_1=0.418746,
    ratio_2=0.154445,  # getThreeGraph
    epsilon=1e-7,
    win_size=1,  # get_color_line_laplace
    delta=7,
    lambda_val=1000,
    tol=1e-7,
    maxit=2000,
):
    I = image.astype(np.float64)
    conv_img = filter_image_3d(I)
    I = I + (I - conv_img)

    t = label_expansion(image, trimap)

    L1 = get_three_graph(I, t, K0, K1, K2, K3, ratio_1, ratio_2)
    L2 = get_color_line_laplace(I, t, epsilon, win_size)

    m, n = trimap.shape
    L = L1 + delta * L2

    M = (trimap == 255) | (trimap == 0).astype(np.uint8)
    G = (trimap == 255).flatten(order="F").reshape(-1, 1).astype(np.uint8)
    Lambda = lambda_val * sparse.diags(M.flatten(order="F"), 0, format="csr")

    initial_guess = trimap.astype(np.float64).flatten(order="F") / 255.0
    Alpha, _ = sparse.linalg.bicgstab(
        L + Lambda, Lambda.dot(G), x0=initial_guess, rtol=tol, maxiter=maxit
    )
    Alpha = Alpha.reshape(-1, 1)

    matte = np.reshape(Alpha, (m, n), order="F")
    matte = np.clip(matte, 0, 1)

    h = fspecial_gauss(9, 0.5)
    matte = ndimage.correlate(matte, h, mode="nearest")

    return matte
