import numpy as np
import scipy.sparse as sparse
from three_layer_graph_alpha_matting.kdtree import KDTree
from numba import njit, prange


def get_three_graph(
    I, trimap, K0, K1, K2, K3, ratio_1=0.418746, ratio_2=0.154445
):
    I = I.copy().astype(np.float32)
    trimap = trimap.copy().astype(np.float32)

    m, n, _ = I.shape
    N = m * n

    # Calculate initial X
    x = np.tile(np.arange(1, n + 1), (m, 1))
    y = np.tile(np.arange(1, m + 1), (n, 1)).T
    featherArray = np.dstack((I, y / 12, x / 12))
    X = featherArray.reshape(N, -1, order="F").astype(np.float32)

    # Calculate K0
    kdtree = KDTree(X)
    D, IDX = kdtree.query(X, k=K0 + 1)

    IDX = IDX[:, 1:]
    D = D[:, 1:].astype(np.float64)

    sumD = get_sumD(D, IDX, K0).reshape(N, 1)
    X = np.hstack((X[:, :3], X[:, 3:5], sumD)).astype(np.float32)

    # Calculate K1
    kdtree = KDTree(X)
    D, IDX = kdtree.query(X, k=K1 + 1)

    IDX = IDX[:, 1:]
    D = D[:, 1:].astype(np.float64)

    sumD = get_sumD(D, IDX, K1).reshape(N, 1)
    X = np.hstack((X[:, :3], X[:, 3:5] / 120, sumD)).astype(np.float32)

    IX = np.argsort(np.sum(D**2, axis=1))[: int(0.9 * N)].reshape(-1, 1)
    flag = np.zeros(m * n, dtype=np.uint8)
    flag[IX] = 1
    flag = flag.reshape(m, n, order="F")
    neighborsArray = IDX.reshape(m, n, K1, order="F")

    # Calculate W1
    W1 = get_graph(I, trimap, neighborsArray, flag)

    # Calculate K2
    kdtree = KDTree(X)
    D, IDX = kdtree.query(X, k=K2 + 1)

    IDX = IDX[:, 1:]
    D = D[:, 1:].astype(np.float64)

    sumD = get_sumD(D, IDX, K2).reshape(N, 1)
    X = np.hstack((X[:, :3], X[:, 3:5] / 12, sumD)).astype(np.float32)

    flag = select_pix(I, D, ratio_1)
    neighborsArray = IDX.reshape(m, n, K2, order="F")

    # Calculate W2
    W2 = get_graph(I, trimap, neighborsArray, flag)

    # Calculate K3
    kdtree = KDTree(X)
    D, IDX = kdtree.query(X, k=K3 + 1)

    IDX = IDX[:, 1:]
    D = D[:, 1:].astype(np.float64)

    flag = select_pix(I, D, ratio_2)
    neighborsArray = IDX.reshape(m, n, K3, order="F")

    # Calculate W3
    W3 = get_graph(I, trimap, neighborsArray, flag)

    # Calculate L
    L = 0.1 * (W3.T @ W3) + 0.3 * (W2.T @ W2) + (W1.T @ W1)
    return L.tocsr()

'''
def get_sumD(D, IDX, K):
    layerNum = 120
    sumD = np.sum(D, axis=1)

    resh_1 = np.reshape(D.T, [-1, 1])
    resh_2 = np.reshape(IDX.T, [-1, 1])
    
    for _ in range(layerNum):
        sumD = resh_1 + sumD[resh_2]
        sumD = np.reshape(sumD.T, [K, -1])  # Here
        sumD = sumD.T                       # Here
        sumD = np.sum(sumD, axis=1)

    sumD = (sumD - np.min(sumD)) / (np.max(sumD) - np.min(sumD))
    feature = sumD * 255 / np.std(sumD, ddof=1)

    return feature
'''

# New optimized version of get_sumD
def get_sumD(D, IDX, K):
    layer_num = 120
    num_pixels = D.shape[0]

    sumD_prev = np.sum(D, axis=1)  # n=0

    row_indices = np.repeat(np.arange(num_pixels), K)
    col_indices = IDX.flatten()
    data = np.ones(len(row_indices))
    neighbor_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_pixels, num_pixels))

    sumD_current = sumD_prev.copy()
    for _ in range(layer_num):
        sumD_current = neighbor_matrix.dot(sumD_current) + sumD_prev

    sumD = sumD_current
    sumD = (sumD - np.min(sumD)) / (np.max(sumD) - np.min(sumD))
    sumD = sumD * 255 / np.std(sumD, ddof=1)
    
    return sumD

'''
def get_graph(I, trimap, neighborsArray, flag, tol=1e-3):
    N = I.shape[0] * I.shape[1]
    K = neighborsArray.shape[2]

    featherArray = I.copy()
    X = featherArray.reshape(N, -1, order="F")

    cond_1 = trimap == 255
    cond_2 = trimap == 0
    cond_3 = flag == 0
    indArray = np.nonzero(~(cond_1 + cond_2 + cond_3).flatten(order="F"))[0].reshape(
        -1, 1
    )

    neighborsIndex = neighborsArray.reshape((N, K), order="F")
    neighbors = X[neighborsIndex.flatten(), :].T.reshape((X.shape[1], K, N), order="F")
    A = neighbors - np.transpose(X).reshape((X.shape[1], 1, N), order="F")

    w_array = np.zeros((K, N))
    identity = np.identity(K)

    for ind in indArray:
        part = np.squeeze(A[:, :, ind])
        C = part.T @ part
        trace = np.trace(C)
        if trace == 0:
            w = np.ones(K)
        else:
            C = C + tol * trace * identity
            w = np.linalg.solve(C, np.ones((K, 1)))
        w = w / np.sum(w)
        w_array[:, ind] = w.reshape(-1, 1)

    w_array = w_array.flatten(order="F")

    indArray_flat = indArray.flatten()
    i_array = np.zeros((K, N), dtype=np.int32)
    i_array[:, indArray_flat] = np.ones((K, 1), dtype=np.int32) * indArray.T
    j_array = np.zeros((K, N), dtype=np.int32)
    j_array[:, indArray_flat] = neighborsIndex[indArray_flat, :].T

    w_array = w_array.flatten(order="F")
    i_array = i_array.flatten(order="F")
    j_array = j_array.flatten(order="F")

    o_array = flag.flatten(order="F")

    O = sparse.diags(o_array, 0, shape=(N, N), format="csr")
    W = sparse.eye(N, format="csr") - sparse.csr_matrix(
        (w_array, (i_array, j_array)), shape=(N, N), dtype=np.float64
    )
    W_res = (O.T @ W).tocsr()

    return W_res
'''

# New optimized version of get_graph
def get_graph(I, trimap, neighborsArray, flag, tol=1e-3):
    N = I.shape[0] * I.shape[1]
    K = neighborsArray.shape[2]

    featherArray = I.copy()
    X = featherArray.reshape(N, -1, order="F")

    cond_1 = trimap == 255
    cond_2 = trimap == 0
    cond_3 = flag == 0
    indArray = np.nonzero(~(cond_1 + cond_2 + cond_3).flatten(order="F"))[0].reshape(
        -1, 1
    )

    neighborsIndex = neighborsArray.reshape((N, K), order="F")
    neighbors = X[neighborsIndex.flatten(), :].T.reshape((X.shape[1], K, N), order="F")
    A = neighbors - np.transpose(X).reshape((X.shape[1], 1, N), order="F")
    A = A.astype(np.float64)
    w_array = compute_weights(A, indArray, tol, K, N)

    w_array = w_array.flatten(order="F")

    indArray_flat = indArray.flatten()
    i_array = np.zeros((K, N), dtype=np.int32)
    i_array[:, indArray_flat] = np.ones((K, 1), dtype=np.int32) * indArray.T
    j_array = np.zeros((K, N), dtype=np.int32)
    j_array[:, indArray_flat] = neighborsIndex[indArray_flat, :].T

    w_array = w_array.flatten(order="F")
    i_array = i_array.flatten(order="F")
    j_array = j_array.flatten(order="F")

    o_array = flag.flatten(order="F")

    O = sparse.diags(o_array, 0, shape=(N, N), format="csr")
    W = sparse.eye(N, format="csr") - sparse.csr_matrix(
        (w_array, (i_array, j_array)), shape=(N, N), dtype=np.float64
    )
    W_res = (O.T @ W).tocsr()

    return W_res

@njit(parallel=True, cache=True)
def compute_weights(A, indArray, tol, K, N):
    w_array = np.zeros((K, N))
    for indIdx in prange(len(indArray)):
        ind = indArray[indIdx, 0]
        
        C = np.empty((K, K))
        b = np.empty(K)
        w = np.empty(K)

        for i in range(K):
            b[i] = 1.0
            w[i] = 1.0
            for j in range(K):
                s = tol * (i == j)
                for k in range(3):
                    s += A[k, i, ind] * A[k, j, ind]
                C[i, j] = s

        L = np.empty((K, K))
        # Copy lower triangular of C into L: np.tril(C, out=L)
        for i in range(K):
            for j in range(K):
                L[i, j] = C[i, j] if i >= j else 0.0

        cholesky(L)

        # Solve L b = w for b
        # This is the other way round to save on a temporary vector
        backsub_lower(L, w, b)
        # Solve L^T w = b for w
        backsub_upper(L.T, b, w)

        w_sum = w.sum()
        for i in range(K):
            w_array[i, ind] = w[i] / w_sum
    return w_array

@njit
def cholesky(L):
    n = L.shape[0]

    for i in range(n):
        for j in range(i):
            for k in range(j):
                L[i, j] -= L[i, k] * L[j, k]
            L[i, j] /= L[j, j]
        for k in range(i):
            L[i, i] -= L[i, k] * L[i, k]
        L[i, i] = np.sqrt(L[i, i])

@njit
def backsub_lower(L, b, x):
    n = L.shape[0]

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

@njit
def backsub_upper(U, b, x):
    n = U.shape[0]

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

def select_pix(img_input, D, selectedRatio):
    m, n, _ = img_input.shape
    N = m * n

    r = int(np.round(0.7 * N))

    B = np.sum(D**2, axis=1)
    B /= B[r]

    IX = np.argsort(B)
    C = np.cumsum(B[IX[:r]]) / (0.7 * N)

    threshold = np.argmin(np.abs(C - selectedRatio))
    IX = IX[:threshold]

    selectedPix = np.zeros(m * n)
    selectedPix[IX] = 1
    selectedPix = selectedPix.reshape(m, n, order="F")

    return selectedPix
