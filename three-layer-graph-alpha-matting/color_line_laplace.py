import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as sparse


def get_color_line_laplace(I, trimap, epsilon=1e-7, win_size=1):
    consts = (trimap == 0) | (trimap == 255)
    consts = ndimage.binary_erosion(consts, 
                                    structure=np.ones((win_size * 2 + 1, win_size * 2 + 1)), 
                                    border_value=1).astype(bool)
    
    I = np.double(I) / 255.0
    h, w, c = I.shape
    img_size = w * h
    neb_size = (win_size * 2 + 1) ** 2
    
    indsM = np.arange(1, img_size + 1).reshape(h, w, order='F')
    tlen = np.sum(1 - consts[win_size : -win_size, win_size : -win_size]) * (neb_size ** 2)
    
    row_inds = np.zeros(tlen, dtype=int).reshape(-1, 1)
    col_inds = np.zeros(tlen, dtype=int).reshape(-1, 1)
    vals = np.zeros(tlen, dtype=np.float64).reshape(-1, 1)
    len_val = 0
    
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
                continue

            win_inds = indsM[i - win_size : i + win_size + 1, j - win_size : j + win_size + 1]
            win_inds = win_inds.flatten(order='F')
            
            winI = I[i - win_size : i + win_size + 1, j - win_size : j + win_size + 1, :]
            winI = winI.reshape(neb_size, c, order='F')

            win_mu = np.mean(winI, axis=0).reshape(1, -1)
            win_var = np.linalg.inv(np.dot(winI.T, winI) / neb_size - np.outer(win_mu, win_mu) + epsilon / neb_size * np.eye(c))
            winI = winI - np.tile(win_mu, (neb_size, 1))
            tvals = (1 + winI @ win_var @ winI.T) / neb_size
            
            row_inds[len_val : neb_size ** 2 + len_val] = np.tile(win_inds, neb_size).reshape(-1, 1, order='F')
            col_inds[len_val : neb_size ** 2 + len_val] = np.tile(win_inds, neb_size).reshape(-1, 9).reshape(-1, 1, order='F')
            
            testing = tvals.flatten().reshape(-1, 1, order='F')
            vals[len_val : neb_size ** 2 + len_val] = testing

            len_val += neb_size ** 2
        
    vals = vals[: len_val]
    row_inds = row_inds[: len_val]
    col_inds = col_inds[: len_val]
    
    A = sparse.coo_matrix((vals.squeeze(), (row_inds.squeeze()-1, col_inds.squeeze()-1)), shape=(img_size, img_size)).tocsr()
    sumA = np.array(A.sum(axis=1))
    A = sparse.spdiags([sumA.flatten()], [0], img_size, img_size, format='csr').tocsr() - A
    return A

