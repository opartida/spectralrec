import numpy as np


def soft_threshold_svd(X, lam):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_shrunk = s - lam
    keep = s_shrunk > 0
    return U[:, keep], s_shrunk[keep], Vt[keep]


def soft_impute(
    users, movies, residuals, shape, lam, max_iter=100, tol=1e-4, warm_start=None
):
    if warm_start:
        U, s, Vt = warm_start
        Z = U @ np.diag(s) @ Vt
    else:
        Z = np.zeros(shape=shape)

    for i in range(max_iter):
        X = np.copy(Z)
        X[users, movies] = residuals
        U_shrunk, s_shrunk, Vt_shrunk = soft_threshold_svd(X=X, lam=lam)
        Z_new = U_shrunk @ np.diag(s_shrunk) @ Vt_shrunk
        change = np.sum((Z_new - Z) ** 2)
        den = np.sum(Z**2)
        if change == 0:
            break

        if den != 0 and change / den < tol:
            break
        Z = Z_new
    return U_shrunk, s_shrunk, Vt_shrunk
