"""NOTEARS linear solver.

Adapted from the public NOTEARS implementation by Zheng et al. The public
runner keeps continuous weights and applies the benchmark threshold outside
this solver.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(
    X: np.ndarray,
    lambda1: float,
    loss_type: str,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.0,
) -> np.ndarray:
    def _loss(W: np.ndarray):
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError(f"unknown loss type: {loss_type}")
        return loss, G_loss

    def _h(W: np.ndarray):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w: np.ndarray):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w: np.ndarray):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    del n
    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)
    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = np.abs(_adj(w_est))
    if w_threshold > 0:
        W_est[np.abs(W_est) < w_threshold] = 0
    np.fill_diagonal(W_est, 0)
    return W_est

