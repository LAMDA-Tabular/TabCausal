import numpy as np


def standardize_x(X, *, clip=10.0, eps=1e-8):
    """Column-wise z-score preprocessing for benchmark data.

    The same transformation is used for observational and interventional rows:
    statistics are computed over all rows in the file, independently per
    variable. Intervention flags are not touched by this helper.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    X = (X - mean) / std
    if clip is not None:
        X = np.clip(X, -float(clip), float(clip))
    return X
