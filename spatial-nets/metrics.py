from typing import Union

import numpy as np
import scipy.sparse as sp

Array = Union[np.ndarray, sp.csr_matrix]


def CPC(F1: Array, F2: Array, rel_tol: float = 1e-3) -> float:
    """
    Calculate the common part of commuters between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs(
        (x - y) / x) < rel_tol, "arrays do not have same sum (up to rel. tol.)"
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    denom_mat = F1 + F2
    num_mat = denom_mat - np.abs(F1 - F2)  # this implements 2*min(F1, F2)

    return np.sum(num_mat) / np.sum(denom_mat)


def CPL(F1: Array, F2: Array, rel_tol: float = 1e-3) -> float:
    """
    Calculate the common part of links between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs(
        (x - y) / x) < rel_tol, "arrays do not have same sum (up to rel. tol.)"
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    bool_F1 = F1 > 0
    bool_F2 = F2 > 0

    # sparse implements np.matrix and * is mat-mat multplication, not elementwise
    if sp.issparse(F1):
        prod = bool_F1.multiply(bool_F2)
    elif sp.issparse(F2):
        prod = bool_F2.multiply(bool_F1)
    else:
        prod = bool_F1 * bool_F2

    return 2 * np.sum(prod) / (np.sum(bool_F1) + np.sum(bool_F2))


def RMSE(F1: Array, F2: Array, rel_tol: float = 1e-3, norm=True) -> float:
    """
    Calculate root-mean-square error between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs(
        (x - y) / x) < rel_tol, "arrays do not have same sum (up to rel. tol.)"
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    N = np.prod(F1.shape)

    diff = F1 - F2
    power = diff.power(2) if sp.issparse(diff) else np.power(diff, 2)
    out = np.sqrt(power.sum() / N)

    if norm:
        out *= N / x

    return out
