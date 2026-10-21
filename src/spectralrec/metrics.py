import math

import numpy as np


def check_arguments(y_true, y_pred):
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("some of the arguments are empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different dimensions")


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_arguments(y_true, y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_arguments(y_true, y_pred)

    return math.sqrt(np.mean((y_true - y_pred) ** 2))
