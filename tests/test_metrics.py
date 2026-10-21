from math import sqrt

import pytest

from spectralrec.metrics import mae, rmse


def test_mae():
    y_true = (1, 3, 6)
    y_pred = (5, 1, 1)
    res = mae(y_true, y_pred)
    assert res == pytest.approx(11 / 3)


def test_rmse():
    y_true = (1, 3, 6)
    y_pred = (5, 1, 1)
    res = rmse(y_true, y_pred)
    assert res == pytest.approx(sqrt(15))


def test_mae_input_with_different_sizes():
    y_true = (4, 5, 6)
    y_pred = (8, 9)

    with pytest.raises(ValueError) as execinfo:
        mae(y_true, y_pred)
    assert "y_true and y_pred have different dimensions" in str(execinfo.value)


def test_mae_empty_inputs():
    with pytest.raises(ValueError) as execinfo:
        mae((), ())
    assert "some of the arguments are empty" in str(execinfo.value)


def test_rmse_input_with_different_sizes():
    y_true = (4, 5, 6)
    y_pred = (8, 9)

    with pytest.raises(ValueError) as execinfo:
        rmse(y_true, y_pred)
    assert "y_true and y_pred have different dimensions" in str(execinfo.value)


def test_rmse_empty_inputs():
    with pytest.raises(ValueError) as execinfo:
        rmse((), ())
    assert "some of the arguments are empty" in str(execinfo.value)
