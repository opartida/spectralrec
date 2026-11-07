import numpy as np
import pytest

from spectralrec.sgd import FactorModel, sgd_step


def test_sgd_step():
    model = FactorModel(
        user_factors=np.array([[1.0, 2.0], [0.5, -1.0]]),
        movie_factors=np.array([[0.0, 1.0], [2.0, 1.0]]),
    )
    u, i = 0, 1
    residual, lr, reg = 5.0, 0.1, 0.5

    sgd_step(model=model, u=u, i=i, residual=residual, lr=lr, reg=reg)

    # Uu <- Uu + lr * [(residual - Uu @ Vi) * Vi - reg * Uu]
    # Uu <- [1.0,2.0] + 0.1 * [(5 - [1.0,2.0] @ [2.0,1.0]) * [2.0,1.0] - 0.5*[1.0,2.0]]
    # Uu <- [1.0,2.0] + 0.1 * [1*[2.0,1.0] - [0.5,1]]
    # Uu <- [1.0, 2.0] + 0.1 * [1.5,0]
    # Uu <- [1.15, 2.0]
    np.testing.assert_array_almost_equal(model.user_factors[u], np.array([1.15, 2.0]))

    # Vi <- Vi + lr * [(residual - Uu @ Vi) * Uu - reg * Vi]
    # Vi <- [2.0,1.0] + 0.1 * [(5 - [1.0,2.0] @ [2.0,1.0]) * [1.0,2.0] - 0.5*[2.0, 1.0]]
    # Vi <- [2.0,1.0] + 0.1 * [1 * [1.0, 2.0] - [1, 0.5]]
    # Vi <- [2.0,1.0] + 0.1 * [0, 1.5]
    # Vi <- [2.0, 1.15]
    np.testing.assert_array_almost_equal(model.movie_factors[i], np.array([2.0, 1.15]))


def test_sgd_step_big_learning_rate():
    model = FactorModel(
        user_factors=np.array([[1.0, 2.0], [0.5, -1.0]]),
        movie_factors=np.array([[0.0, 1.0], [2.0, 1.0]]),
    )
    u, i = 0, 1
    residual, lr, reg = 5.0, 10e3, 0.5

    with pytest.raises(FloatingPointError):
        sgd_step(model=model, u=u, i=i, residual=residual, lr=lr, reg=reg)
        sgd_step(model=model, u=u, i=i, residual=residual, lr=lr, reg=reg)


def test_sgd_step_reg_0():
    model = FactorModel(
        user_factors=np.array([[1.0, 2.0], [0.5, -1.0]]),
        movie_factors=np.array([[0.0, 1.0], [2.0, 1.0]]),
    )
    u, i = 0, 1
    residual, lr, reg = 5.0, 0.1, 0
    sgd_step(model=model, u=u, i=i, residual=residual, lr=lr, reg=reg)

    # Uu <- Uu + lr * [(residual - Uu @ Vi) * Vi - reg * Uu]
    # Uu <- [1.0,2.0] + 0.1 * [(5 - [1.0,2.0] @ [2.0,1.0]) * [2.0,1.0]]
    # Uu <- [1.0,2.0] + 0.1 * [1*[2.0,1.0]]
    # Uu <- [1.0, 2.0] + [0.2, 0.1]
    # Uu <- [1.2, 2.1]
    np.testing.assert_array_almost_equal(model.user_factors[u], np.array([1.2, 2.1]))

    # Vi <- Vi + lr * [(residual - Uu @ Vi) * Uu - reg * Vi]
    # Vi <- [2.0,1.0] + 0.1 * [(5 - [1.0,2.0] @ [2.0,1.0]) * [1.0,2.0]]
    # Vi <- [2.0,1.0] + 0.1 * [1 * [1.0, 2.0]]
    # Vi <- [2.0,1.0] + [0.1, 0.2]
    # Vi <- [2.1, 1.2]
    np.testing.assert_array_almost_equal(model.movie_factors[i], np.array([2.1, 1.2]))
