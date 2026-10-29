# Shapes used in this module:
#   user_factors (U):   (n_users, k)
#   movie_factors (V):  (n_movies, k)
#   users, movies:      (N,)   one entry per (user, movie) pair
#   predictions:        (N,)
import numpy as np


def predict_observed(user_factors, movie_factors, users, movies) -> np.ndarray:
    return np.sum(user_factors[users] * movie_factors[movies], axis=1)


def to_dense(users, movies, values, shape) -> np.ndarray:
    dense_matrix = np.zeros(shape=shape)
    dense_matrix[users, movies] = values
    return dense_matrix
