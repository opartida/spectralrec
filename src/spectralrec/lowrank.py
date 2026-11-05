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


def compute_singular_values(matrix) -> np.ndarray:
    return np.linalg.svd(matrix, compute_uv=False)


class TruncatedSVD:
    def __init__(self, k: int):
        self.k = k

    def fit(self, users, movies, residuals, shape):
        M = to_dense(users=users, movies=movies, values=residuals, shape=shape)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        singular_values_square_root = np.sqrt(s)
        self.initial_user_factors = U * singular_values_square_root
        self.initial_movie_factors = singular_values_square_root * np.transpose(Vt)
        self.truncate(k=self.k)

    def truncate(self, k):
        self.k = k
        self.user_factors = self.initial_user_factors[:, 0:k]
        self.movie_factors = self.initial_movie_factors[:, 0:k]

    def predict(self, users, movies):
        return predict_observed(
            user_factors=self.user_factors,
            movie_factors=self.movie_factors,
            users=users,
            movies=movies,
        )
