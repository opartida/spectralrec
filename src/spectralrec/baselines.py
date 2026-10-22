import numpy as np


class GlobalMean:
    def __init__(self, n_users: int, n_movies: int):
        self.n_users = n_users
        self.n_movies = n_movies

    def fit(self, users, movies, ratings):
        self.global_mean = np.mean(ratings)

    def predict(self, users, movies):
        return np.full(np.size(users), self.global_mean)


class UserMean:
    def __init__(self, n_users: int, n_movies: int):
        self.n_users = n_users
        self.n_movies = n_movies

    def fit(self, users, movies, ratings):
        global_mean = np.mean(ratings)
        ratings_per_user = np.bincount(users, minlength=self.n_users)
        sum_ratings_per_user = np.bincount(users, ratings, minlength=self.n_users)
        out = np.full(self.n_users, global_mean)
        self.user_mean = np.divide(
            sum_ratings_per_user, ratings_per_user, out=out, where=ratings_per_user != 0
        )

    def predict(self, users, movies):
        return self.user_mean[users]
