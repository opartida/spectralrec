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


class BiasBaseline:
    def __init__(self, n_users: int, n_movies: int, reg_movie, reg_user):
        self.n_users = n_users
        self.n_movies = n_movies
        self.reg_movie = reg_movie
        self.reg_user = reg_user

    def fit(self, users, movies, ratings):
        self.global_mean = np.mean(ratings)
        ratings_per_movie = np.bincount(movies, minlength=self.n_movies)
        sum_ratings_per_movie = np.bincount(movies, ratings, minlength=self.n_movies)

        self.movie_bias = np.divide(
            sum_ratings_per_movie - self.global_mean * ratings_per_movie,
            ratings_per_movie + self.reg_movie,
        )

        ratings_per_user = np.bincount(users, minlength=self.n_users)
        residual_sum_per_user = np.bincount(
            users,
            ratings - self.movie_bias[movies] - self.global_mean,
            minlength=self.n_users,
        )

        self.user_bias = np.divide(
            residual_sum_per_user,
            self.reg_user + ratings_per_user,
        )

    def predict(self, users, movies):
        return self.global_mean + self.movie_bias[movies] + self.user_bias[users]
