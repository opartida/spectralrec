from pathlib import Path

import matplotlib.pyplot as plt

from spectralrec.data import load_movies, load_ratings, validate_ratings

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
MOVIES_PATH = Path("data/raw/ml-latest-small/movies.csv")


def main():
    """
    Main function to inspect data.
    """
    # Load data
    ratings = load_ratings(RATINGS_PATH)
    movies = load_movies(MOVIES_PATH)

    validate_ratings(ratings)

    n_users = ratings["userId"].nunique()
    n_catalog_movies = movies["movieId"].nunique()
    n_ratings = ratings["rating"].count()
    n_rated_movies = ratings["movieId"].nunique()

    ratings_per_user = ratings.groupby("userId")["rating"].count()
    ratings_per_movie = ratings.groupby("movieId")["rating"].count()

    print(f"Users: {n_users}")
    print(f"Movies in the catalog: {n_catalog_movies}")
    print(f"Ratings: {n_ratings}")
    print(f"Movies with at least one rating: {n_rated_movies}")

    print(f"Rating density: {n_ratings / (n_users * n_rated_movies):.2%}")

    print(f"Rating distribution:\n{ratings['rating'].value_counts().sort_index()}")

    print(f"Min ratings per user: {ratings_per_user.min()}")

    print(f"Max ratings per user: {ratings_per_user.max()}")

    print(f"Median ratings per user: {ratings_per_user.median()}")

    print(f"Min ratings per movie: {ratings_per_movie.min()}")
    print(f"Max ratings per movie: {ratings_per_movie.max()}")
    print(f"Median ratings per movie: {ratings_per_movie.median()}")

    print(f"Movies with a single rating: {ratings_per_movie.eq(1).sum()}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(ratings_per_movie, bins=50, color="skyblue", edgecolor="black")
    ax.set_title("Ratings per movie")
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of movies")
    ax.set_yscale("log")
    ax.grid(axis="y")

    fig.savefig("reports/figures/ratings_per_movie.png")


if __name__ == "__main__":
    main()
