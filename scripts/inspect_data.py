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

    usuarios_unicos = ratings["userId"].nunique()
    peliculas_catalogo = movies["movieId"].nunique()
    num_valoraciones = ratings["rating"].count()
    peliculas_valoradas = ratings["movieId"].nunique()

    valoraciones_por_usuario = ratings.groupby("userId")["rating"].count()
    valoraciones_por_pelicula = ratings.groupby("movieId")["rating"].count()

    print(f"Num usuarios: {usuarios_unicos}")
    print(f"Películas del catálogo: {peliculas_catalogo}")
    print(f"Num de valoraciones: {num_valoraciones}")
    print(f"Número de películas valoradas: {peliculas_valoradas}")

    print(
        f"densidad de valoraciones: {num_valoraciones / (usuarios_unicos * peliculas_valoradas):.2%}"
    )

    print(f"Distribución ratings:\n{ratings['rating'].value_counts().sort_index()}")

    print(f"Mínimo de valoraciones por usuario: {valoraciones_por_usuario.min()}")

    print(f"Máximo de valoraciones por usuario: {valoraciones_por_usuario.max()}")

    print(f"Mediana de valoraciones por usuario: {valoraciones_por_usuario.median()}")

    print(f"Mínimo de valoraciones por película: {valoraciones_por_pelicula.min()}")
    print(f"Máximo de valoraciones por película: {valoraciones_por_pelicula.max()}")
    print(f"Mediana de valoraciones por película: {valoraciones_por_pelicula.median()}")

    print(
        f"Número de películas con una sola valoración: {valoraciones_por_pelicula.eq(1).sum()}"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(valoraciones_por_pelicula, bins=50, color="skyblue", edgecolor="black")
    ax.set_title("Distribución de valoraciones por película")
    ax.set_xlabel("Número de valoraciones")
    ax.set_ylabel("Número de películas")
    ax.set_yscale("log")
    ax.grid(axis="y")

    fig.savefig("reports/figures/ratings_per_movie.png")


if __name__ == "__main__":
    main()
