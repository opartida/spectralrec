# SpectralRec

Final project based on MIT 18.065.

Goal: predict the rating a user would give to a movie, using the known
ratings of other users and movies.



## Data

Source: MovieLens Latest Small (GroupLens). Downloaded on 2026-10-16.

| Statistic | Value |
|---|---|
| Users | 610 |
| Movies in the catalog | 9742 |
| Ratings | 100836 |
| Movies with at least one rating | 9724 |
| Rating density | 1.70% |
| Min ratings per user | 20 |
| Max ratings per user | 2698 |
| Median ratings per user | 70.5 |
| Min ratings per movie | 1 |
| Max ratings per movie | 329 |
| Median ratings per movie | 3.0 |
| Movies with a single rating | 3446 |

![Ratings per movie](reports/figures/ratings_per_movie.png)

## Train / validation / test split

- Random split (not by date): 80% train, 10% validation, 10% test.
- Split seed: 42
- Rows removed because their user or movie does not appear in train:

| Set | Rows removed | Share |
|---|---|---|
| Validation | 371 | 3.68% |
| Test | 391 | 3.88% |

Why: The model has no information about a movie (or user) that never appears in train, so it cannot predict its ratings. These are mostly rarely-rated movies, whose few ratings all fell into validation or test. Since rare movies are harder to predict, removing them makes the evaluation optimistic, and it says nothing about how the model performs on new or rare movies.
