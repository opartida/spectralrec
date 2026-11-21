# SpectralRec

A low rank matrix with several of their values being unobserved or missing is completed using different approaches, Truncated SVD, SGD, ALS and Soft-Impute. The ratings are stored in a matrix of users x movies in which only 1.7% of the entries are not missing. Recommendation means filling these missing values.

All models below share the same mathematical structure: 

$$\hat r_{ui} = \mu + b_u + b_i + U_u \cdot V_i$$

where $\mu$ is the global mean, $b_u$ and $b_i$ are the user and movie biases, and $U_u$ and $V_i$ are the k-dimensional factor vectors of user $u$ and movie $i$: row $u$ of $U$ and row $i$ of $V$. The low-rank part, $U V^\top$, models the residuals that the biases leave.


Truncated SVD keeps the k factors $U_u$ and $V_i$ with the biggest singular values of the residual matrix and discards the rest. The residual matrix has all the missing values filled with zeros.

SGD, ALS and Soft-Impute minimize a loss function on the observed ratings only. In the case of SGD and ALS, the objective is to find the factors by minimizing the loss function in which the size of the factors is penalized. The goal of Soft-Impute is to directly obtain the residual matrix Z by minimizing the loss function which penalizes its nuclear norm. At each iteration it fills the missing values with the current Z, performs its SVD and subtracts the regularization parameter $\lambda$ from each singular value.

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

The data are not included in this repository: GroupLens distributes them under their own licence (see the dataset's `README.txt`) so download them from the official source as described in `Installation and usage`.


## Installation and usage
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest

```
Download and unzip `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip` into `data/raw/`, so that `data/raw/ml-latest-small/ratings.csv` exists.

Run the scripts to produce the different tables and figures
```bash
python scripts/inspect_data.py
python scripts/baselines.py
python scripts/spectrum.py
python scripts/svd_sweep.py
python scripts/train.py --config configs/sgd.json
python scripts/train.py --config configs/als.json
python scripts/softimpute_path.py
python scripts/als_vs_softimpute.py
python scripts/effective_rank.py   # needs the k = 100 ALS runs from configs/search/
python scripts/evaluate.py --split test
python scripts/recommend.py --user-id 17 --top-n 10
python scripts/interpret.py
```
To reproduce the hyperparameter search, run each file in `configs/search/`:
```bash
python scripts/train.py --config <file>
```
Each execution adds its row to 
`reports/experiments.csv`

## Train / validation / test split

- Random split (not by date): 80% train, 10% validation, 10% test.
- Split seed: 42
- Rows removed because their user or movie does not appear in train:

| Set | Rows removed | Share |
|---|---|---|
| Validation | 371 | 3.68% |
| Test | 391 | 3.88% |

Why: The model has no information about a movie (or user) that never appears in train, so it cannot predict its ratings. These are mostly rarely-rated movies, whose few ratings all fell into validation or test. Since rare movies are harder to predict, removing them makes the evaluation optimistic, and it says nothing about how the model performs on new or rare movies.

Leakage is avoided in four ways:
- The partitioning is performed only once, at the beginning, using a fixed seed, and the user and movie indices are constructed using only the training set.
- The biases (μ, b_u, b_i) are fitted using only the training set, and the validation and test residuals are calculated based on these biases.
- All hyperparameters (k, reg, lr, λ, and the number of epochs) were selected using only the validation set.
- The test set was accessed only once, at the very end, using the configuration frozen beforehand in the Frozen configuration section and in commit 7f7f88d.


## Bias baseline
The baseline model:
$$\hat r_{ui} = \mu + b_u + b_i$$

With a fixed μ, the bias of movie i is determined by 
$$\min_{b}\ \sum_{u \in R(i)} (r_{ui} - \mu - b)^2 + \lambda_i\, b^2
\qquad\Longrightarrow\qquad
b_i = \frac{\sum_{u \in R(i)} (r_{ui} - \mu)}{\lambda_i + n_i}$$


where $R(i)$ is the set of users who rated movie i, and $n_i$ = $|R(i)|$ is the number of such users. 
The bias of user u is computed by
$$\min_{b}\ \sum_{i \in R(u)} (r_{ui} - \mu - b_i - b)^2 + \lambda_u\, b^2
\qquad\Longrightarrow\qquad
b_u = \frac{\sum_{i \in R(u)} (r_{ui} - \mu - b_i)}{\lambda_u + n_u}$$

where $R(u)$ is the set of movies rated by user u, and $n_u$ = $|R(u)|$ is the number of such movies. 
The regularization parameters, chosen on validation, are $\lambda_i = \lambda_u = 3$. The validation RMSE is 0.8554, compared to 1.0335 for the global mean.

All the low-rank models learn the residuals left by this baseline, which serves as the benchmark they must outperform.
## Singular value spectrum
![Spectrum of the full matrix](reports/figures/spectrum.png)
![Spectrum of the dense submatrix](reports/figures/spectrum_dense.png)

Singular values of the residual matrix were compared to the singular values of a matrix where the residuals were permuted.
Also a dense submatrix was extracted from the residuals matrix, using the 50 users and 50 movies with the most train ratings,
and its singular values were computed and compared with the singular values of this matrix with the non zero values permuted. The results were the following:

| Matrix | Real σ₁ | Control σ₁ (20 shuffles) |
|---|---|---|
| Full | 35.55 | 36.81–38.61 |
| Dense 50 × 50 | 10.64 | 7.67–8.70 |

The bigger value of the first singular value in the dense matrix compared to the first singular value of the permuted matrix suggests that $\sigma_1$ captures something the shuffle does not reproduce, although this does not prove it reflects tastes.
In the full residual matrix $\sigma_1$ stays below the control values and this shows the test does not detect patterns and the shuffling changes something else apart from the tastes.

**Proof that $\|M\|_F^2 = \sum_j \sigma_j^2$.** With $M = U \Sigma V^\top$:

$$\|M\|_F^2 = \mathrm{tr}(M^\top M) = \mathrm{tr}(V \Sigma^2 V^\top) = \mathrm{tr}(\Sigma^2 V^\top V) = \mathrm{tr}(\Sigma^2) = \sum_j \sigma_j^2$$


The more zeros in the matrix the less information we can get from the residuals in relation to tastes. SVD treats each missing cell as a real value of 0, so with 98.3% of the cells missing, the spectrum mostly reflects where the ratings are, not which residual sits in each cell. The biggest singular values of the full matrix and the permuted matrix are very similar. In the dense matrix, the first singular value exceeds those of all 20 controls.

## Truncated SVD (Eckart–Young)

|   k | RMSE train | RMSE val |
|-----|------------|----------|
|   1 |     0.7956 |   0.8529 |
|   2 |     0.7874 |   0.8531 |
|   5 |     0.7610 |   0.8503 |
|  10 |     0.7273 |   0.8491 |
|  20 |     0.6771 |   0.8474 |
|  50 |     0.5604 |   0.8490 |
| 100 |     0.4283 |   0.8519 |

Bias baseline (predicted residual 0): validation RMSE 0.8554.

The training RMSE falls steadily with k, from 0.7956 at k = 1 to 0.4283 at k = 100, because the truncated SVD approximates the training residual matrix more closely as k grows. The validation RMSE barely moves: its best value, 0.8474 at k = 20, is less than 0.01 below the bias baseline. The reason is the zero filling. Validation pairs are empty cells of the training matrix, stored as 0. With small k, the approximation of an almost all-zero matrix is small everywhere, and with large k it reproduces those zeros. In both cases, the predicted residuals for validation pairs stay close to 0, so the model predicts almost the same ratings as the biases alone.

![Truncated SVD: RMSE vs k](reports/figures/svd_k.png)

## Matrix factorization with SGD

Same residuals as the truncated SVD, but the loss only uses the observed training ratings, with no zero filling:

$$L(U,V) = \sum_{(u,i)\in\text{train}}\Big[(\tilde r_{ui} - U_u\cdot V_i)^2 + \lambda\big(\lVert U_u\rVert^2 + \lVert V_i\rVert^2\big)\Big]$$

Configuration (`configs/sgd.json`): k = 10, lr = 0.01, reg = 0.1, scale = 0.1, patience = 30, model seed = 0, n_epochs = 500.

![SGD learning curve](reports/figures/learning_curve_sgd.png)

| | Value |
|---|---|
| Best epoch | 51 |
| Validation RMSE at best epoch | 0.8483 |
| Bias baseline (validation) | 0.8554 |
| Time per epoch | 0.47 s |

Validation stops improving after epoch 51. There is overfitting as the validation error increases while the training error keeps decreasing. With reg = 0.05, the validation curve stayed flat at the baseline; with reg = 0.1 it went down to 0.8483, so the regularization is the first hyperparameter I would explore.

## Matrix factorization with ALS

Same loss as SGD with weighted = true using ALS to minimize it:

$$L(U,V) = \sum_{(u,i)\in\text{train}}\Big[(\tilde r_{ui} - U_u\cdot V_i)^2 + \lambda\big(\lVert U_u\rVert^2 + \lVert V_i\rVert^2\big)\Big]$$

Configuration (`configs/als.json`): k = 10, weighted = true, reg = 0.1, scale = 0.1, patience = 30, model seed = 0, n_epochs = 500.

![ALS learning curve](reports/figures/learning_curve_als.png)

| | Value |
|---|---|
| Best iteration | 30 |
| Validation RMSE at best iteration | 0.8526 |
| Bias baseline (validation) | 0.8554 |
| Time per iteration | 0.12 s |

Comparison with SGD on the same split (seed 42), with the same k = 10 and the same loss (reg = 0.1):

| | SGD | ALS |
|---|---|---|
| Best epoch / iteration | 51 | 30 |
| Validation RMSE | 0.8483 | 0.8526 |
| Training RMSE | 0.7032 | 0.5608 |
| Time per epoch / iteration | 0.47 s | 0.12 s |
| Total training time | 38.1 s | 7.3 s |

Both methods minimize the same loss, but ALS solves each half step exactly, so it gets much closer to the minimum of the training loss: its training RMSE (0.5608) is much lower than SGD's (0.7032). Getting closer to that minimum also means fitting the training noise more, and ALS ends up slightly worse on validation (0.8526 vs 0.8483). SGD was stopped at epoch 51 by early stopping, which watches the validation error and stops after `patience` epochs without improvement. Stopping before the minimum acts as extra regularization on top of λ. ALS is also about five times faster: 0.12 s per iteration against 0.47 s per epoch, and 7.3 s in total against 38.1 s.


## Nuclear norm regularization with Soft-Impute

Compute X matrix by copying the observed residual values and the rest 0 or a chosen initialization matrix. Compute SVD of X, shrink its singular values by subtracting $\lambda$ from them and keep factors with resulting singular value positive, compute Z from kept factors and shrunk singular values and compute new X by copying the observed residuals to Z. Iterate until the change in the values of Z from previous Z squared relative to the norm of Z also squared is smaller than tol or a maximum number of iterations is reached.

$$L(Z) =\ \tfrac12 \sum_{(u,i)\in\text{train}}(\tilde r_{ui} - Z_{ui})^2 + \lambda\lVert Z\rVert_*$$

Configuration (`configs/soft.json`): max_iter = 300, tol = 1e-8, split_seed = 42. The $\lambda$ are chosen from a list that starts at the maximum singular value of the residual matrix of train with zeros in the cells without rating, and ends at this singular value divided by 100 with 10 elements evenly spaced on a log scale in total. For each $\lambda$ an initialization matrix is provided being the Z matrix solution computed for the previous $\lambda$.


|        λ | rank | RMSE train | RMSE val | seconds |
|----------|------|------------|----------|---------|
|   35.545 |    0 |     0.8091 |   0.8554 |     0.5 |
|   21.309 |   13 |     0.7609 |   0.8470 |    16.6 |
|   12.774 |   51 |     0.6187 |   0.8377 |    35.9 |
|    7.658 |   88 |     0.4409 |   0.8413 |    63.0 |
|    4.591 |  114 |     0.2915 |   0.8484 |    96.6 |
|    2.752 |  129 |     0.1844 |   0.8544 |   120.4 |
|    1.650 |  140 |     0.1139 |   0.8589 |   115.8 |
|    0.989 |  147 |     0.0696 |   0.8620 |    89.5 |
|    0.593 |  158 |     0.0422 |   0.8639 |    60.7 |
|    0.355 |  172 |     0.0255 |   0.8649 |    36.0 |

![Soft-Impute: RMSE and rank vs λ](reports/figures/softimpute_path.png)

At the first value, λ = σ₁ = 35.545, every singular value is shrunk to zero: Z is the zero matrix (rank 0) and the validation RMSE equals the bias baseline. As λ decreases, Z keeps more singular values and its rank grows up to 172. The training RMSE keeps falling, down to 0.0255, but the validation RMSE reaches its minimum at λ = 12.774 (0.8377, rank 51) and rises below it. From λ ≈ 1.65 on, it is worse than the baseline: small values of λ overfit.

The best validation RMSE, 0.8377, beats the bias baseline (0.8554) and all the previous models: truncated SVD 0.8474 (k = 20), SGD 0.8483 and ALS 0.8526. The comparison favors Soft-Impute, though: it was tuned over 10 values of λ, while SGD and ALS were trained with hand-picked values of reg and might improve with a proper search. At the best λ, Soft-Impute uses 51 factors, more than SGD and ALS (k = 10) or truncated SVD (k = 20), but shrunk instead of whole: truncated SVD with 50 whole factors gave 0.8490, worse than with 20.

Each iteration takes about 0.5 s, the time of one SVD of the full matrix, so each λ takes roughly its number of iterations times 0.5 s. The time per λ grew until λ ≈ 2.75 and then fell, probably because of the warm start: for the smaller values of λ, the solution of the previous λ is already close to the new one, so fewer iterations are needed.



### ALS and Soft-Impute solve the same problem

$$\lVert Z\rVert_* = \min_{AB^\top = Z}\ \tfrac12\big(\lVert A\rVert_F^2 + \lVert B\rVert_F^2\big)$$

The regularization term in ALS, $\big(\lVert A\rVert_F^2 + \lVert B\rVert_F^2\big)$, is double the regularization term in Soft-Impute, $\lVert Z\rVert_*$, for the factorization that is chosen by ALS at the minimum. As the error term is also double, ALS and Soft-Impute end up choosing the same Z.

This equivalence between these two models was checked in an experiment. A synthetic matrix of 30 users and 20 movies is computed from two random matrices with two columns, so the final matrix has rank 2 and some noise added to this matrix. ALS is trained on 300 ratings with weighted = False, k=20 and reg=0.01. Soft-Impute is trained on the same data with the same lambda as the reg parameter in ALS. 

The computed losses below show that ALS loss is exactly double the Soft-Impute loss:

ALS loss: 0.1833334

Soft-Impute loss: 0.0916667

## Hyperparameter search on validation
Starting from a working configuration, I changed one hyperparameter at a time leaving the others constant and selected the one that minimized the validation RMSE. Each run is recorded in experiments.csv with the corresponding commit. Seed noise was measured by repeating the best configuration with four different seeds.

| Model | Chosen configuration | Validation RMSE | Seed noise |
|---|---|---|---|
| Bias baseline | — | 0.8554 | — |
| Truncated SVD | k = 20 | 0.8474 | — (deterministic) |
| Soft-Impute | λ = 12.774 | 0.8377 | — (deterministic) |
| ALS | k = 100, reg = 0.1, weighted | 0.8369 | 0.0004 |
| SGD | k = 150, lr = 0.04, reg = 0.1 | 0.8357 | 0.0016 |

[reports/experiments.csv](reports/experiments.csv)

[configs/best.json](configs/best.json)

Increasing k improves the model only with regularization, because it controls the effective rank; otherwise, the model overfits. As reg increases, the validation RMSE follows a U shape: with too little regularization the model overfits, and with too much it is flattened to the bias baseline. The minimum is at reg = 0.1.
The hyperparameter lr only affects the time it takes to train the model.
Soft-Impute, ALS and SGD have the same performance taking into account the seed noise and truncated SVD is not as good as the others.

## Frozen configuration (before opening test)

Final models are trained on train + validation (split_seed = 42), with a fixed number of epochs where applicable. Code: commit `7f7f88d`, the commit used to run `evaluate.py`.

| Model | Hyperparameters | Epochs | model_seed |
|---|---|---|---|
| Global mean | — | — | — |
| Bias baseline | reg_movie = 3, reg_user = 3 | — | — |
| Truncated SVD | k = 20 | — | — |
| Soft-Impute | λ = 12.774, max_iter = 300, tol = 1e-8 | — | — |
| ALS | k = 100, reg = 0.1, weighted, scale = 0.1 | 7 | 0 |
| SGD | k = 150, lr = 0.04, reg = 0.1, scale = 0.1 | 35 | 0 |

I will compare SGD, the best model in validation, with the bias baseline.

## Test results

Command: `python scripts/evaluate.py --split test`

| Model | Test RMSE | Test MAE |
|---|---|---|
| Global mean | 1.0281 | 0.8161 |
| Bias baseline | 0.8595 | 0.6614 |
| Truncated SVD | 0.8488 | 0.6520 |
| Soft-Impute | 0.8384 | 0.6421 |
| ALS | 0.8383 | 0.6406 |
| SGD | 0.8383 | 0.6409 |

SGD scores RMSE = 0.8383 and baseline model RMSE = 0.8595. SGD outperforms baseline model by 0.0212. A paired bootstrap with 1000 resampling yields a 95% confidence interval [0.0173, 0.0250].

The interval does not include 0, this means the improvement is not explained by the particular ratings that fell in the test set. SGD, ALS and Soft-Impute obtained the same RMSE value and SGD obtained the highest RMSE difference between validation (0.8357) and test (0.8383), this is because SGD was selected for having the lowest validation RMSE, so part of its validation advantage was luck.

## Recommendations and factor interpretation

### Recommendations

Command: `python scripts/recommend.py --user-id 17 --top-n 10`

| movieId | Title | Predicted rating |
|---|---|---|
| 177593 | Three Billboards Outside Ebbing, Missouri (2017) | 4.67 |
| 6460 | Trial, The (Procès, Le) (1962) | 4.63 |
| 1041 | Secrets & Lies (1996) | 4.62 |
| 1178 | Paths of Glory (1957) | 4.60 |
| 1104 | Streetcar Named Desire, A (1951) | 4.59 |
| 3451 | Guess Who's Coming to Dinner (1967) | 4.57 |
| 2239 | Swept Away (Travolti da un insolito destino nell'azzurro mare d'Agosto) (1975) | 4.54 |
| 1204 | Lawrence of Arabia (1962) | 4.53 |
| 106642 | Day of the Doctor, The (2013) | 4.53 |
| 1217 | Ran (1985) | 4.52 |

The movies were recommended using the model Soft-Impute trained with all the data (train + validation + test). The movies that the user already rated are excluded. A list like the above does not imply that the model is able to predict the user ratings. The evaluation of the performance of the model is measured by the test RMSE.

### Factor interpretation

Command: `python scripts/interpret.py`

![Movie map](reports/figures/movie_map.png)

Only movies with at least 50 ratings are considered. The position of a movie on each direction is fitted from its own ratings, so for a movie with very few ratings it is mostly noise. Without this threshold, such movies would fill the extremes of each direction by chance.

The first two directions can be read as:

- **Axis 1:** blockbuster action (*Armageddon*, *Independence Day*, *Twister*) vs. acclaimed or cult films (*The Godfather*, *Pulp Fiction*, *Memento*).
- **Axis 2:** family films (*Toy Story*, *Aladdin*, *E.T. the Extra-Terrestrial*) vs. films for adults (*Se7en*, *American Psycho*, *Natural Born Killers*). The label is approximate: *Schindler's List* also falls on the family side, so the axis is closer to light or consensus cinema vs. dark and violent cinema.

The factors found by SGD yield the same predictions even when multiplied by any invertible matrix $Q$ (with $U$ multiplied by $Q$, and $V$ by the inverse of the transpose of $Q$): $UV^T = (UQ)(VQ^{-T})^T$. Consequently, depending on the random seed used, we may end up with entirely different factors that nonetheless produce the same predictions. SVD, by contrast, selects one specific factorization—one with orthogonal columns ordered by $\sigma$—which is always the same.

## Limitations and next steps

- **Random split, not temporal.** Ratings were split at random, so the models fill gaps in the matrix rather than predict future ratings. A split by date would be a harder and more realistic test.
- **No cold start.** Validation and test rows whose user or movie does not appear in train were removed (3.7% and 3.9% of the rows), so the evaluation says nothing about new users or rarely rated movies, and it is somewhat optimistic.
- **One hyperparameter at a time.** The search changed one hyperparameter at a time; interactions, such as the best reg for a larger k, were not tested.
- **RMSE only.** A recommender is judged by the order of its list as much as by the error of each rating; ranking metrics were not measured.
- **One dataset, one split.** All results come from MovieLens Latest Small and a single split (seed 42). The bootstrap measures the uncertainty due to the test sample, but not the variation across splits.

**Next steps:** a split by date, ranking metrics such as precision@10 and recall@10, and a randomized SVD to speed up Soft-Impute, whose iterations are dominated by a full SVD of the residual matrix (about 0.5 s each).
