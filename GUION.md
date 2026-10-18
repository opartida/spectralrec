# SpectralRec · Guion del proyecto

Proyecto final para **MIT 18.065** *Matrix Methods in Data Analysis, Signal Processing, and Machine Learning* (Gilbert Strang). Construyes tú, paso a paso, un recomendador de películas con el que estudiar sobre datos reales las ideas centrales del curso: SVD, aproximación de rango bajo, mínimos cuadrados con penalización, descenso de gradiente y norma nuclear.

- **Datos:** MovieLens Latest Small (GroupLens).
- **Herramientas:** Python, NumPy, pandas, Matplotlib y pytest, más la biblioteca estándar. Ninguna librería de recomendación ni de machine learning.
- **Regla:** tú escribes el código. El tutor (Claude o Codex) explica, pregunta, revisa tus intentos y da pistas. Solo enseña una solución completa si se la pides después de intentarlo.
- **Resultados:** este guion no te dice qué números vas a obtener. Salga lo que salga, tu trabajo es medirlo bien y explicarlo.

Ábrelo con la vista previa de VS Code (⇧⌘V) para ver las fórmulas.

---

## La pregunta del proyecto

Coloca las notas en una matriz $R$ de usuarios × películas: $R_{ui}$ es la nota que el usuario $u$ dio a la película $i$. Solo conoces una parte de las entradas (en la sesión 04 medirás cuánta). Recomendar es **rellenar los huecos**.

> ¿Cómo se completa una matriz de rango bajo cuando solo se observa una parte de sus entradas?

Modelo común a todo el proyecto:

$$\hat r_{ui} = \underbrace{\mu + b_u + b_i}_{\text{sesgos}} + \underbrace{U_u \cdot V_i}_{\text{rango bajo}}$$

Primero ajustas los sesgos. Después, todos los métodos de rango bajo modelan **el mismo residuo** $\tilde r_{ui} = r_{ui} - (\mu + b_u + b_i)$, para que la comparación sea justa.

Probarás tres respuestas:

1. **Eckart–Young.** Si la matriz estuviera completa, la mejor aproximación de rango $k$ sería la SVD truncada (Lecture 7). ¿Sirve rellenar los huecos con ceros y truncar?
2. **Ajustar solo lo observado.** Medir el error únicamente en las entradas conocidas, penalizando el tamaño de $U$ y $V$. El problema deja de ser convexo; lo resolverás con SGD (Lecture 25) y con mínimos cuadrados alternos, ALS (Lectures 9–10).
3. **Relajación convexa.** Cambiar "rango ≤ k" por "norma nuclear pequeña" (Lecture 8). El algoritmo Soft-Impute usa la SVD en cada paso, y una identidad demuestra que esta vía y la anterior plantean el mismo problema.

Los valores singulares aparecen al principio (diagnóstico), en medio (truncar) y al final (encoger). De ahí el nombre.

---

## 00 · Cómo se usa

Una sesión termina cuando cumples su checklist, no cuando la has leído. En cada una:

1. **Lee** solo lo indicado.
2. **Programa** lo marcado con `TODO`. No copies implementaciones completas de internet.
3. **Comprueba**: `python -m pytest` en verde, más los comandos de la sesión.
4. **Explica** por escrito, con tus palabras, en `reports/notas.md`. Será la base del informe final.
5. **Commit** solo con los tests en verde.

**Prompt fijo para el tutor.** Guárdalo en `AGENTS.md`. Si usas Claude Code, crea también `CLAUDE.md` con la línea `@AGENTS.md`.

```text
Actúa como profesor de este proyecto. Estoy en la sesión N de GUION.md.
No escribas la solución completa, no modifiques archivos y no ejecutes nada por mí.
No me adelantes resultados.
Explícame solo el concepto que pregunte.
Revisa mi intento, señala el primer error y dame una pista.
No avances a la siguiente sesión hasta que cumpla su checklist.
```

**Si te bloqueas:** pega el error completo, muestra solo la función implicada, explica qué esperabas y pide una pista, no la solución.

**Material del curso:** [MIT OCW 18.065, Spring 2018](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) (las "Lectures" citadas son de este curso) y el libro de Strang *Linear Algebra and Learning from Data* (2019).

### Mapa

| Fase | Sesiones | Curso |
|---|---|---|
| A. Datos | 01 entorno y paquete · 02 entender MovieLens · 03 cargar · 04 explorar | |
| B. Evaluación honesta | 05 índices · 06 train/val/test · 07 métricas · 08 sesgos | L8, L9–10 |
| C. Vía espectral | 09 modelo de rango bajo · 10 espectro · 11 SVD truncada | L1–2, L6–7, L17 |
| D. Ajustar lo observado | 12 un paso de SGD · 13 entrenamiento · 14 ALS | L5, L9–10, L21–22, L25 |
| E. Vía convexa | 15 norma nuclear y Soft-Impute | L8, L30 |
| F. Resultados | 16 selección · 17 test · 18 recomendar e interpretar · 19 entrega | |

### Estructura final del repositorio

```text
spectralrec/
├── pyproject.toml
├── README.md
├── GUION.md
├── AGENTS.md
├── configs/              # hiperparámetros en JSON
├── data/raw/             # MovieLens (ignorado por Git)
├── reports/
│   ├── figures/          # figuras generadas por scripts
│   ├── experiments.csv   # registro de experimentos
│   └── notas.md          # tus respuestas
├── results/              # parámetros entrenados (ignorado por Git)
├── scripts/              # puntos de entrada ejecutables
├── src/spectralrec/
│   ├── __init__.py
│   ├── data.py           # carga, validación, índices, partición
│   ├── pipeline.py       # prepara los datos de un experimento
│   ├── metrics.py
│   ├── baselines.py
│   ├── lowrank.py        # predicción, matriz densa, SVD truncada
│   ├── sgd.py
│   ├── als.py
│   ├── softimpute.py
│   └── recommend.py
└── tests/                # test_data.py, test_metrics.py, ...
```

`experiments/` puede servirte para pruebas sueltas que no forman parte del paquete.

---

## Fase A · Datos

### 01 · Entorno y paquete

**Objetivo.** Que `import spectralrec` funcione desde cualquier carpeta y que `python -m pytest` pase desde la raíz.

**Lee.** [venv](https://docs.python.org/3/library/venv.html) · [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) · [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) · [pytest: Get started](https://docs.pytest.org/en/stable/getting-started.html)

**Haz.** En la terminal de VS Code, dentro del repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pytest
```

Crea `src/spectralrec/__init__.py` vacío (convierte la carpeta en un paquete) y `pyproject.toml`:

```toml
[build-system]
requires = ["..."]        # TODO: setuptools, con una versión mínima que admita el modo editable
build-backend = "..."     # TODO

[project]
name = "spectralrec"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = []         # TODO: lo que importa el paquete (pytest no)

[project.optional-dependencies]
dev = []                  # TODO: herramientas solo para desarrollar
```

Instala en modo editable con `python -m pip install -e ".[dev]"` y escribe `tests/test_smoke.py` con un test que solo importe `spectralrec`.

En `.gitignore` deben estar `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `data/raw/` y `results/`.

> **Python 3.9:** no uses `int | None` en anotaciones (es de 3.10; usa `Optional[int]`) ni `match`.

**Comprueba.**

```bash
python -c "import spectralrec; print(spectralrec.__file__)"   # ruta dentro de src/spectralrec/
python -m pytest
git status                                                    # sin .venv, datos ni *.egg-info
```

**Explica.** ¿Por qué el nombre del repo y el del paquete pueden ser distintos? ¿Qué gana el proyecto con `src/`? ¿Por qué pytest va en `dev` y no en `dependencies`?

**Checklist.**
- [ ] `import spectralrec` funciona también desde otra carpeta.
- [ ] `python -m pytest` pasa.
- [ ] `git status` no ofrece añadir datos, el entorno ni archivos generados.

**Commit.** `chore: package skeleton with pyproject and smoke test`

### 02 · Entender MovieLens

**Objetivo.** Poder describir con una frase qué es una fila de `ratings.csv` antes de programar el cargador.

**Haz.** Descarga [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) y descomprímelo en `data/raw/`, de modo que queden `data/raw/ml-latest-small/ratings.csv`, `movies.csv` y `README.txt`.

**Lee.** El [README oficial](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html): *Ratings Data File Structure*, *Movies Data File Structure* y la licencia de uso.

**Escribe en README.md** (sección *Datos*, con la fecha de descarga):
1. ¿Qué representa una fila de `ratings.csv`?
2. ¿En qué escala está `rating` y qué valores puede tomar?
3. ¿`movieId` es la posición de una fila? ¿Son consecutivos los IDs?
4. ¿Qué es `timestamp` y para qué podría servir aquí?
5. ¿Cómo se relacionan `ratings.csv` y `movies.csv`?
6. ¿Por qué los CSV no se suben a GitHub?

**Explica.** La pregunta clave del proyecto: en la matriz usuarios × películas, ¿qué hay en una entrada que nadie ha valorado? ¿Es lo mismo "desconocido" que "cero"?

**Checklist.**
- [ ] El ZIP y los CSV están ignorados por Git.
- [ ] README explica las cuatro columnas de `ratings.csv`.
- [ ] Has respondido a la pregunta clave.

**Commit.** `docs: describe MovieLens data`

### 03 · Cargar los CSV

**Objetivo.** Funciones que lean los CSV y fallen con un mensaje claro si el archivo no es el esperado.

**Lee.** [Tutorial de pandas: leer datos tabulares](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html) · [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) · [pytest: tmp_path](https://docs.pytest.org/en/stable/how-to/tmp_path.html) · [pytest.raises](https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions)

**Programa.** Primero un experimento: lee `ratings.csv` con `pd.read_csv` e imprime `head()`, `shape` y `dtypes`. Cuando entiendas lo que sale, sustitúyelo en `src/spectralrec/data.py` por:

```python
RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
MOVIE_COLUMNS = {"movieId", "title", "genres"}

def load_ratings(path: Path) -> pd.DataFrame:
    ratings = ...  # TODO: leer el CSV
    # TODO: obtener TODAS las columnas que faltan en una sola operación (pista: conjuntos)
    # TODO: si falta alguna, ValueError que diga qué archivo y qué columnas
    return ratings

def load_movies(path: Path) -> pd.DataFrame:
    ...  # TODO: lo mismo con MOVIE_COLUMNS
```

Si la comprobación de columnas se repite en las dos funciones, extráela a una función auxiliar.

**Comprueba.** En `tests/test_data.py`, con CSV diminutos escritos dentro de `tmp_path`:
- un archivo válido devuelve un DataFrame con las filas y columnas esperadas;
- un archivo sin `rating` lanza `ValueError` y el mensaje menciona `rating`;
- ningún test lee `data/raw/`.

**Explica.** ¿Por qué los tests no usan el dataset real? ¿Qué ventaja tiene `tmp_path` frente a una carpeta fija?

**Checklist.**
- [ ] Las funciones devuelven DataFrames.
- [ ] Un CSV incompleto falla diciendo qué falta y en qué archivo.
- [ ] Los tests no dependen de `data/raw/`.

**Commit.** `feat(data): load ratings and movies with column validation`

### 04 · Explorar antes de modelar

**Objetivo.** Conocer tamaño, densidad y forma de los datos, y dejar una ficha reproducible.

**Lee.** Busca en la documentación de pandas: `nunique`, `value_counts`, `isna`, `duplicated`, `groupby(...).size()` y `describe`.

**Programa.**
1. `validate_ratings(df)` en `data.py`: `ValueError` si hay nulos en las columnas necesarias, notas fuera de [0,5, 5] o pares (userId, movieId) repetidos.
2. `scripts/inspect_data.py`, que imprima:
   - el número de filas, de usuarios, de películas del catálogo y de películas valoradas;
   - **densidad** = valoraciones / (usuarios × películas valoradas);
   - distribución de `rating`;
   - valoraciones por usuario y por película: mínimo, mediana y máximo;
   - número de películas con una sola valoración.
3. El mismo script guarda `reports/figures/ratings_per_movie.png`: histograma del número de valoraciones por película, con el eje y logarítmico.

**Comprueba.** Un test por cada fallo de `validate_ratings`, con DataFrames diminutos. No pongas cifras del dataset en los tests: comprueba propiedades, no valores.

**Producto.** Tabla de estadísticas en README (con la fecha de descarga) y la figura.

**Explica.** ¿Qué diferencia hay entre películas del catálogo y películas valoradas? Con la densidad que has medido, ¿cuántas entradas de la matriz conoces de cada 100? ¿Qué problema tendrá un modelo con las películas de una sola valoración?

**Checklist.**
- [ ] La ficha y la figura salen de ejecutar el script.
- [ ] Has guardado estadísticas, no datos.

**Commit.** `feat: data validation and inspection script`

---

## Fase B · Evaluación honesta

### 05 · De IDs a índices

**Objetivo.** Pasar de los IDs de MovieLens a posiciones 0…n−1 de arrays NumPy.

**Lee.** [dataclasses](https://docs.python.org/3/library/dataclasses.html) · [numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)

**Programa.** En `data.py`:

```python
@dataclass
class IndexMaps:
    user_ids: np.ndarray      # posición -> userId
    movie_ids: np.ndarray     # posición -> movieId
    user_to_index: dict       # userId -> posición
    movie_to_index: dict      # movieId -> posición

def build_index_maps(train: pd.DataFrame) -> IndexMaps:
    ...  # TODO: IDs únicos y ordenados de train y los dos diccionarios

def encode_ratings(df: pd.DataFrame, maps: IndexMaps):
    ...  # TODO: devuelve (users, movies, ratings): tres arrays de longitud N
         # TODO: un ID que no esté en los mapas da un error que nombra ese ID
```

**Comprueba.** Con IDs no consecutivos a propósito (usuarios `[10, 30]`, películas `[7, 99]`):
- los índices empiezan en 0 y no tienen huecos;
- codificar y decodificar (`maps.user_ids[users]`) recupera los IDs originales;
- un ID desconocido da un error claro.

**Explica.** ¿Por qué no usar `userId` directamente como número de fila de U? ¿Qué debería pasar si validación contiene un ID que no estaba en train?

**Checklist.**
- [ ] Índices 0…n−1 sin huecos.
- [ ] La ida y vuelta recupera los IDs.
- [ ] Una entidad desconocida da un error claro.

**Commit.** `feat(data): index maps and encoding`

### 06 · Train, validación y test

**Objetivo.** Medir generalización sin entrenar con las respuestas del examen.

**Lee.** [NumPy: Generator](https://numpy.org/doc/stable/reference/random/generator.html)

Partición aleatoria de aproximadamente 80/10/10:
- **Train:** ajusta medias, sesgos y factores.
- **Validación:** elige k, regularización, learning rate y épocas.
- **Test:** se abre **una sola vez**, en la sesión 17.

**Programa.** Practica antes con diez números: `rng = np.random.default_rng(42)`, `order = rng.permutation(10)`, y reparte 8, 1 y 1. Después, en `data.py`:

```python
def split_ratings(df, seed: int, ratios=(0.8, 0.1, 0.1)):
    ...  # TODO: UNA permutación de las filas, cortada en tres trozos

def drop_unknown(df, train):
    ...  # TODO: quita las filas cuyo userId o movieId no estén en train
         #       devuelve (df_filtrado, n_eliminadas)
```

No uses `DataFrame.sample` tres veces: podrías repetir filas.

**Dos semillas.** Una `split_seed` fija para todo el proyecto y otra `model_seed` para inicializar modelos. Así puedes repetir un modelo con otra semilla sin cambiar la partición.

**Decisión** (documéntala en README): con una partición aleatoria predices huecos de la matriz, no el futuro. La partición temporal queda como extensión.

**Comprueba.**
- Los tres conjuntos no comparten filas y su unión es el total.
- La misma semilla da exactamente la misma partición; otra semilla, otra distinta.
- Tras `drop_unknown` no queda ningún ID desconocido para train, y el recuento de eliminadas es correcto en un ejemplo pequeño.

**Explica.** ¿Por qué tres conjuntos y no dos? ¿Qué tipo de películas desaparecen de val/test al filtrar y cómo sesga eso la evaluación?

**Checklist.**
- [ ] Partición reproducible con la semilla.
- [ ] Ningún ID desconocido en val/test.
- [ ] README dice cuántas filas se eliminaron.

**Commit.** `feat(data): reproducible train/val/test split`

### 07 · RMSE y MAE

**Objetivo.** Una regla fiable para comparar modelos.

Para el vector de errores $e = y - \hat y$ de longitud $N$ (normas de la Lecture 8):

$$\text{MAE} = \frac{\lVert e\rVert_1}{N} \qquad \text{RMSE} = \frac{\lVert e\rVert_2}{\sqrt N}$$

**Programa.** En `metrics.py`:

```python
def mae(y_true, y_pred) -> float:
    ...  # TODO: validar formas, calcular y devolver un float

def rmse(y_true, y_pred) -> float:
    ...  # TODO
```

**Comprueba.**
- Con `y_true=[1, 3]` y `y_pred=[2, 1]`, calcula MAE y RMSE a mano antes de escribir el `assert`.
- Longitudes distintas o arrays vacíos → `ValueError`.
- Compara floats con `pytest.approx`.

No redondees predicciones antes de medir: un modelo puede predecir 3,72 aunque las notas vayan de 0,5 en 0,5.

**Explica.** ¿Por qué RMSE castiga más los errores grandes? Construye dos vectores de error con el mismo MAE y distinto RMSE.

**Checklist.**
- [ ] Coinciden con el cálculo a mano.
- [ ] Fallan con formas incompatibles.

**Commit.** `feat(metrics): RMSE and MAE`

### 08 · Medias y sesgos con penalización

**Objetivo.** Fijar el listón que tendrá que superar el rango bajo y resolver tu primer problema de mínimos cuadrados con penalización.

**Lee.** Koren, Bell y Volinsky (2009), apartado *Adding biases* · Lectures 9–10 (mínimos cuadrados y penalización) · [numpy.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html)

**Paso 0: `pipeline.py`.** Escribe `prepare_data(ratings_path, split_seed)`: carga, valida, parte, filtra, construye los mapas **solo con train** y codifica. Todos los scripts la usarán.

**Programa.** En `baselines.py`, tres modelos con la misma interfaz:

```python
class GlobalMean:
    def __init__(self, n_users: int, n_movies: int): ...
    def fit(self, users, movies, ratings): ...     # TODO
    def predict(self, users, movies): ...          # TODO: array de forma (N,)
```

1. `GlobalMean`: siempre la media de train.
2. `UserMean`: la media del usuario en train.
3. `BiasBaseline(reg_movie, reg_user)`: $\mu + b_u + b_i$.

**En papel, antes de programar.** Con $\mu$ fija, el sesgo de la película $i$ resuelve

$$\min_{b}\ \sum_{u \in R(i)} (r_{ui} - \mu - b)^2 + \lambda\, b^2$$

Deriva la solución: es ridge en una dimensión. ¿Qué pasa con $\lambda = 0$ en una película con una sola valoración? ¿Y con $\lambda \to \infty$? Calcula después $b_u$ del mismo modo, con los $b_i$ ya fijados.

**Pista.** `np.bincount(users, weights=..., minlength=n_users)` suma por usuario sin bucles.

**Comprueba.**
- Valores calculados a mano en un ejemplo de 2 usuarios × 3 películas.
- Con λ enorme, `BiasBaseline` predice casi la media global.

**Producto.** `scripts/baselines.py` imprime y guarda en `reports/baselines_val.csv` el RMSE y el MAE de validación de cada modelo. Prueba unos pocos λ, elige por validación y guarda la elección en `configs/baseline.json`.

**Checklist.**
- [ ] Cada modelo usa solo datos de train en `fit`.
- [ ] Los tres comparten `fit`/`predict`.
- [ ] La tabla sale de un comando.

**Commit.** `feat(baselines): global mean, user mean and regularized biases`

---

## Fase C · La vía espectral

### 09 · El modelo de rango bajo

**Objetivo.** Saber qué son U y V y predecir pares (u, i) sin construir la matriz completa.

**Lee.** Lectures 1–2 (rango; el producto como suma de matrices de rango 1) · Koren et al. (2009), *A basic matrix factorization model*

Desde aquí el objetivo es el **residuo** $\tilde r_{ui} = r_{ui} - (\mu + b_u + b_i)$, con el `BiasBaseline` de la sesión 08 ajustado solo con train. Amplía `prepare_data` para que devuelva también los residuos.

**Programa.** Escribe primero las formas como comentario:

```text
U (user_factors):  (n_users, k)
V (movie_factors): (n_movies, k)
users, movies:     (N,)
predicciones:      (N,)
```

En `lowrank.py`:

```python
def predict_observed(user_factors, movie_factors, users, movies) -> np.ndarray:
    ...  # TODO: U[u] · V[i] para cada par, forma (N,). Prohibido construir U @ V.T

def to_dense(users, movies, values, shape) -> np.ndarray:
    ...  # TODO: matriz de ceros con `values` en las posiciones (users, movies)
```

**Pista.** [Indexado avanzado](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing): `U[users]` tiene forma `(N, k)`.

**En papel.** Escribe $UV^\top = \sum_{j=1}^{k} U_{:,j}\, V_{:,j}^\top$. ¿Por qué implica rango ≤ k? ¿Cuántos números tiene el modelo frente a usuarios × películas?

**Comprueba.**
- `predict_observed` coincide con productos escalares hechos a mano (k = 2, 2 usuarios, 3 películas).
- Devuelve N predicciones para N pares.
- `to_dense` coloca cada valor en su sitio y deja ceros en el resto.

**Explica.** ¿Por qué U[u] y V[i] tienen la misma longitud? ¿Qué significa describir una película con k números? ¿Por qué el RMSE calculado sobre residuos es igual al calculado sobre notas?

**Checklist.**
- [ ] Formas escritas y comprobadas por los tests.
- [ ] `predict_observed` no construye la matriz completa.

**Commit.** `feat(lowrank): residuals, observed predictions and dense builder`

### 10 · El espectro

**Objetivo.** Mirar los valores singulares de la matriz de residuos y decidir, con un control, qué dicen.

**Lee.** Lecture 6 (SVD) · Lecture 17 (*Rapidly decreasing singular values*) · [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html): fíjate en `full_matrices`, en `compute_uv` y en que devuelve $V^\top$, no $V$.

**Programa.** `scripts/spectrum.py`:
1. $M$ = `to_dense` de los residuos de train, con ceros en los huecos.
2. Sus valores singulares $\sigma_1 \ge \sigma_2 \ge \dots$
3. **Control:** baraja los residuos entre las posiciones observadas (mismos huecos y mismos valores, sin estructura) y calcula su espectro.
4. Guarda `reports/figures/spectrum.png`: $\sigma_j$ frente a $j$ (eje y logarítmico) de la matriz real y del control, y la energía acumulada $\sum_{j \le k}\sigma_j^2 / \sum_j \sigma_j^2$.
5. Repite la comparación con una submatriz densa: los usuarios y películas con más valoraciones. ¿Cambia la conclusión?

**Comprueba** (con matrices sintéticas, no con MovieLens):
- $BC^\top$ con $B$ de 20×3 y $C$ de 15×3 tiene exactamente 3 valores singulares no despreciables.
- $\lVert M\rVert_F^2 = \sum_j \sigma_j^2$.

**Explica** (bajo la figura, en README). Compara la matriz real con el control. Demuestra con la SVD que $\lVert M\rVert_F^2 = \sum_j \sigma_j^2$. Los ceros de los huecos no son datos: ¿qué papel juegan en el espectro?

**Checklist.**
- [ ] La figura incluye el control y sale del script.
- [ ] La interpretación está escrita junto a la figura.

**Commit.** `feat: singular value spectrum with shuffled control`

### 11 · SVD truncada (Eckart–Young)

**Objetivo.** Tu primer modelo de rango bajo, sin entrenamiento iterativo.

**Lee.** Lecture 7 (*Eckart–Young: the closest rank k matrix to A*)

Si $M = U\Sigma V^\top$, la matriz de rango $k$ más cercana a $M$ (en norma de Frobenius y espectral) es $M_k = U_k \Sigma_k V_k^\top$, y

$$\lVert M - M_k\rVert_F^2 = \sum_{j > k} \sigma_j^2$$

**Programa.** En `lowrank.py`:

```python
class TruncatedSVD:
    def __init__(self, k: int): ...
    def fit(self, users, movies, residuals, shape):
        M = ...                       # TODO
        U, s, Vt = ...                # TODO: SVD reducida
        self.user_factors = ...       # TODO: ¿dónde pones s?
        self.movie_factors = ...       # TODO
    def predict(self, users, movies):
        ...                           # TODO: usa predict_observed
```

Predicción final = sesgos + residuo de rango bajo.

**Producto.** `scripts/svd_sweep.py`: RMSE de validación para varios k (por ejemplo 1, 2, 5, 10, 20, 50, 100), en tabla y en `reports/figures/svd_k.png`. Haz la SVD una vez y trunca para cada k.

**Comprueba.**
- Una matriz sintética completa de rango 2 se reconstruye (casi) exactamente con k = 2.
- Con k = 1, $\lVert M - M_1\rVert_F^2 = \sum_{j>1}\sigma_j^2$: Eckart–Young comprobado numéricamente.

**Explica.**
- ¿Dónde has puesto Σ: en U, en V o repartida como $\sqrt\Sigma$? ¿Cambia la predicción? ¿Cambiaría si penalizaras el tamaño de U y V?
- ¿Qué pasa en train y en validación al aumentar k? ¿Por qué?
- ¿Qué efecto tienen los ceros de los huecos sobre las predicciones?
- Opcional: si cada entrada se observara al azar con probabilidad $p$, $\mathbb E[M/p]$ sería la matriz completa. Prueba a reescalar. ¿Se observan al azar las notas de MovieLens?

**Checklist.**
- [ ] Test de Eckart–Young en verde.
- [ ] Tabla y figura de RMSE de validación frente a k.

**Commit.** `feat(lowrank): truncated SVD predictor`

---

## Fase D · Ajustar solo lo observado

### 12 · La pérdida y un paso de SGD

**Objetivo.** Actualizar una observación correctamente antes de recorrer el dataset.

**Lee.** Lectures 21–22 (descenso de gradiente) · Lecture 25 (SGD) · Koren et al. (2009), *Learning algorithms*

Pérdida sobre los residuos observados $\Omega$:

$$L(U,V) = \sum_{(u,i)\in\Omega}\Big[(\tilde r_{ui} - U_u\cdot V_i)^2 + \lambda\big(\lVert U_u\rVert^2 + \lVert V_i\rVert^2\big)\Big]$$

Fíjate en que la regularización está dentro de la suma. Lo necesitarás en la sesión 14.

**En papel.** Deriva el gradiente del término de una observación respecto de $U_u$ y de $V_i$, escribe las dos actualizaciones y decide qué haces con el factor 2. Pide al tutor que revise la derivación, no que la haga.

**Programa.** En `sgd.py`:

```python
@dataclass
class FactorModel:
    user_factors: np.ndarray   # (n_users, k)
    movie_factors: np.ndarray   # (n_movies, k)

def init_factor_model(n_users, n_movies, k, scale, rng) -> FactorModel:
    ...  # TODO: valores aleatorios pequeños

def sgd_step(model, u, i, residual, lr, reg) -> float:
    prediction = ...                   # TODO
    error = ...                        # TODO
    old_user = ...                     # TODO: copia (¿por qué?)
    model.user_factors[u] += ...       # TODO
    model.movie_factors[i] += ...       # TODO: usa old_user
    # TODO: si el error no es finito o se dispara, lanza FloatingPointError
    return error
```

**Comprueba.**
- Una actualización coincide con tu cálculo en papel (k = 2, números elegidos por ti).
- Con `reg=0` desaparecen los términos de regularización.
- Un learning rate enorme acaba en `FloatingPointError`, no en NaN silenciosos.

**Orden.** Matriz sintética pequeña → una actualización → diez → una época → MovieLens. Si saltas directamente al dataset, un bug puede parecer un modelo malo.

**Explica.** ¿Por qué no inicializar a ceros? Calcula el gradiente en U = V = 0. ¿Y si todas las columnas empiezan iguales? ¿Qué pasaría si actualizas V con la U ya modificada?

**Checklist.**
- [ ] Una actualización coincide con el papel.
- [ ] Sin NaN ni infinitos silenciosos.

**Commit.** `feat(sgd): factor model and single SGD step`

### 13 · Bucle de entrenamiento y curvas

**Objetivo.** Entrenar por épocas, vigilar validación, conservar el mejor estado y dibujarlo.

**Lee.** [Matplotlib: quick start](https://matplotlib.org/stable/users/explain/quick_start.html)

**Programa.**

```python
@dataclass
class TrainingHistory:
    epoch: list        # TODO: cuidado con los valores por defecto mutables (field(default_factory=...))
    train_rmse: list
    val_rmse: list

def train_sgd(train, val, n_users, n_movies, k, lr, reg, scale, n_epochs, patience, seed):
    users, movies, residuals = train
    rng = np.random.default_rng(seed)
    model = init_factor_model(n_users, n_movies, k, scale, rng)
    for epoch in range(n_epochs):
        for j in rng.permutation(len(residuals)):
            ...        # TODO: sgd_step de la observación j
        ...            # TODO: RMSE de train y de validación, guardados en el historial
        ...            # TODO: si mejora, COPIA el estado (no guardes una referencia)
        ...            # TODO: para si no mejora en `patience` épocas
    ...                # TODO: restaura el mejor estado
    return model, history
```

**Comprueba.** Primero con datos sintéticos, en un test: genera $R = BC^\top$ + ruido (rango 2, 30×20), observa la mitad de las entradas y entrena. La pérdida de entrenamiento debe bajar y el RMSE en las entradas ocultas debe acercarse al nivel del ruido.

**Producto.** `scripts/train.py --config configs/sgd.json`:
- lee la configuración (`model`, `split_seed`, `model_seed`, `k`, `lr`, `reg`, `scale`, `n_epochs`, `patience`);
- entrena, añade una fila a `reports/experiments.csv` y guarda los parámetros en `results/` (`np.savez`);
- dibuja `reports/figures/learning_curve_sgd.png`: épocas en x, RMSE en y, train y validación, con leyenda.

Mide el tiempo por época con `time.perf_counter()` y anótalo.

**Cómo leer la curva.**
- Train y validación bajan: aprende.
- Train baja y validación sube: sobreajusta.
- Las dos casi planas: k bajo, demasiada regularización, learning rate pequeño o un bug.
- Saltos o NaN: learning rate alto o error de signo.

**Explica** (bajo la figura). Cuándo deja de mejorar validación, si hay sobreajuste y qué cambio probarás después.

**Checklist.**
- [ ] El orden cambia cada época pero es reproducible con la semilla.
- [ ] Se restaura el mejor modelo de validación.
- [ ] La figura sale de un script y tiene ejes y leyenda.

**Commit.** `feat(sgd): training loop with early stopping and learning curve`

### 14 · ALS: mínimos cuadrados alternos

**Objetivo.** Atacar la misma pérdida con otro método, en el que cada subproblema es un ridge con solución exacta.

**Lee.** Lecture 9 (mínimos cuadrados) · Lecture 10 (penalización) · Lecture 5 (matrices definidas positivas) · [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) · [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)

Con V fija, la pérdida se separa por usuarios. Si $F$ son las filas de V de las películas que valoró el usuario $u$ y $r$ sus residuos:

$$\min_x\ \lVert Fx - r\rVert^2 + \lambda'\lVert x\rVert^2 \quad\Longrightarrow\quad (F^\top F + \lambda' I)\,x = F^\top r$$

Después se fija U y se actualizan las películas del mismo modo.

**Programa.** En `als.py`:

```python
def solve_ridge(F, r, reg) -> np.ndarray:
    A = ...      # TODO: Fᵀ F + reg · I
    b = ...      # TODO: Fᵀ r
    return np.linalg.solve(A, b)
```

**La trampa de λ′.** ¿Qué λ′ corresponde a la pérdida de tu SGD, con la regularización dentro de la suma? (pista: cuenta cuántas veces aparece $\lVert U_u\rVert^2$). Implementa dos variantes con un parámetro `weighted`:
- `weighted=True`: el λ′ que acabas de deducir, para comparar con SGD sobre la misma pérdida;
- `weighted=False`: λ′ = λ, que necesitarás en la sesión 15.

**Agrupa sin buscar cada vez.** Precalcula, para cada usuario y cada película, los índices de sus observaciones. Funciones útiles: `np.argsort`, `np.bincount`, `np.cumsum` y `np.split`.

**Comprueba.**
- `solve_ridge` coincide con la solución de mínimos cuadrados del sistema ampliado `[F; √λ·I] x ≈ [r; 0]` (con `lstsq`): dos caras del mismo problema.
- La agrupación conserva cada observación exactamente una vez.
- En datos sintéticos, la pérdida de entrenamiento no sube tras cada medio paso (usuarios, luego películas).

**Producto.** Amplía `scripts/train.py` con `model: "als"` y genera `learning_curve_als.png`. Compara con SGD en la misma partición, con el mismo k y la misma pérdida: RMSE de validación, iteraciones o épocas, y segundos.

**Explica.** ¿Por qué $F^\top F + \lambda I$ es invertible si $\lambda > 0$? ¿Por qué `solve` y no `inv`? ¿Por qué un medio paso de ALS no puede empeorar la pérdida de entrenamiento? ¿Le pasa lo mismo a una época de SGD?

**Checklist.**
- [ ] Ninguna inversa explícita.
- [ ] Una iteración actualiza usuarios y después películas.
- [ ] SGD y ALS comparados con exactamente la misma partición.

**Commit.** `feat(als): alternating least squares`

---

## Fase E · La vía convexa

### 15 · Norma nuclear y Soft-Impute

**Objetivo.** Resolver la versión convexa del problema y cerrar el círculo con la SVD.

**Lee.** Lecture 8 (norma nuclear) · Lecture 30 (*Completing a rank-one matrix*) · [Mazumder, Hastie y Tibshirani (2010)](https://www.jmlr.org/papers/v11/mazumder10a.html), secciones 1–3

El rango no es convexo. Su sustituto convexo es la **norma nuclear** $\lVert Z\rVert_* = \sum_j \sigma_j(Z)$, la norma ℓ1 de los valores singulares:

$$\min_Z\ \tfrac12 \sum_{(u,i)\in\Omega}(\tilde r_{ui} - Z_{ui})^2 + \lambda\lVert Z\rVert_*$$

**Soft-Impute** repite tres pasos:
1. **Rellenar:** $X$ toma los residuos observados en $\Omega$ y los valores de $Z$ en los huecos.
2. **Encoger:** si $X = U\Sigma V^\top$, entonces $Z \leftarrow U\,\max(\Sigma - \lambda, 0)\,V^\top$.
3. **Parar** cuando $\lVert Z_{\text{nuevo}} - Z\rVert_F^2 / \lVert Z\rVert_F^2$ baje de una tolerancia.

El paso 2 es un Eckart–Young suave: en lugar de quedarte con k valores singulares, restas λ a todos y descartas los que quedan negativos.

**Programa.** En `softimpute.py`:

```python
def soft_threshold_svd(X, lam):
    U, s, Vt = ...           # TODO: SVD reducida
    s_shrunk = ...           # TODO
    keep = ...               # TODO: solo los valores > 0
    return U[:, keep], s_shrunk[keep], Vt[keep]

def soft_impute(users, movies, residuals, shape, lam, max_iter=100, tol=1e-4, warm_start=None):
    ...  # TODO: rellenar, encoger y comprobar convergencia; devuelve (U, s, Vt)
```

**En papel: la identidad que une las fases D y E.**

$$\lVert Z\rVert_* = \min_{AB^\top = Z}\ \tfrac12\big(\lVert A\rVert_F^2 + \lVert B\rVert_F^2\big)$$

- Demuestra que el mínimo es ≤ $\lVert Z\rVert_*$ eligiendo, con la SVD de Z, $A = U\Sigma^{1/2}$ y $B = V\Sigma^{1/2}$.
- La otra desigualdad usa $\lVert AB^\top\rVert_* \le \lVert A\rVert_F\lVert B\rVert_F$ y la desigualdad entre medias aritmética y geométrica. Inténtalo o cita a Mazumder et al. (2010).
- Consecuencia: con k suficientemente grande, la factorización penalizada con $\tfrac{\lambda}{2}(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)$ (tu ALS con `weighted=False`) y Soft-Impute con el mismo λ plantean el mismo problema.

**Comprueba.**
- `soft_threshold_svd` sobre `diag(5, 3, 1)` con λ = 2: valores singulares calculados a mano.
- Si λ ≥ $\sigma_1$ de la matriz de observados (con ceros en los huecos), la solución es la matriz nula. Explica por qué (pista: primera iteración desde Z = 0).
- Identidad numérica: para una Z aleatoria, $A = U\sqrt\Sigma$ y $B = V\sqrt\Sigma$ dan $\tfrac12(\lVert A\rVert_F^2 + \lVert B\rVert_F^2) = \sum_j\sigma_j$.
- Experimento sintético: compara el valor del objetivo de ALS (`weighted=False`, k grande) y el de Soft-Impute con el mismo λ. ¿Cuadra con la identidad?

**Producto.** En MovieLens, recorre un camino de λ decreciente que empiece en $\sigma_1$, usando cada solución como `warm_start` de la siguiente. Guarda `reports/figures/softimpute_path.png` con el RMSE de validación y el rango de Z frente a λ. Mide cuánto tarda cada iteración.

**Explica.** ¿Por qué penalizar la suma de los $\sigma_j$ favorece el rango bajo? (compáralo con ℓ1 y vectores dispersos). Truncar frente a encoger: ¿qué hace cada uno con los valores singulares grandes? ¿Qué rango elige Soft-Impute en el mejor λ y cómo se compara con el mejor k de SGD y ALS?

**Checklist.**
- [ ] Identidad demostrada (al menos la parte con la SVD) y comprobada numéricamente.
- [ ] La figura del camino de λ sale de un script.

**Commit.** `feat(softimpute): nuclear norm regularization via Soft-Impute`

---

## Fase F · Resultados

### 16 · Elegir hiperparámetros sin tocar test

**Objetivo.** Comparar pocas configuraciones, con hipótesis claras, solo en validación.

Una cuadrícula pequeña por modelo: `k` para la SVD truncada; `k`, `reg` y `lr` para SGD; `k` y `reg` para ALS; el camino de λ para Soft-Impute. No lances todas las combinaciones de golpe: parte de una configuración que funcione y cambia una dimensión cada vez.

Cada fila de `reports/experiments.csv`: `run_id, git_commit, model, split_seed, model_seed, k, reg, lr, lam, weighted, best_epoch, train_rmse, val_rmse, val_mae, seconds` (pista para el commit: `git rev-parse --short HEAD`).

**Ruido de semilla.** Repite la mejor configuración de cada modelo con 3 `model_seed` distintas, con la misma partición. Si dos modelos difieren menos que esa variación, no digas que uno es mejor.

Guarda las configuraciones elegidas en `configs/best.json`.

**Explica.** ¿Aumentar k siempre ayuda? ¿Qué pasa con la distancia entre train y validación? ¿Qué regularización evita el sobreajuste sin aplanar el modelo?

> **Prohibido** elegir mirando test. Si lo haces, test se convierte en otra validación y deja de estimar el rendimiento final.

**Checklist.**
- [ ] Cada fila permite repetir el experimento.
- [ ] La elección se justifica con datos de validación.

**Commit.** `feat: hyperparameter search on validation`

### 17 · Abrir test una sola vez

**Objetivo.** El resultado final, sin modificar después los modelos elegidos.

**Antes de abrir test**, congela por escrito en README: commit, semillas, partición, hiperparámetros y mejor época de cada modelo. Decide también si entrenas solo con train o con train + val y un número fijo de épocas.

**Producto.** `scripts/evaluate.py --split test` produce una tabla con `test_RMSE` y `test_MAE` de: media global, sesgos, SVD truncada, SGD, ALS y Soft-Impute.

**Incertidumbre.** Haz un bootstrap emparejado sobre las filas de test (por ejemplo 1000 remuestreos con reemplazo, las mismas filas para los dos modelos) de la diferencia de RMSE entre tu mejor modelo y los sesgos. Da el intervalo del 95 %.

Si el rango bajo no mejora a los sesgos, el proyecto no fracasa: investiga fuga de datos, errores de actualización, regularización o falta de señal, y explícalo.

> **Después de abrir test** no ajustes nada mirando el resultado. Si lo haces, declara que test ha dejado de serlo.

**Checklist.**
- [ ] La tabla sale de un comando reproducible.
- [ ] Ningún modelo cambió tras ver test.
- [ ] La diferencia principal lleva su intervalo.

**Commit.** `feat: final test evaluation`

### 18 · Recomendar e interpretar

**Objetivo.** Una salida comprensible y una lectura geométrica de los factores.

**Parte A · Recomendaciones.** En `recommend.py`, con `scripts/recommend.py --user-id 17 --top-n 10`:

```python
def recommend(model, user_id, maps, movies, seen_movie_ids, n=10) -> pd.DataFrame:
    # TODO: validar que el usuario existe
    # TODO: puntuar las películas conocidas por el modelo
    # TODO: excluir las ya valoradas
    # TODO: ordenar y devolver movieId, title, score
    ...
```

Tests: ninguna recomendada está en el historial del usuario; salen n filas ordenadas; un usuario desconocido da error. Decide qué haces con puntuaciones fuera de [0,5, 5]: recortarlas solo para mostrarlas, nunca para medir sin declararlo.

**Parte B · Interpretar.** Soft-Impute te da Z en forma SVD. En `scripts/interpret.py`, para las dos primeras direcciones (columnas de V):
- lista las 10 películas con valor más alto y las 10 con valor más bajo, solo entre películas con suficientes valoraciones en train (fija tú el umbral);
- guarda `reports/figures/movie_map.png`: las películas más valoradas en el plano de esas dos direcciones, con algunos títulos.

**Explica.** Pon nombre a cada eje, si puedes. ¿Por qué los ejes de la SVD están bien definidos y los U, V de SGD no? (pista: $UV^\top = (UQ)(VQ^{-\top})^\top$ para cualquier Q invertible). ¿Por qué una lista de recomendaciones bonita no demuestra precisión?

**Checklist.**
- [ ] Las recomendaciones tienen títulos y excluyen lo ya visto.
- [ ] La interpretación de los ejes se apoya en ejemplos.

**Commit.** `feat: recommendations and factor interpretation`

### 19 · Entrega

**Objetivo.** Que otra persona reproduzca tus resultados sin recibir los datos.

**README**, en este orden:
1. Pregunta del proyecto y qué se predice.
2. Datos: qué son, cómo descargarlos y por qué no están en el repo.
3. Instalación y comandos exactos.
4. Partición y cómo se evita la fuga.
5. Métodos: sesgos, SVD truncada, SGD, ALS y Soft-Impute.
6. Validación, decisión tomada y tabla final de test con su incertidumbre.
7. Figuras, interpretación, limitaciones y siguiente mejora.

**Informe para 18.065** (ajústalo a lo que pida el curso), a partir de `reports/notas.md`:
1. Completar matrices y por qué rango bajo.
2. Eckart–Young y qué ocurre con los huecos.
3. Factorización penalizada: SGD y ALS (ridge, matrices definidas positivas).
4. Norma nuclear, Soft-Impute y la identidad que une los dos enfoques.
5. Experimentos, resultados e interpretación de los factores.
6. Limitaciones.

**Comandos que deben funcionar desde un clon limpio:**

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/inspect_data.py
python scripts/spectrum.py
python scripts/train.py --config configs/best.json
python scripts/evaluate.py --split test
python scripts/recommend.py --user-id 17 --top-n 10
python scripts/interpret.py
```

Adapta los nombres a tu interfaz, pero que nadie tenga que adivinar cómo se ejecuta.

**Checklist.**
- [ ] Todos los tests pasan.
- [ ] `git status` no contiene datos ni archivos temporales.
- [ ] El README permite reproducir las tablas.
- [ ] El último commit contiene código, tests y explicación.

---

## Extensiones opcionales

- **Partición temporal:** entrena con el pasado y evalúa con el futuro usando `timestamp`.
- **Métricas de ranking:** precision@10 y recall@10.
- **SVD aleatorizada** (Lecture 13) para acelerar Soft-Impute.
- **Usuario nuevo:** con V fija, las valoraciones de un usuario nuevo se convierten en su vector con `solve_ridge`, sin reentrenar.
- **Sesgos aprendidos dentro de SGD**, como en Koren et al. (2009).

## Fuentes

- G. Strang, *Linear Algebra and Learning from Data*, Wellesley-Cambridge Press, 2019.
- MIT OpenCourseWare, 18.065 *Matrix Methods in Data Analysis, Signal Processing, and Machine Learning*, Spring 2018.
- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer* 42(8), 2009.
- R. Mazumder, T. Hastie, R. Tibshirani, "Spectral Regularization Algorithms for Learning Large Incomplete Matrices", *JMLR* 11, 2010.
- E. Candès, B. Recht, "Exact Matrix Completion via Convex Optimization", *Foundations of Computational Mathematics* 9, 2009.
- F. M. Harper, J. A. Konstan, "The MovieLens Datasets: History and Context", *ACM TiiS* 5(4), 2015.
- Documentación oficial de NumPy, pandas, Matplotlib, pytest y la Python Packaging User Guide.

*Guion educativo independiente inspirado en MIT 18.065. No es material oficial del MIT ni de GroupLens.*
