# Notas

## Sesión 02

**En la matriz usuarios × películas, ¿es lo mismo "desconocido" que "cero"?**
No, desconocido es que no se calificó esa película. 0 no es posible y seria un error en los datos


1. **¿Qué representa una fila de `ratings.csv`?**
   La calificacion de una pelicula por un usuario específico

2. **¿En qué escala está `rating` y qué valores puede tomar?**
   0.5-5 con incrementos de 0.5

3. **¿`movieId` es la posición de una fila? ¿Son consecutivos los IDs?**
   no es el id que identifica l apelícula en movies.csv. No son consecutivos.

4. **¿Qué es `timestamp` y para qué podría servir en este proyecto?**
   segundos desde medianoche UTC 1 de enero de 1970. el timestamp permite separar por fechas, entrenar con el pasado y comprobar con el futuro.  

5. **¿Cómo se relacionan `ratings.csv` y `movies.csv`?**
   una fila de ratings contiene el movieId que se califica y ese id identifica la pelicula en movies.csv

6. **¿Por qué los CSV no se suben a GitHub?**
   Por que la licencia del repo puede no coincidir con la de los datos

## Sesión 3

1. **¿Por qué los tests no usan el dataset real?**
    Los datos no están en el repo y se necesitarían para ejecutar los tests. Con pocos datos sabes exactemente el resultado esperado del test
2. **¿Qué ventaja tiene tmp_path frente a una carpeta fija?**
    Se crea especificamente para el test y no hace falta tener ficheros con datos de test en el proyecto. cada test recibe carpeta vacia y asi ningún test puede estropear otro.

## Sesión 4

1. **¿Qué diferencia hay entre películas del catálogo y películas valoradas?**
   Las películas del catálogo son todas las que hay y que pueden recibir una valoración. Las valoradas son las películas del catálogi que han recibido una valoración. Una película sin valoraciones no aparece en la matriz y el modelo no puede aprender de ella.
2. **Con la densidad que has medido, ¿cuántas casillas de la matriz conoces de cada 100?**
   1.70
3. **¿Qué problema tendrá un modelo con las películas de una sola valoración?**
   No se tendrá un indicador robusto. No podemos saber si realmente la película es buena o mala.

## Sesión 5

1. **¿Por qué no usar el userId directamente como número de fila de U?**
   Porque los ids pueden ser numeros no correlativos y si se usara el id como número de fila la matriz podría ser muy grnade si estos ids son grandes
2. **¿Qué debería pasar si validación contiene un ID que no estaba en train?**
   Quitarlo del conjunto y contar cuantas se quitan.

## Sesión 6

1. **¿Por qué tres conjuntos y no dos?**
   Training es para generar el modelo. Validation para seleccionar el mejor modelo y testing para test final del modelo. El modelo se tiene que evaluar con datos que no haya sido usados para hacer la elección del mismo. La evaluacion del modelo estaría inflada ya que pudo obtener una buena evaluacion simplemente por azar.
2. **¿Qué tipo de películas desaparecen de val/test al filtrar y cómo sesga eso la evaluación?**
   Las películas que no tengan usuario o pelicula que no haya sido usado para generar el modelo, es decir, que no hayan aparecido en training data set. Es decir pelícuals con pocas valoraciones. Si tiene una sola valoración si cae en validación o test será eliminada. Al quitarlas la valoración del modelo será mejor porque valorar películas raras es mas dificil. Además, el resultado no dice nada de cómo funciona el modelo con películas raras.

## Sesión 7

1. **¿Por qué RMSE castiga más los errores grandes?**
   Porque cada error se eleva al cuadrado. Si el error se duplica su aportacion a mae se duplica pero su aportación a la suma de cuadrados se cuadriplica
   
2. **Construye dos vectores de error con el mismo MAE y distinto RMSE.**
   error1 = (1, 1, 1)
   error2 = (3, 0, 0)
   mae = 1
   rmse1 = 1
   rmse2 = 1.73

## Sesión 8

$$b_i = \frac{\sum_{u \in R(i)} r_{ui} - n_i\mu}{\lambda_i + n_i}
$$

$$b_u = \frac{\sum_{i \in R(u)} r_{ui} - n_u\mu - \sum_{i \in R(u)} b_i}{\lambda_u + n_u} = 
\frac{\sum_{i \in R(u)} (r_{ui} - b_i - \mu)} {\lambda_u + n_u}
$$

1. **Con $\lambda = 0$: ¿qué es $b_i$? Descríbelo con palabras. ¿Cuánto valdría para una película con una sola valoración de 5 estrellas, si $\mu$ es 3.5? ¿Te fiarías de ese valor?**
Cuanto se desvía la media de la película de la media global. Valdría 5 - 3.5 = 1.5. No me fiaría esa desviación puede ser casualidad no hay suficientes datos para tener un valor robusto.

2. **Con $\lambda$ muy grande: ¿hacia dónde va $b_i$? ¿Y la predicción $\mu + b_i$?**
A 0. A la media mu.

3. **Para verlo de un golpe: escribe $b_i$ como (un factor que depende de $n_i$ y $\lambda$) × (la media de $r_{ui} - \mu$ de esa película). Con el mismo $\lambda$, ¿cuánto vale ese factor para una película con 1 valoración y para una con 300?**

$$b_i = \sum_{u \in R(i)} \frac{r_{ui} - \mu}{\lambda + n_i} = \sum_{u \in R(i)}\frac{ n_i(r_{ui} - \mu)}{n_i(\lambda + n_i)} = 
\frac{n_i}{(\lambda + n_i)}\sum_{u \in R(i)}\frac{(r_{ui} - \mu)}{n_i}
$$

Si el número de valoraciones de la película es mucho mayor $\lambda$ el modelo se fiará de los datos y predicirá la media de la película. Si el número de valoraciones es mucho menor que lambda el modelo predicirá la media global.
Entre medias es una mezcla de las dos, y el peso de la media de la película es el factor $\frac{n_i}{\lambda + n_i}$
