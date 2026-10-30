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

1. **¿Coincide con el orden que predijiste? Si no, ¿qué te sorprende?**
Si, coincide
2. **¿Por qué mejora cada modelo al anterior?**
El UserMean le añade al GlobalMean informacion sobre cada usuario específico. Para predecir la valoración de un usario usa la media de sus valoraciones no la media de todos las valoraciones de todos los usuarios. El BiasBaseline tiene información sobre la forma de valorar pelicuals en comparación con otros usuarios. Si tenemos un usuario con una media de valoraciones altas, UserMean valoraria cualquier película no valorada por este usuario con una valoracion alta, la media del usuario. Sin embargo BiasBaseline la valoraría como la valora la media pero teniendo en cuenta la forma de valorar de este usuario específico con respecto a los otros usuarios. Si este usuario valora mas alto se vería reflejado en la valoración.

3. **Reto opcional: el RMSE de GlobalMean debería coincidir, aproximadamente, con una estadística muy conocida de las notas de validación. ¿Cuál? Pista: el RMSE de predecir siempre la media es la raíz de la media de $(r - \mu)^2$. Puedes comprobarlo con NumPy.**
Desviación típica

4. **¿Cuánto cambia el RMSE si mueves `reg_movie` (de 0 a 10, con `reg_user` fijo)? ¿Y si mueves `reg_user` (con `reg_movie` fijo)? ¿Cuál de los dos $\lambda$ importa más, y por qué, teniendo en cuenta que cada usuario tiene al menos 20 valoraciones y que la mediana por película es 3?**

si mueves reg_movie dejando reg_user fijo cambia mas o menos 0.025 y mueves reg_user y dejas reg_movie fijo no cambia nada. Importa mas reg_movie. 
Pasa porque el modelo da valoraciones usando una mezcla de la media global y la media de la pelicula o del usuario. Lambda controla si tiende mas a la media global o a la de las películas. Si tienes mas valoracions por película o usuario que lambda usará mas la media por película o usuario y si es menos la global.
Todos los usuarios tienen al menos 20 notas, más que cualquier $\lambda$ entre 0 y 10, así que reg_user casi no cambia nada. La mitad de las películas tiene 3 o menos, así que reg_movie sí cambia mucho.

## Sesión 9
1. **¿qué significa describir una película con $k$ números?**
significa extraer unos factores de los datos y representar las valoraciones de un usuario mediante una combinacion lineal de esos factores. Entonces en vez de tener n numeros que representan las vsloraciones de un usario tendriamos k numeros uno por cada factor y donde k sería menor que n. Basicamente queremos encontrar reducción de dimensión donde los axis nuevos que son los factores tengan un sentido posiblemente gusto por terror o comedia y si fueran películas seria la temática de la película
2. **¿Por qué $U_u$ y $V_i$ tienen la misma longitud?**
por que necesitamos compararlos haciendo producto escalar entre ellos y contienen ademas en cada fila la representacion de un usuario/película en términos de los factores, la posición $j$ de los dos vectores habla del mismo factor. En el usuario, cuánto le gusta ese factor; en la película, cuánto lo tiene.
3. **¿Por qué el RMSE calculado sobre los residuos es igual al calculado sobre las notas?**

$$\text{Sobre notas:}\quad r_{ui} - (\mu + b_u + b_i + U_u \cdot V_i) = r_{ui} - \mu - b_u - b_i - U_u \cdot V_i$$

$$\text{Sobre residuos:}\quad (r_{ui} - \mu - b_u - b_i) - U_u \cdot V_i = r_{ui} - \mu - b_u - b_i - U_u \cdot V_i$$

El error de cada pareja es el mismo de las dos formas. Como el RMSE solo depende de los errores (la raíz de la media de sus cuadrados), sale igual.


$$UV^\top = \sum_{j=1}^{k} U_{:,j}\, V_{:,j}^\top$$

1. **¿Por qué cada sumando $U_{:,j}\, V_{:,j}^\top$ tiene rango 1?**
Porque es una multiplicacion columna x fila y por tanto cada fila de la matriz es un multiplo de la fila inicial donde el multiplicador es el coeficiente correspondiente del vector columna.

2. **¿Por qué la suma tiene rango $\le k$?**
podemos ver cada fila como combinacion lineal de k vectores filas donde los coeficientes son los de cada usuario para cada factor por tanto como mucho podrá haber k filas linealmente independientes

3. **¿Cuántos números tiene el modelo frente a usuarios × películas?**
$U$ tiene $n_{\text{users}} \cdot k$ números y $V$ tiene $n_{\text{movies}} \cdot k$, así que el modelo tiene

$$k\,(n_{\text{users}} + n_{\text{movies}})$$

frente a los $n_{\text{users}} \cdot n_{\text{movies}}$ de la tabla completa. Con $n_{\text{users}} = 610$, $n_{\text{movies}} = 9724$ y $k = 10$, el modelo tiene 103340 números y la tabla 5931640, unas 58 veces más.

## Sesión 10
1. **¿En qué se diferenciarán los σ de la matriz real de los del control?**
Espero que la matriz de control tenga menos estructura y por tanto tenga unos valores singulares mas parecidos entre si en tamaño. La matriz original tendrá algunos valores singulares grandes y el resto despreciables frente a estos.
La figura muestra dos curvas muy parecidas que se separan ligeramente en los últimos valores singulares. Los primeros $\sigma$ dependen sobre todo de lo que comparten las dos matrices, es decir, del patrón de huecos, y no de dónde está colocado cada valor, que es donde estarían los gustos.

2. **Si la hipótesis es correcta, ¿qué esperas ver al comparar la submatriz densa con su control?**
Lo contrario que antes las curvas deberian separarse al principio y ser  similares hacia el final.
Si divergen al principio quedando la matriz de inicial por encima al principio lo que indica que tiene mas estructura que la de control.

3. **¿Qué tamaño de submatriz elijo y por qué?**
- Density for (50, 50): 61.8%
- Density for (100, 100): 49.9%
- Density for (200, 200): 33.0%

Elegimos la mas densa es decir submatriz 50x50 ya que queremos que tenga los menos ceros posibles. Tenemos menos valores singulares pero suficientes para ver comportamiento de las curvas.




