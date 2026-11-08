<!-- LTeX: language=es -->

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
   Las películas del catálogo son todas las que hay y que pueden recibir una valoración. Las valoradas son las películas del catálogo que han recibido una valoración. Una película sin valoraciones no aparece en la matriz y el modelo no puede aprender de ella.
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
   Training es para generar el modelo. Validation para seleccionar el mejor modelo y testing para test final del modelo. El modelo se tiene que evaluar con datos que no haya sido usados para hacer la elección del mismo. La evaluation del modelo estaría inflada ya que pudo obtener una buena evaluation simplemente por azar.
2. **¿Qué tipo de películas desaparecen de val/test al filtrar y cómo sesga eso la evaluación?**
   Las películas que no tengan usuario o película que no haya sido usado para generar el modelo, es decir, que no hayan aparecido en training data set. Es decir películas con pocas valoraciones. Si tiene una sola valoración si cae en validación o test será eliminada. Al quitarlas la valoración del modelo será mejor porque valorar películas raras es más difícil. Además, el resultado no dice nada de cómo funciona el modelo con películas raras.

## Sesión 7

1. **¿Por qué RMSE castiga más los errores grandes?**
   Porque cada error se eleva al cuadrado. Si el error se duplica su aportación a mae se duplica pero su aportación a la suma de cuadrados se cuadriplica
   
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

Si el número de valoraciones de la película es mucho mayor $\lambda$ el modelo se fiará de los datos y predecirá la media de la película. Si el número de valoraciones es mucho menor que lambda el modelo predecirá la media global.
Entre medias es una mezcla de las dos, y el peso de la media de la película es el factor $\frac{n_i}{\lambda + n_i}$

1. **¿Coincide con el orden que predijiste? Si no, ¿qué te sorprende?**
Si, coincide
2. **¿Por qué mejora cada modelo al anterior?**
El UserMean le añade al GlobalMean information sobre cada usuario específico. Para predecir la valoración de un usuario usa la media de sus valoraciones no la media de todos las valoraciones de todos los usuarios. El BiasBaseline tiene información sobre la forma de valorar películas en comparación con otros usuarios. Si tenemos un usuario con una media de valoraciones altas, UserMean valoraría cualquier película no valorada por este usuario con una valoración alta, la media del usuario. Sin embargo BiasBaseline la valoraría como la valora la media pero teniendo en cuenta la forma de valorar de este usuario específico con respecto a los otros usuarios. Si este usuario valora más alto se vería reflejado en la valoración.

3. **Reto opcional: el RMSE de GlobalMean debería coincidir, aproximadamente, con una estadística muy conocida de las notas de validación. ¿Cuál? Pista: el RMSE de predecir siempre la media es la raíz de la media de $(r - \mu)^2$. Puedes comprobarlo con NumPy.**
Desviación típica

4. **¿Cuánto cambia el RMSE si mueves `reg_movie` (de 0 a 10, con `reg_user` fijo)? ¿Y si mueves `reg_user` (con `reg_movie` fijo)? ¿Cuál de los dos $\lambda$ importa más, y por qué, teniendo en cuenta que cada usuario tiene al menos 20 valoraciones y que la mediana por película es 3?**

si mueves reg_movie dejando reg_user fijo cambia más o menos 0.025 y mueves reg_user y dejas reg_movie fijo no cambia nada. Importa más reg_movie. 
Pasa porque el modelo da valoraciones usando una mezcla de la media global y la media de la película o del usuario. Lambda controla si tiende más a la media global o a la de las películas. Si tienes más valoraciones por película o usuario que lambda usará más la media por película o usuario y si es menos la global.
Todos los usuarios tienen al menos 20 notas, más que cualquier $\lambda$ entre 0 y 10, así que reg_user casi no cambia nada. La mitad de las películas tiene 3 o menos, así que reg_movie sí cambia mucho.

## Sesión 9
1. **¿qué significa describir una película con $k$ números?**
significa extraer unos factores de los datos y representar las valoraciones de un usuario mediante una combinación lineal de esos factores. Entonces en vez de tener n números que representan las valoraciones de un usuario tendríamos k números uno por cada factor y donde k sería menor que n. Básicamente queremos encontrar reducción de dimensión donde los axis nuevos que son los factores tengan un sentido posiblemente gusto por terror o comedia y si fueran películas seria la temática de la película
2. **¿Por qué $U_u$ y $V_i$ tienen la misma longitud?**
por que necesitamos compararlos haciendo producto escalar entre ellos y contienen además en cada fila la representation de un usuario/película en términos de los factores, la posición $j$ de los dos vectores habla del mismo factor. En el usuario, cuánto le gusta ese factor; en la película, cuánto lo tiene.
3. **¿Por qué el RMSE calculado sobre los residuos es igual al calculado sobre las notas?**

$$\text{Sobre notas:}\quad r_{ui} - (\mu + b_u + b_i + U_u \cdot V_i) = r_{ui} - \mu - b_u - b_i - U_u \cdot V_i$$

$$\text{Sobre residuos:}\quad (r_{ui} - \mu - b_u - b_i) - U_u \cdot V_i = r_{ui} - \mu - b_u - b_i - U_u \cdot V_i$$

El error de cada pareja es el mismo de las dos formas. Como el RMSE solo depende de los errores (la raíz de la media de sus cuadrados), sale igual.


$$UV^\top = \sum_{j=1}^{k} U_{:,j}\, V_{:,j}^\top$$

1. **¿Por qué cada sumando $U_{:,j}\, V_{:,j}^\top$ tiene rango 1?**
Porque es una multiplication columna x fila y por tanto cada fila de la matriz es un múltiplo de la fila inicial donde el multiplicador es el coeficiente correspondiente del vector columna.

2. **¿Por qué la suma tiene rango $\le k$?**
podemos ver cada fila como combination lineal de k vectores filas donde los coeficientes son los de cada usuario para cada factor por tanto como mucho podrá haber k filas linealmente independientes

3. **¿Cuántos números tiene el modelo frente a usuarios × películas?**
$U$ tiene $n_{\text{users}} \cdot k$ números y $V$ tiene $n_{\text{movies}} \cdot k$, así que el modelo tiene

$$k\,(n_{\text{users}} + n_{\text{movies}})$$

frente a los $n_{\text{users}} \cdot n_{\text{movies}}$ de la tabla completa. Con $n_{\text{users}} = 610$, $n_{\text{movies}} = 9724$ y $k = 10$, el modelo tiene 103340 números y la tabla 5931640, unas 58 veces más.

## Sesión 10
1. **¿En qué se diferenciarán los σ de la matriz real de los del control?**
Espero que la matriz de control tenga menos estructura y por tanto tenga unos valores singulares más parecidos entre si en tamaño. La matriz original tendrá algunos valores singulares grandes y el resto despreciables frente a estos.
La figura muestra dos curvas muy parecidas que se separan ligeramente en los últimos valores singulares. Los primeros $\sigma$ dependen sobre todo de lo que comparten las dos matrices, es decir, del patrón de huecos, y no de dónde está colocado cada valor, que es donde estarían los gustos.

First singular value: 35.55
Max control first singular value: 38.61
Min control first singular value: 36.81

el $\sigma_1$ real queda por debajo de los controles, así que el barajado cambia algo más que los gustos

2. **Si la hipótesis es correcta, ¿qué esperas ver al comparar la submatriz densa con su control?**
Lo contrario que antes las curvas deberían separarse al principio y ser  similares hacia el final.
Si divergen al principio quedando la matriz real por encima al principio lo que indica que $\sigma_1$ recoge algo que el barajado no reproduce. En todo el resto de tramos la curva de la matriz de control queda por encima de la matriz real.
El $\sigma_1$ real (10,64) supera el máximo de 20 controles (8,70; mínimo 7,67), así que esa diferencia difícilmente es casualidad del barajado. El resto del espectro no lo he comprobado.


3. **¿Qué tamaño de submatriz elijo y por qué?**
- Density for (50, 50): 61.8%
- Density for (100, 100): 49.9%
- Density for (200, 200): 33.0%

Elegimos la más densa es decir submatriz 50x50 ya que queremos que tenga los menos ceros posibles. Tenemos menos valores singulares pero suficientes para ver comportamiento de las curvas.

## Sesión 11

1. **¿Cómo cambiarán el RMSE de train y el de validación al aumentar k?**
El RMSE de train irá descendiendo ya que usamos su propia matriz de residuos, cuanto mayor sea k más se acercará a los residuos de training. El RMSE de validación debería ir reduciéndose hasta un k a partir del cual empezará a subir al hacerse la aproximación de los residuos muy similar a los de training y no necesariamente a los de validación.
Se observa que el error de train desciende. Pasa de 0.7956 para k=1 a 0.4283 para k=100. La matriz truncada de SVD se aproxima a los valores de los residuos de train cuando k aumenta y por eso el error se reduce. Los pares de validación se rellenan con ceros en la matriz de residuos de training y al aproximarla con la matriz truncada de SVD si usamos k pequeño como la mayor parte de los valores de la matriz son ceros la matriz aproximada tendrá valores muy pequeños y lo mismo ocurrirá para valores grandes de k ya que la matriz aproximada se parecerá más a la matriz de train con relleno que tiene ceros en los pares de validación y por tanto al calcular los residuos que predice la SVD para los pares de validación serán bastante similares entre sí y se acercan a 0. Como los residuos predichos son casi 0, el modelo predice casi lo mismo que los sesgos. En los datos observamos que el mayor valor de los errores en validación es 0.8531 con k=2 y el menor 0.8474 con k=20. Por lo tanto la predicción se cumple tanto para training como para validación. Para validación existe un descenso desde valores pequeños de k hasta valores intermedios de k pero este descenso es muy pequeño respecto al modelo baseline de sesgos (0.8554) y de nuevo vuelve a subir el error cuando k aumenta.

2. **¿Dónde has puesto Σ: en U, en V o repartida como $\sqrt\Sigma$? ¿Cambia la predicción?**
La he repartido como $\sqrt\Sigma$ en los dos factores: `U * sqrt(s)` para los usuarios y `sqrt(s) * Vt.T` para las películas. La predicción no cambia la ponga donde la ponga, porque $\Sigma$ aparece una sola vez en el producto:

$$U_k \Sigma_k V_k^\top = (U_k\Sigma_k)\,V_k^\top = U_k\,(\Sigma_k V_k^\top) = (U_k\sqrt{\Sigma_k})(\sqrt{\Sigma_k}\,V_k^\top)$$

Ejemplo con k = 1, σ = 4, u = 0.5 y v = 0.25: con Σ en el usuario, 2 × 0.25 = 0.5; en la película, 0.5 × 1 = 0.5; repartida, 1 × 0.5 = 0.5.

**¿Cambiaría si penalizaras el tamaño de U y V?**
Sí. Si el modelo minimiza error + λ × (suma de los cuadrados de los números de U y V), la predicción es la misma, pero el coste no. Ejemplo con un usuario, una película y σ = 4 (u = v = 1): con Σ entera en el usuario, los factores son 4 y 1, y el tamaño es 4² + 1² = 17; repartida, los factores son 2 y 2, y el tamaño es 2² + 2² = 8. Las dos opciones predicen 4, pero un modelo penalizado prefiere la repartida, porque paga menos por el tamaño.

3. **¿Qué efecto tienen los ceros de los huecos sobre las predicciones?**
Los residuos que predice la matriz truncada en los huecos, rellenos con 0, son casi 0. El efecto es que no ayudan a mejorar el modelo de sesgos al usar la matriz truncada de residuos. Ver pregunta 1.

4. **si cada entrada se observara al azar con probabilidad $p$, $\mathbb E[M/p]$ sería la matriz completa. Prueba a reescalar. ¿Se observan al azar las notas de MovieLens?**

## Sesión 12

1. **Derivación del paso de SGD**
Pérdida de una observación $(u, i)$:

$$\ell_{ui} = (\tilde r_{ui} - U_u\cdot V_i)^2 + \lambda\big(\lVert U_u\rVert^2 + \lVert V_i\rVert^2\big)$$

Gradientes, con la regla de la cadena. Como $\lVert U_u\rVert^2 = \sum_j U_{uj}^2$, su derivada es $2U_u$:

$$\frac{\partial \ell_{ui}}{\partial U_u} = -2(\tilde r_{ui} - U_u\cdot V_i)\,V_i + 2\lambda U_u$$

$$\frac{\partial \ell_{ui}}{\partial V_i} = -2(\tilde r_{ui} - U_u\cdot V_i)\,U_u + 2\lambda V_i$$

Actualizaciones: un paso en contra del gradiente. El factor 2 aparece en los dos términos, así que lo saco como factor común y lo meto en el learning rate $\eta$:

$$U_u \leftarrow U_u + \eta\,\big[(\tilde r_{ui} - U_u\cdot V_i)\,V_i - \lambda\,U_u\big]$$

$$V_i \leftarrow V_i + \eta\,\big[(\tilde r_{ui} - U_u\cdot V_i)\,U_u - \lambda\,V_i\big]$$

En las dos fórmulas, $U_u$ y $V_i$ a la derecha son los valores de antes del paso.

Comprobación con números (`test_sgd_step`): con $U_u = (1, 2)$, $V_i = (2, 1)$, $\tilde r_{ui} = 5$, $\eta = 0.1$ y $\lambda = 0.5$, la predicción es 4 y el error, 1. Después del paso, $U_u = (1.15,\ 2.0)$ y $V_i = (2.0,\ 1.15)$.

2. **¿Por qué no se inicializa a ceros?**
Con $U_u = V_i = 0$, los dos gradientes se anulan:

$$\frac{\partial \ell_{ui}}{\partial U_u} = -2(\tilde r_{ui} - 0)\cdot 0 + 2\lambda\cdot 0 = 0, \qquad \frac{\partial \ell_{ui}}{\partial V_i} = -2(\tilde r_{ui} - 0)\cdot 0 + 2\lambda\cdot 0 = 0$$

Así que $U_u \leftarrow U_u + 0$ y $V_i \leftarrow V_i + 0$: el paso no hace nada y el modelo se queda en el mismo punto para siempre.


3. **¿Y si todas las columnas empiezan iguales?**
$$\frac{\partial \ell_{ui}}{\partial U_u}\Bigg|_{U_u = (a,\ a),\ V_i = (b,\ b)} = -2(\tilde r_{ui} - 2ab)\,(b,\ b) + 2\lambda\,(a,\ a)$$

$$\frac{\partial \ell_{ui}}{\partial V_i}\Bigg|_{U_u = (a,\ a),\ V_i = (b,\ b)} = -2(\tilde r_{ui} - 2ab)\,(a,\ a) + 2\lambda\,(b,\ b)$$

Las dos componentes de cada gradiente son iguales, así que el paso suma lo mismo a las dos y siguen iguales. Por ejemplo, con $U_u = (1, 1)$, $V_i = (2, 2)$, $\tilde r_{ui} = 5$, $\eta = 0.1$ y $\lambda = 0.5$, sale $U_u = (1.15,\ 1.15)$. Como en cada paso vuelve a pasar lo mismo, las columnas no se separan nunca.

Dos columnas iguales dicen lo mismo: es como tener un solo factor. Si el factor 1 fuera «intriga» y el 2 «humor», toda película tendría tanta intriga como humor y a todo usuario le gustarían igual; no se podría representar a alguien con $(2, -1)$. En el producto, $UV^\top = 2\,U_{:,1}V_{:,1}^\top$, de rango 1: con k = 2 en el código, el modelo predice como uno con k = 1. Por eso se inicializa con números aleatorios, para que cada columna empiece distinta.

4. **¿Qué pasaría si actualizas V con la U ya modificada?**
El gradiente se calcula en el punto $U_u$, $V_i$ y por tanto en la fórmula aparece el punto $U_u$, $V_i$ donde se ha calculado el gradiente y es el valor que hay que utilizar. El que se ha calculado en este paso ya no es el punto original sino el nuevo punto que ha calculado el algoritmo, donde en el paso siguiente se calculará el nuevo gradiente. Si actualizara $V_i$ con la $U_u$ nueva, ya no sería el paso de gradiente que derivé: con los números del test saldría $V_i = (2.015,\ 1.15)$ en vez de $(2.0,\ 1.15)$.

## Sesión 13

1. **¿Cómo esperas que se muevan las curvas de train y de validación a lo largo de las épocas?**
Las dos curvas deberían de decrecer. La curva de training debería decrecer en todas las épocas. La de validación debería decrecer hasta una "época mejor" y a partir de ahí empezar a crecer. Esto se debe a que el modelo empieza a memorizar los datos de training y por tanto no es capaz o cada vez generaliza menos y por tanto su predicción en los residuos de validación será peor. Usando un valor de regularización que permita al modelo generalizar y no solo memorizar los valores de train (reg = 0.1) se observa que las curvas hacen lo que se describe en la predicción. Con reg = 0.05 el modelo deja a los factores mas libertad es decir no los intenta forzar a que sean pequeños, y por tanto el modelo tenderá a memorizar los datos de training y por tanto el error en validación no decrecerá.

2. **¿En qué se parecerá o se diferenciará de lo que viste con la SVD truncada?**
En este caso la curva de validación debería decrecer en cada época. La SVD truncada intentaba ajustar unos residuos que eran 0 para los valores no observados y tanto si usábamos un k pequeño como grande siempre el valor del modelo para el residuo de estos pares no observados era cercano a cero, por tanto similar al modelo baseline. Ahora lo que se hace es ajustar los valores de los residuos observados solamente y los factores que se encuentran de esta manera no están forzados a ajustar los residuos no observados a 0 y por tanto se espera que la predicción de los residuos no observados usando estos factores del modelo sea más similar a los valores reales.
**Observación.** No se cumplió. Con reg = 0.1, el mejor RMSE de validación de SGD es 0.8483 (época 51). Mejora al baseline de sesgos (0.8554), pero queda prácticamente igual que la SVD truncada con k = 20 (0.8474), incluso un poco peor. Una diferencia de 0.001 es tan pequeña que podría deberse a la semilla; lo comprobaré en la sesión 16 repitiendo con varias `model_seed`. Con esta configuración, ajustar solo los residuos observados no ha bastado para mejorar a la SVD truncada. Queda por ver si cambiando reg, k o lr cambia el resultado.

