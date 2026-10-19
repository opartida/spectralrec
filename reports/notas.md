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

## Sesion 5

1. **¿Por qué no usar el userId directamente como número de fila de U?**
   Porque los ids pueden ser numeros no correlativos y si se usara el id como número de fila la matriz podría ser muy grnade si estos ids son grandes
2. **¿Qué debería pasar si validación contiene un ID que no estaba en train?**
   Quitarlo del conjunto y contar cuantas se quitan.

## Sesion 6

1. **¿Por qué tres conjuntos y no dos?**
   Training es para generar el modelo. Validation para seleccionar el mejor modelo y testing para test final del modelo. El modelo se tiene que evaluar con datos que no haya sido usados para hacer la elección del mismo. La evaluacion del modelo estaría inflada ya que pudo obtener una buena evaluacion simplemente por azar.
2. **¿Qué tipo de películas desaparecen de val/test al filtrar y cómo sesga eso la evaluación?**
   Las películas que no tengan usuario o pelicula que no haya sido usado para generar el modelo, es decir, que no hayan aparecido en training data set. Es decir pelícuals con pocas valoraciones. Si tiene una sola valoración si cae en validación o test será eliminada. Al quitarlas la valoración del modelo será mejor porque valorar películas raras es mas dificil. Además, el resultado no dice nada de cómo funciona el modelo con películas raras.
