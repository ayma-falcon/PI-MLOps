# Proyecto Integrado MLOps

## Introducción
En el marco de este proyecto, se me encomendó la responsabilidad de desempeñar el papel de un Ingeniero MLOps. Mi rol específico implica actuar como un Data Scientist en el contexto de la plataforma Steam. La tarea principal consiste en desarrollar un sistema de recomendación de videojuegos destinado a los usuarios de la plataforma. Se espera que parta desde cero, desempeñando de manera ágil funciones de Data Engineer para lograr un Producto Mínimo Viable (MVP) para la finalización del proyecto.

## Extracción, Transformación y Carga ([ETL](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/ETL_reviews.ipynb))
![ETL](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/_src/ETL.jpg)

#### En el proceso de Extracción, Transformación y Carga (ETL), se llevaron a cabo las siguientes tareas:
> Nota: Los archivos JSON utilizados para ETL se encuentran en la carpeta [datasets](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/tree/main/datasets) comprimidos, en caso de querer ejecutar el codigo se deben descomprimir

* **Desanidamiento de datasets:** Inicialmente, se abordó la desanidación de datasets en formato JSON con el fin de convertirlos en datos manejables. Además, dentro del DataFrame obtenido tras el primer desanidamiento, se identificaron columnas que también requerían desanidamiento adicional.

* **Gestión de datos nulos o innecesarios:** Se implementó una gestión de datos que implicó la eliminación de columnas consideradas innecesarias para las funciones y el modelo de recomendación. Asimismo, se identificaron filas que contenían datos nulos que no podían ser rellenados y por este motivo fueron eliminadas.

* **Fechas:** En relación a los campos que contenían fechas en el formato YYYY-MM-AA, se procedió a simplificarlos al extraer únicamente el año (YYYY) y crear una nueva columna con esta información. Posteriormente, la columna original con el formato YYYY-MM-AA fue eliminada, ya que no era necesaria para el propósito de la aplicación. En los casos en los que se encontraron datos que no cumplían con el formato estándar (YYYY-MM-DD) y resultaban en valores nulos, se optó por sustituir esos valores nulos utilizando la moda de la columna respectiva.

* **Columna 'sentiment_analysis':** Se ha creado una nueva columna denominada 'sentiment_analysis' al aplicar un análisis de sentimiento utilizando procesamiento de lenguaje natural (NLP). Esta nueva columna sigue una escala específica, asignando el valor '0' en caso de una valoración negativa, '1' para neutral, y '2' para una valoración positiva. La finalidad de esta nueva columna es reemplazar la columna original 'user_reviews.review'. Este reemplazo se llevó a cabo con el propósito de facilitar tanto la tarea de los modelos de aprendizaje automático como el análisis de datos, simplificando la interpretación de las valoraciones de los usuarios en el conjunto de datos.

* **Carga en formato Parquet:** Dado que los DataFrames en formato CSV resultaban excesivamente voluminosos, lo cual podía complicar la gestión de archivos, se tomó la decisión de cargarlos en formato Parquet. Esta elección permitió optimizar la eficiencia de almacenamiento y acceso a los datos, facilitando así el manejo de la información.

## Analisis Exploratorio de Datos ([EDA](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/EDA.ipynb))
![EDA](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/_src/EDA.jpg)

En el proceso de Exploración de Datos (EDA), se realizaron análisis y evaluaciones específicas en relación a los datos esenciales para el modelo de recomendación. El objetivo principal de este análisis fue comprender cómo estos datos se relacionaban entre sí y si se identificaban patrones significativos. Estos hallazgos desempeñaron un papel fundamental en la fase de diseño de las funciones, ya que proporcionaron información crucial sobre cuáles serían los datos más apropiados esperar a la hora de ponerlas a prueba.

## Desarrollo [API](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/main.py)
![FastApi-Render](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/_src/FastApi-Render.png)
> Nota: Dado que estamos utilizando la versión gratuita de Render, es importante destacar que todas las funciones se ejecutan utilizando una muestra del código que corresponde a los primeros 40,000 datos de los DataFrames. Esta medida se toma para evitar posibles errores relacionados con la capacidad de procesamiento limitada disponible en Render.

En esta fase del proyecto, se plantea la disponibilización de los datos de la empresa mediante el uso del framework FastAPI. Las consultas que se proponen son las siguientes:

* def `PlayTimeGenre`(genero: str): Devuelve el año con mas horas jugadas para dicho género.
Ejemplo de generos para usar: Action, Indie, Casual, Simulation
* def `UserForGenre`(genero: str): Devuelve el usuario que acumula más horas jugadas para el género dado y la cantidad total de horas jugadas.
Ejemplo de generos para usar: Action, Indie, Casual, Simulation
* def `UsersRecommend`(año: int): Devuelve el top 3 de juegos más recomendados por usuarios para el año dado.
* def `UsersNotRecommend`(año: int): Devuelve el top 3 de juegos menos recomendados por usuarios para el año dado.
* def `sentiment_analysis`(año: int): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

## Modelo de recomendacion ([Machine Learning](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/ML.ipynb))
![ML](https://github.com/ayma-falcon/Proyecto-Integrador-MLOps/blob/main/_src/ML.png)
> Nota: De manera similar a las funciones, en los modelos de recomendación, también se emplearon muestras de los DataFrames equivalentes a los primeros 5000 datos. Esto se hizo para garantizar un rendimiento óptimo y evitar problemas de capacidad de procesamiento en el entorno de Render.

Para la fase final del proyecto, se requiere la implementación de dos modelos de recomendación que posteriormente serán cargados en la API. Los modelos propuestos y su funcionamiento son los siguientes:
* def **recomendacion_juego**(id de producto): Ingresando el id de producto, se recibe una lista con 5 juegos recomendados similares al ingresado.
Ejemplo de id para usar: 761140, 643980, 670290, 767400
* def **recomendacion_usuario**(id de usuario): Ingresando el id de un usuario, se recibe una lista con 5 juegos recomendados para dicho usuario.
Ejemplo de id para usar: evcentric, doctr

## Complementos
* **[Video explicativo](https://youtu.be/fk8gFYpSM4M):** Para obtener una comprensión más completa de este proyecto, les extiendo una cordial invitación para que vean mi video explicativo. En este video, se muestra la ejecución de la API y se ofrece una breve explicación de su funcionamiento.

* **[Link API](https://pi-mlops-aymara-falcon.onrender.com/docs):** Link de la API en la cual se puede hacer uso de las funciones y el modelo de recomendación
