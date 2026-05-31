# Decisiones de Diseno del Modulo 3

## Enfoque

Se implemento un recomendador hibrido en PyTorch. El modelo combina embeddings de usuario y destino con atributos de contenido inferidos desde los CSV del dataset. Esta decision permite cubrir el entregable de filtrado colaborativo y, cuando hay metadata de destinos, aprovechar senales como ciudad, pais, categoria, presupuesto o variables numericas.

## Datos e inferencia de esquema

El dataset de Kaggle puede venir con varios CSV y nombres de columnas no necesariamente alineados con el repositorio. Por eso el cargador infiere alias frecuentes:

- Usuario: `user_id`, `customer_id`, `traveler_id`, `client_id`.
- Destino: `destination_id`, `destination`, `destination_name`, `place_id`, `city`.
- Preferencia: `rating`, `score`, `stars`.
- Orden temporal: `timestamp`, `date`, `travel_date`.

Si no existe rating explicito, cada interaccion se interpreta como feedback implicito positivo.

## Entrenamiento

El problema se formula como prediccion binaria de interaccion. Las interacciones observadas son positivas y se generan muestras negativas por usuario excluyendo destinos ya vistos. La perdida usada es `BCEWithLogitsLoss`.

La division train/valid/test se realiza por usuario para que la evaluacion mida recomendacion real: dado el historial parcial del usuario, el modelo debe recuperar destinos retenidos.

## Evaluacion

Las metricas de clasificacion clasicas no capturan bien el ordenamiento de recomendaciones. Se reportan metricas top-K:

- `precision@5` y `precision@10`.
- `recall@5` y `recall@10`.
- `hit_rate@5` y `hit_rate@10`.
- `map@5` y `map@10`.
- `ndcg@5` y `ndcg@10`.

Estas metricas permiten analizar satisfaccion aproximada, cobertura de preferencias y calidad del orden del ranking.

En este dataset existen multiples `DestinationID` para un mismo lugar turistico. Por esa razon, la evaluacion principal considera aciertos por nombre de destino (`Name`) cuando esta metadata existe. Esto evita penalizar recomendaciones equivalentes como dos filas distintas de "Leh Ladakh".

## Artefactos

El checkpoint guarda pesos, mapeos usuario/destino, metadata de destinos, historial visto e `item_features`. Esto permite generar recomendaciones sin reconstruir el dataset original.

## Riesgos

El arranque frio para usuarios no vistos no esta resuelto en esta version; el CLI falla explicitamente si el usuario no estuvo en entrenamiento. Para produccion se recomienda agregar un fallback de popularidad filtrado por preferencias declaradas.
