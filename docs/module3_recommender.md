# Modulo 3: Sistema de Recomendacion de Destinos de Viaje

Este modulo entrena un recomendador personalizado para sugerir destinos dentro de las rutas de transporte de la empresa a partir de interacciones usuario-destino.

## Dataset

Dataset recomendado:

https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset

Descarga:

```bash
python scripts/download_data.py --module module3 --output-dir data/raw
```

El cargador busca archivos CSV en `data/raw/module3_recommender` e infiere columnas comunes como `user_id`, `customer_id`, `destination_id`, `destination`, `rating`, `score` y `timestamp`. Si existen CSVs separados de destinos, los usa como metadatos de contenido.

## Entrenamiento

```bash
python scripts/train_module3_recommender.py ^
  --data-dir data/raw/module3_recommender ^
  --output-dir models/module3_recommender ^
  --epochs 20 ^
  --batch-size 256
```

Archivos generados:

- `models/module3_recommender/best_model.pth`: checkpoint PyTorch con embeddings, metadatos e historial de usuarios.
- `models/module3_recommender/history.csv`: perdida de entrenamiento y metricas de validacion.
- `models/module3_recommender/evaluation/metrics.json`: `precision@k`, `recall@k`, `hit_rate@k`, `map@k` y `ndcg@k`.
- `models/module3_recommender/evaluation/examples.json`: ejemplos de recomendaciones para usuarios de prueba.
- `models/module3_recommender/training_result.json`: resumen del entrenamiento.

## Evaluacion

```bash
python scripts/evaluate_module3_recommender.py ^
  --data-dir data/raw/module3_recommender ^
  --checkpoint models/module3_recommender/best_model.pth
```

Metricas principales:

- `precision@10`: proporcion de destinos recomendados que aparecen en el holdout del usuario.
- `recall@10`: proporcion de destinos relevantes recuperados entre las 10 primeras sugerencias.
- `ndcg@10`: calidad del ordenamiento, premiando aciertos en posiciones altas.
- `map@10`: precision promedio del ranking para cada usuario.

La evaluacion se calcula a nivel de nombre de destino cuando existe metadata `Name`, porque el dataset contiene varios `DestinationID` para el mismo lugar turistico.

## Recomendaciones

```bash
python scripts/recommend_module3_destinations.py ^
  --checkpoint models/module3_recommender/best_model.pth ^
  --user-id 12 ^
  --user-id 28 ^
  --top-k 5
```

La salida JSON incluye ranking, identificador de destino, nombre legible cuando esta disponible, score y metadatos del destino.

## Analisis de efectividad

Usar `evaluation/metrics.json` y `evaluation/examples.json` para contrastar:

- Usuarios satisfechos: aproximados por `hit_rate@10` y `recall@10`, que indican si el sistema recupera destinos que el usuario efectivamente eligio.
- Incremento potencial de demanda: rutas recomendadas con alta frecuencia pueden priorizarse en campanas o capacidad operativa.
- Diversidad de destinos: revisar los ejemplos y la distribucion de recomendaciones para detectar sobreexposicion de destinos populares.
- Calidad del ranking: `ndcg@10` alto indica que los destinos relevantes aparecen temprano, reduciendo friccion para el usuario.

## Limitaciones

El modulo aprende de interacciones historicas; usuarios o destinos nuevos requieren suficiente metadata de contenido o una politica de arranque frio basada en popularidad. Los datos crudos y pesos no se versionan por defecto.
