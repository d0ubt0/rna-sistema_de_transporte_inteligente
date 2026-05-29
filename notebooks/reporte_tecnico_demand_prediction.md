A continuación, se detalla un informe técnico exhaustivo del código y de la arquitectura del modelo de aprendizaje profundo contenidos en el archivo `01_eda_demand.ipynb`.

---

### 1. Descripción General del Proyecto
El objetivo principal del código es construir un pipeline de Machine Learning y Deep Learning de extremo a extremo para predecir la demanda de transporte público (número de pasajeros). El sistema aborda la naturaleza compleja de las series temporales de transporte mediante la simulación avanzada de datos y el posterior entrenamiento de una **Red Neuronal Recurrente (RNN) basada en LSTM (Long Short-Term Memory)** combinada con **Capas de Embeddings** en PyTorch para capturar interacciones tanto temporales como categóricas.

---

### 2. Generación de Datos Sintéticos Avanzados
Antes del modelo, el script genera un dataset sintético altamente realista (`demanda_transporte.csv`) que simula **7,500 registros** en total (1,500 días de historial para 5 rutas distintas: Ruta A, B, C, D y E). La simulación inyecta patrones de comportamiento del mundo real:
* **Tendencia Estocástica:** Implementa un incremento continuo modelado como un *Random Walk* con perturbaciones normales para cada ruta de forma independiente.
* **Estacionalidad Múltiple:** Factores multiplicativos con ruido gaussiano para los días de la semana y los meses del año.
* **Persistencia Climática:** Simulación de condiciones del clima (Soleado, Nublado, Lluvia) donde la probabilidad actual depende del historial del día anterior (simulando cadenas de Markov de persistencia). Además, penaliza la demanda si hay lluvia consecutiva o lluvia en fines de semana.
* **Efectos de Calendario y Pagos:** Incrementos fijos por la cercanía a días festivos (ventanas pre y post festivo con un +10% de boost) y picos de demanda ajustados en días específicos de pago (días 1, 15 y 30 de cada mes).
* **Componentes de Ruido Complejos:** Se añade ruido autorregresivo $AR(1)$ para simular dependencia temporal de corto plazo en los errores, junto a un componente de **ruido heterocedástico** (la varianza del error aumenta a medida que la demanda absoluta es mayor).

---

### 3. Preprocesamiento y Preparación de Secuencias Temporales
Para acondicionar los datos de manera óptima para la red neuronal, el código realiza las siguientes transformaciones:
1. **Codificación Categórica:** Utiliza `LabelEncoder` de scikit-learn para transformar las variables textuales `ruta` y `clima` en índices numéricos discretos (`route_id` y `clima_id`). Estos índices se guardan mediante artefactos serializados (`.pkl`).
2. **Normalización Numérica:** Aplica `MinMaxScaler` de manera independiente a las características continuas (`dia_semana`, `mes`, `festivo`, etc.) y a la variable objetivo (`pasajeros`) para escalar todos los valores numéricos al rango $[0, 1]$, facilitando la convergencia del optimizador en PyTorch.
3. **Construcción de Ventanas Deslizantes (`build_sequences`):** Define una longitud de secuencia fija (`SEQ_LENGTH = 30`). La función agrupa los datos cronológicamente por ruta y extrae bloques continuos de 30 días de historial para predecir el día 31:
    * **Entrada de la serie ($X$):** Matriz de dimensiones `(N, 30, n_features)` que contiene el comportamiento continuo e histórico de los últimos 30 días.
    * **Variables del Día Objetivo:** Extrae el `route_id` y el `clima_id` correspondientes específicamente al día de la predicción, separándolos para que alimenten a las capas de representación densa.
4. **Dataset y Carga de Lotes PyTorch (`TransportDataset`):** Hereda de `torch.utils.data.Dataset` y convierte las estructuras en tensores nativos (`torch.float32` para datos continuos y `torch.long` para los índices categóricos de embeddings). Los datos se empaquetan en un `DataLoader` con un tamaño de lote (`BATCH_SIZE`) de 32 y mecanismos de fijación de semillas (`seed_worker`) que aseguran la perfecta reproducibilidad de los experimentos.

---

### 4. Arquitectura Detallada del Modelo (`TransportLSTM`)
La clase `TransportLSTM` define una topología híbrida avanzada que procesa en paralelo variables secuenciales continuas e información categórica estática/dinámica del paso objetivo.

```
Estructura de Flujo del Modelo:
[X continuos (30 días)] ──> [ Capa LSTM (2 capas, Dim:128) ] ──> [Último Estado Oculto] ──┐
[ID de Ruta Objetivo ] ──> [ Capa Embedding (Dim: 8)     ] ─────────────────────────────┼─> [Concatenación (Dim: 140)] ──> [Capa Lineal (140->64)] ──> ReLU ──> Dropout ──> [Capa Lineal (64->1)] ──> Predicción Pasajeros
[ID de Clima Objetivo] ──> [ Capa Embedding (Dim: 4)     ] ─────────────────────────────┘
```

#### Componentes Internos de la Red:
* **Capas de Embedding Categórico:**
  * `route_embedding`: Un bloque `nn.Embedding` enfocado en aprender representaciones vectoriales densas para las 5 rutas disponibles, mapeando cada ID discreto a un espacio continuo de **8 dimensiones** (`route_embedding_dim=8`).
  * `clima_embedding`: Un bloque `nn.Embedding` encargado de proyectar las categorías de clima a un vector denso de **4 dimensiones** (`clima_embedding_dim=4`).
* **Bloque Recurrente Temporal (LSTM):**
  * `self.lstm = nn.LSTM(...)`: Configurada con `input_size=4` (características continuas rezagadas), un tamaño oculto de **128 neuronas** (`hidden_size=128`), organizada de forma apilada con **2 capas recurrentes** (`num_layers=2`), procesamiento orientado por lotes primero (`batch_first=True`) y una probabilidad de **dropout del 20%** (`dropout=0.2`) entre capas para regularizar el aprendizaje.
* **Mecanismo de Fusión de Características (Forward Step):**
  * La red toma las secuencias de entrada y las hace pasar por la LSTM, abstrayendo únicamente el vector del último paso temporal de la secuencia: `lstm_out[:, -1, :]`. Este vector condensa toda la memoria histórica de los 30 días previos.
  * Recupera los vectores de embedding para la ruta y el clima correspondientes al día objetivo.
  * Ejecuta una fusión por concatenación: `torch.cat([lstm_out, route_embed, clima_embed], dim=1)`. La dimensión resultante de este vector combinado es de exactamente **140 componentes** ($128 \text{ (LSTM)} + 8 \text{ (Ruta)} + 4 \text{ (Clima)} = 140$).
* **Capa de Salida de Regresión (Fully Connected - FC):**
  * Consiste en un bloque `nn.Sequential` que toma el vector fusionado de 140 dimensiones y lo reduce mediante una transformación lineal a 64 neuronas (`nn.Linear(140, 64)`).
  * Aplica una activación no lineal de tipo `nn.ReLU()` seguida de una capa de desconexión aleatoria (`nn.Dropout(0.2)`) para mitigar el sobreajuste.
  * Concluye con una capa lineal final de salida `nn.Linear(64, 1)` encargada de proyectar el resultado en un único valor continuo: la cantidad escalar normalizada de pasajeros.

El número total de parámetros entrenables dentro de esta arquitectura asciende a **209,845 parámetros**.

---

### 5. Estrategia de Entrenamiento y Control de Sobreajuste
El entrenamiento está estructurado con prácticas modernas de optimización para garantizar la robustez del modelo:
* **Función de Pérdida y Optimización:** Minimiza el Error Cuadrático Medio (MSE) utilizando el optimizador Adam con una tasa de aprendizaje inicial estándar de `1.00e-03`.
* **Programador Dinámico de LR (Scheduler):** Se incluye un `scheduler.step(avg_val_loss)` que monitorea la pérdida de validación en cada época, reduciendo dinámicamente la tasa de aprendizaje si el modelo entra en un estancamiento (*plateau*).
* **Early Stopping Riguroso:** Aunque el límite máximo está fijado en 50 épocas, se implementa un contador de paciencia (`PATIENCE`). Si el error de validación deja de mejorar durante el número límite de épocas consecutivas, el proceso se interrumpe prematuramente para resguardar la capacidad de generalización. Los mejores pesos de la red se guardan de inmediato en el disco bajo el archivo `demand_prediction/best_model.pth`.

---

### 6. Evaluación de Métricas y Resultados Obtenidos
Al finalizar el entrenamiento, el código cambia al estado de inferencia (`model.eval()`) deshabilitando el cálculo de gradientes (`torch.no_grad()`) para medir con precisión el rendimiento sobre secuencias de prueba futuras independientes. Las métricas calculadas son el error cuadrático medio (RMSE), el error absoluto medio (MAE) y el error porcentual absoluto medio (MAPE):

| Segmentación por Ruta | RMSE (Pasajeros) | MAE (Pasajeros) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Ruta A** | 151.04 | 119.34 | 8.47% |
| **Ruta B** | 213.37 | 161.85 | 7.59% |
| **Ruta C** | 120.07 | 94.36 | 10.33% |
| **Ruta D** | 176.55 | 132.15 | 7.36% |
| **Ruta E** | 125.12 | 100.36 | 8.86% |
| **MÉTRICA GLOBAL** | **160.99** | **121.61** | **8.52%** |

*Nota técnica: Los resultados muestran que el modelo posee un alto nivel de precisión predictiva, con un margen de error porcentual global promedio de apenas un **8.52%** frente a la demanda real simulada.*
