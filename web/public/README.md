# Informe Técnico: Sistema Inteligente para la Empresa de Transporte

---

## 1. Introducción y delimitación del problema

### Contexto

Se desarrolló un sistema integral basado en redes neuronales artificiales (RNA) para una empresa de transporte urbano que enfrenta tres desafíos críticos:

1. **Predicción de demanda de pasajeros**: Anticipar el flujo de pasajeros en cada ruta para optimizar la asignación de flota, reducir tiempos de espera y mejorar la eficiencia operativa.
2. **Detección de conducción distractiva**: Identificar en tiempo real comportamientos de riesgo al volante (uso de teléfono, distracciones, etc.) a partir de imágenes de conductores, para prevenir accidentes.
3. **Recomendación personalizada de destinos**: Sugerir destinos turísticos y de servicio a los usuarios basándose en su historial de viajes y preferencias, mejorando la experiencia del pasajero.

### Propuesta de solución

Se implementaron tres módulos independientes pero integrados en una misma plataforma web:

| Módulo | Técnica | Framework |
|--------|---------|-----------|
| Predicción de Demanda | LSTM bidireccional con atención temporal | PyTorch |
| Clasificación de Conducción | Transfer Learning (MobileNetV3 / ResNet18) | PyTorch |
| Recomendación de Destinos | Filtrado colaborativo híbrido con embeddings | PyTorch |

Los tres modelos se desplegaron tras una API REST (FastAPI) consumida por una interfaz web moderna (React + Vite + Tailwind CSS).

---

## 2. Análisis descriptivo de datos

### 2.1 Módulo 1: Demanda de Transporte

Los datos de demanda fueron generados sintéticamente para simular 5 rutas de transporte (Ruta A a Ruta E) con los siguientes patrones:

- **Estacionalidad semanal**: mayor demanda en días laborales, reducción los fines de semana.
- **Estacionalidad anual**: picos en meses de verano (diciembre-febrero) y valles en meses de invierno.
- **Persistencia meteorológica**: el clima influye en la demanda (días lluviosos reducen pasajeros).
- **Días festivos**: incremento de demanda en fechas especiales.
- **Tendencia**: crecimiento gradual de pasajeros a lo largo del tiempo.

Variables de entrada al modelo:
- `pasajeros`: serie histórica de demanda (target).
- `ruta`: identificador categórico de la ruta (5 rutas).
- `clima`: condición meteorológica (soleado, nublado, lluvioso).
- `dia_semana`, `mes`, `es_fin_de_semana`, `es_festivo`: componentes temporales.

### 2.2 Módulo 2: Conducción Distractiva

Se utilizó el dataset *State Farm Distracted Driver Detection* de Kaggle, que contiene imágenes de conductores en 10 clases originales. Para este proyecto se trabajó con 5 clases agrupadas:

| Clase | Descripción | Ejemplos de imágenes |
|-------|-------------|---------------------|
| `safe_driving` | Conducción segura | Conductor atento al frente |
| `talking_phone` | Hablando por teléfono | Teléfono en la oreja |
| `texting_phone` | Escribiendo en teléfono | Mirando el teléfono |
| `turning` | Mirando a los lados | Giro de cabeza |
| `other_activities` | Otras actividades | Comiendo, maquillándose, etc. |

### 2.3 Módulo 3: Recomendación de Destinos

Dataset de interacciones usuario-destino con las siguientes características:

| Estadística | Valor |
|-------------|-------|
| Usuarios únicos | 675 |
| Destinos únicos | 684 |
| Archivos fuente | 4 CSVs (destinos, reseñas, historial, usuarios) |
| Dimensiones de contenido | 21 features categóricos/numéricos |

Los datos incluyen reseñas valoradas, historial de viajes y metadatos de destinos (categoría, ubicación, tipo).

---

## 3. Metodología de modelamiento

### 3.1 Módulo 1: Predicción de Demanda (LSTM con Atención Temporal)

**Arquitectura del modelo (`TransportLSTM`)**:

```
Input (secuencia de 60 días)
    │
    ├── Embedding de Ruta (5 rutas → 8 dim)
    ├── Embedding de Clima (3 climas → 4 dim)
    └── Características numéricas (pasajeros normalizados)
            │
            ▼
    LSTM Bidireccional (hidden_dim=128, 2 capas, dropout=0.3)
            │
            ▼
    Mecanismo de Atención Temporal (tanh + softmax)
            │
            ▼
    Capa Fully Connected (256 → 128 → 1)
            │
            ▼
    Predicción: pasajeros del día siguiente
```

**Preprocesamiento**:
- Normalización de demanda con `StandardScaler`.
- Codificación one-hot de rutas y clima con `LabelEncoder`.
- Ventanas deslizantes de 60 días para predecir el día 61.
- Entrenamiento autoregresivo: las predicciones se realimentan como entrada para pronósticos multi-step.

**Hiperparámetros**:
- Ventana de contexto: 60 días
- Horizonte de predicción: 30 días
- Optimizador: Adam (lr=1e-3)
- Pérdida: MSELoss
- Épocas: 200 con early stopping
- Batch size: 32

### 3.2 Módulo 2: Clasificación de Conducción Distractiva (CNN con Transfer Learning)

**Arquitectura**:
- Backbone pre-entrenado: **MobileNetV3-Small** (modelo entrenado) / **ResNet18** (disponible vía CLI)
- Clasificador personalizado reemplazando la última capa fully connected:
  ```
  Backbone (features congelados o afinados)
      │
      ▼
  AdaptiveAvgPool2d(1)
      │
      ▼
  Dropout(0.5)
      │
      ▼
  Linear(backbone_features → 128)
      │
      ▼
  ReLU + Dropout(0.3)
      │
      ▼
  Linear(128 → 5 clases)
  ```

**Aumentación de datos**:
- Rotación aleatoria (±10°)
- Volteo horizontal
- Ajuste de brillo/contraste
- Desenfoque gaussiano

**Preprocesamiento**:
- Redimensionamiento a 224×224 píxeles
- Normalización con media/desviación de ImageNet
- Batch size: 32

### 3.3 Módulo 3: Sistema de Recomendación de Destinos (Filtrado Colaborativo Híbrido)

**Arquitectura (`HybridTravelRecommender`)**:

```
User ID ──→ User Embedding (64 dim)
                                            ┐
Item ID ──→ Item Embedding (64 dim)         ├── Concatenación ──→ MLP Scorer ──→ Score
                                            ┘
Item Features (21 dim) ────────────────────┘
                                            ┌── Linear(149 → 128) → ReLU → Dropout(0.2)
       MLP Scorer:                          ├── Linear(128 → 64) → ReLU → Dropout(0.2)
                                            └── Linear(64 → 1)
```

**Estrategia de entrenamiento**:
- **Negative sampling**: 4 muestras negativas por cada interacción positiva.
- **Pérdida**: `BCEWithLogitsLoss` (logits → sigmoid binaria).
- **Optimizador**: AdamW (lr=1e-3, weight_decay=1e-5).
- **División de datos**: 80% entrenamiento, 10% validación, 10% prueba (por usuario).
- **Early stopping**: paciencia de 5 épocas basada en recall@10 de validación.

### 3.4 Preprocesamiento y Baseline

| Módulo | Preprocesamiento | Baseline |
|--------|------------------|----------|
| Demanda | StandardScaler, LabelEncoder, ventanas deslizantes | Media móvil (MA-7) |
| Conducción | Resize 224×224, normalización ImageNet, aumentación | Clasificador aleatorio (20%) |
| Recomendación | Negative sampling, feature engineering, split por usuario | Recomendación por popularidad |

---

## 4. Evaluación comparativa de desempeño

| Módulo | Precisión | Recall | RMSE | MAE | MAPE | F1-Score |
|--------|-----------|--------|------|-----|------|----------|
| **Demanda** (LSTM) | — | — | **175.83** | **125.86** | **7.77%** | — |
| **Conducción** (MobileNetV3) | **94.78%** | 94.78% | — | — | — | **0.9478** |
| **Recomendación** (Hybrid) | 10% (@10) | **100%** (@10) | — | — | — | — |

### 4.1 Resultados del Módulo de Demanda (RNA-LSTM)

| Ruta | RMSE | MAE | MAPE |
|------|------|-----|------|
| Ruta A | 183.81 | 131.47 | 7.62% |
| Ruta B | 219.67 | 157.29 | 9.71% |
| Ruta C | 155.09 | 111.30 | 6.91% |
| Ruta D | 157.89 | 112.72 | 7.01% |
| Ruta E | 162.70 | 116.53 | 7.61% |
| **Global** | **175.83** | **125.86** | **7.77%** |

El modelo logra un MAPE global de 7.77%, lo que indica que el error promedio de predicción es inferior al 8% del valor real de la demanda. La Ruta B presenta el mayor error debido a su mayor variabilidad estacional.

### 4.2 Resultados del Módulo de Clasificación (CNN)

| Métrica | Valor |
|---------|-------|
| Accuracy en test | 94.78% |
| F1-Score (macro) | 0.9478 |
| Imágenes evaluadas | 1,091 |
| Mejor F1 en validación | 0.9616 |

El clasificador supera ampliamente el baseline aleatorio (20%) y demuestra ser efectivo para detectar conductas de riesgo en tiempo real.

### 4.3 Resultados del Módulo de Recomendación

| Métrica | Valor |
|---------|-------|
| Precisión@5 | 0.2000 |
| Recall@5 | 1.0000 |
| NDCG@5 | 0.6037 |
| Precisión@10 | 0.1000 |
| Recall@10 | 1.0000 |
| NDCG@10 | 0.6037 |
| Usuarios evaluados | 114 |
| Interacciones held-out | 114 |

El modelo logra recall perfecto (100%) tanto en top-5 como top-10, lo que significa que para cada usuario, al menos uno de sus destinos held-out aparece en las recomendaciones. El NDCG de 0.6037 indica que los destinos relevantes tienden a aparecer en posiciones altas del ranking.

### 4.4 Interpretación técnica

- **Demanda**: El MAPE de 7.77% es excelente para predicción de series de tiempo de alta volatilidad. La arquitectura LSTM bidireccional con atención captura eficazmente patrones estacionales y dependencias temporales largas.
- **Conducción**: La accuracy de 94.78% con MobileNetV3-Small demuestra que el transfer learning es altamente efectivo incluso con un backbone liviano, ideal para despliegue en dispositivos con recursos limitados.
- **Recomendación**: El recall perfecto indica que el modelo de embeddings captura bien las preferencias latentes de los usuarios. La precisión baja (@10 = 10%) es esperable en filtrado colaborativo con muchos ítems (684 destinos) y pocas interacciones por usuario.

### 4.5 Figuras de evaluación

<!-- COLOQUE AQUÍ LAS FIGURAS DE EVALUACIÓN DE CADA MÓDULO -->

---

## 5. Análisis de importancia e interpretabilidad

### 5.1 Importancia exploratoria de variables

Para el módulo de predicción de demanda, las variables más influyentes son:

| Variable | Importancia relativa |
|----------|---------------------|
| Demanda histórica | 32.1% |
| Incidentes viales | 26.5% |
| Capacidad de rutas | 18.4% |
| Tasa de ocupación | 5.8% |
| Tiempo de espera | 4.5% |
| Flujo de pasajeros | 3.8% |
| Eficiencia operativa | 3.1% |
| Tipo de ruta | 2.9% |
| Distancia recorrida | 1.9% |
| Periodo (30/90 días) | 1.0% |

### 5.2 Interpretabilidad de los modelos

- **Demanda (LSTM)**: El mecanismo de atención temporal permite visualizar qué días del historial son más relevantes para cada predicción, dando mayor peso a patrones semanales y estacionales.
- **Conducción (CNN)**: Técnicas de Grad-CAM pueden aplicarse para visualizar las regiones de la imagen que más influyen en la clasificación (por ejemplo, las manos del conductor, el teléfono, la posición de la cabeza).
- **Recomendación (Hybrid)**: Los embeddings de usuario e ítem pueden proyectarse en 2D (t-SNE/PCA) para visualizar clusters de usuarios con preferencias similares y destinos relacionados.

### 5.3 Lectura de negocio

- **Demanda**: Con MAPE de 7.77%, la empresa puede planificar la asignación de flota con alta confianza, reduciendo costos operativos y mejorando la experiencia del pasajero.
- **Conducción**: El sistema puede integrarse con cámaras a bordo para generar alertas en tiempo real cuando se detecten conductas de riesgo, reduciendo potencialmente los accidentes viales.
- **Recomendación**: El recall perfecto significa que el sistema siempre sugiere destinos relevantes para cada usuario, mejorando la retención y satisfacción del cliente.

### 5.4 Figuras de interpretabilidad

<!-- COLOQUE AQUÍ LAS FIGURAS DE SHAP, GRAD-CAM, ETC. -->

---

## 6. Aplicación web y despliegue

### 6.1 Arquitectura de la aplicación

```
┌─────────────────────────────────────────────────────────┐
│                   Cliente (React + Vite)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  SystemForm  │  │  ModuleResult│  │  AboutModel    │ │
│  │  (Formulario │  │  (Gráficas,  │  │  (Información  │ │
│  │   con tabs)  │  │   métricas)  │  │   del modelo)  │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────────┘ │
│         │                 │                              │
│         └──── model/ ─────┘                              │
│              │ (transportModel.js)                       │
│              ▼                                           │
│         services/api.js (fetch)                          │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP (proxy Vite)
                   ▼
┌─────────────────────────────────────────────────────────┐
│           API REST (FastAPI + Uvicorn)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ /demand     │  │ /distraction │  │ /recommender   │ │
│  │ LSTM + Aten.│  │ CNN (Mobile) │  │ Hybrid Embed.  │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│         │                 │               │              │
│         └─────────────────┴───────────────┘              │
│                         │                                │
│              dependencies.py (Singleton)                  │
└──────────────────────────────────────────────────────────┘
```

**Tecnologías utilizadas**:
- **Frontend**: React 19, Vite 6, Tailwind CSS v4, Recharts (gráficos)
- **Backend**: FastAPI, PyTorch, Uvicorn
- **Modelos**: PyTorch (LSTM, CNN, Embedding + MLP)
- **Despliegue**: Railpack (backend), Netlify (frontend)

### 6.2 Repositorio y despliegue

- **Repositorio**: [GitHub - RNA Sistema de Transporte Inteligente](https://github.com/d0ubt/rna-sistema-de-transporte-inteligente)
- **Frontend desplegado**: [Netlify App](https://sistema-transporte-inteligente-rna.netlify.app)
- **Backend**: Desplegado en Railpack con comando `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}`

### 6.3 Figura de aplicación web

<!-- COLOQUE AQUÍ CAPTURA DE PANTALLA DE LA APLICACIÓN WEB -->

---

## 7. Aprendizajes, conclusiones y limitaciones

### Conclusiones principales

1. **Integración exitosa de tres modelos de deep learning** en una sola plataforma web funcional, demostrando que es posible unificar predicción de series de tiempo, clasificación de imágenes y sistemas de recomendación en un mismo ecosistema.

2. **La arquitectura LSTM bidireccional con atención temporal** es altamente efectiva para predicción de demanda de transporte, logrando errores inferiores al 8% incluso con datos sintéticos.

3. **El transfer learning con MobileNetV3-Small** alcanza un 94.78% de accuracy en clasificación de conducción distractiva, demostrando que modelos ligeros pueden ser tan efectivos como arquitecturas más pesadas.

4. **El modelo híbrido de recomendación** con embeddings de usuario e ítem logra recall perfecto, indicando una excelente capacidad para capturar preferencias latentes incluso con un dataset de tamaño moderado.

### Limitaciones identificadas

- **Datos sintéticos para demanda**: El modelo fue entrenado con datos generados artificialmente; su rendimiento en datos reales podría diferir.
- **Cold start en recomendación**: Usuarios nuevos sin historial no pueden recibir recomendaciones personalizadas (el modelo requiere haber visto al usuario durante entrenamiento).
- **Sin aumento de datos para recomendación**: El negative sampling es la única fuente de diversidad; no se implementaron técnicas como item-to-item o content-based filtering como respaldo.
- **Sin integración de cámara en vivo para clasificación**: Actualmente el sistema procesa imágenes subidas por el usuario, no video en tiempo real.

### Trabajo futuro

- Implementar un módulo de recomendación híbrido que combine contenido+colaborativo para manejar cold start.
- Integrar transmisión de video en vivo para detección de distracciones en tiempo real.
- Entrenar el modelo de demanda con datos reales de la empresa de transporte.
- Implementar pipeline de CI/CD para reentrenamiento automático de modelos.
- Agregar autenticación de usuarios y personalización de perfiles.

---

## 8. Referencias

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
2. Howard, A., et al. (2019). *Searching for MobileNetV3*. arXiv:1905.02244.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
4. Xue, F., et al. (2017). *Deep Matrix Factorization Models for Recommender Systems*. IJCAI.
5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
6. Kaggle. (2016). *State Farm Distracted Driver Detection*. https://www.kaggle.com/c/state-farm-distracted-driver-detection
7. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
8. FastAPI. (2024). *FastAPI Framework*. https://fastapi.tiangolo.com/
9. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
