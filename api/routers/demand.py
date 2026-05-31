
import numpy as np
import pandas as pd
from pathlib import Path
from api.dependencies import get_demand_model_infra
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.module1_demand.predictor import forecast_autoregressive

router = APIRouter(prefix="/demand", tags=["Predicción de Demanda"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMAND_DATA_PATH = PROJECT_ROOT / "data" / "demanda_transporte.csv"
HOLIDAYS = {(1, 1), (5, 1), (7, 28), (7, 29), (12, 25)}


class DemandRequest(BaseModel):
    route_id: int = Field(..., ge=0, le=4, description="ID de ruta codificada (0–4)")
    steps: int = Field(default=30, ge=1, le=30, description="Número de pasos a predecir")
    sequence: list[list[float]] | None = Field(
        default=None,
        description=(
            "Opcional. Matriz 30x4 con los últimos 30 días de datos normalizados. "
            "Columnas: [dia_semana, mes, festivo, pasajeros]. "
            "Si no se envía, el API usa el histórico local."
        ),
    )
    clima_id: int | None = Field(
        default=None, ge=0, le=2,
        description="ID de clima para el paso único (compatibilidad hacia atrás)",
    )
    future_features: list[list[float]] | None = Field(
        default=None,
        description="Matriz steps×3 con [dia_semana, mes, festivo] normalizados para cada paso futuro",
    )
    future_clima_ids: list[int] | None = Field(
        default=None,
        description="Lista de clima_ids para cada paso futuro",
    )


def _is_holiday(date):
    return int((date.month, date.day) in HOLIDAYS)


def _load_route_history(route_id, route_encoder):
    if not DEMAND_DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No existe el histórico de demanda: {DEMAND_DATA_PATH}",
        )

    route_name = route_encoder.inverse_transform([route_id])[0]
    df = pd.read_csv(DEMAND_DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    route_df = (
        df[df["ruta"] == route_name]
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    if len(route_df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"No hay al menos 30 registros históricos para {route_name}",
        )

    return route_name, route_df


def _build_default_forecast_inputs(
    route_df,
    steps,
    feature_scaler,
    target_scaler,
    clima_encoder,
):
    recent = route_df.tail(30).copy()
    scaled_features = feature_scaler.transform(
        recent[["dia_semana", "mes", "festivo"]]
    )
    scaled_target = target_scaler.transform(recent[["pasajeros"]])
    sequence = np.column_stack([scaled_features, scaled_target]).astype(np.float32)

    last_date = recent["fecha"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )
    future_raw = pd.DataFrame({
        "fecha": future_dates,
        "dia_semana": future_dates.weekday,
        "mes": future_dates.month,
        "festivo": [_is_holiday(date) for date in future_dates],
    })

    future_features = feature_scaler.transform(
        future_raw[["dia_semana", "mes", "festivo"]]
    )

    month_mode = (
        route_df.groupby("mes")["clima"]
        .agg(lambda values: values.mode().iloc[0])
        .to_dict()
    )
    fallback_clima = route_df["clima"].mode().iloc[0]
    future_climas = [
        month_mode.get(int(month), fallback_clima)
        for month in future_raw["mes"]
    ]
    future_clima_ids = clima_encoder.transform(future_climas)

    history = [
        {
            "fecha": row.fecha.strftime("%Y-%m-%d"),
            "pasajeros": int(row.pasajeros),
        }
        for row in recent.itertuples()
    ]

    return sequence, future_features, future_clima_ids, future_raw, history


@router.get("/metadata")
async def demand_metadata():
    try:
        model, scalers, _device = get_demand_model_infra()

        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")

        feature_scaler = scalers["feature"]
        target_scaler = scalers["target"]
        route_encoder = scalers["route"]
        clima_encoder = scalers["clima"]

        return {
            "sequence_length": 30,
            "forecast_horizon": 30,
            "feature_columns": ["dia_semana", "mes", "festivo", "pasajeros"],
            "future_feature_columns": ["dia_semana", "mes", "festivo"],
            "feature_min": feature_scaler.data_min_.tolist(),
            "feature_max": feature_scaler.data_max_.tolist(),
            "target_min": float(target_scaler.data_min_[0]),
            "target_max": float(target_scaler.data_max_[0]),
            "routes": [
                {"id": int(index), "name": name}
                for index, name in enumerate(route_encoder.classes_)
            ],
            "climas": [
                {"id": int(index), "name": name}
                for index, name in enumerate(clima_encoder.classes_)
            ],
            "model": "TransportLSTMAttention",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_demand(data: DemandRequest):
    try:
        model, scalers, device = get_demand_model_infra()

        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")

        target_scaler = scalers["target"]
        route_encoder = scalers["route"]

        steps = data.steps
        feature_scaler = scalers["feature"]
        clima_encoder = scalers["clima"]
        route_name = route_encoder.inverse_transform([data.route_id])[0]
        future_raw = None
        history = None

        if data.sequence is None:
            route_name, route_df = _load_route_history(
                data.route_id,
                route_encoder,
            )
            seq, future_features, clima_ids, future_raw, history = (
                _build_default_forecast_inputs(
                    route_df=route_df,
                    steps=steps,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    clima_encoder=clima_encoder,
                )
            )
        else:
            # Resolver clima_ids
            if data.future_clima_ids:
                clima_ids = data.future_clima_ids
            elif data.clima_id is not None:
                clima_ids = [data.clima_id] * steps
            else:
                clima_ids = [1] * steps

            # Resolver future_features
            if data.future_features:
                future_features = data.future_features
            else:
                future_features = [[0.5, 0.5, 0.0]] * steps

            if len(clima_ids) != steps or len(future_features) != steps:
                raise HTTPException(
                    status_code=400,
                    detail=f"future_features y future_clima_ids deben tener longitud {steps}",
                )

            seq = np.array(data.sequence, dtype=np.float32)
            if seq.shape != (30, 4):
                raise HTTPException(
                    status_code=400,
                    detail="sequence debe tener forma 30x4",
                )

        pred_norm = forecast_autoregressive(
            model=model,
            sequence=seq,
            route_id=data.route_id,
            future_features=future_features,
            future_clima_ids=clima_ids,
            device=device,
        )
        pred_abs = target_scaler.inverse_transform(
            pred_norm.reshape(-1, 1)
        ).flatten()
        predictions = [round(float(value), 2) for value in pred_abs]
        prediction_rows = []
        if future_raw is not None:
            prediction_rows = [
                {
                    "fecha": row.fecha.strftime("%Y-%m-%d"),
                    "prediccion": predictions[index],
                }
                for index, row in enumerate(future_raw.itertuples())
            ]

        return {
            "predicciones": predictions,
            "prediccion_pasajeros": predictions[0] if steps == 1 else None,
            "ruta": route_name,
            "historico": history,
            "pronostico": prediction_rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
