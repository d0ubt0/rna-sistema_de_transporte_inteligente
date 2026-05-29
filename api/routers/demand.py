
import numpy as np
import torch
from dependencies import get_demand_model_infra
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/demand", tags=["Predicción de Demanda"])

class DemandRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description=(
            "Matriz 30x4 con los últimos 30 días de datos normalizados. "
            "Columnas: [dia_semana, mes, festivo, pasajeros]. "
            "Cada columna debe estar normalizada con los mismos scalers usados en entrenamiento."
        ),
    )
    route_id: int = Field(..., ge=0, le=4, description="ID de ruta codificada (0–4)")
    clima_id: int | None = Field(
        default=None, ge=0, le=2,
        description="ID de clima para el paso único (compatibilidad hacia atrás)",
    )
    steps: int = Field(default=1, ge=1, le=30, description="Número de pasos a predecir")
    future_features: list[list[float]] | None = Field(
        default=None,
        description="Matriz steps×3 con [dia_semana, mes, festivo] normalizados para cada paso futuro",
    )
    future_clima_ids: list[int] | None = Field(
        default=None,
        description="Lista de clima_ids para cada paso futuro",
    )

@router.post("/predict")
async def predict_demand(data: DemandRequest):
    try:
        model, scalers, device = get_demand_model_infra()

        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")

        target_scaler = scalers["target"]
        route_encoder = scalers["route"]

        steps = data.steps

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

        seq = np.array(data.sequence, dtype=np.float32).reshape(1, 30, 4)
        predictions = []

        for i in range(steps):
            seq_tensor = torch.tensor(seq, device=device)
            route_tensor = torch.tensor([data.route_id], device=device)
            clima_tensor = torch.tensor([clima_ids[i]], device=device)

            with torch.no_grad():
                pred_norm = model(seq_tensor, route_tensor, clima_tensor)
                pred_norm = pred_norm.cpu().numpy().flatten()[0]

            pred_abs = target_scaler.inverse_transform(
                np.array([[pred_norm]])
            )[0, 0]
            predictions.append(round(float(pred_abs), 2))

            if i < steps - 1:
                ff = future_features[i]
                new_row = np.array(
                    [[ff[0], ff[1], ff[2], pred_norm]], dtype=np.float32
                )
                seq = np.concatenate(
                    [seq[:, 1:, :], new_row.reshape(1, 1, 4)], axis=1
                )

        return {
            "predicciones": predictions,
            "prediccion_pasajeros": predictions[0] if steps == 1 else None,
            "ruta": route_encoder.inverse_transform([data.route_id])[0],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
