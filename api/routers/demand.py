
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_demand_model_infra

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
    clima_id: int = Field(..., ge=0, le=2, description="ID de clima codificado (0–2)")

@router.post("/predict")
async def predict_demand(data: DemandRequest):
    try:
        model, scalers, device = get_demand_model_infra()

        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")

        target_scaler = scalers["target"]
        route_encoder = scalers["route"]

        seq_array = np.array(data.sequence, dtype=np.float32).reshape(1, 30, 4)

        seq_tensor = torch.tensor(seq_array, device=device)
        route_tensor = torch.tensor([data.route_id], device=device)
        clima_tensor = torch.tensor([data.clima_id], device=device)

        with torch.no_grad():
            pred_norm = model(seq_tensor, route_tensor, clima_tensor)
            pred_norm = pred_norm.cpu().numpy().flatten()[0]

        pred_abs = target_scaler.inverse_transform(
            np.array([[pred_norm]])
        )[0, 0]

        return {
            "prediccion_pasajeros": round(float(pred_abs), 2),
            "ruta": route_encoder.inverse_transform([data.route_id])[0],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
