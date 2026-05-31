# api/routers/recommender.py
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.dependencies import get_recommender_model

router = APIRouter(prefix="/recommender", tags=["Recommender"])


@router.get("/health")
async def health():
    model = get_recommender_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de recomendación no disponible")
    return {"status": "online", "model": "module3_recommender"}


@router.get("/info")
async def info():
    model = get_recommender_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de recomendación no disponible")
    num_users = len(model.user_to_idx)
    num_items = len(model.idx_to_item)
    return {
        "num_users": num_users,
        "num_items": num_items,
        "sample_users": list(model.user_to_idx.keys())[:20],
        "sample_destinations": [model._display_name(i) for i in range(min(20, num_items))],
    }


@router.get("/recommend")
async def recommend(
    user_id: str = Query(..., description="ID del usuario para generar recomendaciones"),
    top_k: int = Query(6, ge=1, le=50, description="Número de recomendaciones a retornar"),
    exclude_seen: bool = Query(True, description="Excluir destinos ya visitados"),
):
    model = get_recommender_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de recomendación no disponible")

    if user_id not in model.user_to_idx:
        sample = list(model.user_to_idx.keys())[:5]
        raise HTTPException(
            status_code=404,
            detail=f"Usuario '{user_id}' no encontrado. Usuarios disponibles (muestra): {sample}",
        )

    try:
        recommendations = model.recommend(
            user_id=user_id,
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
        return {
            "user_id": user_id,
            "top_k": top_k,
            "recommendations": recommendations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}")
