# api/dependencies.py
import os
import sys

import joblib
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.module1_demand.model import TransportLSTM

# ============================================================
# FUNCIÓN DE CARGA GLOBAL
# ============================================================
MODEL_DIR = os.path.join(API_DIR, "models", "demand")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Variables globales para el Singleton
_model = None
_scalers = {}

def get_demand_model_infra():
    """Carga el modelo y scalers una sola vez en memoria (Pattern Singleton)"""
    global _model, _scalers

    if _model is None:
        try:
            _scalers["feature"] = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
            _scalers["target"] = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))
            _scalers["route"] = joblib.load(os.path.join(MODEL_DIR, "route_encoder.pkl"))
            _scalers["clima"] = joblib.load(os.path.join(MODEL_DIR, "clima_encoder.pkl"))

            _model = TransportLSTM(
                num_routes=len(_scalers["route"].classes_),
                num_climas=len(_scalers["clima"].classes_)
            ).to(DEVICE)

            weights = torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True)
            _model.load_state_dict(weights)
            _model.eval()
            print(f"✓ Modelo de demanda cargado exitosamente en {DEVICE} desde dependencies.py")
        except Exception as e:
            print(f"❌ Error cargando infraestructura de demanda: {e}")

    return _model, _scalers, DEVICE
