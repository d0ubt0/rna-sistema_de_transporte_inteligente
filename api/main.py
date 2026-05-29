from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import demand, distraction, recommender

app = FastAPI(
    title="Sistema Inteligente de Transporte",
    description="API de Predicción, Seguridad Vial y Recomendación",
    version="1.0.0"
)

# Configuración de CORS para permitir conexiones desde un frontend web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusión de módulos (Routers)
app.include_router(demand.router)
app.include_router(distraction.router)
app.include_router(recommender.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Sistema Integrado de Transporte activo",
        "endpoints": ["/demand", "/distraction", "/recommender"]
    }
