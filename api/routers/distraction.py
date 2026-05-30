from pathlib import Path
from tempfile import NamedTemporaryFile

from api.dependencies import get_distraction_classifier
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from src.module2_distraction.classifier import preventive_measure_for

router = APIRouter(prefix="/distraction", tags=["Detección de Distracción"])


@router.get("/health")
async def distraction_health():
    classifier = get_distraction_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de distraccion no disponible",
        )

    return {
        "status": "ready",
        "model": "module2_distraction",
        "classes": classifier.class_names,
        "image_size": classifier.image_size,
        "device": str(classifier.device),
    }


@router.get("/classes")
async def get_distraction_classes():
    classifier = get_distraction_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de distraccion no disponible",
        )

    return {
        "classes": [
            {
                "id": class_name,
                "label": class_name,
                "preventive_measure": preventive_measure_for(class_name),
            }
            for class_name in classifier.class_names
        ]
    }


@router.post("/predict")
async def predict_distraction(file: UploadFile = File(...)):
    classifier = get_distraction_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de distraccion no disponible",
        )

    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen",
        )

    suffix = Path(file.filename or "").suffix or ".jpg"
    temp_path = None

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="La imagen esta vacia")

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        prediction = classifier.predict(temp_path)
        preventive_measure = preventive_measure_for(prediction.predicted_label)

        return {
            "filename": file.filename,
            "predicted_label": prediction.predicted_label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "preventive_measure": preventive_measure,
        }
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
