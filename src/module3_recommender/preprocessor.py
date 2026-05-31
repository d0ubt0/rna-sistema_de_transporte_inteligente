
from __future__ import annotations

from .data_loader import RecommenderDataConfig, create_data_loaders


def preprocess_pipeline(
    data_dir: str,
    batch_size: int = 256,
    val_split: float = 0.1,
    test_split: float = 0.1,
    negative_samples: int = 4,
    seed: int = 42,
):
    """Compatibilidad con la estructura del modulo 1: prepara datos del recomendador."""
    return create_data_loaders(
        RecommenderDataConfig(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            negative_samples=negative_samples,
            seed=seed,
        )
    )
