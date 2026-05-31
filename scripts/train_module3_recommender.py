from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module3_recommender import RecommenderTrainingConfig, train_recommender_model
from src.shared.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modulo 3 de recomendacion de destinos.")
    parser.add_argument("--data-dir", required=True, help="Directorio con CSVs del dataset de Kaggle.")
    parser.add_argument("--output-dir", default="models/module3_recommender")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-samples", type=int, default=4)
    parser.add_argument("--min-positive-rating", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda o vacio para autodetectar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RecommenderTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        min_positive_rating=args.min_positive_rating,
        device=args.device,
    )
    result = train_recommender_model(config)
    save_json(result.to_dict(), Path(args.output_dir) / "training_result.json")
    print(f"Modelo guardado en: {result.best_model_path}")
    print(f"Metricas de prueba en: {result.metrics_path}")


if __name__ == "__main__":
    main()
