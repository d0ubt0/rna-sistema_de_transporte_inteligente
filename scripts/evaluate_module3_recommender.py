from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module3_recommender.data_loader import RecommenderDataConfig, create_data_loaders
from src.module3_recommender.evaluator import TravelRecommenderEvaluator
from src.module3_recommender.model import build_model
from src.shared.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua un checkpoint del modulo 3.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="models/module3_recommender/evaluation_external")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["metadata"]
    model_config = metadata["model_config"]
    data = create_data_loaders(
        RecommenderDataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=metadata["config"].get("seed", 42),
            negative_samples=metadata["config"].get("negative_samples", 4),
        )
    )
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = TravelRecommenderEvaluator(
        model=model,
        records=data.test_records,
        item_features=data.item_features,
        idx_to_user=data.idx_to_user,
        idx_to_item=data.idx_to_item,
        item_metadata=data.item_metadata,
        train_history=data.train_history,
        device=device,
        output_dir=args.output_dir,
    ).evaluate()
    save_json(result.to_dict(), Path(args.output_dir) / "evaluation_result.json")
    print(f"Precision@10: {result.metrics.get('precision@10', 0.0):.4f}")
    print(f"Recall@10: {result.metrics.get('recall@10', 0.0):.4f}")
    print(f"NDCG@10: {result.metrics.get('ndcg@10', 0.0):.4f}")


if __name__ == "__main__":
    main()
