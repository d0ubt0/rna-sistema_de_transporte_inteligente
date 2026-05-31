from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module3_recommender import TravelDestinationRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera recomendaciones de destinos para usuarios.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--user-id", action="append", required=True, help="Puede repetirse para varios usuarios.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommender = TravelDestinationRecommender(args.checkpoint, device=args.device)
    output = {
        user_id: recommender.recommend(user_id=user_id, top_k=args.top_k)
        for user_id in args.user_id
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
