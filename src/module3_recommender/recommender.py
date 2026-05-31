
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import build_model


class TravelDestinationRecommender:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        metadata = checkpoint["metadata"]
        model_config = metadata["model_config"]
        self.model = build_model(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.item_features = checkpoint["item_features"].float()
        self.idx_to_user = metadata["idx_to_user"]
        self.idx_to_item = metadata["idx_to_item"]
        self.user_to_idx = metadata["user_to_idx"]
        self.item_metadata = metadata["item_metadata"]
        self.train_history = {
            int(user): set(items)
            for user, items in metadata.get("train_history", {}).items()
        }
        self.full_history = {
            int(user): set(items)
            for user, items in metadata.get("full_history", {}).items()
        }

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        if user_id not in self.user_to_idx:
            raise KeyError(f"Usuario no visto durante entrenamiento: {user_id}")
        user_idx = self.user_to_idx[user_id]
        scores = self.model.score_all_items(user_idx, self.item_features, self.device)
        if exclude_seen:
            seen = self.full_history.get(user_idx, set())
            if seen:
                scores[list(seen)] = -torch.inf
        indices: list[int] = []
        identities: set[str] = set()
        for item_idx in torch.argsort(scores, descending=True).tolist():
            if not torch.isfinite(scores[item_idx]):
                continue
            identity = str(self._display_name(item_idx)).strip().lower()
            if identity in identities:
                continue
            indices.append(item_idx)
            identities.add(identity)
            if len(indices) >= top_k:
                break
        probabilities = torch.sigmoid(scores[indices]).tolist()
        return [
            self._format_recommendation(item_idx, rank + 1, float(probabilities[rank]))
            for rank, item_idx in enumerate(indices)
        ]

    def _display_name(self, item_idx: int) -> str:
        metadata = self.item_metadata[item_idx] if item_idx < len(self.item_metadata) else {}
        return (
            metadata.get("Name")
            or metadata.get("name")
            or metadata.get("destination")
            or metadata.get("Destination")
            or metadata.get("destination_name")
            or metadata.get("Destination Name")
            or metadata.get("item_key")
            or self.idx_to_item[item_idx]
        )

    def _format_recommendation(self, item_idx: int, rank: int, score: float) -> dict[str, Any]:
        metadata = self.item_metadata[item_idx] if item_idx < len(self.item_metadata) else {}
        return {
            "rank": rank,
            "destination_id": self.idx_to_item[item_idx],
            "destination": self._display_name(item_idx),
            "score": score,
            "metadata": metadata,
        }
