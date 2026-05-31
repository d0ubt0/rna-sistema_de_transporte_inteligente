
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.shared.base_evaluator import EvaluationResult
from src.shared.utils import ensure_dir, save_json

from .data_loader import InteractionRecord
from .model import HybridTravelRecommender


@dataclass
class RecommendationExample:
    user_id: str
    heldout_destination: str | None
    recommendations: list[dict[str, Any]]


class TravelRecommenderEvaluator:
    def __init__(
        self,
        model: HybridTravelRecommender,
        records: list[InteractionRecord],
        item_features: torch.Tensor,
        idx_to_user: list[str],
        idx_to_item: list[str],
        item_metadata: list[dict[str, Any]],
        train_history: dict[int, set[int]],
        device: torch.device | str,
        output_dir: str | Path,
        k_values: tuple[int, ...] = (5, 10),
        max_examples: int = 5,
    ) -> None:
        self.model = model
        self.records = records
        self.item_features = item_features
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.item_metadata = item_metadata
        self.train_history = train_history
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.k_values = k_values
        self.max_examples = max_examples

    def evaluate(self) -> EvaluationResult:
        ensure_dir(self.output_dir)
        self.model.to(self.device)
        records_by_user: dict[int, set[int]] = {}
        for record in self.records:
            records_by_user.setdefault(record.user_idx, set()).add(record.item_idx)

        rows: list[dict[str, float]] = []
        examples: list[RecommendationExample] = []
        max_k = max(self.k_values)

        for user_idx, relevant_items in records_by_user.items():
            ranked_items = self._rank_for_user(user_idx, max_k)
            ranked_destinations = [self._item_identity(item_idx) for item_idx in ranked_items]
            relevant_destinations = {
                self._item_identity(item_idx)
                for item_idx in relevant_items
            }
            rows.append(_metrics_for_user(ranked_destinations, relevant_destinations, self.k_values))
            if len(examples) < self.max_examples:
                examples.append(
                    RecommendationExample(
                        user_id=self.idx_to_user[user_idx],
                        heldout_destination=self._display_name(next(iter(relevant_items))),
                        recommendations=[
                            self._format_recommendation(item_idx, position + 1)
                            for position, item_idx in enumerate(ranked_items[:max_k])
                        ],
                    )
                )

        metrics = _aggregate(rows, self.k_values)
        metrics["evaluated_users"] = len(records_by_user)
        metrics["heldout_interactions"] = len(self.records)
        metrics_path = self.output_dir / "metrics.json"
        save_json(metrics, metrics_path)

        examples_path = self.output_dir / "examples.json"
        save_json(
            {
                "examples": [
                    {
                        "user_id": example.user_id,
                        "heldout_destination": example.heldout_destination,
                        "recommendations": example.recommendations,
                    }
                    for example in examples
                ]
            },
            examples_path,
        )
        return EvaluationResult(
            metrics=metrics,
            examples_path=str(examples_path),
            report_path=str(metrics_path),
        )

    def _rank_for_user(self, user_idx: int, limit: int) -> list[int]:
        scores = self.model.score_all_items(user_idx, self.item_features, self.device)
        seen_items = self.train_history.get(user_idx, set())
        if seen_items:
            scores[list(seen_items)] = -torch.inf
        sorted_items = torch.argsort(scores, descending=True).tolist()
        ranked_items: list[int] = []
        ranked_identities: set[str] = set()
        for item_idx in sorted_items:
            if not torch.isfinite(scores[item_idx]):
                continue
            identity = self._item_identity(item_idx)
            if identity in ranked_identities:
                continue
            ranked_items.append(item_idx)
            ranked_identities.add(identity)
            if len(ranked_items) >= limit:
                break
        return ranked_items

    def _item_identity(self, item_idx: int) -> str:
        return str(self._display_name(item_idx)).strip().lower()

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

    def _format_recommendation(self, item_idx: int, rank: int) -> dict[str, Any]:
        metadata = self.item_metadata[item_idx] if item_idx < len(self.item_metadata) else {}
        return {
            "rank": rank,
            "destination_id": self.idx_to_item[item_idx],
            "destination": self._display_name(item_idx),
            "metadata": metadata,
        }


def _metrics_for_user(
    ranked_items: list[str],
    relevant_items: set[str],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in k_values:
        top_k = ranked_items[:k]
        seen_hits: set[str] = set()
        hits = []
        for item in top_k:
            hit = item in relevant_items and item not in seen_hits
            hits.append(1.0 if hit else 0.0)
            if hit:
                seen_hits.add(item)
        hit_count = float(sum(hits))
        result[f"precision@{k}"] = hit_count / max(k, 1)
        result[f"recall@{k}"] = hit_count / max(len(relevant_items), 1)
        result[f"hit_rate@{k}"] = 1.0 if hit_count > 0 else 0.0
        result[f"map@{k}"] = _average_precision(hits, len(relevant_items))
        result[f"ndcg@{k}"] = _ndcg(hits, min(len(relevant_items), k))
    return result


def _aggregate(rows: list[dict[str, float]], k_values: tuple[int, ...]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for k in k_values for metric in (f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", f"map@{k}", f"ndcg@{k}")}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _average_precision(hits: list[float], relevant_count: int) -> float:
    if relevant_count == 0:
        return 0.0
    score = 0.0
    hits_seen = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            hits_seen += 1.0
            score += hits_seen / index
    return score / min(relevant_count, len(hits))


def _ndcg(hits: list[float], ideal_hits: int) -> float:
    if ideal_hits == 0:
        return 0.0
    dcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    return float(dcg / ideal) if ideal else 0.0
