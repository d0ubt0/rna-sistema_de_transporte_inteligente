
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from src.shared.base_trainer import BaseTrainer, TrainingResult
from src.shared.utils import ensure_dir, get_device, save_json, set_seed

from .data_loader import RecommenderDataConfig, RecommenderDataModule, create_data_loaders
from .evaluator import TravelRecommenderEvaluator
from .model import build_model


@dataclass
class RecommenderTrainingConfig:
    data_dir: str
    output_dir: str = "models/module3_recommender"
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    negative_samples: int = 4
    max_content_categories: int = 80
    min_positive_rating: float | None = None
    seed: int = 42
    val_split: float = 0.1
    test_split: float = 0.1
    device: str | None = None


class TravelRecommenderTrainer(BaseTrainer):
    def __init__(self, config: RecommenderTrainingConfig) -> None:
        super().__init__(config.output_dir)
        self.config = config
        self.device = torch.device(config.device) if config.device else get_device()

    def train(self) -> TrainingResult:
        set_seed(self.config.seed)
        ensure_dir(self.output_dir)
        data = create_data_loaders(
            RecommenderDataConfig(
                data_dir=self.config.data_dir,
                batch_size=self.config.batch_size,
                val_split=self.config.val_split,
                test_split=self.config.test_split,
                negative_samples=self.config.negative_samples,
                max_content_categories=self.config.max_content_categories,
                min_positive_rating=self.config.min_positive_rating,
                seed=self.config.seed,
            )
        )
        model = build_model(
            num_users=len(data.idx_to_user),
            num_items=len(data.idx_to_item),
            content_dim=data.item_features.shape[1],
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        best_metric = -1.0
        best_model_path = self.output_dir / "best_model.pth"
        history: list[dict[str, float]] = []
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(model, data, criterion, optimizer)
            val_metrics = TravelRecommenderEvaluator(
                model=model,
                records=data.val_records,
                item_features=data.item_features,
                idx_to_user=data.idx_to_user,
                idx_to_item=data.idx_to_item,
                item_metadata=data.item_metadata,
                train_history=data.train_history,
                device=self.device,
                output_dir=self.output_dir / "validation",
                max_examples=0,
            ).evaluate().metrics
            val_recall = float(val_metrics.get("recall@10", 0.0))
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_precision_at_10": float(val_metrics.get("precision@10", 0.0)),
                "val_recall_at_10": val_recall,
                "val_ndcg_at_10": float(val_metrics.get("ndcg@10", 0.0)),
            }
            history.append(row)
            self._save_history(history)
            print(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"loss={train_loss:.4f} val_recall@10={val_recall:.4f} "
                f"val_ndcg@10={row['val_ndcg_at_10']:.4f}"
            )
            if val_recall > best_metric:
                best_metric = val_recall
                epochs_without_improvement = 0
                self._save_checkpoint(model, data, best_model_path)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.patience:
                break

        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluation = TravelRecommenderEvaluator(
            model=model,
            records=data.test_records,
            item_features=data.item_features,
            idx_to_user=data.idx_to_user,
            idx_to_item=data.idx_to_item,
            item_metadata=data.item_metadata,
            train_history=data.train_history,
            device=self.device,
            output_dir=self.output_dir / "evaluation",
        ).evaluate()

        metadata = {
            "config": asdict(self.config),
            "num_users": len(data.idx_to_user),
            "num_items": len(data.idx_to_item),
            "content_dim": int(data.item_features.shape[1]),
            "best_val_recall_at_10": best_metric,
            "schema": data.schema,
            "source_files": data.source_files,
            "history": history,
            "test_metrics": evaluation.metrics,
        }
        save_json(metadata, self.output_dir / "metadata.json")
        return TrainingResult(
            best_model_path=str(best_model_path),
            metrics_path=evaluation.report_path,
            history=history,
            metadata=metadata,
        )

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        data: RecommenderDataModule,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        item_features = data.item_features.to(self.device)
        for user_ids, item_ids, _, labels in data.train_loader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            labels = labels.to(self.device)
            features = item_features[item_ids]
            optimizer.zero_grad(set_to_none=True)
            logits = model(user_ids, item_ids, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        return total_loss / max(total_samples, 1)

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        data: RecommenderDataModule,
        path: Path,
    ) -> None:
        ensure_dir(path.parent)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "config": asdict(self.config),
                    "model_config": asdict(model.config),
                    "idx_to_user": data.idx_to_user,
                    "idx_to_item": data.idx_to_item,
                    "item_to_idx": data.item_to_idx,
                    "user_to_idx": data.user_to_idx,
                    "item_metadata": data.item_metadata,
                    "train_history": {str(key): sorted(value) for key, value in data.train_history.items()},
                    "full_history": {str(key): sorted(value) for key, value in data.full_history.items()},
                    "popularity": data.popularity.tolist(),
                    "schema": data.schema,
                },
                "item_features": data.item_features,
            },
            path,
        )

    def _save_history(self, history: list[dict[str, float]]) -> None:
        path = self.output_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


def train_recommender_model(config: RecommenderTrainingConfig) -> TrainingResult:
    return TravelRecommenderTrainer(config).train()
