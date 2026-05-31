
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.shared.base_model import BaseTorchModel


@dataclass
class TravelRecommenderConfig:
    num_users: int
    num_items: int
    content_dim: int = 0
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.2


class HybridTravelRecommender(BaseTorchModel):
    """Hybrid neural recommender for user-destination interactions."""

    def __init__(self, config: TravelRecommenderConfig) -> None:
        super().__init__()
        self.config = config
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        input_dim = config.embedding_dim * 2 + config.content_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self._reset_parameters()

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        parts = [user_vector, item_vector]
        if self.config.content_dim > 0:
            if item_features is None:
                raise ValueError("item_features es requerido cuando content_dim > 0.")
            parts.append(item_features)
        features = torch.cat(parts, dim=1)
        return self.scorer(features).squeeze(1)

    def score_all_items(
        self,
        user_id: int,
        item_features: torch.Tensor,
        device: torch.device | str,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        self.eval()
        device = torch.device(device)
        item_count = self.config.num_items
        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, item_count, batch_size):
                end = min(start + batch_size, item_count)
                items = torch.arange(start, end, dtype=torch.long, device=device)
                users = torch.full_like(items, fill_value=user_id)
                batch_features = (
                    item_features[start:end].to(device)
                    if self.config.content_dim > 0
                    else None
                )
                scores.append(self(users, items, batch_features).detach().cpu())
        return torch.cat(scores)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)


def build_model(
    num_users: int,
    num_items: int,
    content_dim: int = 0,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> HybridTravelRecommender:
    return HybridTravelRecommender(
        TravelRecommenderConfig(
            num_users=num_users,
            num_items=num_items,
            content_dim=content_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    )
