
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


USER_ALIASES = ("user_id", "userid", "user id", "customer_id", "customerid", "traveler_id", "traveller_id", "client_id")
ITEM_ALIASES = ("destination_id", "destinationid", "place_id", "placeid", "item_id", "product_id", "destination", "destination_name", "location", "city")
RATING_ALIASES = (
    "rating",
    "ratings",
    "score",
    "stars",
    "user_rating",
    "review_rating",
    "experience_rating",
    "experiencerating",
)
TIME_ALIASES = (
    "timestamp",
    "date",
    "travel_date",
    "traveldate",
    "visit_date",
    "visitdate",
    "visited_at",
    "created_at",
)


@dataclass
class RecommenderDataConfig:
    data_dir: str
    batch_size: int = 256
    val_split: float = 0.1
    test_split: float = 0.1
    negative_samples: int = 4
    max_content_categories: int = 80
    min_positive_rating: float | None = None
    seed: int = 42


@dataclass
class InteractionRecord:
    user_idx: int
    item_idx: int
    rating: float


@dataclass
class RecommenderDataModule:
    train_loader: DataLoader
    val_records: list[InteractionRecord]
    test_records: list[InteractionRecord]
    item_features: torch.Tensor
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_user: list[str]
    idx_to_item: list[str]
    item_metadata: list[dict[str, Any]]
    train_history: dict[int, set[int]]
    full_history: dict[int, set[int]]
    popularity: np.ndarray
    source_files: list[str]
    schema: dict[str, str]


class TravelInteractionDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[int, int, float]],
        item_features: torch.Tensor,
    ) -> None:
        self.samples = samples
        self.item_features = item_features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, item_idx, label = self.samples[index]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            self.item_features[item_idx].float(),
            torch.tensor(label, dtype=torch.float32),
        )


def create_data_loaders(config: RecommenderDataConfig) -> RecommenderDataModule:
    root = Path(config.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    frames = _read_csvs(root)
    interactions, destinations, schema = _resolve_tables(frames)
    interactions = _prepare_interactions(interactions, schema, config.min_positive_rating)
    if interactions.empty:
        raise ValueError("No se encontraron interacciones positivas usuario-destino.")

    item_metadata = _build_item_metadata(interactions, destinations, schema["item"])
    user_values = sorted(interactions[schema["user"]].astype(str).unique())
    item_values = sorted(item_metadata["item_key"].astype(str).unique())
    item_metadata = (
        item_metadata.assign(item_key=item_metadata["item_key"].astype(str))
        .drop_duplicates("item_key", keep="first")
        .set_index("item_key")
        .reindex(item_values)
        .reset_index()
    )
    user_to_idx = {value: index for index, value in enumerate(user_values)}
    item_to_idx = {value: index for index, value in enumerate(item_values)}

    interactions["user_idx"] = interactions[schema["user"]].astype(str).map(user_to_idx)
    interactions["item_idx"] = interactions[schema["item"]].astype(str).map(item_to_idx)
    records = [
        InteractionRecord(int(row.user_idx), int(row.item_idx), float(row.rating_value))
        for row in interactions.itertuples()
    ]

    train_records, val_records, test_records = _split_by_user(
        records,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
    )
    train_history = _history(train_records)
    full_history = _history(records)
    popularity = _popularity(train_records, len(item_values))
    item_features = _build_item_features(
        item_metadata,
        max_categories=config.max_content_categories,
    )
    samples = _negative_sample(
        train_records,
        train_history=train_history,
        num_items=len(item_values),
        negatives_per_positive=config.negative_samples,
        seed=config.seed,
    )
    dataset = TravelInteractionDataset(samples, item_features)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return RecommenderDataModule(
        train_loader=loader,
        val_records=val_records,
        test_records=test_records,
        item_features=item_features,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_user=user_values,
        idx_to_item=item_values,
        item_metadata=item_metadata.fillna("").to_dict(orient="records"),
        train_history=train_history,
        full_history=full_history,
        popularity=popularity,
        source_files=[str(path) for path in frames],
        schema=schema,
    )


def _read_csvs(root: Path) -> dict[Path, pd.DataFrame]:
    csv_files = sorted(root.rglob("*.csv")) if root.is_dir() else [root]
    frames = {}
    for path in csv_files:
        try:
            frames[path] = pd.read_csv(path)
        except UnicodeDecodeError:
            frames[path] = pd.read_csv(path, encoding="latin-1")
    if not frames:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {root}")
    return frames


def _resolve_tables(frames: dict[Path, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, str]]:
    interaction_parts: list[pd.DataFrame] = []
    interaction_paths: set[Path] = set()
    for path, frame in frames.items():
        schema = {
            "user": _find_column(frame, USER_ALIASES),
            "item": _find_column(frame, ITEM_ALIASES),
            "rating": _find_column(frame, RATING_ALIASES),
            "time": _find_column(frame, TIME_ALIASES),
        }
        if schema["user"] is None or schema["item"] is None:
            continue

        part = pd.DataFrame(
            {
                "__user_id": frame[schema["user"]],
                "__item_id": frame[schema["item"]],
                "__rating": frame[schema["rating"]] if schema["rating"] else 1.0,
                "__time": frame[schema["time"]] if schema["time"] else pd.NaT,
            }
        )
        part["__source_file"] = str(path)
        interaction_parts.append(part)
        interaction_paths.add(path)

    if not interaction_parts:
        raise ValueError(
            "No pude inferir columnas de usuario y destino. Use nombres como user_id y destination_id."
        )

    interactions = pd.concat(interaction_parts, ignore_index=True)
    best_schema = {
        "user": "__user_id",
        "item": "__item_id",
        "rating": "__rating",
        "time": "__time",
    }

    item_column = "__item_id"
    destinations = None
    for path, frame in frames.items():
        if path in interaction_paths:
            continue
        destination_key = _find_column(frame, ITEM_ALIASES)
        if destination_key is not None:
            destinations = frame.copy().rename(columns={destination_key: item_column})
            break

    return interactions, destinations, best_schema


def _prepare_interactions(
    interactions: pd.DataFrame,
    schema: dict[str, str],
    min_positive_rating: float | None,
) -> pd.DataFrame:
    frame = interactions.copy()
    frame = frame.dropna(subset=[schema["user"], schema["item"]])
    frame["rating_value"] = pd.to_numeric(frame[schema["rating"]], errors="coerce").fillna(1.0)
    threshold = min_positive_rating
    if threshold is None and frame["rating_value"].max() > 1:
        threshold = float(frame["rating_value"].median())
    if threshold is not None:
        frame = frame[frame["rating_value"] >= threshold]
    sort_columns = [schema["user"]]
    if schema.get("time") and schema["time"] in frame.columns:
        frame["_time"] = pd.to_datetime(frame[schema["time"]], errors="coerce")
        sort_columns.append("_time")
    return frame.sort_values(sort_columns).drop_duplicates([schema["user"], schema["item"]], keep="last")


def _build_item_metadata(
    interactions: pd.DataFrame,
    destinations: pd.DataFrame | None,
    item_column: str,
) -> pd.DataFrame:
    metadata = interactions.drop_duplicates(item_column).copy()
    if destinations is not None:
        metadata = metadata[[item_column]].merge(destinations, on=item_column, how="left")
    metadata = metadata.rename(columns={item_column: "item_key"})
    return metadata


def _build_item_features(metadata: pd.DataFrame, max_categories: int) -> torch.Tensor:
    frame = metadata.copy()
    candidate_columns = [
        column for column in frame.columns
        if column != "item_key" and not column.lower().endswith("id")
    ]
    if not candidate_columns:
        return torch.zeros((len(frame), 0), dtype=torch.float32)

    feature_parts: list[np.ndarray] = []
    for column in candidate_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy(dtype=np.float32)
            std = float(values.std()) or 1.0
            feature_parts.append(((values - float(values.mean())) / std).reshape(-1, 1))
            continue
        normalized = series.fillna("unknown").astype(str).map(_normalize_text)
        top_values = normalized.value_counts().head(max_categories).index
        for value in top_values:
            feature_parts.append((normalized == value).astype(np.float32).to_numpy().reshape(-1, 1))

    if not feature_parts:
        return torch.zeros((len(frame), 0), dtype=torch.float32)
    return torch.tensor(np.concatenate(feature_parts, axis=1), dtype=torch.float32)


def _negative_sample(
    records: list[InteractionRecord],
    train_history: dict[int, set[int]],
    num_items: int,
    negatives_per_positive: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    rng = np.random.default_rng(seed)
    samples = [(record.user_idx, record.item_idx, 1.0) for record in records]
    all_items = np.arange(num_items)
    for record in records:
        positives = train_history.get(record.user_idx, set())
        candidates = np.setdiff1d(all_items, np.fromiter(positives, dtype=int), assume_unique=False)
        if len(candidates) == 0:
            continue
        size = min(negatives_per_positive, len(candidates))
        for item_idx in rng.choice(candidates, size=size, replace=False).tolist():
            samples.append((record.user_idx, int(item_idx), 0.0))
    rng.shuffle(samples)
    return samples


def _split_by_user(
    records: list[InteractionRecord],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[InteractionRecord], list[InteractionRecord], list[InteractionRecord]]:
    rng = np.random.default_rng(seed)
    by_user: dict[int, list[InteractionRecord]] = {}
    for record in records:
        by_user.setdefault(record.user_idx, []).append(record)

    train: list[InteractionRecord] = []
    val: list[InteractionRecord] = []
    test: list[InteractionRecord] = []
    for user_records in by_user.values():
        user_records = list(user_records)
        rng.shuffle(user_records)
        if len(user_records) < 3:
            train.extend(user_records)
            continue
        test_count = max(1, int(len(user_records) * test_split))
        val_count = max(1, int(len(user_records) * val_split))
        test.extend(user_records[:test_count])
        val.extend(user_records[test_count:test_count + val_count])
        train.extend(user_records[test_count + val_count:])
    if not train:
        raise ValueError("Dataset insuficiente para crear entrenamiento.")
    return train, val, test


def _history(records: list[InteractionRecord]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for record in records:
        result.setdefault(record.user_idx, set()).add(record.item_idx)
    return result


def _popularity(records: list[InteractionRecord], num_items: int) -> np.ndarray:
    counts = np.zeros(num_items, dtype=np.float32)
    for record in records:
        counts[record.item_idx] += 1.0
    if counts.max() > 0:
        counts = counts / counts.max()
    return counts


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize_name(column): column for column in frame.columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()) or "unknown"
