
from pathlib import Path

from PIL import Image

from src.module2_distraction.data_loader import (
    DistractionDataConfig,
    create_data_loaders,
)
from src.module3_recommender.data_loader import (
    RecommenderDataConfig,
    create_data_loaders as create_recommender_loaders,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(120, 40, 80)).save(path)


def test_distraction_data_loader_splits_imagefolder(tmp_path):
    for class_name in ["safe", "phone"]:
        for index in range(5):
            _write_image(tmp_path / class_name / f"{index}.jpg")

    data = create_data_loaders(
        DistractionDataConfig(
            data_dir=str(tmp_path),
            image_size=32,
            batch_size=2,
            val_split=0.2,
            test_split=0.2,
        )
    )

    assert data.class_names == ["phone", "safe"]
    assert len(data.train_loader.dataset) == 6
    assert len(data.val_loader.dataset) == 2
    assert len(data.test_loader.dataset) == 2


def test_recommender_data_loader_infers_schema(tmp_path):
    interactions = tmp_path / "interactions.csv"
    interactions.write_text(
        "\n".join(
            [
                "user_id,destination_id,rating,date",
                "u1,d1,5,2026-01-01",
                "u1,d2,4,2026-01-02",
                "u1,d3,5,2026-01-03",
                "u2,d2,5,2026-01-01",
                "u2,d3,4,2026-01-02",
                "u2,d4,5,2026-01-03",
            ]
        ),
        encoding="utf-8",
    )
    destinations = tmp_path / "destinations.csv"
    destinations.write_text(
        "\n".join(
            [
                "destination_id,city,category,budget",
                "d1,Bogota,Cultura,100",
                "d2,Medellin,Negocios,120",
                "d3,Cartagena,Playa,220",
                "d4,Cali,Cultura,90",
            ]
        ),
        encoding="utf-8",
    )

    data = create_recommender_loaders(
        RecommenderDataConfig(
            data_dir=str(tmp_path),
            batch_size=2,
            negative_samples=1,
        )
    )

    assert data.idx_to_user == ["u1", "u2"]
    assert len(data.idx_to_item) == 4
    assert data.item_features.shape[0] == 4
    assert len(data.train_loader.dataset) > 0
