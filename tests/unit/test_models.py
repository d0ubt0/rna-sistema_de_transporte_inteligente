
import torch

from src.module1_demand.model import TransportLSTM
from src.module2_distraction.model import build_model
from src.module3_recommender.model import build_model as build_recommender_model


def test_transport_lstm_forward_shape():
    model = TransportLSTM(num_routes=5, num_climas=3)
    model.eval()

    with torch.no_grad():
        output = model(
            torch.randn(2, 30, 4),
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
        )

    assert output.shape == (2, 1)


def test_driver_distraction_model_forward_shape():
    model = build_model(num_classes=4, architecture="resnet18", pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 4)


def test_travel_recommender_forward_shape():
    model = build_recommender_model(
        num_users=3,
        num_items=5,
        content_dim=2,
        embedding_dim=8,
        hidden_dim=16,
    )
    output = model(
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
        torch.randn(2, 2),
    )

    assert output.shape == (2,)
