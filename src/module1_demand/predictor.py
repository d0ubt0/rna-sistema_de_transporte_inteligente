import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def build_future_test_sequences(
    train_df,
    test_df,
    feature_cols,
    seq_length,
):
    X, routes, climas, y = [], [], [], []

    for route_id in train_df["route_id"].unique():

        train_route = train_df[
            train_df["route_id"] == route_id
        ].sort_values("fecha")

        test_route = test_df[
            test_df["route_id"] == route_id
        ].sort_values("fecha")

        if len(test_route) == 0:
            continue

        combined = pd.concat(
            [train_route, test_route],
            ignore_index=True,
        )

        feats = combined[feature_cols].values
        clima_ids = combined["clima_id"].values
        targets = combined["pasajeros"].values

        split_idx = len(train_route)

        for i in range(split_idx, len(combined)):
            start = i - seq_length

            if start < 0:
                continue

            X.append(feats[start:i])
            routes.append(route_id)
            climas.append(clima_ids[i])
            y.append(targets[i])

    return (
        np.array(X),
        np.array(routes),
        np.array(climas),
        np.array(y),
    )


def predict(model, test_loader, device):
    all_preds = []
    all_reals = []
    all_routes = []

    model.eval()

    with torch.no_grad():
        for X_batch, routes_batch, climas_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            routes_batch = routes_batch.to(device)
            climas_batch = climas_batch.to(device)

            outputs = model(X_batch, routes_batch, climas_batch)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_reals.extend(y_batch.numpy().flatten())
            all_routes.extend(routes_batch.cpu().numpy().flatten())

    return (
        np.array(all_preds),
        np.array(all_reals),
        np.array(all_routes),
    )


def forecast_autoregressive(
    model,
    sequence,
    route_id,
    future_features,
    future_clima_ids,
    device,
):
    """
    Pronostica varios pasos usando cada predicción como parte del
    siguiente contexto temporal.

    sequence:
        Array (seq_length, 4) normalizado con
        [dia_semana, mes, festivo, pasajeros].
    future_features:
        Array (steps, 3) normalizado con
        [dia_semana, mes, festivo].
    future_clima_ids:
        Array/list (steps,) con el clima esperado de cada día futuro.
    """

    seq = np.asarray(sequence, dtype=np.float32).copy()
    future_features = np.asarray(future_features, dtype=np.float32)
    future_clima_ids = np.asarray(future_clima_ids, dtype=np.int64)

    if seq.ndim != 2 or seq.shape[1] != 4:
        raise ValueError("sequence debe tener forma (seq_length, 4)")

    if future_features.ndim != 2 or future_features.shape[1] != 3:
        raise ValueError("future_features debe tener forma (steps, 3)")

    if len(future_features) != len(future_clima_ids):
        raise ValueError("future_features y future_clima_ids deben tener la misma longitud")

    predictions = []
    model.eval()

    with torch.no_grad():
        for step_features, clima_id in zip(future_features, future_clima_ids):
            seq_tensor = torch.tensor(seq[None, :, :], device=device)
            route_tensor = torch.tensor([route_id], dtype=torch.long, device=device)
            clima_tensor = torch.tensor([clima_id], dtype=torch.long, device=device)

            pred_norm = model(seq_tensor, route_tensor, clima_tensor)
            pred_norm = float(pred_norm.cpu().numpy().flatten()[0])
            predictions.append(pred_norm)

            new_row = np.array(
                [step_features[0], step_features[1], step_features[2], pred_norm],
                dtype=np.float32,
            )
            seq = np.concatenate([seq[1:], new_row[None, :]], axis=0)

    return np.array(predictions, dtype=np.float32)
