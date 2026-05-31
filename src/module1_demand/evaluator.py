# type: ignore

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def denormalize(values, target_scaler):
    pas_min = target_scaler.data_min_[0]
    pas_max = target_scaler.data_max_[0]
    return values * (pas_max - pas_min) + pas_min


def compute_metrics_by_route(
    all_preds,
    all_reals,
    all_routes,
    route_encoder,
    target_scaler=None,
):
    if target_scaler is not None:
        reals_real = denormalize(all_reals, target_scaler)
        preds_real = denormalize(all_preds, target_scaler)
    else:
        reals_real = all_reals
        preds_real = all_preds

    resultados = []

    for route_id in sorted(np.unique(all_routes)):
        mask = all_routes == route_id
        y_r = reals_real[mask]
        y_p = preds_real[mask]

        rmse = np.sqrt(mean_squared_error(y_r, y_p))
        mae = mean_absolute_error(y_r, y_p)
        mape = np.mean(np.abs((y_r - y_p) / (y_r + 1e-8))) * 100

        nombre_ruta = route_encoder.inverse_transform([route_id])[0]

        resultados.append({
            "ruta": nombre_ruta,
            "route_id": route_id,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "y_real": y_r,
            "y_pred": y_p,
        })

    return resultados


def compute_global_metrics(reals_real, preds_real):
    rmse = np.sqrt(mean_squared_error(reals_real, preds_real))
    mae = mean_absolute_error(reals_real, preds_real)
    mape = np.mean(
        np.abs((reals_real - preds_real) / (reals_real + 1e-8))
    ) * 100
    return rmse, mae, mape


def print_metrics_table(resultados, reals_real, preds_real):
    print("\n" + "=" * 60)
    print("      MÉTRICAS DE EVALUACIÓN POR RUTA")
    print("=" * 60)

    print(
        f"{'Ruta':<12}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'MAPE (%)':>14}"
    )
    print("-" * 60)

    for r in resultados:
        print(
            f"{r['ruta']:<12}"
            f"{r['rmse']:>12.2f}"
            f"{r['mae']:>12.2f}"
            f"{r['mape']:>14.2f}%"
        )

    rmse_g, mae_g, mape_g = compute_global_metrics(reals_real, preds_real)

    print("-" * 60)
    print(
        f"{'GLOBAL':<12}"
        f"{rmse_g:>12.2f}"
        f"{mae_g:>12.2f}"
        f"{mape_g:>14.2f}%"
    )
    print("=" * 60)

    return rmse_g, mae_g, mape_g


def plot_predictions(resultados, save_path=None):
    import matplotlib.pyplot as plt
    num_rutas = len(resultados)

    fig, axes = plt.subplots(
        num_rutas,
        1,
        figsize=(14, 4 * num_rutas),
    )

    if num_rutas == 1:
        axes = [axes]

    for ax, r in zip(axes, resultados): #type: ignore
        n_puntos = len(r["y_real"])
        x = np.arange(n_puntos)
        ax.plot(x, r["y_real"], label="Real", linewidth=1.2, alpha=0.9)#type: ignore
        ax.plot(
            x,
            r["y_pred"],
            label="Predicción",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )
        ax.fill_between(x, r["y_real"], r["y_pred"], alpha=0.12)

        ax.set_title(
            f"{r['ruta']}  —  "
            f"RMSE: {r['rmse']:.1f} | "
            f"MAE: {r['mae']:.1f} | "
            f"MAPE: {r['mape']:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Muestra futura")
        ax.set_ylabel("Pasajeros")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.suptitle(
        "Predicción vs Demanda Real por Ruta",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"\n✓ Gráfica guardada: {save_path}")


def plot_metrics_comparison(df_metricas, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metricas = ["rmse", "mae", "mape"]
    titulos = [
        "RMSE (pasajeros)",
        "MAE (pasajeros)",
        "MAPE (%)",
    ]

    for ax, metrica, titulo in zip(axes, metricas, titulos):
        valores = df_metricas[metrica].values
        rutas = df_metricas["ruta"].values

        bars = ax.bar(rutas, valores)
        media = valores.mean()

        ax.axhline(
            media,
            linestyle="--",
            linewidth=1.2,
            label=f"Media: {media:.1f}",
        )

        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + media * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Ruta")
        ax.set_ylabel(titulo)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Comparativa de Métricas por Ruta",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"✓ Gráfica guardada: {save_path}")


def plot_heatmap_error(resultados, save_path=None, n_muestras=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = min(
        n_muestras,
        min(len(r["y_real"]) for r in resultados),
    )

    error_matrix = np.array([
        np.abs(r["y_real"][:n] - r["y_pred"][:n])
        for r in resultados
    ])

    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(
        error_matrix,
        ax=ax,
        cmap="YlOrRd",
        yticklabels=[r["ruta"] for r in resultados],
        xticklabels=False,
        cbar_kws={
            "label": "Error Absoluto (pasajeros)",
        },
    )

    ax.set_title(
        "Heatmap de Error Absoluto "
        "— Primeras 100 Muestras Futuras",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Muestra futura")
    ax.set_ylabel("Ruta")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"✓ Gráfica guardada: {save_path}")


def build_summary_df(resultados):
    df_metricas = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_real", "y_pred")}
        for r in resultados
    ])

    print("\nTabla resumen:")
    print(df_metricas.to_string(index=False))

    return df_metricas


def plot_learning_curve(history, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train", linewidth=1.5)
    plt.plot(history["val_loss"], label="Validation", linewidth=1.5)
    plt.axvline(
        len(history["train_loss"]) - 1,
        color="gray", ls=":", alpha=0.5,
        label=f"Stop epoch {len(history['train_loss'])}"
    )
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.show()

    if save_path:
        print(f"✓ Curva de aprendizaje guardada: {save_path}")


def run_evaluation(
    model,
    test_loader,
    device,
    target_scaler,
    route_encoder,
    history=None,
    output_dir="models/demand",
):
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Inference ---
    model.eval()
    all_preds = []
    all_reals = []
    all_routes = []

    with torch.no_grad():
        for X_batch, routes_batch, climas_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            routes_batch = routes_batch.to(device)
            climas_batch = climas_batch.to(device)

            outputs = model(X_batch, routes_batch, climas_batch)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_reals.extend(y_batch.numpy().flatten())
            all_routes.extend(routes_batch.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_reals = np.array(all_reals)
    all_routes = np.array(all_routes)

    # --- 2. Denormalize ---
    reals_real = denormalize(all_reals, target_scaler)
    preds_real = denormalize(all_preds, target_scaler)

    # --- 3. Metrics per route ---
    resultados = compute_metrics_by_route(
        all_preds, all_reals, all_routes, route_encoder,
        target_scaler=target_scaler,
    )

    # --- 4. Global metrics ---
    rmse_g, mae_g, mape_g = print_metrics_table(resultados, reals_real, preds_real)

    # --- 5. Summary DataFrame ---
    df_metricas = build_summary_df(resultados)

    # --- 6. Save metrics JSON ---
    metrics_dict = {
        "global": {
            "rmse": round(float(rmse_g), 2),
            "mae": round(float(mae_g), 2),
            "mape": round(float(mape_g), 2),
        },
        "por_ruta": [
            {
                "ruta": r["ruta"],
                "route_id": int(r["route_id"]),
                "rmse": round(float(r["rmse"]), 2),
                "mae": round(float(r["mae"]), 2),
                "mape": round(float(r["mape"]), 2),
            }
            for r in resultados
        ],
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n✓ Métricas guardadas: {metrics_path}")

    # --- 8. Save summary CSV ---
    csv_path = os.path.join(output_dir, "metrics_por_ruta.csv")
    df_metricas.to_csv(csv_path, index=False)
    print(f"✓ CSV guardado: {csv_path}")

    # --- 9. Save per-route predictions CSV ---
    rows = []
    for r in resultados:
        for i in range(len(r["y_real"])):
            rows.append({
                "ruta": r["ruta"],
                "route_id": int(r["route_id"]),
                "muestra": i,
                "real": round(float(r["y_real"][i]), 2),
                "prediccion": round(float(r["y_pred"][i]), 2),
                "error_abs": round(float(abs(r["y_real"][i] - r["y_pred"][i])), 2),
            })
    preds_df = pd.DataFrame(rows)
    preds_csv = os.path.join(output_dir, "predicciones_detalle.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"✓ Predicciones detalladas guardadas: {preds_csv}")

    # --- 10. Save plots ---
    plot_predictions(
        resultados,
        save_path=os.path.join(output_dir, "prediccion_vs_real_por_ruta.png"),
    )
    plot_metrics_comparison(
        df_metricas,
        save_path=os.path.join(output_dir, "comparativa_metricas_por_ruta.png"),
    )
    plot_heatmap_error(
        resultados,
        save_path=os.path.join(output_dir, "heatmap_error_por_ruta.png"),
    )

    # --- 12. Learning curve (if history provided) ---
    if history is not None:
        plot_learning_curve(
            history,
            save_path=os.path.join(output_dir, "curva_aprendizaje.png"),
        )
        history_csv = os.path.join(output_dir, "training_history.csv")
        pd.DataFrame(history).to_csv(history_csv, index=False)
        print(f"✓ Historial de entrenamiento guardado: {history_csv}")

    print("\n" + "=" * 60)
    print("      EVALUACIÓN COMPLETA — TODOS LOS ARTEFACTOS GUARDADOS")
    print("=" * 60)

    return {
        "resultados": resultados,
        "df_metricas": df_metricas,
        "reals_real": reals_real,
        "preds_real": preds_real,
        "global_metrics": metrics_dict["global"],
    }
