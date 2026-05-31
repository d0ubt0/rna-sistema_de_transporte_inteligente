import torch
from data_generator import generate_data
from data_loader import TransportDataset, build_sequences, create_dataloader
from evaluator import run_evaluation
from model import TransportLSTM
from predictor import build_future_test_sequences
from preprocessor import preprocess_pipeline
from trainer import train_model

DATA_PATH = "data/demanda_transporte.csv"
SEQ_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 80
LR = 0.0007
OUTPUT_DIR = "models/demand"

if __name__ == "__main__":
    print("Generando datos sintéticos...")
    generate_data(DATA_PATH, web_output_csv="web/public/data/demanda_transporte.csv")

    print("Preprocesando...")
    pipe = preprocess_pipeline(DATA_PATH, save=True, max_year=2026, output_dir=OUTPUT_DIR)

    print("Construyendo secuencias de entrenamiento...")
    X_train, routes_train, climas_train, y_train = build_sequences(
        pipe["train_df"], pipe["feature_cols"], seq_length=SEQ_LENGTH
    )
    X_val, routes_val, climas_val, y_val = build_sequences(
        pipe["test_df"], pipe["feature_cols"], seq_length=SEQ_LENGTH
    )

    train_loader = create_dataloader(
        TransportDataset(X_train, routes_train, climas_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = create_dataloader(
        TransportDataset(X_val, routes_val, climas_val, y_val),
        batch_size=BATCH_SIZE,
    )

    model = TransportLSTM(
        num_routes=len(pipe["route_encoder"].classes_),
        num_climas=len(pipe["clima_encoder"].classes_),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    model = model.to(device)

    print("Entrenando modelo...")
    history = train_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR,
        output_dir=OUTPUT_DIR,
    )

    print("\nCargando mejor modelo para evaluación...")
    model.load_state_dict(
        torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=device, weights_only=True)
    )

    print("Construyendo secuencias de test (forecasting futuro real)...")
    X_test, routes_test, climas_test, y_test = build_future_test_sequences(
        pipe["train_df"], pipe["test_df"],
        pipe["feature_cols"], seq_length=SEQ_LENGTH,
    )
    test_loader = create_dataloader(
        TransportDataset(X_test, routes_test, climas_test, y_test),
        batch_size=BATCH_SIZE,
    )

    print("Evaluando modelo...\n")
    run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        target_scaler=pipe["target_scaler"],
        route_encoder=pipe["route_encoder"],
        history=history,
        output_dir=OUTPUT_DIR,
    )

    print("\n✓ Entrenamiento y evaluación completados exitosamente.")
