import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

log = logging.getLogger(__name__)

# ======================================================
# FUNÇÃO AUXILIAR – CRIA JANELAS TEMPORAIS
# ======================================================
def create_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


# ======================================================
# TREINAMENTO E SALVAMENTO DO MODELO
# ======================================================
def train_and_save(
    symbol: str,
    df: pd.DataFrame,
    window: int = 60,
    epochs: int = 10,
    batch_size: int = 32
):
    log.info("Iniciando treinamento LSTM para %s", symbol)

    # Localiza a coluna Close
    close_col = next((c for c in df.columns if "Close" in str(c)), None)
    if close_col is None:
        raise ValueError("Coluna 'Close' não encontrada.")

    close = df[[close_col]].values.astype("float32")

    # Normalização
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    if len(scaled) <= window:
        raise ValueError(f"Dados insuficientes para janela {window}")

    X, y = create_sequences(scaled, window)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # MODELO (forma correta, sem warning)
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1], 1)),
        LSTM(50, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    # TREINO
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 🔥 MÉTRICAS CORRETAS (ERA ISSO QUE FALTAVA)
    val_loss = history.history["val_loss"][-1]
    val_mae = history.history["val_mae"][-1]

    # SALVAMENTO
    os.makedirs("app/models", exist_ok=True)

    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    log.info(
        "Treinamento finalizado para %s | val_mae=%.6f | val_loss=%.6f",
        symbol, val_mae, val_loss
    )

    # ✅ RETORNO CORRETO (SEM NameError)
    return {
        "model_path": model_path,
        "val_mae": float(val_mae),
        "val_loss": float(val_loss)
    }


# ======================================================
# CARREGAMENTO DO MODELO
# ======================================================
def load_model_and_scaler(symbol: str):
    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado.")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ======================================================
# PREVISÃO (INFERÊNCIA)
# ======================================================
def forecast_from_series(
    closes: list[float],
    symbol: str,
    horizon: int = 1,
    window: int = 60
):
    model, scaler = load_model_and_scaler(symbol)

    arr = np.array(closes, dtype="float32").reshape(-1, 1)
    scaled = scaler.transform(arr)

    seq = scaled[-window:].reshape(1, window, 1)

    preds = []
    current_seq = seq.copy()

    for _ in range(horizon):
        p = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(p)
        new_val = np.array(p).reshape(1, 1, 1)
        current_seq = np.concatenate([current_seq[:, 1:, :], new_val], axis=1)

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1).tolist()

    return preds_inv
