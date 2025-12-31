import os
import joblib
import numpy as np
import pandas as pd
import time
import logging

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

# ======================================================
# CONFIGURAÇÃO DE LOG
# ======================================================
log = logging.getLogger(__name__)

# ======================================================
# CALLBACK DE LOG POR EPOCH (MLOps friendly)
# ======================================================
class TrainingLogger(Callback):
    """
    Callback personalizado para logar métricas a cada epoch.
    Não interfere no treinamento nem na API.
    """

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()
        log.info("Treinamento iniciado.")

    def on_epoch_end(self, epoch, logs=None):
        log.info(
            "Epoch %d | loss=%.6f | mae=%.6f | val_loss=%.6f | val_mae=%.6f",
            epoch + 1,
            logs.get("loss"),
            logs.get("mae"),
            logs.get("val_loss"),
            logs.get("val_mae"),
        )

    def on_train_end(self, logs=None):
        elapsed = time.perf_counter() - self.start_time
        log.info("Treinamento finalizado em %.2f segundos.", elapsed)


# ======================================================
# FUNÇÃO AUXILIAR – CRIA JANELAS TEMPORAIS
# ======================================================
def create_sequences(values, window):
    """
    Converte uma série temporal em pares (X, y) para LSTM.
    """
    X, y = [], []

    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Formato exigido pela LSTM: (amostras, janela, features)
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
    """
    Treina um modelo LSTM, loga métricas detalhadas
    e salva o modelo e o scaler.
    """
    total_start = time.perf_counter()
    log.info("Iniciando pipeline de treinamento para %s", symbol)

    # --------------------------------------------------
    # 1. Localizar coluna Close
    # --------------------------------------------------
    close_col = next((c for c in df.columns if "Close" in str(c)), None)
    if close_col is None:
        raise ValueError("Coluna 'Close' não encontrada.")

    close = df[[close_col]].values.astype("float32")
    log.info("Total de registros utilizados: %d", len(close))

    # --------------------------------------------------
    # 2. Normalização
    # --------------------------------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    if len(scaled) <= window:
        raise ValueError(f"Dados insuficientes para janela {window}")

    # --------------------------------------------------
    # 3. Criação das sequências
    # --------------------------------------------------
    X, y = create_sequences(scaled, window)
    log.info("Sequências criadas | X=%s | y=%s", X.shape, y.shape)

    # --------------------------------------------------
    # 4. Split treino / validação
    # --------------------------------------------------
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    log.info(
        "Split de dados | treino=%d | validação=%d",
        len(X_train), len(X_val)
    )

    # --------------------------------------------------
    # 5. Definição do modelo LSTM
    # --------------------------------------------------
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

    log.info("Modelo LSTM compilado com sucesso.")

    # --------------------------------------------------
    # 6. Treinamento (COM LOGS POR EPOCH)
    # --------------------------------------------------
    train_start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[TrainingLogger()],
        verbose=0  # logs ficam padronizados no logging
    )

    train_time = time.perf_counter() - train_start

    # --------------------------------------------------
    # 7. Métricas finais
    # --------------------------------------------------
    val_loss = history.history["val_loss"][-1]
    val_mae = history.history["val_mae"][-1]

    log.info(
        "Resumo final | epochs=%d | val_mae=%.6f | val_mse=%.6f | tempo=%.2fs",
        epochs,
        val_mae,
        val_loss,
        train_time
    )

    # --------------------------------------------------
    # 8. Salvamento do modelo e scaler
    # --------------------------------------------------
    os.makedirs("app/models", exist_ok=True)

    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    total_time = time.perf_counter() - total_start

    log.info(
        "Pipeline completo finalizado para %s em %.2f segundos.",
        symbol,
        total_time
    )

    # --------------------------------------------------
    # 9. Retorno (API permanece compatível)
    # --------------------------------------------------
    return {
        "model_path": model_path,
        "val_mae": float(val_mae),
        "val_loss": float(val_loss),
        "epochs": epochs,
        "train_time_seconds": round(train_time, 2),
        "total_time_seconds": round(total_time, 2)
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

    start = time.perf_counter()

    for _ in range(horizon):
        p = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(p)
        new_val = np.array(p).reshape(1, 1, 1)
        current_seq = np.concatenate(
            [current_seq[:, 1:, :], new_val],
            axis=1
        )

    elapsed = time.perf_counter() - start
    log.info(
        "Inferência concluída | symbol=%s | horizon=%d | tempo=%.4fs",
        symbol, horizon, elapsed
    )

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1).tolist()

    return preds_inv
