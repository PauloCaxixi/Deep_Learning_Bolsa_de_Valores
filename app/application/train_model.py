# ======================================================
# IMPORTAÇÕES
# ======================================================

# Manipulação de caminhos e arquivos no sistema operacional
import os

# Serialização de objetos Python (usado para salvar o scaler)
import joblib

# Operações matemáticas e vetoriais
import numpy as np

# Manipulação de dados em DataFrame
import pandas as pd

# Normalização de dados para o intervalo [0, 1]
from sklearn.preprocessing import MinMaxScaler

# TensorFlow e Keras (deep learning)
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sistema de logs da aplicação
import logging

# Logger padrão do módulo
log = logging.getLogger(__name__)


# ======================================================
# FUNÇÃO AUXILIAR – CRIA JANELAS TEMPORAIS
# ======================================================
def create_sequences(values, window):
    """
    Transforma uma série temporal em janelas deslizantes
    para treinamento de redes LSTM.

    Exemplo:
    window = 3
    values = [1,2,3,4,5]

    X = [[1,2,3], [2,3,4]]
    y = [4,5]
    """

    # Listas que armazenarão as sequências de entrada (X)
    # e os valores alvo (y)
    X, y = [], []

    # Percorre os dados respeitando o tamanho da janela
    for i in range(window, len(values)):
        # Janela de dados passados
        X.append(values[i - window:i, 0])

        # Valor seguinte (o que o modelo deve prever)
        y.append(values[i, 0])

    # Converte listas em arrays NumPy
    X = np.array(X)
    y = np.array(y)

    # Redimensiona X para o formato esperado pela LSTM:
    # (amostras, passos de tempo, features)
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
    Treina um modelo LSTM com dados históricos
    e salva o modelo e o scaler em disco.
    """

    log.info("Iniciando treinamento LSTM para %s", symbol)

    # --------------------------------------------------
    # IDENTIFICA A COLUNA 'Close'
    # --------------------------------------------------
    # Procura qualquer coluna que contenha a palavra "Close"
    close_col = next((c for c in df.columns if "Close" in str(c)), None)

    if close_col is None:
        raise ValueError("Coluna 'Close' não encontrada.")

    # Extrai os preços de fechamento e converte para float32
    close = df[[close_col]].values.astype("float32")

    # --------------------------------------------------
    # NORMALIZAÇÃO DOS DADOS
    # --------------------------------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    # Verificação de tamanho mínimo
    if len(scaled) <= window:
        raise ValueError(f"Dados insuficientes para janela {window}")

    # Criação das sequências temporais
    X, y = create_sequences(scaled, window)

    # --------------------------------------------------
    # DIVISÃO TREINO / VALIDAÇÃO (80% / 20%)
    # --------------------------------------------------
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # --------------------------------------------------
    # DEFINIÇÃO DO MODELO LSTM
    # --------------------------------------------------
    # Uso explícito de Input para evitar warnings do Keras
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1], 1)),
        LSTM(50, activation="tanh"),
        Dense(1)
    ])

    # Compilação do modelo
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    # --------------------------------------------------
    # TREINAMENTO DO MODELO
    # --------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # --------------------------------------------------
    # EXTRAÇÃO DAS MÉTRICAS FINAIS
    # --------------------------------------------------
    val_loss = history.history["val_loss"][-1]
    val_mae = history.history["val_mae"][-1]

    # --------------------------------------------------
    # SALVAMENTO DO MODELO E SCALER
    # --------------------------------------------------
    os.makedirs("app/models", exist_ok=True)

    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    log.info(
        "Treinamento finalizado para %s | val_mae=%.6f | val_loss=%.6f",
        symbol, val_mae, val_loss
    )

    # Retorno das métricas e caminho do modelo
    return {
        "model_path": model_path,
        "val_mae": float(val_mae),
        "val_loss": float(val_loss)
    }


# ======================================================
# CARREGAMENTO DO MODELO
# ======================================================
def load_model_and_scaler(symbol: str):
    """
    Carrega o modelo LSTM treinado e o scaler correspondente.
    """

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
    """
    Realiza previsões futuras com base em uma série
    de preços de fechamento.
    """

    # Carrega modelo e scaler
    model, scaler = load_model_and_scaler(symbol)

    # Converte lista em array e normaliza
    arr = np.array(closes, dtype="float32").reshape(-1, 1)
    scaled = scaler.transform(arr)

    # Última janela usada como entrada inicial
    seq = scaled[-window:].reshape(1, window, 1)

    preds = []
    current_seq = seq.copy()

    # Previsão recursiva (multi-step)
    for _ in range(horizon):
        p = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(p)

        # Insere a nova previsão no fim da sequência
        new_val = np.array(p).reshape(1, 1, 1)
        current_seq = np.concatenate(
            [current_seq[:, 1:, :], new_val],
            axis=1
        )

    # Desnormaliza as previsões
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1).tolist()

    return preds_inv
