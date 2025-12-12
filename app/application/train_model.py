# app/application/train_model.py

import os
import joblib  # Usado para salvar e carregar o escalonador (Scaler)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Normaliza os dados entre 0 e 1
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

log = logging.getLogger(__name__)

# --- FUNÇÃO AUXILIAR: CRIAÇÃO DE SEQUÊNCIAS ---
def create_sequences(values, window):
    """
    Transforma uma série temporal em janelas deslizantes.
    Ex: Se a janela é 60, usa os dias 1-60 para prever o dia 61.
    """
    X, y = [], []
    for i in range(window, len(values)):
        # X: contém os dados da janela (passado)
        X.append(values[i-window:i, 0])
        # y: contém o valor alvo (o que queremos prever)
        y.append(values[i, 0])
    X = np.array(X)
    y = np.array(y)
    # Reshape necessário para o Keras: (amostras, passos de tempo, características)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


# --- FUNÇÃO: TREINAMENTO ---
def train_and_save(symbol: str, df: pd.DataFrame, window: int=60, epochs:int=10, batch_size:int=32):
    """Treina o modelo LSTM e salva os arquivos (.keras e .joblib)."""
    log.info("Iniciando treinamento da LSTM para %s", symbol)
    
    # Identifica a coluna de fechamento (Close) mesmo em formatos diferentes do yfinance
    close_col = next((col for col in df.columns if "Close" in str(col)), None)
    if close_col is None:
        raise ValueError("Coluna 'Close' não encontrada.")

    # Converte os dados para float32 (formato preferido pelo TensorFlow)
    close = df[[close_col]].values.astype("float32")

    # Normalização: LSTMs são sensíveis à escala dos dados. 
    # Transformamos os preços em valores entre 0 e 1.
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    # Verifica se há dados suficientes para criar ao menos uma janela
    if len(scaled) <= window:
        raise ValueError(f"Dados insuficientes para janela de {window}.")

    # Prepara os dados para o modelo
    X, y = create_sequences(scaled, window)

    # Divisão: 80% para treino e 20% para validação (teste)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Arquitetura do Modelo:
    # 1. Camada LSTM com 50 neurônios para aprender padrões temporais.
    # 2. Camada Dense (saída) para prever o valor único final.
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], 1), activation="tanh"),
        Dense(1)
    ])

    # Compilação: Adam é o otimizador e MSE (Erro Quadrático Médio) é a função de perda.
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # Treinamento real: o modelo tenta ajustar seus pesos aos dados.
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val), verbose=1) 

    # Salva o modelo treinado e o escalonador (essencial para inverter a normalização depois)
    os.makedirs("app/models", exist_ok=True)
    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    return {"model_path": model_path, "mae": float(mae), "mse": float(loss)}


# --- FUNÇÃO: CARREGAMENTO ---
def load_model_and_scaler(symbol: str):
    """Carrega o modelo e o escalonador salvos anteriormente."""
    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado.")
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --- FUNÇÃO: PREVISÃO (INFERÊNCIA) ---
def forecast_from_series(closes: list[float], symbol: str, horizon: int = 1, window: int = 60):
    """Usa o modelo carregado para prever 'horizon' dias no futuro."""
    model, scaler = load_model_and_scaler(symbol)

    # Prepara os dados recebidos para o formato que o modelo entende (escalonado)
    arr = np.array(closes).reshape(-1,1).astype("float32")
    scaled_all = scaler.transform(arr)

    # Pega os últimos dados (tamanho da janela) para começar a previsão
    seq = scaled_all[-window:].reshape(1, window, 1)

    preds = []
    current_seq = seq.copy()
    
    # Loop de Previsão Recursiva:
    # Prevemos o dia T+1, adicionamos esse valor na janela e prevemos o dia T+2.
    for i in range(horizon):
        p = model.predict(current_seq, verbose=0)[0,0]
        preds.append(p)
        # 'Roda' a janela: remove o primeiro elemento e adiciona a nova previsão no final
        new_val = np.array(p).reshape(1,1,1)
        current_seq = np.concatenate([current_seq[:,1:,:], new_val], axis=1)
    
    # Inverte a normalização para voltar aos preços reais (Ex: 0.85 -> R$ 34,50)
    preds = np.array(preds).reshape(-1,1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1).tolist()
    
    return preds_inv