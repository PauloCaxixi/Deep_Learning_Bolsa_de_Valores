# app/application/train_model.py
import os
import math
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# criar pasta models se não existir
os.makedirs("app/models", exist_ok=True)

def create_lstm_model(window: int):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def series_to_windows(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i, 0])
        y.append(series[i, 0])
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    return X[..., None], y  # X shape -> (N, window, 1)

def train_and_save(symbol: str, df, window=60, epochs=20, batch_size=32, model_dir="app/models"):
    """
    Treina LSTM para o Close da DataFrame df (index datetime e coluna 'Close').
    Salva modelo e scaler em model_dir com prefix symbol.
    Retorna dict com métricas e caminhos.
    """
    closes = df[["Close"]].astype("float32").values  # shape (T,1)

    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closes)

    # train/val/test split (temporal)
    T = len(scaled)
    test_size = int(0.1 * T)
    val_size = int(0.1 * T)
    train_size = T - val_size - test_size

    train = scaled[:train_size]
    val = scaled[train_size:train_size + val_size]
    test = scaled[train_size + val_size:]

    # construir janelas
    Xtr, ytr = series_to_windows(train, window)
    Xva, yva = series_to_windows(np.concatenate([train[-window:], val], axis=0), window)
    Xte, yte = series_to_windows(np.concatenate([train[-window:], val[-window:] if len(val) >= window else val, test], axis=0), window)

    model = create_lstm_model(window)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)

    # avaliar
    pred_te_scaled = model.predict(Xte, verbose=0).reshape(-1, 1)
    yte_scaled = yte.reshape(-1, 1)
    pred_te = scaler.inverse_transform(pred_te_scaled)
    yte_inv = scaler.inverse_transform(yte_scaled)

    mae = mean_absolute_error(yte_inv, pred_te)
    rmse = math.sqrt(mean_squared_error(yte_inv, pred_te))
    mape = float((np.abs((yte_inv - pred_te) / yte_inv).mean()) * 100)

    # salvar artefatos
    model_path = os.path.join(model_dir, f"{symbol}_lstm.keras")
    scaler_path = os.path.join(model_dir, f"{symbol}_scaler.joblib")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "train_size": len(Xtr),
        "val_size": len(Xva),
        "test_size": len(Xte)
    }

def load_model_and_scaler(symbol: str, model_dir="app/models"):
    model_path = os.path.join(model_dir, f"{symbol}_lstm.keras")
    scaler_path = os.path.join(model_dir, f"{symbol}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train first.")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def forecast_from_series(closes: list, symbol: str, horizon: int = 1, window: int = 60, model_dir="app/models"):
    """
    Autoregressive forecast: recebe list de closes (float), usa modelo salvo para prever 'horizon' passos.
    Expects at least 'window' values in closes.
    """
    model, scaler = load_model_and_scaler(symbol, model_dir=model_dir)
    arr = np.array(closes, dtype="float32").reshape(-1, 1)
    scaled = scaler.transform(arr)
    if len(scaled) < window:
        raise ValueError(f"Need at least {window} historical closes to predict")
    cur = scaled[-window:].reshape(1, window, 1).astype("float32")
    preds = []
    for _ in range(horizon):
        p = model.predict(cur, verbose=0)[0, 0]
        preds.append(p)
        # shift
        cur = np.concatenate([cur[:, 1:, :], np.array([[[p]]], dtype="float32")], axis=1)
    preds_inv = scaler.inverse_transform(np.array(preds, dtype="float32").reshape(-1, 1)).ravel().tolist()
    return preds_inv
