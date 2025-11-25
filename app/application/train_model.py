# app/application/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# helper to prepare sequences
def create_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_and_save(symbol: str, df: pd.DataFrame, window: int=60, epochs:int=10, batch_size:int=32):
    """
    Trains an LSTM on df['Close'] and saves scaler and model.
    Returns metrics dict (mae, mse) on a small validation split.
    """
    # ensure Close available
    close_col = None
    for col in df.columns:
        if "Close" in str(col):
            close_col = col
            break
    if close_col is None:
        raise ValueError("Close column not found")

    close = df[[close_col]].values.astype("float32")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    # ensure we have at least window+1 rows
    if len(scaled) <= window:
        raise ValueError(f"Not enough data to train with window={window}. Have {len(scaled)} rows.")

    X, y = create_sequences(scaled, window)

    # train/validation split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # build model
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], 1), activation="tanh", return_sequences=False),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    # train
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    # evaluate
    loss, mae = model.evaluate(X_val, y_val, verbose=0)

    # save model & scaler
    os.makedirs("app/models", exist_ok=True)
    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")
    # prefer Keras native format
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return {"model_path": model_path, "scaler_path": scaler_path, "mae": float(mae), "mse": float(loss)}

def load_model_and_scaler(symbol: str):
    model_path = os.path.join("app", "models", f"{symbol}_lstm.keras")
    scaler_path = os.path.join("app", "models", f"{symbol}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def forecast_from_series(closes: list[float], symbol: str, horizon: int = 1, window: int = 60):
    """
    Given a list of close floats (most recent last), predict next `horizon` days.
    This function will:
      - load model & scaler (if available)
      - scale input using scaler
      - create sliding window from closes
      - predict horizon steps iteratively (rolling)
    Returns a list of predicted floats (de-scaled).
    """
    # ensure we have enough data (we will let caller decide to reduce window)
    model, scaler = load_model_and_scaler(symbol)

    # prepare numpy array for scaling
    arr = np.array(closes).reshape(-1,1).astype("float32")
    scaled_all = scaler.transform(arr)

    # if scaled_all length < window, the caller should have reduced window accordingly
    if len(scaled_all) < window:
        raise ValueError("scaled array shorter than window")

    # start sequence = last `window` values
    seq = scaled_all[-window:].reshape(1, window, 1)

    preds = []
    current_seq = seq.copy()
    for _ in range(horizon):
        p = model.predict(current_seq, verbose=0)[0,0]
        preds.append(p)
        # roll: append p, drop first
        current_seq = np.concatenate([current_seq[:,1:,:], np.array(p).reshape(1,1,1)], axis=1)

    # inverse transform preds
    preds = np.array(preds).reshape(-1,1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1).tolist()
    return preds_inv
