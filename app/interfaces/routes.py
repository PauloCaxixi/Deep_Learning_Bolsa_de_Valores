# app/interfaces/routes.py
from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from time import perf_counter
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ..application.get_data import GetData
from ..application.train_model import train_and_save, forecast_from_series, load_model_and_scaler

import os
import pandas as pd
import yfinance as yf  # used for fallback fetch

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# ---- prometheus metrics simples ----
REQUEST_COUNT = Counter("app_requests_total", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["method", "endpoint"])

# ---- models para requests ----
class TrainIn(BaseModel):
    symbol: str
    start: str = "2018-01-01"
    end: str = "2024-07-20"
    window: int = 60
    epochs: int = 20
    batch_size: int = 32

class PredictIn(BaseModel):
    symbol: str
    closes: list[float] = None  # se fornecido, será usado para prever
    horizon: int = 1
    window: int = 60

# ---- helper para extrair coluna Close de forma robusta ----
def extract_close_series(df: pd.DataFrame) -> pd.Series | None:
    # 1) direct 'Close' series
    if "Close" in df.columns and isinstance(df["Close"], pd.Series):
        return df["Close"]

    # 2) MultiIndex columns: try first level == 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if str(col[0]) == "Close":
                series = df[col]
                if isinstance(series, pd.Series):
                    return series

        # if df['Close'] is a sub-DataFrame, pick first numeric
        if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
            subdf = df["Close"]
            for c in subdf.columns:
                if pd.api.types.is_numeric_dtype(subdf[c]):
                    return subdf[c]

    # 3) fallback: any column name containing 'Close'
    for col in df.columns:
        if "Close" in str(col):
            series = df[col]
            if isinstance(series, pd.Series):
                return series

    return None

# ---- simple HTML UI ----
@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---- home example returning data ----
@router.get("/home")
async def read_home():
    return {"ok": True, "info": "Use /v1/train to train and /v1/predict to predict."}

# ---- small debug endpoint: return how many closes exist for symbol ----
@router.get("/data-count")
async def data_count(symbol: str):
    info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
    df = GetData(info).QueryDf()
    if df is None:
        raise HTTPException(status_code=404, detail="No data")
    closes_series = extract_close_series(df)
    if closes_series is None:
        raise HTTPException(status_code=500, detail="No Close column")
    return {"symbol": symbol, "rows": len(df), "closes": len(closes_series)}

# ---- train endpoint ----
@router.post("/train")
async def train(payload: TrainIn):
    method = "POST"; endpoint = "/v1/train"
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()
    try:
        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(payload.symbol, payload.start, payload.end)
        df = GetData(info).QueryDf()
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data for symbol / period")

        result = train_and_save(payload.symbol, df, window=payload.window, epochs=payload.epochs, batch_size=payload.batch_size)
        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        return JSONResponse({"status": "trained", "metrics": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- predict endpoint ----
@router.post("/predict")
async def predict(payload: PredictIn):
    method = "POST"; endpoint = "/v1/predict"
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()
    try:
        # 1) prepare closes (either from payload or fetch)
        if payload.closes is None:
            info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(payload.symbol, None, None)
            df = GetData(info).QueryDf()
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No data for symbol")

            closes_series = extract_close_series(df)
            if closes_series is None:
                # fallback: try yfinance directly with a period request
                df2 = yf.download(payload.symbol, period=f"{max(payload.window*2,180)}d", auto_adjust=True)
                if df2 is None or df2.empty:
                    raise HTTPException(status_code=500, detail="Could not retrieve closes")
                closes_series = extract_close_series(df2)
                if closes_series is None:
                    raise HTTPException(status_code=500, detail="Could not find Close column in fetched data")

            closes = closes_series.astype(float).tolist()
        else:
            closes = payload.closes

        # 2) final length check
        if len(closes) < payload.window:
            raise HTTPException(status_code=400, detail=f"Need at least {payload.window} historical closes to predict, got {len(closes)}")

        # 3) forecast
        preds = forecast_from_series(closes, payload.symbol, horizon=payload.horizon, window=payload.window)
        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        return {"symbol": payload.symbol, "predictions": preds}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Train first.")
    except HTTPException:
        raise
    except Exception as e:
        # return clearer error to client and log in uvicorn
        raise HTTPException(status_code=500, detail=str(e))

# ---- prometheus metrics endpoint ----
@router.get("/metrics")
async def metrics():
    data = generate_latest()
    return HTMLResponse(content=data, media_type=CONTENT_TYPE_LATEST)

# ---- simple UI form handler (POST from HTML) ----
@router.post("/ui/predict", response_class=HTMLResponse)
async def ui_predict(request: Request, symbol: str = Form(...), horizon: int = Form(1), window: int = Form(60)):
    """
    UI handler that does NOT train inside the HTTP request.
    It checks if a model exists; if not, shows a friendly page instructing to call /v1/train first.
    """
    try:
        model_dir = "app/models"
        model_path = os.path.join(model_dir, f"{symbol}_lstm.keras")

        # If model not present, show an HTML instructing the user to call /v1/train via the docs button
        if not os.path.exists(model_path):
            # Render a page telling user to train first (link to docs/train)
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol})

        # If model exists, fetch closes and predict (same robust logic as /v1/predict)
        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
        df = GetData(info).QueryDf()
        closes_series = extract_close_series(df)
        if closes_series is None:
            # fallback to instructive page
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error": "Could not get 'Close' series from Yahoo Finance."})
        closes = closes_series.astype(float).tolist()
        if len(closes) < window:
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error": f"Not enough history ({len(closes)} rows). Please run /v1/train with a longer start date."})

        preds = forecast_from_series(closes, symbol, horizon=horizon, window=window)
        return templates.TemplateResponse("result.html", {"request": request, "symbol": symbol, "predictions": preds})
    except FileNotFoundError:
        return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error": "Model not found. Please train first."})
    except Exception as e:
        # show friendly error page (and log trace in server console)
        return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error": str(e)})


@router.get("/history")
async def history(symbol: str, limit: int = 60):
    """
    Return the last `limit` close prices for a symbol as JSON.
    Example: /v1/history?symbol=AAPL&limit=60
    """
    try:
        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
        df = GetData(info).QueryDf()
        if df is None or df.empty:
            return JSONResponse(status_code=404, content={"detail": "No data for symbol"})

        closes_series = extract_close_series(df)
        if closes_series is None:
            return JSONResponse(status_code=500, content={"detail": "Close column not found"})

        # df index may be Date index; ensure we output dates + closes
        # If DataFrame has a DatetimeIndex, reset index for dates
        series_df = df.reset_index()[["Date"]] if "Date" in df.reset_index().columns else None

        # construct list of {date, close}
        rows = []
        # use the df we fetched (reset_index to ensure Date is a column)
        df2 = df.reset_index()
        # find appropriate 'Close' column name
        close_col = None
        for col in df2.columns:
            if "Close" in str(col):
                close_col = col
                break
        if close_col is None:
            return JSONResponse(status_code=500, content={"detail":"Could not find Close column"})

        # take last `limit` rows
        tail = df2.tail(limit)
        for _, r in tail.iterrows():
            # Date may be pd.Timestamp -> convert to ISO date string
            d = r["Date"]
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            rows.append({"date": date_str, "close": float(r[close_col])})
        return {"symbol": symbol, "history": rows}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})