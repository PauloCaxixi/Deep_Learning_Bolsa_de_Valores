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
import yfinance as yf
import logging # ADICIONADO

log = logging.getLogger(__name__) # Logger para o módulo

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# MÉTICAS PROMETHEUS
REQUEST_COUNT = Counter("app_requests_total", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["method", "endpoint"])

class TrainIn(BaseModel):
    symbol: str
    start: str = "2018-01-01"
    end: str = "2025-12-09"
    window: int = 60
    epochs: int = 10
    batch_size: int = 32

class PredictIn(BaseModel):
    symbol: str
    closes: list[float] = None
    horizon: int = 1
    window: int = 60

# helper to extract Close
def extract_close_series(df: pd.DataFrame) -> pd.Series | None:
    if "Close" in df.columns and isinstance(df["Close"], pd.Series):
        return df["Close"]
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if str(col[0]) == "Close":
                return df[col]
    for col in df.columns:
        if "Close" in str(col):
            return df[col]
    return None

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # CORREÇÃO: As variáveis 'logo_path' e 'logo_exists' precisam ser definidas
    # antes de serem usadas no TemplateResponse.
    logo_path = "/mnt/data/5ecf1705-570e-4634-bf3e-8feff82b4532.png"
    
    # 1. Definição da variável faltante. 
    # Mantenha como 'False' ou use 'os.path.exists(logo_path)' se o arquivo realmente existir.
    logo_exists = os.path.exists(logo_path)
    
    return templates.TemplateResponse("index.html", {"request": request, "logo_path": logo_path if logo_exists else None})

@router.get("/data-count")
async def data_count(symbol: str):
    info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
    df = GetData(info).QueryDf()
    if df is None or df.empty:
        log.warning("Nenhum dado encontrado para /data-count para %s.", symbol)
        raise HTTPException(status_code=404, detail="No data")
    series = extract_close_series(df)
    if series is None:
        log.error("Coluna 'Close' não encontrada para %s após consulta.", symbol)
        raise HTTPException(status_code=500, detail="No Close column")
    log.info("Consulta /data-count bem-sucedida para %s. %d linhas.", symbol, len(df))
    return {"symbol": symbol, "rows": len(df), "closes": len(series)}

@router.post("/train")
async def train(payload: TrainIn):
    method = "POST"; endpoint = "/v1/train"
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()
    log.info("Processando requisição de treino para o símbolo: %s", payload.symbol) 
    
    try:
        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(payload.symbol, payload.start, payload.end)
        
        query_start = perf_counter()
        df = GetData(info).QueryDf()
        query_time = perf_counter() - query_start
        
        log.info("Consulta de dados concluída em %.4f segundos. Linhas obtidas: %d", query_time, len(df) if df is not None else 0)

        if df is None or df.empty:
            log.warning("Nenhum dado encontrado para %s no período especificado.", payload.symbol)
            raise HTTPException(status_code=404, detail="No data for symbol/period")
        
        train_start = perf_counter()
        result = train_and_save(payload.symbol, df, window=payload.window, epochs=payload.epochs, batch_size=payload.batch_size)
        train_time = perf_counter() - train_start
        
        log.info("Treinamento concluído para %s em %.4f segundos. MAE: %.4f", payload.symbol, train_time, result.get("mae", -1))

        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        log.info("Requisição /train finalizada com sucesso em %.4f segundos.", elapsed) 
        return JSONResponse({"status": "trained", "metrics": result})
    except Exception as e:
        log.error("Erro ao processar /train para %s: %s", payload.symbol, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(payload: PredictIn):
    method = "POST"; endpoint = "/v1/predict"
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()
    log.info("Processando requisição de previsão para o símbolo: %s", payload.symbol)
    
    try:
        if payload.closes is None:
            log.info("Nenhum 'closes' fornecido, consultando Yahoo Finance.")
            info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(payload.symbol, None, None)
            df = GetData(info).QueryDf()
            
            series = extract_close_series(df)
            if series is None:
                log.warning("Tentativa secundária de consulta YF devido à coluna 'Close' ausente.")
                df2 = yf.download(payload.symbol, period=f"{max(payload.window*4,360)}d", auto_adjust=True)
                if df2 is None or df2.empty:
                    raise HTTPException(status_code=500, detail="Could not retrieve closes")
                series = extract_close_series(df2)
                if series is None:
                    raise HTTPException(status_code=500, detail="Could not find Close column")
            closes = series.astype(float).tolist()
        else:
            closes = payload.closes

        available = len(closes)
        used_window = payload.window
        if available < payload.window:
            used_window = available
            log.warning("Janela reduzida para %d, pois apenas %d dados históricos estão disponíveis.", used_window, available)
        if used_window <= 0:
            log.error("Dados históricos insuficientes (%d) para previsão.", available)
            raise HTTPException(status_code=400, detail=f"No historical closes available for {payload.symbol}")

        predict_start = perf_counter()
        preds = forecast_from_series(closes, payload.symbol, horizon=payload.horizon, window=used_window)
        predict_time = perf_counter() - predict_start
        
        log.info("Previsão concluída em %.4f segundos. Próximo valor previsto: %.4f", predict_time, preds[0])

        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        log.info("Requisição /predict finalizada com sucesso em %.4f segundos.", elapsed)
        return {"symbol": payload.symbol, "predictions": preds, "used_window": used_window, "available_history": available}
    except FileNotFoundError:
        log.warning("Tentativa de previsão para %s, mas o modelo não foi encontrado.", payload.symbol)
        raise HTTPException(status_code=404, detail="Model not found. Train first.")
    except HTTPException:
        raise
    except Exception as e:
        log.error("Erro ao processar /predict para %s: %s", payload.symbol, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def history(symbol: str, limit: int = 90):
    try:
        log.info("Buscando histórico de %d dias para %s.", limit, symbol)
        
        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
        df = GetData(info).QueryDf()
        if df is None or df.empty:
            log.warning("Nenhum dado encontrado para /history para %s.", symbol)
            return JSONResponse(status_code=404, content={"detail":"No data for symbol"})
        series = extract_close_series(df)
        if series is None:
            log.error("Coluna 'Close' não encontrada em /history para %s.", symbol)
            return JSONResponse(status_code=500, content={"detail":"Close column not found"})
        df2 = df.reset_index()
        close_col = None
        for col in df2.columns:
            if "Close" in str(col):
                close_col = col
                break
        if close_col is None:
            return JSONResponse(status_code=500, content={"detail":"Could not find Close column"})
        tail = df2.tail(limit)
        rows = []
        for _, r in tail.iterrows():
            d = r["Date"]
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            rows.append({"date": date_str, "close": float(r[close_col])})
        
        log.info("Histórico de %d dias retornado para %s.", len(rows), symbol)
        return {"symbol": symbol, "history": rows}
    except Exception as e:
        log.error("Erro ao processar /history para %s: %s", symbol, str(e), exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})

@router.get("/metrics")
async def metrics():
    data = generate_latest()
    log.info("Métricas do Prometheus solicitadas e geradas.")
    return HTMLResponse(content=data, media_type=CONTENT_TYPE_LATEST)

@router.post("/ui/predict", response_class=HTMLResponse)
async def ui_predict(request: Request, symbol: str = Form(...), horizon: int = Form(1), window: int = Form(60)):
    method = "POST"; endpoint = "/v1/ui/predict"
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()
    log.info("Processando requisição UI para %s (Horizon: %d)", symbol, horizon)
    
    try:
        model_dir = "app/models"
        model_path = os.path.join(model_dir, f"{symbol}_lstm.keras")
        if not os.path.exists(model_path):
            log.warning("Modelo não encontrado para %s.", symbol)
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol})

        info = __import__("app.domain.entities.information", fromlist=["Info"]).Info(symbol, None, None)
        df = GetData(info).QueryDf()
        series = extract_close_series(df)
        if series is None:
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error":"Could not get 'Close' series from Yahoo Finance."})
        closes = series.astype(float).tolist()
        available = len(closes)
        used_window = window if available >= window else available
        if used_window <= 0:
            return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error":"No historical data available."})
        
        preds = forecast_from_series(closes, symbol, horizon=horizon, window=used_window)
        
        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        log.info("Requisição UI concluída. Próximo valor: %.4f", preds[0])
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "predictions": preds,
            "used_window": used_window,
            "available_history": available
        })
    except FileNotFoundError:
        log.warning("FileNotFoundError na UI para %s.", symbol)
        return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error":"Model not found. Please train first."})
    except Exception as e:
        log.error("Erro inesperado na UI para %s: %s", symbol, str(e), exc_info=True)
        return templates.TemplateResponse("needs_training.html", {"request": request, "symbol": symbol, "error": str(e)})