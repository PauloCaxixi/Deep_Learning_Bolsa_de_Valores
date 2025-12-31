# ===============================
# IMPORTAÇÕES
# ===============================

# FastAPI: cria rotas, recebe requisições e retorna respostas HTTP
from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

# Jinja2: renderiza páginas HTML
from fastapi.templating import Jinja2Templates

# Pydantic: valida os dados recebidos no corpo da requisição
from pydantic import BaseModel

# Mede tempo de execução (performance)
from time import perf_counter

# Prometheus: métricas de monitoramento
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Camada de aplicação: busca dados, treina modelo e faz previsões
from ..application.get_data import GetData
from ..application.train_model import train_and_save, forecast_from_series, load_model_and_scaler

# Bibliotecas auxiliares
import os                    # Manipulação de arquivos e caminhos
import pandas as pd          # Manipulação de dados
import yfinance as yf        # Busca dados financeiros
import logging               # Registro de logs (mensagens)

# Cria o logger para este arquivo
log = logging.getLogger(__name__)

# ===============================
# CONFIGURAÇÃO DO ROUTER E TEMPLATES
# ===============================

# Router do FastAPI para registrar os endpoints
router = APIRouter()

# Diretório onde ficam os arquivos HTML
templates = Jinja2Templates(directory="app/templates")

# ===============================
# MÉTRICAS DO PROMETHEUS
# ===============================

# Conta quantas requisições foram feitas
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total API requests",
    ["method", "endpoint"]
)

# Mede quanto tempo cada requisição demora
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["method", "endpoint"]
)

# ===============================
# MODELOS DE ENTRADA (Pydantic)
# ===============================

# Dados esperados para treinar o modelo
class TrainIn(BaseModel):
    symbol: str              # Símbolo da ação
    start: str = "2018-01-01"
    end: str = "2025-12-09"
    window: int = 60          # Janela de tempo
    epochs: int = 10          # Quantidade de épocas
    batch_size: int = 32      # Tamanho do lote

# Dados esperados para previsão
class PredictIn(BaseModel):
    symbol: str              # Símbolo da ação
    closes: list[float] = None  # Lista de preços (opcional)
    horizon: int = 1          # Quantos passos prever
    window: int = 60          # Janela usada na previsão

# ===============================
# FUNÇÃO AUXILIAR
# ===============================

# Extrai a coluna "Close" do DataFrame, independentemente do formato
def extract_close_series(df: pd.DataFrame) -> pd.Series | None:
    # Caso padrão
    if "Close" in df.columns and isinstance(df["Close"], pd.Series):
        return df["Close"]

    # Caso colunas MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if str(col[0]) == "Close":
                return df[col]

    # Busca genérica
    for col in df.columns:
        if "Close" in str(col):
            return df[col]

    return None

# ===============================
# ROTA PRINCIPAL (HTML)
# ===============================

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Caminho do logo
    logo_path = "/mnt/data/5ecf1705-570e-4634-bf3e-8feff82b4532.png"

    # Verifica se o arquivo existe
    logo_exists = os.path.exists(logo_path)

    # Renderiza a página inicial
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "logo_path": logo_path if logo_exists else None
        }
    )

# ===============================
# ROTA PARA CONTAR DADOS
# ===============================

@router.get("/data-count")
async def data_count(symbol: str):
    # Cria objeto Info dinamicamente
    info = __import__(
        "app.domain.entities.information",
        fromlist=["Info"]
    ).Info(symbol, None, None)

    # Busca dados
    df = GetData(info).QueryDf()

    # Validação
    if df is None or df.empty:
        log.warning("Nenhum dado encontrado para %s", symbol)
        raise HTTPException(status_code=404, detail="No data")

    # Extrai a coluna Close
    series = extract_close_series(df)
    if series is None:
        raise HTTPException(status_code=500, detail="No Close column")

    # Retorna informações básicas
    return {
        "symbol": symbol,
        "rows": len(df),
        "closes": len(series)
    }

# ===============================
# ROTA DE TREINAMENTO
# ===============================

@router.post("/train")
async def train(payload: TrainIn):
    method = "POST"
    endpoint = "/v1/train"

    # Incrementa contador
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()

    try:
        # Cria objeto de consulta
        info = __import__(
            "app.domain.entities.information",
            fromlist=["Info"]
        ).Info(payload.symbol, payload.start, payload.end)

        # Busca dados
        df = GetData(info).QueryDf()
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data")

        # Treina o modelo e salva
        result = train_and_save(
            payload.symbol,
            df,
            window=payload.window,
            epochs=payload.epochs,
            batch_size=payload.batch_size
        )

        # Registra tempo
        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)

        # Retorna resultado
        return JSONResponse({
            "status": "trained",
            "metrics": result
        })

    except Exception as e:
        log.error("Erro no treinamento", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# ROTA DE PREVISÃO (API)
# ===============================

@router.post("/predict")
async def predict(payload: PredictIn):
    method = "POST"
    endpoint = "/v1/predict"

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    t0 = perf_counter()

    try:
        # Se não vierem preços, busca no Yahoo Finance
        if payload.closes is None:
            info = __import__(
                "app.domain.entities.information",
                fromlist=["Info"]
            ).Info(payload.symbol, None, None)

            df = GetData(info).QueryDf()
            series = extract_close_series(df)

            # Segunda tentativa via yfinance
            if series is None:
                df2 = yf.download(
                    payload.symbol,
                    period=f"{max(payload.window * 4, 360)}d",
                    auto_adjust=True
                )
                series = extract_close_series(df2)

            closes = series.astype(float).tolist()
        else:
            closes = payload.closes

        # Ajusta janela se necessário
        available = len(closes)
        used_window = min(payload.window, available)

        if used_window <= 0:
            raise HTTPException(status_code=400, detail="No historical data")

        # Executa previsão
        preds = forecast_from_series(
            closes,
            payload.symbol,
            horizon=payload.horizon,
            window=used_window
        )

        elapsed = perf_counter() - t0
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)

        return {
            "symbol": payload.symbol,
            "predictions": preds,
            "used_window": used_window,
            "available_history": available
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# ROTA DE HISTÓRICO
# ===============================

@router.get("/history")
async def history(symbol: str, limit: int = 90):
    try:
        info = __import__(
            "app.domain.entities.information",
            fromlist=["Info"]
        ).Info(symbol, None, None)

        df = GetData(info).QueryDf()
        series = extract_close_series(df)

        df2 = df.reset_index()
        close_col = next(col for col in df2.columns if "Close" in str(col))

        tail = df2.tail(limit)
        rows = []

        for _, r in tail.iterrows():
            rows.append({
                "date": str(r["Date"]),
                "close": float(r[close_col])
            })

        return {"symbol": symbol, "history": rows}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ===============================
# ROTA DE MÉTRICAS
# ===============================

@router.get("/metrics")
async def metrics():
    data = generate_latest()
    return HTMLResponse(content=data, media_type=CONTENT_TYPE_LATEST)

# ===============================
# ROTA DE PREVISÃO VIA INTERFACE WEB
# ===============================

@router.post("/ui/predict", response_class=HTMLResponse)
async def ui_predict(
    request: Request,
    symbol: str = Form(...),
    horizon: int = Form(1),
    window: int = Form(60)
):
    try:
        model_path = f"app/models/{symbol}_lstm.keras"

        # Se não existir modelo, pede treinamento
        if not os.path.exists(model_path):
            return templates.TemplateResponse(
                "needs_training.html",
                {"request": request, "symbol": symbol}
            )

        # Busca dados
        info = __import__(
            "app.domain.entities.information",
            fromlist=["Info"]
        ).Info(symbol, None, None)

        df = GetData(info).QueryDf()
        series = extract_close_series(df)
        closes = series.astype(float).tolist()

        used_window = min(window, len(closes))

        preds = forecast_from_series(
            closes,
            symbol,
            horizon=horizon,
            window=used_window
        )

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "symbol": symbol,
                "predictions": preds,
                "used_window": used_window,
                "available_history": len(closes)
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "needs_training.html",
            {
                "request": request,
                "symbol": symbol,
                "error": str(e)
            }
        )
