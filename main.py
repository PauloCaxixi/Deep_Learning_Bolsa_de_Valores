# main.py

from fastapi import FastAPI
from app.interfaces.routes import router as api_router
import logging
import sys

# ===============================
# CONFIGURAÇÃO GLOBAL DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

log = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="Prever Fechamento de Ação")
    app.include_router(api_router, prefix="/v1")
    return app

app = create_app()

log.info("Aplicação FastAPI criada e rotas incluídas.")
