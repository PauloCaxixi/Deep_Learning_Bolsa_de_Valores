# main.py

# Importa a classe principal do FastAPI, que gerencia o servidor web.
from fastapi import FastAPI
# Importa o objeto 'router' que contém todos os endpoints (rotas /v1/train, /v1/predict, etc.) 
# definidos no seu arquivo app/interfaces/routes.py.
from app.interfaces.routes import router as api_router
# Módulo padrão do Python para registro de eventos (logs).
import logging
# Módulo padrão para acessar variáveis e funções relacionadas ao interpretador, 
# usado aqui para direcionar logs para a saída padrão (console).
import sys

# --- CONFIGURAÇÃO CENTRAL DO LOGGING ---
def configure_logging():
    """Define o formato e o destino das mensagens de log da aplicação."""
    # Obtém o logger raiz.
    log = logging.getLogger()
    # Cria um 'Handler' para enviar as mensagens de log para a saída padrão (console).
    handler = logging.StreamHandler(sys.stdout)
    
    # Define o formato de cada linha de log, incluindo tempo, nível, nome do módulo e a mensagem.
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    
    # Adiciona o handler ao logger para que as mensagens sejam exibidas.
    # 
    
    # Proteção simples contra adicionar handlers repetidos, comum em ambientes com 'reload'.
    if not log.handlers:
        log.addHandler(handler)
        
    # Define o nível mínimo de log a ser registrado. INFO inclui mensagens de nível INFO, WARNING e ERROR.
    log.setLevel(logging.INFO) 
# --------------------------------------

def create_app() -> FastAPI:
    """Função que instancia e configura o objeto principal do FastAPI."""
    # Instancia o FastAPI, definindo o título que aparecerá na documentação (ex: /docs).
    app = FastAPI(title="Prever Fechamento de Ação")
    # Inclui o conjunto de rotas importado, aplicando o prefixo '/v1' a todos os endpoints.
    app.include_router(api_router, prefix="/v1")
    # Retorna a instância da aplicação configurada.
    return app

# 1. Configurar o log antes de criar a app
# Chama a função para configurar o sistema de logs antes de iniciar a aplicação.
configure_logging() 

# Cria a instância da aplicação FastAPI.
app = create_app()
# Registra uma mensagem no nível INFO, confirmando que a aplicação carregou suas rotas.
logging.info("Aplicação FastAPI criada e rotas incluídas.")

# Bloco padrão do Python que verifica se o script está sendo executado diretamente.
if __name__ == "__main__":
    # Importa o servidor de desenvolvimento e produção ASGI (Asynchronous Server Gateway Interface).
    import uvicorn
    # Inicia o servidor Uvicorn.
    # "main:app" - Diz ao Uvicorn para procurar a variável 'app' no arquivo 'main.py'.
    # host="0.0.0.0" - Permite acesso de qualquer IP (pode ser acessado por localhost ou rede).
    # port=8000 - Define a porta de escuta do servidor.
    # reload=True - Monitora mudanças nos arquivos e recarrega o servidor automaticamente (útil em desenvolvimento).
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)