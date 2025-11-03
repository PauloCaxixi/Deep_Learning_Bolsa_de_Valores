Tech_Challenge_4 — README

Summary of what was done (up to Nov 02, 2025)

This README documents the edits and fixes applied to the FastAPI project so the API returns stock data from Yahoo Finance as a JSON list and is runnable with uvicorn.

High-level goals achieved
- Make the FastAPI app run with a single entrypoint and proper router inclusion.
- Fix import/package issues so modules under `src` / `app` are importable when running as a script.
- Ensure `GetData` fetches data with `yfinance` and returns JSON-serializable native types.
- Fix the API route so it returns a list of records (not the Info object) and avoid FastAPI serialization errors.

Files added / edited (what and why)

- `main.py`
  - Purpose: Application entrypoint — creates the FastAPI app and runs uvicorn when executed directly.
  - Changes: Cleaned duplicate `create_app` definitions and now imports the application router from `app.interfaces.schemas`.
  - Behavior: Includes the router at prefix `/v1` and runs uvicorn on `0.0.0.0:8000` when `python main.py` is used.

- `app/interfaces/schemas.py`
  - Purpose: API interface definition for endpoints.
  - Changes: Converted to expose an `APIRouter()` named `router` (instead of creating a separate FastAPI instance) so `main.py` can include it.
  - Contents: A sample `GET /` route returning a welcome message.

- `app/interfaces/routes.py`
  - Purpose: Actual API routes used by the app (included under `/v1`).
  - Changes: Fixed the `/home` route to call `GetData(dt).QueryData()` and return the processed data (JSON-safe list) under the `dados` key.
  - Behavior: Instantiates `Info(...)`, passes it to `GetData`, and returns the fetched data.

- `app/application/get_data.py`
  - Purpose: Encapsulates data fetching from Yahoo Finance using `yfinance`.
  - Changes made:
    - `QueryData` now calls `yf.download(..., auto_adjust=True)` to avoid FutureWarning.
    - DataFrame rows are converted to a list of dictionaries (records) with keys coerced to strings.
    - Values are converted safely to JSON-serializable types:
      - `pandas.Timestamp` -> `YYYY-MM-DD` string
      - `numpy` scalars -> `float` (where appropriate)
      - Python ints/floats -> `float`
      - Other types -> `str` fallback
    - This prevents FastAPI's jsonable_encoder errors (e.g., unhashable keys, types that cannot be encoded by default).

- `src/__init__.py`, `src/domain/__init__.py`, `src/domain/entities/__init__.py`
  - Purpose: Mark directories as Python packages to support imports while running as a module.
  - Changes: Empty `__init__.py` files were added.

Why these changes were necessary
- The FastAPI app originally had two FastAPI instances (one in `main.py` and one in `schemas.py`) which caused inconsistent routing and contributed to confusion when including routers.
- Calling instance methods incorrectly (e.g. class method vs instance call) was fixed for `GetData` to ensure `self` context is available.
- Returning the `Info` instance from the route caused the API to show only the input parameters; the real data fetched from Yahoo Finance must be returned instead.
- Pandas and numpy types are not always JSON-serializable out of the box; explicitly converting keys and values to native types prevents runtime encoding errors.
 # Tech_Challenge_4 — README

Resumo do que foi feito (até 02/11/2025)

Este README descreve as mudanças realizadas no projeto FastAPI para que a API retorne dados de ações (Yahoo Finance) como uma lista JSON e seja executável com uvicorn.

Principais objetivos alcançados
- Fazer a aplicação FastAPI rodar com um único ponto de entrada e inclusão correta do router.
- Corrigir problemas de importação/packaging para que módulos em `src` / `app` sejam importáveis quando o projeto for executado.
- Garantir que `GetData` busque dados com `yfinance` e retorne tipos nativos serializáveis em JSON.
- Corrigir a rota API para retornar uma lista de registros (em vez do objeto `Info`) e evitar erros de serialização do FastAPI.

Arquivos editados / adicionados (o que foi feito e por quê)

- `main.py`
  - Objetivo: Ponto de entrada da aplicação — cria a instância FastAPI e executa o uvicorn quando chamado diretamente.
  - Mudanças: Removidas definições duplicadas de `create_app` e importado o `router` de `app.interfaces.schemas`.
  - Comportamento: Inclui o router com prefixo `/v1` e executa o uvicorn em `0.0.0.0:8000` quando `python main.py` é usado.

- `app/interfaces/schemas.py`
  - Objetivo: Definição da interface (endpoints) da API.
  - Mudanças: Alterado para expor um `APIRouter()` chamado `router` (em vez de criar um FastAPI separado), permitindo que `main.py` inclua as rotas.
  - Conteúdo: Rota de exemplo `GET /` que retorna uma mensagem de boas-vindas.

- `app/interfaces/routes.py`
  - Objetivo: Rotas reais da API (incluídas sob `/v1`).
  - Mudanças: Corrigida a rota `/home` para instanciar `Info(...)`, chamar `GetData(dt).QueryData()` e retornar a lista processada (segura para JSON) na chave `dados`.

- `app/application/get_data.py`
  - Objetivo: Encapsular a busca de dados do Yahoo Finance usando `yfinance`.
  - Mudanças aplicadas:
    - `QueryData` passa a chamar `yf.download(..., auto_adjust=True)` para evitar FutureWarning.
    - As linhas do DataFrame são convertidas para uma lista de dicionários (records) com chaves convertidas em strings.
    - Valores são convertidos de forma segura para tipos serializáveis em JSON:
      - `pandas.Timestamp` -> string no formato `YYYY-MM-DD`
      - scalars do `numpy` -> `float` quando aplicável
      - ints/floats do Python -> `float`
      - Outros tipos -> `str` como fallback
    - Isso evita erros do `jsonable_encoder` do FastAPI (por exemplo, chaves não-hashable ou tipos não-encodáveis).

- `src/__init__.py`, `src/domain/__init__.py`, `src/domain/entities/__init__.py`
  - Objetivo: Marcar diretórios como pacotes Python para suportar importações ao executar como módulo.
  - Mudanças: Arquivos `__init__.py` vazios foram adicionados.

Por que essas mudanças foram necessárias
- O projeto originalmente tinha duas instâncias FastAPI (uma em `main.py` e outra em `schemas.py`), o que causava roteamento inconsistente e confusão ao incluir routers.
- Chamadas incorretas a métodos de instância (por exemplo, chamar método de classe sem instanciar) foram ajustadas para garantir que `self` esteja disponível.
- Retornar o objeto `Info` na rota fazia com que a API mostrasse apenas os parâmetros de entrada; era necessário retornar os dados reais obtidos do Yahoo Finance.
- Tipos do pandas e numpy não são automaticamente serializáveis em JSON; converter explicitamente chaves/valores para tipos nativos evita erros em tempo de execução.

Como executar a API (PowerShell)
1. Ative seu ambiente virtual (caso não esteja ativo):

   # PowerShell
   .\env\Scripts\Activate.ps1

2. Instale dependências (se necessário):

   pip install "fastapi" "uvicorn[standard]" "yfinance" "pandas" "numpy"

3. Inicie o servidor:

   # executar como script (main.py chama uvicorn.run internamente)
   python main.py

   # ou executar uvicorn diretamente (método alternativo recomendado)
   python -m uvicorn main:app --reload

4. Abra a interface no navegador:
   - Swagger UI: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc
   - `/v1/home` (retorna lista de dados da ação): http://127.0.0.1:8000/v1/home

Formato esperado de resposta para `GET /v1/home`
```json
{
  "dados": [
    {
      "Date": "2018-01-01",
      "Open": 123.45,
      "High": 125.67,
      "Low": 122.89,
      "Close": 124.56,
      "Volume": 1000000.0
    }
  ]
}
```

Notas de troubleshooting e erros comuns
- Erros de importação (ModuleNotFoundError / "No module named 'app' or 'domain'"):
  - Garanta que você execute o comando a partir da raiz do repositório onde `main.py` está localizado.
  - Se executar com `-m`, confirme que o diretório atual contém o módulo `main.py` no caminho correto.

- Pacotes ausentes em tempo de execução:
  - O linter estático pode mostrar `pandas`/`numpy` não encontrados; assegure-se de que estão instalados no mesmo venv usado para rodar o servidor.

- Erros do `jsonable_encoder` do FastAPI (TypeError: cannot use 'list' as a dict key):
  - Esse erro foi causado por retornar objetos com chaves não-string. A função `get_data.QueryData` agora força chaves para strings e converte valores para tipos nativos.

- FutureWarning do `yfinance` sobre mudança no `auto_adjust`:
  - Foi definido `auto_adjust=True` explicitamente para silenciar o aviso e retornar preços ajustados.

Próximas melhorias sugeridas (baixo risco)
- Limitar o número de linhas retornadas durante o desenvolvimento: retornar `results[:n]` em `QueryData` para respostas mais rápidas.
- Adicionar um modelo de resposta Pydantic em `routes.py` para definir e validar o schema de saída do endpoint `/v1/home`.
- Adicionar testes unitários para `GetData.QueryData` usando mock de `yfinance.download` para tornar os testes determinísticos e rápidos.
- Adicionar logging para erros e operações bem-sucedidas.
- Implementar cache para evitar downloads repetidos do mesmo símbolo/período durante o desenvolvimento.

Posso também:
- Adicionar um pequeno script de teste em `tests/` que importe `GetData` e imprima um registro de exemplo processado.
- Adicionar um modelo Pydantic de resposta e ajustar a rota para usá-lo (recomendado para APIs de produção).

Próximos passos
Se quiser que eu implemente alguma das melhorias sugeridas ou gere o script de teste / modelo Pydantic, diga qual e eu implemento em seguida.
