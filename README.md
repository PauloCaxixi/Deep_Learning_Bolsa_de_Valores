# 📈 AI Stock Price Predictor

## Previsor de Preço de Ações com LSTM e FastAPI

Este projeto implementa uma solução completa de Machine Learning aplicada à previsão de preços de ações, utilizando uma Rede Neural Recorrente do tipo LSTM (Long Short-Term Memory), exposta por meio de uma API REST desenvolvida com FastAPI.

A aplicação cobre todo o ciclo de vida do modelo (treinamento, persistência, inferência e monitoramento), seguindo boas práticas de MLOps e uma arquitetura em camadas.

---

## ✨ Tecnologias Utilizadas

* API Framework: FastAPI
* Servidor ASGI: Uvicorn
* Machine Learning: TensorFlow / Keras (LSTM)
* Processamento de Dados: Pandas, NumPy
* Coleta de Dados Financeiros: yfinance
* Normalização: MinMaxScaler (scikit-learn)
* Persistência de Modelos: .keras + joblib
* Monitoramento: Prometheus
* Logging: Módulo logging do Python

---

## 🚀 Execução Local do Projeto

### 1️⃣ Criação do Ambiente Virtual

O uso de ambiente virtual é fortemente recomendado, especialmente devido às dependências do TensorFlow.

```bash
# Criação do ambiente virtual
python -m venv tf_env

# Ativação (Windows PowerShell)
.\tf_env\Scripts\Activate

# Ativação (Linux/macOS)
source tf_env/bin/activate
```

---

### 2️⃣ Instalação das Dependências

Instale todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

⚠️ Certifique-se de que o arquivo se chame corretamente requirements.txt.

---

## 📂 Estrutura do Projeto

```text
Tech_Challenge_4/
├── main.py
├── app/
│   ├── templates/
│   │   ├── index.html
│   │   ├── needs_training.html
│   │   └── result.html
│   ├── models/                # Modelos LSTM e scalers salvos
│   ├── application/           # Camada de aplicação
│   │   ├── __init__.py
│   │   ├── get_data.py
│   │   └── train_model.py
│   ├── domain/                # Camada de domínio
│   │   ├── __init__.py
│   │   └── entities/
│   │       └── information.py
│   └── interfaces/            # Camada de interfaces (API)
│       └── routes.py
```

A estrutura segue uma arquitetura em camadas, separando responsabilidades de forma clara.

---

## ▶️ Iniciando o Servidor

Com o ambiente virtual ativado, execute:

```bash
python main.py
```

ou

```bash
uvicorn main:app --reload
```

A aplicação estará disponível em:

```
http://localhost:8000
```

A documentação interativa da API pode ser acessada em:

```
http://localhost:8000/docs
```

---

## 🧠 Fluxo de Uso da Aplicação

### 🔹 Passo 1 — Treinamento do Modelo

Antes de realizar previsões, é necessário treinar o modelo para o ativo desejado.

* Acesse /docs
* Utilize o endpoint POST /v1/train
* Informe os parâmetros de treinamento

Exemplo de payload:

```json
{
  "symbol": "AAPL",
  "start": "2010-01-01",
  "end": "2025-12-31",
  "window": 60,
  "epochs": 15
}
```

O modelo treinado e o scaler correspondente são automaticamente salvos em app/models.

---

### 🔹 Passo 2 — Realizar Previsão

* Acesse [http://localhost:8000/v1](http://localhost:8000/v1)
* Informe o symbol previamente treinado
* Clique em Predict

A API carregará o modelo salvo, buscará os dados históricos recentes e retornará a previsão de preço.

---

## 📊 Monitoramento e Logging (MLOps)

### 1️⃣ Métricas com Prometheus

O endpoint /v1/metrics expõe métricas para monitoramento em tempo real:

* Contagem de Requisições: app_requests_total
* Latência: app_request_latency_seconds

Essas métricas podem ser coletadas por um servidor Prometheus e visualizadas via Grafana.

---

### 2️⃣ Logging de Eventos

O sistema utiliza o módulo logging para rastreamento detalhado:

* Treinamento: início/fim do treino, métricas finais (MAE e Loss)
* Inferência: tempo de execução e valores previstos
* Erros: exceções capturadas com log.error

---

## 🛠️ Notas Técnicas de Implementação

### Modelo LSTM

* Uma camada LSTM com 50 unidades
* Função de ativação tanh
* Saída contínua para regressão

### Normalização

* MinMaxScaler aplicado antes do treinamento
* Scaler salvo junto com o modelo

### Persistência

* Modelo salvo em formato .keras
* Scaler salvo via joblib

### Arquitetura

* Interfaces (rotas / API)
* Aplicação (lógica de ML e dados)
* Domínio (entidades de negócio)

---

## 🤝 Contribuição

Contribuições são bem-vindas. Exemplos de melhorias futuras:

* Suporte a múltiplos modelos (Prophet, ARIMA, XGBoost)
* Autenticação e controle de acesso na API
* Otimização de hiperparâmetros
* Cache de previsões
* Pipeline CI/CD para automação de treinamento