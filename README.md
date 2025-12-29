# 📈 AI Stock Price Predictor (Previsor de Preço de Ações com LSTM)

Este é um projeto de Machine Learning (MLOps) que utiliza uma Rede Neural Recorrente (LSTM) para prever o preço de fechamento de ações, servido através de uma API REST robusta construída com FastAPI. O sistema inclui um pipeline de treinamento, inferência e monitoramento de desempenho.

## ✨ Tecnologias Principais

* **Framework:** FastAPI
* **Modelo de ML:** TensorFlow/Keras (LSTM - Long Short-Term Memory)
* **Estrutura de Dados:** Pandas
* **Coleta de Dados:** `yfinance`
* **Monitoramento:** Prometheus (Métricas de Latência e Contagem)
* **Logging:** Módulo `logging` do Python para rastreamento de eventos e erros.

## 🚀 Como Executar o Projeto Localmente

Siga estes passos para configurar e executar o projeto no seu ambiente.

### 1. Configuração do Ambiente Virtual

É crucial usar um ambiente virtual para gerenciar as dependências corretamente (especialmente o TensorFlow).

```bash
# Cria o ambiente virtual chamado 'tf_env'
python -m venv tf_env 

# Ativa o ambiente virtual (Windows PowerShell)
.\tf_env\Scripts\Activate

# (Ou no Linux/macOS)
# source tf_env/bin/activate


Instalação das Dependências
Instale todas as bibliotecas necessárias, incluindo TensorFlow, FastAPI, Uvicorn e Prometheus.

Bash

# Instala as bibliotecas de ML, API e MLOps
pip install -r requirments.txt


Estrutura do Projeto
Certifique-se de que a estrutura de pastas esteja correta:

Tech_Challenge_4/
├── main.py
├── app/
│   ├── templates/
│   │   ├── index.html
│   │   ├── needs_training.html
│   │   └── result.html
│   ├── models/  <-- Modelos e Scalers salvos aqui após o treino
│   ├── application/
│   │   ├── __init__.py
│   │   ├── get_data.py
│   │   └── train_model.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── entities/
│   │       └── information.py
│   └── interfaces/
│       └── routes.py


Iniciando o Servidor
Com o ambiente ativado, inicie o servidor FastAPI/Uvicorn:

Bash

python main.py

O servidor estará disponível em: http://localhost:8000


🧠 Fluxo de Trabalho (Como Usar)
Passo 1: Treinar o Modelo
Antes de prever, você deve treinar o modelo para o símbolo desejado (ex: PETR4.SA ou AAPL).

Acesse http://localhost:8000/docs.

Expanda o endpoint POST /v1/train.

Preencha o Request body com os parâmetros desejados (o padrão é geralmente um bom ponto de partida).

Clique em "Execute".

Exemplo de Payload:

JSON

{
  "symbol": "AAPL",
  "start": "2010-01-01",
  "end": "2025-12-31",
  "window": 60,
  "epochs": 15
}
Passo 2: Fazer Previsão
Acesse a interface em http://localhost:8000/v1.

Digite o Symbol (o mesmo usado no treinamento).

Clique em Predict.

A API carregará o modelo salvo e o scaler, consultará os dados históricos recentes e fará a previsão.

📊 Monitoramento e Logging (MLOps)
Este projeto utiliza duas abordagens de monitoramento:

1. Prometheus Metrics (Métricas de Desempenho)
O endpoint /v1/metrics expõe métricas de latência e contagem de requisições, que podem ser coletadas por um servidor Prometheus (e visualizadas em um dashboard Grafana) para monitorar a saúde da API em tempo real.

Contagem: app_requests_total

Latência: app_request_latency_seconds

2. Logging (Rastreamento de Eventos e Erros)
O módulo logging do Python registra eventos importantes no console, facilitando a depuração e o rastreamento do fluxo de dados:

Treinamento: Informações sobre o carregamento de dados, início/fim do model.fit e métricas finais (MAE/MSE).

Previsão: Tempo de inferência e os valores previstos.

Erros: Captura Exceptions e HTTPExceptions com detalhes, usando log.error().

🛑 Notas de Implementação
LSTM: O modelo utiliza uma camada LSTM de 50 unidades e é treinado para regressão (prever um valor numérico contínuo).

Escalonamento: O MinMaxScaler é crucial para normalizar os dados antes do treinamento e desnormalizar as previsões, sendo salvo junto com o modelo usando joblib.

Estrutura de Arquitetura: O projeto segue vagamente uma arquitetura em camadas, separando Interfaces (rotas), Aplicação (lógica de ML e dados) e Domínio (entidades).

🤝 Contribuição
Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias, como:

Adicionar suporte a mais modelos (ex: Prophet, ARIMA).

Implementar autenticação na API.

Otimizar o desempenho do train_model.py.