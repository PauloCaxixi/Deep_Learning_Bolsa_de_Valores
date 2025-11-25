# Deep Learning Bolsa de Valores

Previsão de preços com LSTM usando Flask + Streamlit.

## Como rodar

```bash
1 ► python -m venv env
2 ► source env/bin/activate
3 ► pip install -r requirements.txt
4 ► sudo docker build -t meu-dashboard .
5 ► sudo docker run -p 5000:5000 -v "$(pwd)/static:/app/static" meu-dashboard
