FROM python:3.10-slim

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Garante que o diretório static exista
RUN mkdir -p /app/static

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta do Flask
EXPOSE 5000

# Executa o app
CMD ["python", "app.py"]
