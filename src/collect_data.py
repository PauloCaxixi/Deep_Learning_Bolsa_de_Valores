import yfinance as yf
import pandas as pd
from datetime import datetime
from src.db_utils import save_to_db

def collect_stock_data(ticker="PETR4.SA", start="2018-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print(f"📊 Coletando dados de {ticker} de {start} até {end}...")

    data = yf.download(ticker, start=start, end=end, progress=False)

    # Alguns retornos do yfinance vêm como tupla (df, metadata)
    if isinstance(data, tuple):
        df = data[0]
    else:
        df = data

    # 🔧 Garante que a data vire uma coluna
    df.reset_index(inplace=True)

    if not isinstance(df, pd.DataFrame):
        raise ValueError("❌ O yfinance não retornou um DataFrame válido.")

    # Padroniza nomes das colunas (evita erro se for MultiIndex)
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    print(f"✅ {len(df)} registros coletados.")
    return df

if __name__ == "__main__":
    try:
        df = collect_stock_data()
        save_to_db(df, "raw_petr4")
        print("💾 Dados salvos no banco SQLite.")
    except Exception as e:
        print(f"❌ Erro: {e}")
