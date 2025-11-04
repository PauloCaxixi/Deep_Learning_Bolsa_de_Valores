import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.db")

def save_to_db(df, table_name):
    if df.empty:
        raise ValueError("DataFrame vazio — nada para salvar.")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # garante que pasta /data existe

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Tabela '{table_name}' salva com sucesso em {DB_PATH}.")

def read_from_db(table_name):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Banco de dados não encontrado: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    print(f"Tabela '{table_name}' carregada ({len(df)} registros).")
    return df
