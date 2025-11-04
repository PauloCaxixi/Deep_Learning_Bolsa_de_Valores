from src.collect_data import collect_stock_data
from src.db_utils import save_to_db, read_from_db
from src.preprocess import preprocess_data

print("Iniciando pipeline de coleta e pré-processamento...")

df_raw = collect_stock_data(ticker="PETR4.SA", start="2018-01-01")
save_to_db(df_raw, "raw_petr4")

df_scaled, scaler = preprocess_data("raw_petr4")
