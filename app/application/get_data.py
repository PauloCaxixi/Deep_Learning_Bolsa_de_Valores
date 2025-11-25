# app/application/get_data.py
import yfinance as yf
import pandas as pd
import numpy as np
from ..domain.entities.information import Info

class GetData:
    def __init__(self, info: Info):
        self.symbol = info.symbol
        self.start_date = info.start_date
        self.end_date = info.end_date

    def QueryData(self):
        df = yf.download(self.symbol, self.start_date, self.end_date, auto_adjust=True)
        df = df.dropna()
        # Mantém colunas relevantes: Date, Open, High, Low, Close, Volume
        df_reset = df.reset_index()
        # converte em lista de dicionários (JSON-serializable)
        df_dict = df_reset.to_dict(orient="records")
        results = []
        for item in df_dict:
            row = {}
            for k, v in item.items():
                if isinstance(v, (pd.Timestamp,)):
                    row[str(k)] = v.strftime("%Y-%m-%d")
                elif isinstance(v, (np.generic,)):
                    try:
                        row[str(k)] = float(v)
                    except Exception:
                        row[str(k)] = str(v)
                elif isinstance(v, (int, float)):
                    row[str(k)] = float(v)
                else:
                    row[str(k)] = str(v) if v is not None else None
            results.append(row)
        return results

    def QueryDf(self):
        """Retorna o DataFrame bruto (útil para treino)."""
        df = yf.download(self.symbol, self.start_date, self.end_date, auto_adjust=True)
        return df.dropna()
