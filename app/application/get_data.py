#Importa a biblioteca yfinance e a entidade Info
import yfinance as yf 
import pandas as pd
import numpy as np
from ..domain.entities.information import Info

#Classe para obter dados do Yahoo Finance
class GetData:
    #Construtor que recebe um objeto Info
    def __init__(self, Info):
        self.symbol = Info.symbol
        self.start_date = Info.start_date
        self.end_date = Info.end_date

    #Método para consultar os dados
    def QueryData(self):
        df = yf.download(self.symbol, self.start_date, self.end_date, auto_adjust=True)
        # Convert DataFrame to dict with dates as strings
        df_dict = df.reset_index().to_dict(orient='records')
        results = []
        for item in df_dict:
            row = {}
            for key, value in item.items():
                # ensure key is a string (avoid unhashable keys like lists)
                k = str(key)
                # handle pandas Timestamp / datetime
                if isinstance(value, (pd.Timestamp,)):
                    row[k] = value.strftime('%Y-%m-%d')
                # handle numpy numbers
                elif isinstance(value, (np.generic,)):
                    try:
                        row[k] = float(value)
                    except Exception:
                        row[k] = str(value)
                # handle plain python numbers
                elif isinstance(value, (int, float)):
                    row[k] = float(value)
                else:
                    # fallback to string for any other types
                    row[k] = str(value) if value is not None else None
            results.append(row)
        return results

#Exemplo de uso
#dt = Info('PETR4.SA', '2018-01-01', '2024-07-20')

#Instantia o GetData com os parâmetros fornecidos
#dados = GetData(dt).QueryData()
#print(dados)