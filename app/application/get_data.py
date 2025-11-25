# app/application/get_data.py
import yfinance as yf
import pandas as pd
from ..domain.entities.information import Info

class GetData:
    def __init__(self, info: Info):
        self.symbol = info.symbol
        self.start = info.start_date
        self.end = info.end_date

    def QueryDf(self) -> pd.DataFrame:
        # If start is None, yfinance will fetch a recent period.
        # We prefer to fetch a long period if start is None to ensure enough history.
        try:
            if self.start:
                df = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)
            else:
                # fetch 720 days (~3 years) by default to get enough history
                df = yf.download(self.symbol, period="720d", auto_adjust=True)
            if df is None:
                return pd.DataFrame()
            df = df.dropna()
            return df
        except Exception as e:
            # return empty DataFrame on error (caller handles)
            return pd.DataFrame()
