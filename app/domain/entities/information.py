# app/domain/entities/information.py
class Info:
    def __init__(self, symbol: str, start_date: str | None = None, end_date: str | None = None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
