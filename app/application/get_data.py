# app/application/get_data.py

# --- IMPORTS DE INFRAESTRUTURA E DOMÍNIO ---
# Biblioteca para baixar dados financeiros (a fonte externa de dados).
import yfinance as yf
# Biblioteca para manipulação e estruturação dos dados (DataFrame).
import pandas as pd
# Importa a entidade 'Info' da camada de domínio, que contém o que deve ser consultado.
from ..domain.entities.information import Info

# --- CLASSE DE SERVIÇO DE DADOS ---
class GetData:
    """
    Serviço da Camada de Aplicação responsável por obter dados 
    históricos de preços de ações usando yfinance.
    """
    
    def __init__(self, info: Info):
        """
        Construtor. Recebe a Entidade 'Info' e extrai os parâmetros de consulta.
        
        :param info: Objeto da camada de Domínio contendo o símbolo e as datas.
        """
        # Armazena os parâmetros de consulta.
        self.symbol = info.symbol
        self.start = info.start_date
        self.end = info.end_date

    def QueryDf(self) -> pd.DataFrame:
        """
        Executa a consulta no yfinance e retorna um DataFrame do Pandas 
        com os dados históricos de preço.
        """
        # Se 'start' for None, o yfinance usa o parâmetro 'period'.
        try:
            if self.start:
                # Se as datas de início/fim foram fornecidas (uso pelo /train), 
                # baixa o histórico no período específico.
                # auto_adjust=True: ajusta automaticamente para splits de ações, etc.
                df = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)
            else:
                # Se nenhuma data for fornecida (uso pelo /predict ou /history), 
                # baixa um período recente padrão (cerca de 3 anos) para garantir dados suficientes.
                df = yf.download(self.symbol, period="720d", auto_adjust=True)
            
            # Verificação básica se a consulta falhou.
            if df is None:
                return pd.DataFrame()
            
            # Remove linhas com valores ausentes (NaN) após o download.
            # Isso garante que a série de preços esteja limpa antes de ser usada pelo ML.
            df = df.dropna()
            
            # Retorna o DataFrame limpo. 
            return df
            
        except Exception as e:
            # Em caso de qualquer erro (ex: símbolo inválido, problema de conexão),
            # retorna um DataFrame vazio. O chamador (ex: /train ou /predict) 
            # é responsável por lidar com esse erro.
            return pd.DataFrame()