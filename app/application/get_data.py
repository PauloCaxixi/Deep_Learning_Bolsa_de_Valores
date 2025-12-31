# ==========================================
# app/application/get_data.py
# ==========================================

# --- IMPORTAÇÕES ---

# Biblioteca que acessa dados financeiros do Yahoo Finance
import yfinance as yf

# Biblioteca usada para trabalhar com tabelas de dados (DataFrame)
import pandas as pd

# Importa a entidade Info da camada de domínio
# Ela contém as informações necessárias para a consulta (símbolo e datas)
from ..domain.entities.information import Info


# ==========================================
# CLASSE GetData
# ==========================================

class GetData:
    """
    Classe da camada de Aplicação responsável por buscar
    dados históricos de ações usando o Yahoo Finance (yfinance).
    """

    def __init__(self, info: Info):
        """
        Construtor da classe.

        Recebe um objeto Info (camada de domínio) e extrai
        os parâmetros necessários para a consulta.

        :param info: Objeto Info contendo símbolo, data inicial e final
        """

        # Símbolo da ação (ex: AAPL, PETR4.SA)
        self.symbol = info.symbol

        # Data inicial da consulta (pode ser None)
        self.start = info.start_date

        # Data final da consulta (pode ser None)
        self.end = info.end_date


    def QueryDf(self) -> pd.DataFrame:
        """
        Executa a consulta no Yahoo Finance e retorna
        um DataFrame do Pandas com os dados históricos.
        """

        try:
            # ------------------------------------------------
            # CASO 1: Datas de início e fim foram informadas
            # (Normalmente usado no treinamento do modelo)
            # ------------------------------------------------
            if self.start:
                # Baixa dados do período específico
                # auto_adjust=True ajusta automaticamente preços
                # para eventos como split de ações
                df = yf.download(
                    self.symbol,
                    start=self.start,
                    end=self.end,
                    auto_adjust=True
                )

            # ------------------------------------------------
            # CASO 2: Nenhuma data foi informada
            # (Usado em previsão ou histórico)
            # ------------------------------------------------
            else:
                # Baixa um período padrão recente (720 dias ≈ 2 anos)
                # Garante que haja dados suficientes para previsão
                df = yf.download(
                    self.symbol,
                    period="720d",
                    auto_adjust=True
                )

            # ------------------------------------------------
            # Verificação básica de erro
            # ------------------------------------------------
            # Se por algum motivo o retorno for None,
            # devolve um DataFrame vazio
            if df is None:
                return pd.DataFrame()

            # ------------------------------------------------
            # Limpeza dos dados
            # ------------------------------------------------
            # Remove linhas com valores ausentes (NaN)
            # Isso evita erros nos cálculos do modelo de ML
            df = df.dropna()

            # Retorna o DataFrame pronto para uso
            return df

        except Exception as e:
            # ------------------------------------------------
            # Tratamento de erro
            # ------------------------------------------------
            # Se ocorrer qualquer erro:
            # - símbolo inválido
            # - falha de conexão
            # - erro inesperado
            #
            # Retorna um DataFrame vazio
            # A camada que chamou (ex: /train ou /predict)
            # decide como lidar com isso
            return pd.DataFrame()
