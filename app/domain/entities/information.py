# app/domain/entities/information.py

# A Camada de Domínio é responsável por representar os conceitos de negócio.
# A classe 'Info' encapsula os dados necessários para uma consulta de preço de ações.

class Info:
    """
    Entidade de Domínio que armazena as informações básicas 
    necessárias para realizar uma consulta de dados financeiros.
    """
    
    # Método construtor da classe. É chamado quando um novo objeto Info é criado.
    def __init__(self, symbol: str, start_date: str | None = None, end_date: str | None = None):
        """
        Inicializa a entidade com o símbolo da ação e o período de tempo opcional.
        
        :param symbol: O ticker da ação (ex: 'AAPL', 'PETR4.SA').
        :param start_date: Data de início da consulta (opcional).
        :param end_date: Data de fim da consulta (opcional).
        """
        # Atribui o ticker da ação à instância.
        self.symbol = symbol
        # Atribui a data de início. O ' | None' indica que pode ser uma string ou None (ausente).
        self.start_date = start_date
        # Atribui a data de fim.
        self.end_date = end_date