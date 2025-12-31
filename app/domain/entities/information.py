# ==========================================
# app/domain/entities/information.py
# ==========================================

# --------------------------------------------------
# CAMADA DE DOMÍNIO
# --------------------------------------------------
# A camada de domínio representa os conceitos do negócio.
# Ela NÃO acessa banco de dados, NÃO chama APIs externas
# e NÃO contém regras técnicas.
#
# Aqui ficam apenas estruturas que representam
# informações importantes do problema.
# --------------------------------------------------


class Info:
    """
    Entidade de Domínio.

    Esta classe representa as informações básicas
    necessárias para consultar dados financeiros
    de uma ação (ticker + período).
    """

    def __init__(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None
    ):
        """
        Construtor da entidade Info.

        Ele é chamado sempre que um novo objeto Info é criado.

        :param symbol: Código da ação (ex: 'AAPL', 'PETR4.SA')
        :param start_date: Data inicial da consulta (opcional)
        :param end_date: Data final da consulta (opcional)
        """

        # ------------------------------------------
        # Símbolo da ação
        # ------------------------------------------
        # Exemplo:
        #   "AAPL"       -> Apple (NASDAQ)
        #   "PETR4.SA"   -> Petrobras (B3 Brasil)
        self.symbol = symbol

        # ------------------------------------------
        # Data de início da consulta
        # ------------------------------------------
        # Pode ser:
        # - uma string no formato 'YYYY-MM-DD'
        # - None (quando não se deseja limitar por data)
        self.start_date = start_date

        # ------------------------------------------
        # Data de fim da consulta
        # ------------------------------------------
        # Também pode ser:
        # - uma string no formato 'YYYY-MM-DD'
        # - None
        self.end_date = end_date
