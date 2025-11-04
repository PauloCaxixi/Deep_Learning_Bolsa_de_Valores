import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.db_utils import read_from_db, save_to_db

def preprocess_data(table_name="raw_petr4"):
    df = read_from_db(table_name)

    # Garante nomes de colunas minúsculos e sem espaços
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    # Confere se existe alguma coluna parecida com "date"
    date_col = None
    for c in df.columns:
        if "date" in c:
            date_col = c
            break

    if not date_col:
        raise KeyError("Nenhuma coluna de data encontrada no dataframe!")

    # Ordena corretamente
    df = df.sort_values(date_col)

    # Seleciona apenas colunas numéricas para normalizar
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric_df)
    df_scaled = pd.DataFrame(scaled, columns=numeric_df.columns)

    # Mantém a coluna de data no resultado final
    df_scaled[date_col] = df[date_col].values

    # (Opcional) Salva tabela processada no banco
    save_to_db(df_scaled, "scaled_petr4")
    print(f"Tabela 'scaled_petr4' salva com sucesso ({len(df_scaled)} registros).")

    return df_scaled, scaler
