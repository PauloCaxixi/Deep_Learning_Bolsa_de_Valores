import os
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend headless para Docker/servidor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import time

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def make_prediction(symbol, start_date, end_date):
    try:
        print(f"🔍 Buscando dados para {symbol} de {start_date} até {end_date}")
        acao = yf.Ticker(symbol)
        df = acao.history(start=start_date, end=end_date)

        if df.empty:
            return "Dados não encontrados para esse símbolo."

        close_prices = df['Close'].values.reshape(-1, 1)
        info = acao.info

        last_price = float(close_prices[-1][0])
        volume = int(info.get("volume", int(df['Volume'].iloc[-1])))
        open_price = float(info.get("open", float(df['Open'].iloc[-1])))
        high = float(info.get("dayHigh", float(df['High'].max())))
        low = float(info.get("dayLow", float(df['Low'].min())))
        pe_ratio = info.get("trailingPE")
        dividend_yield = info.get("dividendYield")
        target_price = info.get("targetMeanPrice")

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        window_size = 60
        X, y = create_sequences(scaled_data, window_size)
        if len(X) == 0:
            return "Dados insuficientes para treinamento."

        model = Sequential([
            Input(shape=(window_size, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        model.save('lstm_model.keras')

        last_sequence = X[-1].reshape(1, window_size, 1)
        prediction = model.predict(last_sequence)
        predicted_price = float(scaler.inverse_transform(prediction)[0][0])

        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)

        try:
            print("📈 Gerando gráfico...")
            plt.figure(figsize=(16, 6))
            plt.plot(close_prices, label='Histórico de Preço', color='blue')
            plt.plot(len(close_prices) - 1, last_price, 'bo')  # último ponto real
            plt.plot(len(close_prices), predicted_price, 'ro', label='Previsão')
            plt.title(f'{symbol} | Previsão de Preço', fontsize=16)
            plt.xlabel('Dias', fontsize=12)
            plt.ylabel('Preço de Fechamento (R$)', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True)
            plt.legend()

            lines = [
                f"Último Preço: R$ {last_price:.2f}",
                f"Previsão: R$ {predicted_price:.2f}",
                f"Volume: {volume:,}",
                f"Abertura: R$ {open_price:.2f}",
                f"Máxima: R$ {high:.2f}",
                f"Mínima: R$ {low:.2f}",
                f"P/E Ratio: {round(pe_ratio, 2) if pe_ratio else 'N/A'}",
                f"Dividend Yield: {round(dividend_yield * 100, 2)}%" if dividend_yield else "Dividend Yield: N/A",
                f"Preço-Alvo 1Y: R$ {round(target_price, 2)}" if target_price else "Preço-Alvo 1Y: N/A"
            ]
            textstr = "\n".join(lines)
            plt.gcf().text(0.72, 0.5, textstr, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))

            plt.tight_layout()
            # Nome único para evitar cache
            fname = f"plot_{int(time.time())}.png"
            plot_path = os.path.join(static_dir, fname)
            print(f"💾 Salvando gráfico em: {plot_path}")
            plt.savefig(plot_path)
            plt.close()

            # Verificação real
            exists = os.path.isfile(plot_path)
            print(f"🧪 Arquivo existe? {exists}")
            if not exists:
                return "Falha ao salvar o gráfico."

            return {
                "prediction": round(predicted_price, 2),
                "last_price": round(last_price, 2),
                "volume": volume,
                "open_price": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
                "target_price": round(target_price, 2) if target_price else None,
                "plot_filename": fname  # passa o nome ao template
            }

        except Exception as e:
            print(f"❌ Erro ao gerar gráfico: {e}")
            return f"Erro ao processar gráfico: {str(e)}"

    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return f"Erro ao processar: {str(e)}"