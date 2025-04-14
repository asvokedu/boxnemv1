# main.py
import os
import time
import threading
import pandas as pd
from datetime import datetime
from utils import calculate_technical_indicators
from collect_data import get_binance_klines, get_all_usdt_symbols
import joblib
from collections import defaultdict

MODEL_PATH = 'models'
INTERVAL = '1h'
SYMBOLS = get_all_usdt_symbols()

# Dictionary untuk menyimpan hasil prediksi global
prediction_results = defaultdict(list)

def analyze_symbol(symbol):
    try:
        end_time = datetime.utcnow()
        start_time = end_time - pd.Timedelta(hours=100)

        raw_klines = get_binance_klines(symbol, INTERVAL, start_time, end_time)
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            '_1', '_2', '_3', '_4', '_5', '_6'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df[['timestamp', 'symbol', 'close', 'volume']].astype({'close': float, 'volume': float})

        df = calculate_technical_indicators(df)

        if df.empty or df.isnull().any().any():
            print(f"⚠️ Data tidak cukup untuk {symbol}, dilewati.")
            return

        latest_data = df.iloc[-1:][['rsi', 'macd', 'signal_line', 'support', 'resistance']]
        model_file = os.path.join(MODEL_PATH, f"{symbol.replace('/', '')}_model.pkl")
        encoder_file = os.path.join(MODEL_PATH, f"{symbol.replace('/', '')}_label_encoder.pkl")

        if not os.path.exists(model_file) or not os.path.exists(encoder_file):
            print(f"⚠️ Model atau encoder tidak ditemukan untuk {symbol}.")
            return

        model = joblib.load(model_file)
        label_encoder = joblib.load(encoder_file)

        prediction_encoded = model.predict(latest_data)[0]
        prediction = label_encoder.inverse_transform([int(prediction_encoded)])[0]

        close_price = df.iloc[-1]['close']
        rsi = latest_data['rsi'].values[0]

        prediction_results[prediction].append({
            'symbol': symbol,
            'price': close_price,
            'rsi': rsi
        })

    except Exception as e:
        print(f"❌ Gagal menganalisis {symbol}: {e}")

def print_grouped_predictions():
    print("\n📈 Hasil Prediksi Kelompok:")
    for label, entries in prediction_results.items():
        print(f"\n📌 {label.upper()} ({len(entries)} aset)")
        for item in sorted(entries, key=lambda x: x['rsi'], reverse=True):
            print(f" - {item['symbol']} | Harga: {item['price']:.4f} | RSI: {item['rsi']:.2f}")
    print("\n==============================\n")

def run_analysis():
    global prediction_results
    prediction_results.clear()

    print("🔁 Memulai analisis real-time...")
    threads = []
    for symbol in SYMBOLS:
        thread = threading.Thread(target=analyze_symbol, args=(symbol,))
        thread.start()
        threads.append(thread)
        time.sleep(0.2)  # Batasi laju threading agar tidak overload

    for thread in threads:
        thread.join()

    print_grouped_predictions()

if __name__ == "__main__":
    while True:
        run_analysis()
        print("🕐 Menunggu 1 jam sebelum analisis berikutnya...")
        time.sleep(3600)
