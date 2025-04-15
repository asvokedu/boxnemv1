# main.py
import os
import time
import threading
import subprocess
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

def ensemble_predict(symbol, latest_data):
    try:
        name = symbol.replace("/", "")
        model_path = os.path.join(MODEL_PATH, f"{name}_model.pkl")
        model_new_path = os.path.join(MODEL_PATH, f"{name}_model_new.pkl")
        encoder_path = os.path.join(MODEL_PATH, f"{name}_label_encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return None

        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)

        if os.path.exists(model_new_path):
            model_new = joblib.load(model_new_path)
            pred1 = model.predict_proba(latest_data)
            pred2 = model_new.predict_proba(latest_data)
            ensemble_pred = (pred1 + pred2) / 2
            pred_class = ensemble_pred.argmax(axis=1)[0]
        else:
            pred_class = model.predict(latest_data)[0]

        prediction = label_encoder.inverse_transform([int(pred_class)])[0]
        return prediction
    except Exception as e:
        print(f"⚠️ Gagal ensemble predict untuk {symbol}: {e}")
        return None

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
        prediction = ensemble_predict(symbol, latest_data)

        if prediction is None:
            print(f"⚠️ Prediksi gagal untuk {symbol}.")
            return

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

def prepare_dataset_and_model():
    if not os.path.exists("training_data/training_dataset.csv"):
        print("📦 Dataset belum ditemukan. Menjalankan generate_training_dataset.py...")
        subprocess.run(["python3", "generate_training_dataset.py"])
    else:
        print("✅ Dataset sudah tersedia.")

    print("🧠 Melatih model...")
    subprocess.run(["python3", "train_model.py"])

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
    prepare_dataset_and_model()

    while True:
        run_analysis()
        print("🕐 Menunggu 1 jam sebelum analisis berikutnya...")
        time.sleep(3600)
