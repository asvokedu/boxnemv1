import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from collect_data import get_all_usdt_symbols
from utils import calculate_technical_indicators, generate_label

# Fungsi untuk mengambil data historis klines (candlestick) dari Binance
def get_binance_klines(symbol, interval, start_time, end_time):
    """
    Mengambil data kline (candlestick) dari Binance dalam rentang waktu tertentu
    dan secara otomatis melakukan pagination jika data lebih dari 1000 candle.
    """
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    data = []

    # Konversi ke timestamp dalam milidetik
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            data.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.1)  # Hindari rate-limit

        except requests.exceptions.RequestException as e:
            raise Exception(f"📡 Gagal mengambil data Binance untuk {symbol}: {e}")

    return data


# Ambil semua simbol USDT dari Binance
symbols = get_all_usdt_symbols()

# Setup waktu sekarang
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)
min_candle_count = 720

# Cek apakah dataset lama sudah ada
existing_data_path = "training_data/training_dataset.csv"
if os.path.exists(existing_data_path):
    print("📂 Dataset lama ditemukan. Memuat dan memperbarui data...")
    existing_df = pd.read_csv(existing_data_path, parse_dates=['timestamp'])
else:
    existing_df = pd.DataFrame()
    print("📄 Tidak ditemukan dataset lama. Mengambil data 1 tahun penuh...")

dataset = []

# Loop untuk tiap simbol
for symbol in symbols:
    try:
        print(f"🔍 Memproses simbol {symbol}...")

        # Tentukan waktu mulai berdasarkan data lama jika ada
        if not existing_df.empty and symbol in existing_df['symbol'].unique():
            last_time = existing_df[existing_df['symbol'] == symbol]['timestamp'].max()
            start_time_symbol = pd.to_datetime(last_time) + timedelta(hours=1)
        else:
            start_time_symbol = start_time

        # Ambil data baru
        raw_klines = get_binance_klines(symbol, "1h", start_time_symbol, end_time)

        # Filter simbol yang belum listing minimal 30 hari
        if len(raw_klines) < min_candle_count and existing_df.empty:
            raise ValueError("Data tidak mencukupi (kurang dari 30 hari)")

        if not raw_klines:
            print(f"⚠️ Tidak ada data baru untuk {symbol}")
            continue

        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            '_1', '_2', '_3', '_4', '_5', '_6'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df[['timestamp', 'symbol', 'close', 'volume']].astype({'close': float, 'volume': float})

        # Hitung indikator teknikal dan label
        df = calculate_technical_indicators(df)
        df = generate_label(df)

        dataset.append(df)

    except Exception as e:
        print(f"❌ Gagal mengambil data untuk {symbol}: {e}")

# Simpan atau update dataset
if dataset:
    new_df = pd.concat(dataset, ignore_index=True)

    # Gabungkan dengan data lama jika ada
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['timestamp', 'symbol'], inplace=True)
    else:
        combined_df = new_df

    os.makedirs("training_data", exist_ok=True)
    combined_df.to_csv("training_data/training_dataset.csv", index=False)
    print("✅ Dataset pelatihan diperbarui dan disimpan ke training_data/training_dataset.csv")
else:
    print("⚠️ Tidak ada data baru yang berhasil dikumpulkan.")
