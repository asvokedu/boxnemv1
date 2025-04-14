import os
import pandas as pd
from datetime import datetime, timedelta
from collect_data import get_all_usdt_symbols, get_binance_klines
from utils import calculate_technical_indicators, generate_label

# Ambil semua simbol USDT dari Binance
symbols = get_all_usdt_symbols()

# Waktu mulai dan akhir (1 tahun ke belakang dari sekarang)
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)

dataset = []

print("📥 Mengambil data historis 1 tahun untuk semua aset USDT...")
for symbol in symbols:
    try:
        print(f"🔍 Mengambil data untuk {symbol}...")
        raw_klines = get_binance_klines(symbol, "1h", start_time, end_time)

        # Skip jika data tidak cukup
        if len(raw_klines) < 100:
            raise ValueError("Data tidak cukup")

        # Proses data mentah menjadi DataFrame
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            '_1', '_2', '_3', '_4', '_5', '_6'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df[['timestamp', 'symbol', 'close', 'volume']].astype({'close': float, 'volume': float})

        # Hitung indikator teknikal
        df = calculate_technical_indicators(df)

        # Tambahkan label target
        df = generate_label(df)

        dataset.append(df)
    except Exception as e:
        print(f"❌ Gagal mengambil data untuk {symbol}: {e}")

# Gabungkan semua data simbol ke satu DataFrame
if dataset:
    final_df = pd.concat(dataset, ignore_index=True)

    # Buat folder training_data jika belum ada
    os.makedirs("training_data", exist_ok=True)

    # Simpan dataset ke folder training_data
    final_df.to_csv("training_data/training_dataset.csv", index=False)
    print("✅ Dataset pelatihan disimpan sebagai training_data/training_dataset.csv")
else:
    print("⚠️ Tidak ada data yang berhasil dikumpulkan.")
