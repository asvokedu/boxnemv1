# ==== collect_data.py ====
import requests
import time
import datetime

def get_all_usdt_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    symbols = []
    for s in response.json()['symbols']:
        if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
            symbols.append(s['symbol'])
    return symbols

def get_binance_klines(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    df_total = []
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        df_total.extend(data)
        start_ts = data[-1][0] + 1
        time.sleep(0.5)
    return df_total
