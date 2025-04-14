# ==== utils.py ====
import pandas as pd
import numpy as np
import ta

def calculate_technical_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    df['support'] = df['close'].rolling(window=20).min()
    df['resistance'] = df['close'].rolling(window=20).max()
    df.dropna(inplace=True)
    return df

def generate_label(df, threshold=0.02):
    df = df.copy()
    df['future_close'] = df['close'].shift(-6)
    df['price_change'] = (df['future_close'] - df['close']) / df['close']

    conditions = [
        (df['price_change'] >= threshold),
        (df['price_change'] <= -threshold)
    ]
    choices = ['AGGRESSIVE BUY', 'SELL']
    df['label'] = np.select(conditions, choices, default='WAIT')
    df.drop(columns=['future_close', 'price_change'], inplace=True)
    return df
