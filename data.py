import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
 
 
TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META"]
 
PERIOD_MAP = {
    "1D": ("1d", "5m"),
    "1W": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}
 
 
def fetch_stock_data(ticker: str, period: str = "1M") -> pd.DataFrame:
    yf_period, interval = PERIOD_MAP.get(period, ("1mo", "1d"))
    stock = yf.Ticker(ticker)
    df = stock.history(period=yf_period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df
 
 
def fetch_multiple(tickers: list = TICKERS, period: str = "1M") -> dict:
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_stock_data(t, period)
        except Exception as e:
            print(f"Error fetching {t}: {e}")
    return result
 
 
def fetch_live_quote(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    hist = stock.history(period="2d", interval="1d")
    if len(hist) < 2:
        return {}
    prev_close = hist["Close"].iloc[-2]
    current = hist["Close"].iloc[-1]
    change = current - prev_close
    change_pct = (change / prev_close) * 100
    return {
        "ticker": ticker,
        "price": round(current, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(hist["Volume"].iloc[-1]),
        "high": round(hist["High"].iloc[-1], 2),
        "low": round(hist["Low"].iloc[-1], 2),
        "prev_close": round(prev_close, 2),
    }
 
 
def fetch_watchlist(tickers: list = TICKERS) -> list:
    quotes = []
    for t in tickers:
        try:
            q = fetch_live_quote(t)
            if q:
                quotes.append(q)
        except Exception as e:
            print(f"Error fetching quote for {t}: {e}")
    return quotes