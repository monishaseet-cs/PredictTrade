import pandas as pd
import numpy as np
 
 
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()
 
 
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()
 
 
def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower
 
 
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
 
 
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
 
 
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d
 
 
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()
 
 
def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cumulative_tpv = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative_tpv / cumulative_vol
 
 
def annualised_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    return rolling_std * np.sqrt(252) * 100
 
 
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_20"] = sma(df["close"], 20)
    df["sma_50"] = sma(df["close"], 50)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"])
    df["rsi"] = rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["stoch_k"], df["stoch_d"] = stochastic(df["high"], df["low"], df["close"])
    df["atr"] = atr(df["high"], df["low"], df["close"])
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    df["volatility"] = annualised_volatility(df["close"])
    return df
 
 
def get_rsi_signal(rsi_val: float) -> tuple:
    if rsi_val > 70:
        return "Overbought", "red"
    elif rsi_val < 30:
        return "Oversold", "green"
    return "Neutral", "gray"
 
 
def get_trend_signal(close: float, sma20: float, sma50: float) -> tuple:
    if close > sma20 and sma20 > sma50:
        return "Bullish", "green"
    elif close < sma20 and sma20 < sma50:
        return "Bearish", "red"
    return "Sideways", "orange"
 
 
def get_bollinger_signal(close: float, upper: float, lower: float) -> str:
    if close > upper:
        return "Overbought"
    elif close < lower:
        return "Oversold"
    return "Within Bands"