import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
 
 
FEATURE_COLS = [
    "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_upper", "bb_mid", "bb_lower",
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "atr", "volatility",
    "volume", "open", "high", "low",
]
 
MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR":               SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
}
 
 
def prepare_features(df: pd.DataFrame, horizon: int = 1):
    df = df.copy().dropna()
    df["target"] = df["close"].shift(-horizon)
    df = df.dropna()
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["target"].values
    return X, y, available
 
 
def train_model(df: pd.DataFrame, model_name: str = "Random Forest", horizon: int = 5):
    X, y, features = prepare_features(df, horizon)
    if len(X) < 30:
        return None, None, {}, features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = MODELS.get(model_name, MODELS["Random Forest"])
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    metrics = {
        "mae":      round(mean_absolute_error(y_test, preds), 4),
        "rmse":     round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "r2":       round(r2_score(y_test, preds), 4),
        "accuracy": round(max(0, r2_score(y_test, preds)) * 100, 2),
    }
    return model, scaler, metrics, features
 
 
def predict_future(df: pd.DataFrame, model, scaler, features: list, days: int = 10):
    df = df.copy().dropna()
    available = [c for c in features if c in df.columns]
    last_row   = df[available].iloc[-1].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)
    predictions, upper_bounds, lower_bounds = [], [], []
    current = last_scaled.copy()
    last_close = df["close"].iloc[-1]
    std_pct = df["close"].pct_change().std()
    price = last_close
    for i in range(days):
        pred = float(model.predict(current)[0])
        noise = price * std_pct * np.random.randn() * 0.3
        price = pred + noise
        margin = price * std_pct * 2
        predictions.append(round(price, 2))
        upper_bounds.append(round(price + margin, 2))
        lower_bounds.append(round(price - margin, 2))
        current = current * (1 + np.random.randn(current.shape[1]) * 0.001)
    dates = pd.bdate_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({
        "date":      [d.strftime("%Y-%m-%d") for d in dates],
        "predicted": predictions,
        "upper":     upper_bounds,
        "lower":     lower_bounds,
    })
 
 
def compare_models(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    results = []
    for name in MODELS:
        try:
            _, _, metrics, _ = train_model(df, name, horizon)
            results.append({"Model": name, **metrics})
        except Exception as e:
            results.append({"Model": name, "mae": None, "rmse": None, "r2": None, "accuracy": None})
    return pd.DataFrame(results).sort_values("r2", ascending=False)
 
 
def get_feature_importance(model, features: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame({"feature": features, "importance": [0] * len(features)})
    imp_pct = (imp / imp.sum() * 100).round(2)
    return pd.DataFrame({"feature": features, "importance": imp_pct}).sort_values("importance", ascending=False)
 
 
def get_risk_metrics(df: pd.DataFrame) -> dict:
    close = df["close"].dropna()
    returns = close.pct_change().dropna()
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.05) / annual_vol if annual_vol != 0 else 0
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    max_dd = drawdown.min()
    beta = annual_vol / 0.18
    return {
        "annual_return": round(annual_return * 100, 2),
        "annual_volatility": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "beta": round(beta, 2),
        "risk_level": "High" if annual_vol > 0.30 else "Moderate" if annual_vol > 0.20 else "Low",
    }
 
 
def correlation_matrix(data_dict: dict) -> pd.DataFrame:
    closes = {}
    for ticker, df in data_dict.items():
        if "close" in df.columns and len(df) > 10:
            closes[ticker] = df["close"]
    if not closes:
        return pd.DataFrame()
    combined = pd.DataFrame(closes).dropna()
    return combined.corr().round(2)