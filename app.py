import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

from data import fetch_stock_data, fetch_watchlist, fetch_live_quote, TICKERS
from indicators import add_all_indicators, get_rsi_signal, get_trend_signal, get_bollinger_signal
from model import (
    train_model, predict_future, compare_models,
    get_feature_importance, get_risk_metrics, correlation_matrix,
)
from auth import check_auth, show_login_page, logout, get_user_watchlist, update_watchlist

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace !important; }
  .stApp { background: #0d0f14; }
  .block-container { padding: 1.5rem 2rem; max-width: 100%; }
  .metric-card {
    background: #1a1f2e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 18px 20px; height: 100%;
  }
  .metric-label { font-size: 12px; color: #8b93a8; margin-bottom: 6px; }
  .metric-value { font-size: 26px; font-weight: 700; color: #e8eaf0; letter-spacing: -0.5px; }
  .metric-sub   { font-size: 12px; margin-top: 6px; }
  .green  { color: #00c896 !important; }
  .red    { color: #ff4757 !important; }
  .cyan   { color: #00d4ff !important; }
  .amber  { color: #ffa502 !important; }
  .ticker-bar {
    background: #131720; border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 8px 24px; display: flex; gap: 32px; overflow: hidden;
    font-size: 12px; font-family: 'JetBrains Mono', monospace;
  }
  .ticker-item { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
  .section-title { font-size: 28px; font-weight: 700; color: #e8eaf0; margin-bottom: 4px; }
  .section-sub   { font-size: 13px; color: #8b93a8; margin-bottom: 20px; }
  .alert-card {
    background: #1e2535; border-radius: 8px; padding: 12px 14px;
    margin-bottom: 10px; border-left: 3px solid;
  }
  .stTabs [data-baseweb="tab-list"] { background: #131720; border-bottom: 1px solid rgba(255,255,255,0.07); gap: 0; }
  .stTabs [data-baseweb="tab"] { color: #8b93a8; font-size: 13px; padding: 14px 20px; background: transparent; border: none; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
  .stButton > button {
    background: #00d4ff; color: #000; font-weight: 700;
    border: none; border-radius: 8px; padding: 10px 24px;
    font-family: 'JetBrains Mono', monospace;
  }
  .stButton > button:hover { background: #00b8e6; }
  h1, h2, h3, h4 { color: #e8eaf0 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#131720",
    font=dict(family="JetBrains Mono", color="#8b93a8", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b93a8")),
)

# ── Auth check ────────────────────────────────────────────────────────────────
if not check_auth():
    show_login_page()
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
username = st.session_state.username

def metric_card(label, value, sub, sub_color="green"):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub {sub_color}">{sub}</div>
    </div>"""

def make_candle_fig(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#00c896", decreasing_line_color="#ff4757",
        increasing_fillcolor="#00c896", decreasing_fillcolor="#ff4757", name="Price"), row=1, col=1)
    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], line=dict(color="#00d4ff", width=1), name="SMA 20"), row=1, col=1)
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], line=dict(color="#ffa502", width=1), name="SMA 50"), row=1, col=1)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(color="rgba(124,111,247,0.5)", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(color="rgba(124,111,247,0.5)", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(124,111,247,0.05)", name="BB Lower"), row=1, col=1)
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color="#7c6ff7", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff4757", line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00c896", line_width=0.8, row=2, col=1)
    if "macd_hist" in df.columns:
        colors = ["#00c896" if v >= 0 else "#ff4757" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], marker_color=colors, name="MACD Hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="#00d4ff", width=1), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], line=dict(color="#ffa502", width=1), name="Signal"), row=3, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=520, showlegend=True,
                      yaxis2=dict(title="RSI", range=[0, 100], gridcolor="rgba(255,255,255,0.04)"),
                      yaxis3=dict(title="MACD", gridcolor="rgba(255,255,255,0.04)"))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# ── Top navbar ────────────────────────────────────────────────────────────────
nav1, nav2, nav3 = st.columns([3, 6, 3])
with nav1:
    st.markdown('<span style="color:#00d4ff;font-size:20px;font-weight:700;">↗ PredictTrade</span>', unsafe_allow_html=True)
with nav2:
    now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    st.markdown(f'<span style="color:#4a5168;font-size:12px;">🕐 {now}</span>', unsafe_allow_html=True)
with nav3:
    col_user, col_logout = st.columns([2, 1])
    with col_user:
        st.markdown(f'<span style="color:#00c896;">● Live</span>&nbsp;<span style="color:#8b93a8;font-size:12px;">👤 {username}</span>', unsafe_allow_html=True)
    with col_logout:
        if st.button("Logout", key="logout_btn"):
            logout()

# ── Live ticker bar ───────────────────────────────────────────────────────────
ticker_placeholder = st.empty()
items = ""
for t in TICKERS:
    try:
        q = fetch_live_quote(t)
        if q:
            col = "#00c896" if q["change"] >= 0 else "#ff4757"
            arrow = "▲" if q["change"] >= 0 else "▼"
            items += f'<span class="ticker-item"><span style="color:#e8eaf0;font-weight:700;">{t}</span>&nbsp;<span style="color:{col};">${q["price"]:.2f} {arrow} {abs(q["change_pct"]):.2f}%</span></span>'
    except:
        pass
ticker_placeholder.markdown(f'<div class="ticker-bar">{items}</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "ticker"       not in st.session_state: st.session_state.ticker       = "AAPL"
if "period"       not in st.session_state: st.session_state.period       = "1M"
if "auto_refresh" not in st.session_state: st.session_state.auto_refresh = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_market, tab_ai, tab_risk, tab_tech, tab_profile = st.tabs([
    "📊  Market Overview",
    "🧠  AI Predictions",
    "⚠  Risk Management",
    "📉  Technical Analysis",
    "👤  My Profile",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown('<div class="section-title">Market Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time market intelligence and portfolio performance monitoring</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 5, 3])
    with ctrl1:
        st.radio("Market", ["US", "International", "Crypto"], horizontal=True, label_visibility="collapsed")
    with ctrl2:
        period = st.radio("Time Range", ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"], horizontal=True, label_visibility="collapsed")
        st.session_state.period = period
    with ctrl3:
        ticker = st.selectbox("Stock", TICKERS, index=TICKERS.index(st.session_state.ticker), label_visibility="collapsed")
        st.session_state.ticker = ticker
        auto_ref = st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_ref

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Portfolio Value", "$2,847,392.50", "↗ +$47,283.20 (+1.69%)"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Daily P&L", "+$12,847.30", "↗ +$2,341.50 (+0.82%)"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Market Sentiment", "72/100", "↗ +5 pts (+7.46%)", "cyan"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Active Positions", "24", "↗ +3 (+14.29%)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, watch_col = st.columns([3, 1])

    with chart_col:
        with st.spinner("Loading price data..."):
            df = fetch_stock_data(ticker, period)
            df = add_all_indicators(df)
        st.markdown(f"**Price Chart** — {ticker} ({period})")
        st.plotly_chart(make_candle_fig(df), use_container_width=True)

    with watch_col:
        st.markdown(f"**Watchlist** — {username}'s stocks")
        user_watchlist = get_user_watchlist(username)
        for t in user_watchlist:
            try:
                q = fetch_live_quote(t)
                if q:
                    col = "#00c896" if q["change"] >= 0 else "#ff4757"
                    arrow = "↗" if q["change"] >= 0 else "↘"
                    st.markdown(f"""
                    <div style="background:#1a1f2e;border-left:3px solid {col};border-radius:8px;
                                padding:10px 12px;margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e8eaf0;font-weight:700;font-size:13px;">{t}</span>
                        <span style="color:{col};font-size:11px;">{arrow} {q['change_pct']:+.2f}%</span>
                      </div>
                      <div style="color:#e8eaf0;font-size:18px;font-weight:700;margin:4px 0;">${q['price']:.2f}</div>
                      <div style="display:flex;gap:12px;font-size:11px;color:#8b93a8;margin-top:4px;">
                        <span>H <span style="color:#00c896;">${q['high']}</span></span>
                        <span>L <span style="color:#ff4757;">${q['low']}</span></span>
                      </div>
                    </div>""", unsafe_allow_html=True)
            except:
                pass

    if auto_ref:
        time.sleep(30)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="section-title">AI Prediction Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Machine learning-powered market forecasts with explainable AI insights</div>', unsafe_allow_html=True)

    pred_col, settings_col = st.columns([3, 1])

    with settings_col:
        st.markdown("**Prediction Settings**")
        pred_ticker = st.selectbox("Stock", TICKERS, key="pred_ticker")
        model_name  = st.selectbox("ML Model", ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting", "SVR"], index=2)
        horizon     = st.slider("Forecast Days", 5, 30, 10)
        confidence  = st.slider("Confidence Threshold %", 0, 100, 75)
        st.markdown("---")
        st.markdown("**Model Status**")
        st.markdown('<span style="color:#00c896;">● Active</span>&nbsp;&nbsp;<span style="color:#4a5168;font-size:11px;">v2.5.1</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#8b93a8;font-size:12px;">Updated: {datetime.now().strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)
        st.button("↻ Refresh Predictions", use_container_width=True)

    with pred_col:
        with st.spinner("Training model..."):
            df_pred = fetch_stock_data(pred_ticker, "6M")
            df_pred = add_all_indicators(df_pred)
            model, scaler, metrics, features = train_model(df_pred, model_name, horizon)

        if model is not None:
            future         = predict_future(df_pred, model, scaler, features, horizon)
            history_prices = df_pred["close"].tail(30).values
            history_dates  = df_pred.index[-30:]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=history_dates, y=history_prices,
                                      line=dict(color="#00d4ff", width=2), name="Actual Price"))
            future_dates = pd.to_datetime(future["date"])
            fig2.add_trace(go.Scatter(x=future_dates, y=future["predicted"],
                                      line=dict(color="#7c6ff7", width=1.5, dash="dash"), name="Predicted"))
            fig2.add_trace(go.Scatter(x=future_dates, y=future["upper"],
                                      line=dict(color="rgba(0,200,150,0.5)", width=1), name="Upper Bound"))
            fig2.add_trace(go.Scatter(x=future_dates, y=future["lower"],
                                      line=dict(color="rgba(255,71,87,0.5)", width=1),
                                      fill="tonexty", fillcolor="rgba(0,200,150,0.07)", name="Lower Bound"))
            fig2.add_vline(x=str(history_dates[-1]), line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                               title=dict(text=f"{pred_ticker} · {horizon}-Day Forecast · {confidence}% Confidence",
                                          font=dict(color="#8b93a8", size=13)))
            st.plotly_chart(fig2, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            current_price = df_pred["close"].iloc[-1]
            pred_price    = future["predicted"].iloc[-1]
            pred_ret      = (pred_price - current_price) / current_price * 100
            with m1:
                st.markdown(metric_card("Current Price", f"${current_price:.2f}", "Live price"), unsafe_allow_html=True)
            with m2:
                c = "green" if pred_ret >= 0 else "red"
                st.markdown(metric_card("Predicted", f"${pred_price:.2f}", f"{pred_ret:+.1f}% expected", c), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card("Accuracy", f"{metrics.get('accuracy', 0):.1f}%", f"R²={metrics.get('r2', 0):.3f}", "cyan"), unsafe_allow_html=True)
            with m4:
                risk = get_risk_metrics(df_pred)
                c = "red" if risk["risk_level"] == "High" else "amber" if risk["risk_level"] == "Moderate" else "green"
                st.markdown(metric_card("Risk Level", risk["risk_level"], f"Vol: {risk['annual_volatility']:.1f}%", c), unsafe_allow_html=True)

            st.markdown("<br>**Feature Importance**", unsafe_allow_html=True)
            fi = get_feature_importance(model, features)
            fig_fi = px.bar(fi.head(10), x="importance", y="feature", orientation="h",
                            color="importance", color_continuous_scale=["#7c6ff7", "#00d4ff"],
                            labels={"importance": "Importance %", "feature": ""})
            fig_fi.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
                                 height=280, coloraxis_showscale=False,
                                 yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("**Model Comparison**")
            with st.spinner("Comparing models..."):
                comparison = compare_models(df_pred, horizon)
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            st.markdown("**Forecast Table**")
            st.dataframe(future, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="section-title">Risk Management Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comprehensive portfolio risk assessment and monitoring tools</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        risk_tickers = st.multiselect("Portfolio Stocks", TICKERS, default=["AAPL", "MSFT", "TSLA", "GOOGL"])
    with rc2:
        st.selectbox("Calculation Method", ["Value at Risk (VaR)", "Expected Shortfall", "Monte Carlo"])
    with rc3:
        st.selectbox("Lookback Period", ["30 Days", "60 Days", "90 Days", "180 Days", "1 Year"])

    if st.button("Recalculate Risk"):
        st.cache_data.clear()

    if risk_tickers:
        with st.spinner("Loading portfolio data..."):
            port_data = {}
            for t in risk_tickers:
                try:
                    port_data[t] = add_all_indicators(fetch_stock_data(t, "6M"))
                except:
                    pass

        primary = port_data.get(risk_tickers[0])
        if primary is not None:
            risk = get_risk_metrics(primary)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(metric_card("Portfolio Beta", str(risk["beta"]), "vs benchmark", "amber"), unsafe_allow_html=True)
            with r2:
                c = "green" if risk["sharpe_ratio"] > 1 else "amber"
                st.markdown(metric_card("Sharpe Ratio", str(risk["sharpe_ratio"]), f"Ann. Return: {risk['annual_return']:.1f}%", c), unsafe_allow_html=True)
            with r3:
                st.markdown(metric_card("Max Drawdown", f"{risk['max_drawdown']:.1f}%", "Peak-to-trough", "red"), unsafe_allow_html=True)
            with r4:
                c = "red" if risk["annual_volatility"] > 30 else "amber" if risk["annual_volatility"] > 20 else "green"
                st.markdown(metric_card("Volatility", f"{risk['annual_volatility']:.1f}%", risk["risk_level"] + " Risk", c), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            corr_col, alert_col = st.columns([3, 2])

            with corr_col:
                st.markdown("**Correlation Matrix**")
                corr_df = correlation_matrix(port_data)
                if not corr_df.empty:
                    fig_corr = px.imshow(corr_df, color_continuous_scale=["#7c6ff7", "#1a1f2e", "#ff4757"],
                                         zmin=-1, zmax=1, text_auto=True, aspect="auto")
                    fig_corr.update_layout(**PLOTLY_LAYOUT, height=340)
                    st.plotly_chart(fig_corr, use_container_width=True)

            with alert_col:
                st.markdown("**Risk Alerts**")
                for title, desc, col, impact in [
                    ("Correlation Spike", "TSLA & NVDA correlation 0.89 above avg 0.48", "#ff4757", "Critical"),
                    ("Volatility Warning", "Portfolio vol exceeds 25% threshold.", "#ffa502", "High"),
                    ("Drawdown Alert", "TSLA shows 8.2% drawdown from recent high.", "#ffa502", "High"),
                    ("RSI Extreme", "AAPL RSI 74.2 — potential overbought.", "#00c896", "Low"),
                ]:
                    st.markdown(f"""
                    <div class="alert-card" style="border-left-color:{col}">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:{col};font-size:12px;font-weight:700;">△ {title}</span>
                        <span style="color:#4a5168;font-size:10px;">{impact}</span>
                      </div>
                      <p style="color:#8b93a8;font-size:11px;margin:0;line-height:1.5;">{desc}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Rolling Volatility**", unsafe_allow_html=True)
            vol_fig = go.Figure()
            for t, df_t in port_data.items():
                if "volatility" in df_t.columns:
                    vol_fig.add_trace(go.Scatter(x=df_t.index, y=df_t["volatility"], name=t, mode="lines"))
            vol_fig.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Annualised Volatility (%)")
            st.plotly_chart(vol_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    st.markdown('<div class="section-title">Technical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Advanced charting with pattern detection and signal generation</div>', unsafe_allow_html=True)

    ta_c1, ta_c2, ta_c3, ta_c4 = st.columns([2, 2, 2, 2])
    with ta_c1:
        ta_ticker = st.selectbox("Symbol", TICKERS, key="ta_ticker")
    with ta_c2:
        ta_period = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M", "6M"], index=2, key="ta_period")
    with ta_c3:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"])
    with ta_c4:
        indicators_sel = st.multiselect("Indicators",
            ["SMA 20", "SMA 50", "Bollinger Bands", "RSI", "MACD", "Volume"],
            default=["SMA 20", "SMA 50", "RSI", "MACD"])

    with st.spinner("Loading technical data..."):
        df_ta = add_all_indicators(fetch_stock_data(ta_ticker, ta_period))

    last_close = df_ta["close"].iloc[-1]
    last_rsi   = df_ta["rsi"].iloc[-1] if "rsi" in df_ta.columns else 50
    last_sma20 = df_ta["sma_20"].iloc[-1] if "sma_20" in df_ta.columns else last_close
    last_sma50 = df_ta["sma_50"].iloc[-1] if "sma_50" in df_ta.columns else last_close
    last_vol   = df_ta["volatility"].iloc[-1] if "volatility" in df_ta.columns else 0
    macd_val   = df_ta["macd"].iloc[-1] if "macd" in df_ta.columns else 0

    rsi_sig, _   = get_rsi_signal(last_rsi)
    trend_sig, _ = get_trend_signal(last_close, last_sma20, last_sma50)

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(metric_card("Last Price", f"${last_close:.2f}", f"SMA20: ${last_sma20:.2f}"), unsafe_allow_html=True)
    with s2:
        c = "green" if trend_sig == "Bullish" else "red" if trend_sig == "Bearish" else "amber"
        st.markdown(metric_card("Trend", trend_sig, f"SMA50: ${last_sma50:.2f}", c), unsafe_allow_html=True)
    with s3:
        c = "red" if rsi_sig == "Overbought" else "green" if rsi_sig == "Oversold" else "cyan"
        st.markdown(metric_card("RSI Signal", rsi_sig, f"RSI: {last_rsi:.1f}", c), unsafe_allow_html=True)
    with s4:
        bb_sig = get_bollinger_signal(last_close, df_ta["bb_upper"].iloc[-1], df_ta["bb_lower"].iloc[-1]) if "bb_upper" in df_ta.columns else "N/A"
        st.markdown(metric_card("Bollinger", bb_sig, f"Vol: {last_vol:.1f}%"), unsafe_allow_html=True)
    with s5:
        c = "green" if macd_val > 0 else "red"
        st.markdown(metric_card("MACD", f"{macd_val:.3f}", "Bullish" if macd_val > 0 else "Bearish", c), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rows = 1 + ("RSI" in indicators_sel) + ("MACD" in indicators_sel) + ("Volume" in indicators_sel)
    fig_ta = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                           row_heights=[0.5] + [0.17] * (rows - 1), vertical_spacing=0.02)

    if chart_type == "Candlestick":
        fig_ta.add_trace(go.Candlestick(x=df_ta.index, open=df_ta["open"], high=df_ta["high"],
                                         low=df_ta["low"], close=df_ta["close"],
                                         increasing_line_color="#00c896", decreasing_line_color="#ff4757",
                                         increasing_fillcolor="#00c896", decreasing_fillcolor="#ff4757", name="Price"), row=1, col=1)
    elif chart_type == "Line":
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["close"], line=dict(color="#00d4ff", width=2), name="Close"), row=1, col=1)
    else:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["close"], fill="tozeroy",
                                     fillcolor="rgba(0,212,255,0.08)", line=dict(color="#00d4ff", width=2), name="Close"), row=1, col=1)

    if "SMA 20" in indicators_sel and "sma_20" in df_ta.columns:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["sma_20"], line=dict(color="#00d4ff", width=1), name="SMA 20"), row=1, col=1)
    if "SMA 50" in indicators_sel and "sma_50" in df_ta.columns:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["sma_50"], line=dict(color="#ffa502", width=1), name="SMA 50"), row=1, col=1)
    if "Bollinger Bands" in indicators_sel and "bb_upper" in df_ta.columns:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["bb_upper"], line=dict(color="rgba(124,111,247,0.5)", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["bb_lower"], line=dict(color="rgba(124,111,247,0.5)", width=1, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(124,111,247,0.05)", name="BB Lower"), row=1, col=1)

    cur_row = 2
    if "RSI" in indicators_sel and "rsi" in df_ta.columns:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["rsi"], line=dict(color="#7c6ff7", width=1.5), name="RSI"), row=cur_row, col=1)
        fig_ta.add_hline(y=70, line_dash="dot", line_color="#ff4757", line_width=0.8, row=cur_row, col=1)
        fig_ta.add_hline(y=30, line_dash="dot", line_color="#00c896", line_width=0.8, row=cur_row, col=1)
        cur_row += 1
    if "MACD" in indicators_sel and "macd_hist" in df_ta.columns:
        colors = ["#00c896" if v >= 0 else "#ff4757" for v in df_ta["macd_hist"]]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta["macd_hist"], marker_color=colors, name="MACD Hist"), row=cur_row, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["macd"], line=dict(color="#00d4ff", width=1), name="MACD"), row=cur_row, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["macd_signal"], line=dict(color="#ffa502", width=1), name="Signal"), row=cur_row, col=1)
        cur_row += 1
    if "Volume" in indicators_sel and "volume" in df_ta.columns:
        vol_colors = ["#00c896" if df_ta["close"].iloc[i] >= df_ta["open"].iloc[i] else "#ff4757" for i in range(len(df_ta))]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta["volume"], marker_color=vol_colors, name="Volume"), row=cur_row, col=1)

    fig_ta.update_layout(**PLOTLY_LAYOUT, height=620 + rows * 60, showlegend=True)
    fig_ta.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_ta, use_container_width=True)

    st.markdown("**Pattern Alerts**")
    p1, p2, p3, p4 = st.columns(4)
    for col_obj, sym, pattern, conf, color in [
        (p1, ta_ticker, "Golden Cross",     "87% confidence", "#00c896"),
        (p2, "TSLA",    "Head & Shoulders",  "92% confidence", "#ff4757"),
        (p3, "MSFT",    "Cup & Handle",      "78% confidence", "#ffa502"),
        (p4, "GOOGL",   "Triangle Breakout", "65% confidence", "#8b93a8"),
    ]:
        with col_obj:
            st.markdown(f"""
            <div style="background:#1a1f2e;border-left:3px solid {color};border-radius:8px;padding:12px 14px;">
              <div style="color:{color};font-size:12px;font-weight:700;">{sym}</div>
              <div style="color:#8b93a8;font-size:11px;margin:4px 0;">{pattern}</div>
              <div style="color:#e8eaf0;font-size:11px;">{conf}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MY PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown('<div class="section-title">My Profile</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Welcome back, {username}!</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;padding:30px;">
          <div style="width:70px;height:70px;border-radius:50%;background:#7c6ff7;
                      display:flex;align-items:center;justify-content:center;
                      font-size:28px;font-weight:700;color:#fff;margin:0 auto 16px;">
            {username[0].upper()}
          </div>
          <div style="font-size:18px;font-weight:700;color:#e8eaf0;">{username}</div>
          <div style="font-size:12px;color:#8b93a8;margin-top:4px;">
            {st.session_state.user_data.get('email', '')}
          </div>
          <div style="font-size:11px;color:#4a5168;margin-top:8px;">
            Member since {st.session_state.user_data.get('created_at', 'N/A')}
          </div>
        </div>""", unsafe_allow_html=True)

    with p2:
        st.markdown("**My Watchlist**")
        user_watchlist = get_user_watchlist(username)
        new_watchlist  = st.multiselect("Edit your watchlist", TICKERS, default=user_watchlist)
        if st.button("Save Watchlist"):
            update_watchlist(username, new_watchlist)
            st.success("Watchlist updated!")

        st.markdown("<br>**Account Stats**", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(metric_card("Watchlist", str(len(user_watchlist)), "stocks tracked", "cyan"), unsafe_allow_html=True)
        with a2:
            st.markdown(metric_card("Session", "Active", "logged in", "green"), unsafe_allow_html=True)
        with a3:
            st.markdown(metric_card("Data", "Live", "Yahoo Finance", "green"), unsafe_allow_html=True)