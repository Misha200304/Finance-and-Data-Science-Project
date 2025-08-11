# --- S&P 500 ML App (TS-CV, Quantiles, Feature Upgrades, Two-Stock Compare) ---
import base64
from datetime import date, timedelta
import json
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- UI header ----------------
st.set_page_config(page_title="S&P 500 ML App", layout="wide")
st.title("S&P 500 App")
st.caption("Overview • ML Forecast • Free AI Suggestion (Ollama) — educational, not investment advice")

# ---------------- Small helpers ----------------
def _sort_cols_desc(df):
    if df is None or df.empty: return df
    cols = list(df.columns)
    try:
        dts = pd.to_datetime(cols, errors="coerce")
        order = pd.Series(range(len(cols)), index=cols)
        order.loc[~dts.isna()] = pd.Series(dts[~dts.isna()]).rank(ascending=False, method="first").values
        cols = sorted(cols, key=lambda c: order[c])
    except Exception:
        pass
    return df[cols]

def _row_latest(df, names):
    if df is None or df.empty: return None
    df = _sort_cols_desc(df)
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if not s.empty: return float(s.iloc[0])
    return None

def _row_sum_last(df, names, k):
    if df is None or df.empty: return None
    df = _sort_cols_desc(df)
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s) >= 1: return float(s.iloc[:k].sum())
    return None

def _latest_col_date(df):
    if df is None or df.empty: return None
    try:
        dts = pd.to_datetime(df.columns, errors="coerce")
        dts = dts[~dts.isna()]
        if len(dts) == 0: return None
        return dts.max().date().isoformat()
    except Exception:
        return None

def compute_rsi(s, window=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = -d.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def compute_obv(close, volume):
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * volume).cumsum()
    return obv

# ---------------- Free AI via Ollama ----------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"

def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False

def ai_verdict_ollama(symbol, metrics) -> dict | None:
    prompt = f"""
You are an equity research assistant. Based ONLY on the short-term signal and risk/fundamentals below,
return JSON with:
- verdict: "Buy", "Hold", or "Avoid"
- rationale: 2-4 short bullet points.

Data:
Predicted 5d return: {metrics['pred_ret']:.2%}
5d target price: {metrics['target_price']:.2f}
5d range (low–high): {metrics['low_price']:.2f} – {metrics['high_price']:.2f}
Holdout MAE (5d): {metrics['mae']:.2%}
Holdout R²: {metrics['r2']:.2f}
Ann. volatility: {metrics['ann_vol']:.2%}
Sharpe (rf=0): {metrics['sharpe0']:.2f}
Revenue growth YoY: {('%.2f%%' % (metrics['rev_growth']*100)) if metrics['rev_growth'] is not None else 'N/A'}
ROE: {('%.0f%%' % (metrics['roe']*100)) if metrics['roe'] is not None else 'N/A'}
Debt/Equity: {('%.2f' % metrics['de']) if metrics['de'] is not None else 'N/A'}

Rules:
- 'Buy' only if predicted 5d return is clearly positive and the band isn't heavily negative.
- 'Hold' if signal is weak/uncertain or band straddles zero.
- 'Avoid' if predicted return is negative or risk looks high.
Return ONLY valid JSON. No prose.
""".strip()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are precise. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        return json.loads(text)
    except Exception:
        return None

def rule_based_verdict(metrics) -> dict:
    pred = metrics["pred_ret"]
    low5 = (metrics["low_price"] / metrics["last_price"]) - 1
    high5 = (metrics["high_price"] / metrics["last_price"]) - 1
    r2 = metrics["r2"]; mae = metrics["mae"]
    if pred is None:
        v, why = "Hold", ["No signal available"]
    elif pred > 0.01 and low5 > -0.02 and r2 > -0.05 and mae < 0.05:
        v, why = "Buy", ["Positive 5d signal", "Downside limited in band", "Model error acceptable"]
    elif pred < -0.005 and high5 < 0.02:
        v, why = "Avoid", ["Negative short-term signal", "Upside limited in band"]
    else:
        v, why = "Hold", ["Signal weak/uncertain", "Band straddles zero or error high"]
    return {"verdict": v, "rationale": why}

# ---------------- Data loaders ----------------
@st.cache_data
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for t in pd.read_html(url, header=0):
        if any(c in t.columns for c in ["Symbol","Ticker symbol"]):
            df = t.copy(); break
    df.columns = (df.columns.str.strip()
                  .str.replace(r"\[.*\]", "", regex=True)
                  .str.replace("\xa0"," ", regex=False))
    df = df.rename(columns={"Ticker symbol":"Symbol","Company":"Security","GICS sector":"GICS Sector"})
    df = df[[c for c in ["Symbol","Security","GICS Sector"] if c in df.columns]].dropna(how="all")
    if "Symbol" in df.columns: df["Symbol"] = df["Symbol"].astype(str).str.replace(".","-", regex=False)
    return df

@st.cache_data
def get_hist(symbol, start):
    try:
        d = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
    except Exception: d = pd.DataFrame()
    if d.empty: return pd.DataFrame()
    d = d.reset_index()
    if "Date" not in d.columns: d = d.rename(columns={d.columns[0]:"Date"})
    return d

@st.cache_data
def get_close(symbol, start):
    try:
        d = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
        s = d["Close"].rename(symbol)
    except Exception: s = pd.Series(dtype=float, name=symbol)
    return s

@st.cache_data
def get_earnings_dates(symbol: str, limit: int = 12):
    try:
        t = yf.Ticker(symbol)
        edf = t.get_earnings_dates(limit=limit)
        if edf is None or edf.empty: return []
        col = [c for c in edf.columns if "Earnings" in c and "Date" in c]
        if col: dt = pd.to_datetime(edf[col[0]], errors="coerce")
        else: dt = pd.to_datetime(edf.index, errors="coerce")
        return [d.date() for d in dt.dropna().tolist()]
    except Exception:
        return []

@st.cache_data
def get_fundamentals_manual(symbol: str):
    t = yf.Ticker(symbol)
    try: q_is = t.quarterly_income_stmt.copy()
    except Exception: q_is = pd.DataFrame()
    try: a_is = t.income_stmt.copy()
    except Exception: a_is = pd.DataFrame()
    try: q_bs = t.quarterly_balance_sheet.copy()
    except Exception: q_bs = pd.DataFrame()
    try: a_bs = t.balance_sheet.copy()
    except Exception: a_bs = pd.DataFrame()
    for df in (q_is, a_is, q_bs, a_bs):
        if df is not None and not df.empty: df.index = df.index.str.strip()

    # EPS
    net_income_ttm = _row_sum_last(q_is, ["Net Income","Net Income Common Stockholders","NetIncome"], 4)
    if net_income_ttm is None:
        net_income_ttm = _row_latest(a_is, ["Net Income","Net Income Common Stockholders","NetIncome"])
    try: shares = (t.info or {}).get("sharesOutstanding")
    except Exception: shares = None
    if shares is None:
        shares = _row_latest(a_bs, ["Ordinary Shares Number","Share Issued"]) or _row_latest(q_bs, ["Ordinary Shares Number","Share Issued"])
    eps = (net_income_ttm / shares) if (net_income_ttm is not None and shares not in (None,0)) else None

    # Price & P/E
    price = None
    try:
        fi = getattr(t, "fast_info", None)
        price = (fi.get("last_price") if hasattr(fi, "get") else None) or getattr(fi, "lastPrice", None)
    except Exception: pass
    if price is None:
        try: price = float(t.history(period="1d")["Close"].iloc[-1])
        except Exception: price = None
    pe = (price / eps) if (price is not None and eps not in (None,0)) else None

    # Revenue growth YoY (annual or sum of last 4 quarters)
    rev_latest = _row_latest(a_is, ["Total Revenue","TotalRevenue"])
    rev_prev = None
    if a_is is not None and not a_is.empty and "Total Revenue" in a_is.index:
        s = _sort_cols_desc(a_is).loc["Total Revenue"].dropna()
        if len(s) >= 2: rev_prev = float(s.iloc[1])
    if rev_latest is None and q_is is not None and not q_is.empty and "Total Revenue" in q_is.index:
        qs = _sort_cols_desc(q_is).loc["Total Revenue"].dropna().astype(float)
        if len(qs) >= 8:
            rev_latest = qs.iloc[:4].sum()
            rev_prev   = qs.iloc[4:8].sum()
    rev_growth = (rev_latest - rev_prev) / rev_prev if (rev_latest is not None and rev_prev not in (None,0)) else None

    # ROE
    eq_latest = _row_latest(a_bs, ["Total Stockholder Equity","Total Equity Gross Minority Interest","TotalEquity"])
    eq_prev = None
    if a_bs is not None and not a_bs.empty and any(n in a_bs.index for n in ["Total Stockholder Equity","Total Equity Gross Minority Interest","TotalEquity"]):
        for n in ["Total Stockholder Equity","Total Equity Gross Minority Interest","TotalEquity"]:
            if n in a_bs.index:
                s = _sort_cols_desc(a_bs).loc[n].dropna().astype(float)
                if len(s) >= 2: eq_prev = float(s.iloc[1]); break
    avg_equity = (eq_latest + eq_prev)/2 if (eq_latest is not None and eq_prev is not None) else eq_latest
    roe = (net_income_ttm / avg_equity) if (net_income_ttm is not None and avg_equity not in (None,0)) else None

    # Debt/Equity
    debt_latest = _row_latest(a_bs, ["Total Debt","TotalDebt"]) or _row_latest(q_bs, ["Total Debt","TotalDebt"])
    equity_latest = eq_latest or _row_latest(q_bs, ["Total Stockholder Equity","TotalEquity"])
    de = (debt_latest / equity_latest) if (debt_latest is not None and equity_latest not in (None,0)) else None

    # As-of date
    asof_candidates = [
        _latest_col_date(a_is), _latest_col_date(q_is),
        _latest_col_date(a_bs), _latest_col_date(q_bs)
    ]
    asof = next((d for d in asof_candidates if d is not None), None)

    return {"eps": eps, "pe": pe, "rev_growth": rev_growth, "roe": roe, "de": de, "asof": asof}

# Performance KPIs from start date (for compare)
@st.cache_data
def perf_kpis(symbol: str, start):
    s = get_close(symbol, start)
    if s is None or s.empty:
        return None, None
    rets = s.pct_change().dropna()
    cum = (1 + rets).cumprod()
    days = (s.index[-1] - s.index[0]).days
    years = max(days / 365.25, 1e-9)
    total_return = float(cum.iloc[-1] - 1)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else np.nan
    sharpe = float((rets.mean() / rets.std() * np.sqrt(252))) if rets.std() > 0 else 0.0
    dd = s / s.cummax() - 1; mdd = float(dd.min())
    return {
        "Total return": total_return,
        "CAGR": cagr, "Volatility": vol, "Sharpe rf=0": sharpe, "Max drawdown": mdd
    }, cum.rename(symbol)

# ---------------- Sidebar ----------------
sp = load_sp500()
st.sidebar.subheader("Search")
query = st.sidebar.text_input("Company or ticker", "Apple")
benchmark = st.sidebar.text_input("Benchmark", "SPY")
start_date = st.sidebar.date_input("Start date", value=date(2024, 1, 1))
sectors = sorted(sp["GICS Sector"].dropna().unique()) if "GICS Sector" in sp.columns else []
picked_sectors = st.sidebar.multiselect("Sector", sectors, sectors)

# Optional: show Ollama status
st.sidebar.write("Ollama:", "✅ running" if ollama_available() else "❌ not running")

# ---------------- Company table ----------------
tbl = sp[sp["GICS Sector"].isin(picked_sectors)] if picked_sectors else sp.copy()
st.header("Companies")
st.write(f"{tbl.shape[0]} rows × {tbl.shape[1]} cols")
st.dataframe(tbl, use_container_width=True)
csv = tbl.to_csv(index=False).encode(); b64 = base64.b64encode(csv).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="sp500.csv">Download CSV</a>', unsafe_allow_html=True)

def search(q, t, k=12):
    if not q.strip() or t.empty: return t.head(k)
    m = pd.Series(False, index=t.index)
    if "Security" in t.columns: m |= t["Security"].astype(str).str.contains(q, case=False, na=False)
    if "Symbol"   in t.columns: m |= t["Symbol"].astype(str).str.contains(q,   case=False, na=False)
    return t.loc[m].head(k)

res = search(query, sp)
opts = [f"{r.get('Security', r['Symbol'])} ({r['Symbol']})" for _, r in res.iterrows()] if not res.empty else []
sel = st.sidebar.selectbox("Pick", opts) if opts else None
symbol = sel.split("(")[-1].rstrip(")") if sel else None
if symbol: st.sidebar.success(f"Selected: {symbol}")

# ---------------- Forecast core ----------------
def _build_features(df: pd.DataFrame, earnings_dates: list[date]) -> pd.DataFrame:
    """
    df: OHLCV dataframe with Date index and columns: Open, High, Low, Close, Volume
    returns: features dataframe indexed by Date
    """
    close = df["Close"]
    high, low, vol = df["High"], df["Low"], df["Volume"]

    feats = pd.DataFrame(index=df.index)
    feats["ret_1d"] = close.pct_change()
    feats["mom_5"]  = close.pct_change(5)
    feats["mom_20"] = close.pct_change(20)   # additional momentum
    feats["sma5"]   = close.rolling(5).mean() / close - 1
    feats["sma20"]  = close.rolling(20).mean() / close - 1
    feats["vol20"]  = close.pct_change().rolling(20).std() * np.sqrt(252)
    feats["rsi14"]  = compute_rsi(close, 14) / 100.0

    # New features
    feats["atr14"]  = compute_atr(high, low, close, 14) / close
    feats["obv"]    = compute_obv(close, vol)
    feats["obv_z"]  = (feats["obv"] - feats["obv"].rolling(20).mean()) / feats["obv"].rolling(20).std()

    roll_max20 = close.rolling(20).max()
    roll_min20 = close.rolling(20).min()
    feats["dist_high20"] = close / roll_max20 - 1
    feats["dist_low20"]  = close / roll_min20 - 1

    # Day-of-week cyclical encoding (Mon=0..Sun=6) — using Series so .clip works
    w = pd.Series(df.index.dayofweek, index=df.index).clip(lower=0, upper=6).astype(float)
    feats["dow_sin"] = np.sin(2*np.pi*w/5.0)
    feats["dow_cos"] = np.cos(2*np.pi*w/5.0)

    # Earnings window dummy (within +/- 3 business days)
    earn_flag = pd.Series(0, index=df.index, dtype=float)
    for ed in earnings_dates:
        win = pd.bdate_range(pd.Timestamp(ed) - BDay(3), pd.Timestamp(ed) + BDay(3))
        idx = earn_flag.index.intersection(win)
        earn_flag.loc[idx] = 1.0
    feats["earn_win"] = earn_flag

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

@st.cache_data
def run_forecast(symbol: str, model_choice: str, start_date_cache_key: str) -> dict:
    """
    Train selected model with TimeSeries CV (n_splits=5), predict next 5 trading days,
    and compute metrics + (quantile) forecast band.
    """
    # --- load full hist for modeling ---
    try:
        hist_full = yf.Ticker(symbol).history(period="max", auto_adjust=True)
    except Exception:
        hist_full = pd.DataFrame()
    needed = {"Close","High","Low","Volume"}
    if hist_full.empty or not needed.issubset(hist_full.columns):
        return {"ok": False, "error": "Not enough price history"}

    hist_full = hist_full.dropna(subset=list(needed)).copy()
    hist_full.index = pd.to_datetime(hist_full.index)

    # --- features & target ---
    earnings = get_earnings_dates(symbol, limit=12)
    feats = _build_features(hist_full, earnings).dropna()
    fwd5 = (hist_full["Close"].shift(-5) / hist_full["Close"] - 1).rename("fwd_5d_ret")
    data = pd.concat([feats, fwd5], axis=1).dropna()

    warn = "Short history after feature engineering; results may be noisy." if len(data) < 350 else None

    X = data.drop(columns=["fwd_5d_ret"])
    y = data["fwd_5d_ret"]
    X_live = feats.loc[[feats.index.max()], X.columns]

    # --- build model by choice ---
    def build_model(choice: str):
        if choice == "Linear Regression":
            return LinearRegression()
        elif choice == "Random Forest":
            return RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        elif choice == "SVR (RBF)":
            return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=2.0, epsilon=0.001))
        else:
            return LinearRegression()

    tscv = TimeSeriesSplit(n_splits=5)
    mae_list, r2_list, residuals = [], [], []
    is_quantile = model_choice.startswith("Gradient Boosting")

    for tr, va in tscv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if is_quantile:
            model_cv = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42, n_estimators=400, max_depth=3)
        else:
            model_cv = build_model(model_choice)
        model_cv.fit(X_tr, y_tr)
        y_hat = model_cv.predict(X_va)
        mae_list.append(mean_absolute_error(y_va, y_hat))
        r2_list.append(r2_score(y_va, y_hat))
        residuals.extend((y_va - y_hat).tolist())

    cv_mae = float(np.mean(mae_list)) if mae_list else None
    cv_r2  = float(np.mean(r2_list)) if r2_list else None
    rmse_all = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None

    last_price = float(hist_full["Close"].iloc[-1])

    if is_quantile:
        gb50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42, n_estimators=400, max_depth=3)
        gb10 = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=42, n_estimators=400, max_depth=3)
        gb90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42, n_estimators=400, max_depth=3)
        gb50.fit(X, y); gb10.fit(X, y); gb90.fit(X, y)
        next_week_ret = float(gb50.predict(X_live)[0])
        low_5d  = max(float(gb10.predict(X_live)[0]), -0.95)
        high_5d = float(gb90.predict(X_live)[0])
        model_used = "Gradient Boosting (Quantile P10/P50/P90)"
    else:
        model_all = build_model(model_choice)
        model_all.fit(X, y)
        next_week_ret = float(model_all.predict(X_live)[0])
        if rmse_all is not None and not math.isnan(rmse_all):
            low_5d  = max(next_week_ret - 1.96 * rmse_all, -0.95)
            high_5d = next_week_ret + 1.96 * rmse_all
        else:
            low_5d, high_5d = next_week_ret, next_week_ret
        model_used = model_choice

    # --- 5-day price path & band ---
    daily_rate = (1.0 + next_week_ret) ** (1/5) - 1
    low_dr  = (1.0 + low_5d) ** (1/5) - 1
    high_dr = (1.0 + high_5d) ** (1/5) - 1

    future_dates = pd.bdate_range(hist_full.index[-1] + BDay(1), periods=5)
    future_prices = [last_price * ((1 + daily_rate) ** (i+1)) for i in range(5)]
    band_low  = [last_price * ((1 + low_dr)  ** (i+1)) for i in range(5)]
    band_high = [last_price * ((1 + high_dr) ** (i+1)) for i in range(5)]

    daily = hist_full["Close"].pct_change().dropna()
    ann_vol = daily.std() * np.sqrt(252)
    sharpe0 = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    f = get_fundamentals_manual(symbol)

    return {
        "ok": True, "warn": " | ".join(filter(None, [warn])),
        "close": hist_full[["Close"]].copy(),
        "future_dates": future_dates,
        "future_prices": future_prices,
        "band_low": band_low, "band_high": band_high,
        "next_week_ret": next_week_ret, "last_price": last_price,
        "mae": cv_mae, "r2": cv_r2, "rmse": rmse_all,
        "ann_vol": ann_vol, "sharpe0": sharpe0,
        "fund": f,
        "target_price": future_prices[-1],
        "low_price": band_low[-1], "high_price": band_high[-1],
        "model_used": model_used,
        "cv_mae": cv_mae, "cv_r2": cv_r2,
    }

def plain_english_explainer(verdict: dict, m: dict) -> str:
    badge = (verdict or {}).get("verdict", "Hold")
    pred = m.get("pred_ret", 0.0)
    mae  = m.get("mae", 0.0)
    r2   = m.get("r2", 0.0)
    low  = (m.get("low_price", m["last_price"])  / m["last_price"]) - 1
    high = (m.get("high_price", m["last_price"]) / m["last_price"]) - 1

    band_txt = f"{low:+.1%} to {high:+.1%}"
    pred_txt = f"{pred:+.2%}"

    if badge == "Buy": headline = "We expect a small **rise** this week."
    elif badge == "Avoid": headline = "We expect a **drop** or too much downside risk."
    else: headline = "The signal is **unclear**—better to wait."

    sure_txt = ("This model isn’t very sure" if (r2 is None or abs(r2) < 0.05 or (mae is not None and mae > 0.03))
                else "Confidence is moderate")

    extras = []
    if m.get("rev_growth") is not None: extras.append(f"Revenue growth YoY: {m['rev_growth']:.0%}")
    if m.get("roe") is not None: extras.append(f"ROE: {m['roe']:.0%}")
    if m.get("de") is not None: extras.append(f"Debt/Equity: {m['de']:.2f}")

    bullets = [
        f"**This week’s guess:** {pred_txt} (small move).",
        f"**Best / worst case (range):** {band_txt}.",
        f"**How sure is this?** MAE ≈ {mae:.2%} (TS-CV), R² = {0.0 if r2 is None else r2:.2f}. {sure_txt}.",
    ]
    if extras: bullets.append("**Quick fundamentals:** " + " • ".join(extras))
    return f"{headline}\n\n" + "\n".join(f"- {b}" for b in bullets)

# ---------------- Tabs ----------------
if symbol:
    tab_overview, tab_predict, tab_ai, tab_compare = st.tabs(["Overview", "Predict next week", "AI suggestion", "Model compare"])

    # ----- 1) Overview -----
    with tab_overview:
        dh = get_hist(symbol, start_date)
        if dh.empty:
            st.error("No data for this symbol or start date.")
        else:
            st.subheader(f"{symbol} price & volume")
            # Price
            fig, ax = plt.subplots(figsize=(12,4))
            ax.fill_between(dh["Date"], dh["Close"], alpha=0.25)
            ax.plot(dh["Date"], dh["Close"], linewidth=2)
            ax.set_title(f"{symbol} Close"); ax.set_xlabel("Date"); ax.set_ylabel("Price")
            st.pyplot(fig, clear_figure=True)
            # Volume
            if "Volume" in dh.columns:
                fig, ax = plt.subplots(figsize=(12,3))
                ax.bar(dh["Date"], dh["Volume"])
                ax.set_title(f"{symbol} Volume"); ax.set_xlabel("Date"); ax.set_ylabel("Shares")
                st.pyplot(fig, clear_figure=True)

            s, b = get_close(symbol, start_date), get_close(benchmark, start_date)
            pair = pd.concat([s, b], axis=1).dropna()
            if pair.empty:
                st.error("No overlapping data for symbol and benchmark.")
            else:
                rets = pair.pct_change().dropna()
                cum = (1 + rets).cumprod()
                days = (pair.index[-1] - pair.index[0]).days
                years = max(days / 365.25, 1e-9)
                total_return = cum[symbol].iloc[-1] - 1
                cagr = (pair[symbol].iloc[-1] / pair[symbol].iloc[0]) ** (1 / years) - 1
                vol = rets[symbol].std() * np.sqrt(252)
                sharpe = (rets[symbol].mean() / rets[symbol].std() * np.sqrt(252)) if rets[symbol].std() > 0 else 0.0
                dd = pair[symbol] / pair[symbol].cummax() - 1; mdd = dd.min()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total return", f"{total_return:.2%}")
                c2.metric("CAGR", f"{cagr:.2%}")
                c3.metric("Volatility", f"{vol:.2%}")
                c4.metric("Sharpe rf=0", f"{sharpe:.2f}")
                c5.metric("Max drawdown", f"{mdd:.2%}")

                st.subheader(f"{symbol} vs {benchmark}: cumulative performance")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(cum.index, cum[symbol], label=symbol, linewidth=2)
                ax.plot(cum.index, cum[benchmark], label=benchmark, linewidth=2)
                ax.set_title("Cumulative return (1 = start)")
                ax.set_xlabel("Date"); ax.set_ylabel("Growth of $1"); ax.legend()
                st.pyplot(fig, clear_figure=True)

                st.subheader(f"{symbol} fundamentals (manual)")
                f = get_fundamentals_manual(symbol)
                f1, f2, f3, f4, f5 = st.columns(5)
                f1.metric("EPS (ttm)", f"{f.get('eps'):.2f}" if f.get("eps") is not None else "—")
                f2.metric("P/E (ttm)", f"{f.get('pe'):.2f}" if f.get("pe") is not None else "—")
                f3.metric("Rev growth (YoY)", f"{f.get('rev_growth'):.2%}" if f.get("rev_growth") is not None else "—")
                f4.metric("ROE (ttm)", f"{f.get('roe'):.0%}" if f.get("roe") is not None else "—")
                f5.metric("Debt/Equity", f"{f.get('de'):.2f}" if f.get("de") is not None else "—")
                st.caption("As-of: " + (f.get("asof") or "—"))

    # ----- 2) Predict next week -----
    with tab_predict:
        st.markdown("Predict **next 5 trading days** return and show the **forecast path** on the price chart.")

        model_choice = st.selectbox(
            "Model",
            ["Linear Regression", "Random Forest", "Gradient Boosting (Quantile P10/P50/P90)", "SVR (RBF)"],
            index=0,
            help="Pick the ML model to generate the 5-day forecast."
        )

        if st.button("Run prediction"):
            out = run_forecast(symbol, model_choice=model_choice, start_date_cache_key=str(start_date))
            if not out["ok"]:
                st.error(out["error"])
            else:
                if out["warn"]: st.warning(out["warn"])
                st.caption(f"Model used: **{out['model_used']}** | TS-CV MAE: {out['cv_mae']:.2%} | R²: {out['cv_r2']:.2f}")

                # KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted next week return", f"{out['next_week_ret']:.2%}")
                c2.metric("Target price (5d)", f"{out['target_price']:,.2f}")
                c3.metric("Range (≈95%)", f"{out['low_price']:,.2f} – {out['high_price']:,.2f}")

                # Chart (6-month history + forecast + band)
                hist_tail = out["close"].tail(126).copy()
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(hist_tail.index, hist_tail["Close"], linewidth=2, label="Historical")
                ax.plot(out["future_dates"], out["future_prices"], linewidth=2, linestyle="--", label="Forecast (5 days)")
                ax.fill_between(out["future_dates"], out["band_low"], out["band_high"], alpha=0.2, label="Forecast range")
                ax.set_title(f"{symbol}: price with 5-day forecast")
                ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
                st.pyplot(fig, clear_figure=True)

                st.caption("Bands are **quantiles** for Gradient Boosting; otherwise derived from TS-CV residual RMSE (≈95%).")

    # ----- 3) AI suggestion -----
    with tab_ai:
        st.markdown("Let AI turn the ML signal + risk + fundamentals into a simple **Buy / Hold / Avoid** suggestion.")
        live = ollama_available()
        st.caption(f"Ollama status: {'✅ running' if live else '❌ not running'} (local LLM)")

        if st.button("Generate AI suggestion"):
            out = run_forecast(symbol, model_choice="Linear Regression", start_date_cache_key=str(start_date))
            if not out["ok"]:
                st.error(out["error"]); st.stop()
            if out["warn"]: st.warning(out["warn"])

            f = out["fund"]
            metrics = {
                "pred_ret": out["next_week_ret"],
                "target_price": out["target_price"],
                "low_price": out["low_price"],
                "high_price": out["high_price"],
                "last_price": out["last_price"],
                "mae": out["mae"], "r2": out["r2"],
                "ann_vol": out["ann_vol"], "sharpe0": out["sharpe0"],
                "rev_growth": f.get("rev_growth"), "roe": f.get("roe"), "de": f.get("de")
            }

            verdict = None; used_llm = False
            try:
                if live:
                    verdict = ai_verdict_ollama(symbol, metrics)
                    used_llm = verdict is not None
            except Exception as e:
                st.info(f"Ollama call failed, using rule-based fallback. ({e})")
            if not verdict: verdict = rule_based_verdict(metrics)

            badge = verdict.get("verdict", "Hold")
            icon = {"Buy": "✅", "Hold": "🟨", "Avoid": "⛔"}.get(badge, "ℹ️")
            st.subheader(f"AI suggestion: {icon} {badge}")

            for r in verdict.get("rationale", []):
                st.write("•", r)

            st.markdown("### In plain English")
            st.markdown(plain_english_explainer(verdict, metrics))

            st.caption("Source: " + ("Local LLM (Ollama)" if used_llm else "Rule-based fallback") + ". Educational tool — not investment advice.")
            with st.expander("Details (debug)"):
                st.write("Ollama used:", used_llm)
                st.json(verdict)

    # ----- 4) Model compare (now supports 2-stock comparison) -----
    with tab_compare:
        st.markdown("Pick **two stocks** to compare on **performance & risk KPIs** and **forecast metrics**. Then view charts.")

        symbols_list = sorted(sp["Symbol"].unique().tolist())
        # Default A = sidebar symbol; B = SPY if available, else next symbol
        default_a = symbols_list.index(symbol) if (symbol in symbols_list) else 0
        default_b = symbols_list.index("SPY") if ("SPY" in symbols_list) else (0 if default_a != 0 else 1)

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            stock_a = st.selectbox("Stock A", symbols_list, index=default_a)
        with col_sel2:
            stock_b = st.selectbox("Stock B", symbols_list, index=default_b)

        model_for_metrics = st.selectbox(
            "Model for forecast metrics",
            ["Linear Regression", "Random Forest", "Gradient Boosting (Quantile P10/P50/P90)", "SVR (RBF)"],
            index=0
        )

        if st.button("Run comparison"):
            # --- Performance KPIs from start_date ---
            kpi_a, cum_a = perf_kpis(stock_a, start_date)
            kpi_b, cum_b = perf_kpis(stock_b, start_date)

            if (kpi_a is None) or (kpi_b is None):
                st.error("Could not fetch price data for one of the symbols.")
            else:
                # --- Forecast metrics using same model ---
                out_a = run_forecast(stock_a, model_choice=model_for_metrics, start_date_cache_key=str(start_date))
                out_b = run_forecast(stock_b, model_choice=model_for_metrics, start_date_cache_key=str(start_date))

                # Summary table (no Winner column)
                rows = []


                def row_from(label, a_val, b_val):
                    rows.append({"Metric": label, stock_a: a_val, stock_b: b_val})


                # Performance KPIs
                row_from("Total return (since start)", kpi_a["Total return"], kpi_b["Total return"])
                row_from("CAGR", kpi_a["CAGR"], kpi_b["CAGR"])
                row_from("Volatility (ann.)", kpi_a["Volatility"], kpi_b["Volatility"])
                row_from("Sharpe rf=0", kpi_a["Sharpe rf=0"], kpi_b["Sharpe rf=0"])
                row_from("Max drawdown", kpi_a["Max drawdown"], kpi_b["Max drawdown"])

                # Forecast metrics
                row_from(f"Predicted 5d return ({model_for_metrics})", out_a.get("next_week_ret"),
                         out_b.get("next_week_ret"))
                row_from("TS-CV MAE (lower=better)", out_a.get("cv_mae"), out_b.get("cv_mae"))
                row_from("TS-CV R²", out_a.get("cv_r2"), out_b.get("cv_r2"))

                dfc = pd.DataFrame(rows)

                # Format percentage-like metrics
                pct_metrics = {
                    "Total return (since start)",
                    "CAGR",
                    "Volatility (ann.)",
                    f"Predicted 5d return ({model_for_metrics})",
                    "TS-CV MAE (lower=better)",
                }
                def fmt(x, is_pct=False):
                    if x is None or pd.isna(x): return "—"
                    return f"{x:.2%}" if is_pct else (f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x))
                for col in [stock_a, stock_b]:
                    dfc[col] = dfc.apply(lambda r: fmt(r[col], r["Metric"] in pct_metrics), axis=1)

                st.subheader("Side-by-side comparison")
                st.dataframe(dfc, use_container_width=True)

                # Charts: cumulative performance & KPI bars
                st.subheader("Cumulative performance (growth of $1)")
                # Align indices
                both = pd.concat([cum_a, cum_b], axis=1).dropna()
                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(both.index, both[stock_a], label=stock_a, linewidth=2)
                ax.plot(both.index, both[stock_b], label=stock_b, linewidth=2)
                ax.set_xlabel("Date"); ax.set_ylabel("Growth of $1"); ax.legend()
                st.pyplot(fig, clear_figure=True)

                st.subheader("Risk/return KPIs")
                labels = ["Total return", "CAGR", "Volatility", "Sharpe rf=0", "Max drawdown"]
                a_vals = [kpi_a["Total return"], kpi_a["CAGR"], kpi_a["Volatility"], kpi_a["Sharpe rf=0"], kpi_a["Max drawdown"]]
                b_vals = [kpi_b["Total return"], kpi_b["CAGR"], kpi_b["Volatility"], kpi_b["Sharpe rf=0"], kpi_b["Max drawdown"]]
                x = np.arange(len(labels)); width = 0.35
                fig2, ax2 = plt.subplots(figsize=(12,4))
                ax2.bar(x - width/2, a_vals, width, label=stock_a)
                ax2.bar(x + width/2, b_vals, width, label=stock_b)
                ax2.set_xticks(x, labels)
                ax2.set_ylabel("Value (fractions are %)")
                ax2.legend()
                st.pyplot(fig2, clear_figure=True)

                # Fundamentals snapshot
                st.subheader("Fundamentals snapshot")
                fa = get_fundamentals_manual(stock_a)
                fb = get_fundamentals_manual(stock_b)
                frows = [
                    ["EPS (ttm)", fa.get("eps"), fb.get("eps")],
                    ["P/E (ttm)", fa.get("pe"), fb.get("pe")],
                    ["Rev growth (YoY)", fa.get("rev_growth"), fb.get("rev_growth")],
                    ["ROE (ttm)", fa.get("roe"), fb.get("roe")],
                    ["Debt/Equity", fa.get("de"), fb.get("de")],
                    ["As-of", fa.get("asof"), fb.get("asof")],
                ]
                fdf = pd.DataFrame(frows, columns=["Metric", stock_a, stock_b])
                # Format %
                def fmt_fund(val, pct=False):
                    if val is None or pd.isna(val): return "—"
                    return f"{val:.2%}" if pct else (f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val))
                for i, m in enumerate(fdf["Metric"]):
                    if m in ["Rev growth (YoY)", "ROE (ttm)"]:
                        fdf.loc[i, stock_a] = fmt_fund(fdf.loc[i, stock_a], True)
                        fdf.loc[i, stock_b] = fmt_fund(fdf.loc[i, stock_b], True)
                    elif m not in ["As-of"]:
                        fdf.loc[i, stock_a] = fmt_fund(fdf.loc[i, stock_a], False)
                        fdf.loc[i, stock_b] = fmt_fund(fdf.loc[i, stock_b], False)
                st.dataframe(fdf, use_container_width=True)

# ---------------- If no symbol chosen ----------------
else:
    st.info("Pick a company from the sidebar to get started.")
