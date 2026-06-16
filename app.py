import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import time
from datetime import datetime

# ── Konfiguracja strony ─────────────────────────────────────
st.set_page_config(
    page_title="📈 GPW Analyzer — Feigin Capital",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1e2130; border-radius: 8px; padding: 10px; }
    .metric-green { color: #00e676 !important; }
    .metric-red { color: #ff5252 !important; }
    .metric-yellow { color: #ffeb3b !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .signal-buy { background: #1b5e20; color: #69f0ae; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .signal-sell { background: #b71c1c; color: #ff8a80; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .signal-hold { background: #333; color: #ccc; padding: 4px 12px; border-radius: 20px; }
    .ai-box { background: #1a237e; border-left: 4px solid #3f51b5; padding: 16px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── Spółki GPW + NewConnect ──────────────────────────────────
GPW_BLUE_CHIPS = {
    "PKN Orlen": "PKN.WA",
    "PKO Bank Polski": "PKO.WA",
    "CD Projekt": "CDR.WA",
    "XTB": "XTB.WA",
    "KGHM": "KGH.WA",
    "Allegro": "ALE.WA",
    "Dino Polska": "DNP.WA",
    "LPP": "LPP.WA",
    "Cyfrowy Polsat": "CPS.WA",
    "PZU": "PZU.WA",
    "mBank": "MBK.WA",
    "Santander Bank Polska": "SPL.WA",
    "Pekao": "PEO.WA",
    "JSW": "JSW.WA",
    "Orlen": "PKN.WA",
}

NEWCONNECT = {
    "PlayWay": "PLW.WA",
    "Ten Square Games": "TEN.WA",
    "CI Games": "CIG.WA",
    "Bloober Team": "BLO.WA",
    "Varsav Game Studios": "VGS.WA",
}

ALL_STOCKS = {**GPW_BLUE_CHIPS, **NEWCONNECT}

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Ustawienia")
    st.divider()

    # Wybór grupy
    group = st.selectbox("Segment", ["GPW Blue Chips", "NewConnect", "Wszystkie"])
    if group == "GPW Blue Chips":
        available = GPW_BLUE_CHIPS
    elif group == "NewConnect":
        available = NEWCONNECT
    else:
        available = ALL_STOCKS

    # Wybór spółek
    selected_names = st.multiselect(
        "Spółki do analizy",
        options=list(available.keys()),
        default=list(available.keys())[:8]
    )

    st.divider()

    # Parametry
    period = st.selectbox("Okres danych", ["1mo", "2mo", "3mo", "6mo", "1y"], index=2)
    rsi_period = st.slider("RSI period", 7, 21, 14)
    rsi_oversold = st.slider("RSI — wyprzedanie (KUP)", 20, 40, 30)
    rsi_overbought = st.slider("RSI — wykupienie (SPRZEDAJ)", 60, 85, 70)

    st.divider()

    # API Key Claude
    claude_key = st.text_input("Anthropic API Key (do AI rekomendacji)", type="password",
                                help="Klucz z console.anthropic.com")

    st.divider()
    st.caption(f"🕐 Aktualizacja: {datetime.now().strftime('%H:%M:%S')}")

# ── Funkcje ──────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache 5 minut
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception:
        return None

def compute_indicators(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']

    # RSI
    df['RSI'] = RSIIndicator(close, window=rsi_period).rsi()

    # MACD
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_pct'] = bb.bollinger_pband()

    # SMA
    df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator() if len(df) >= 50 else np.nan

    # Zmiana %
    df['Change_1d'] = close.pct_change() * 100
    df['Change_5d'] = close.pct_change(5) * 100
    df['Change_22d'] = close.pct_change(22) * 100

    return df

def score_stock(df: pd.DataFrame, rsi_oversold: int, rsi_overbought: int) -> dict:
    last = df.iloc[-1]
    score = 0
    signals = []

    rsi = last.get('RSI', 50)
    macd_diff = last.get('MACD_diff', 0)
    bb_pct = last.get('BB_pct', 0.5)
    change_1d = last.get('Change_1d', 0)
    change_5d = last.get('Change_5d', 0)

    # RSI signals
    if rsi < rsi_oversold:
        score += 3
        signals.append(f"🟢 RSI={rsi:.1f} — wyprzedanie (sygnał KUP)")
    elif rsi > rsi_overbought:
        score -= 3
        signals.append(f"🔴 RSI={rsi:.1f} — wykupienie (sygnał SPRZEDAJ)")
    elif rsi < 45:
        score += 1
        signals.append(f"🟡 RSI={rsi:.1f} — neutralny, lekko niedowartościowany")
    else:
        signals.append(f"⚪ RSI={rsi:.1f} — neutralny")

    # MACD
    if macd_diff > 0 and last.get('MACD', 0) < 0:
        score += 2
        signals.append("🟢 MACD crossover pozytywny")
    elif macd_diff < 0 and last.get('MACD', 0) > 0:
        score -= 2
        signals.append("🔴 MACD crossover negatywny")
    elif macd_diff > 0:
        score += 1
        signals.append("🟡 MACD — momentum wzrostowe")
    else:
        score -= 1
        signals.append("🟡 MACD — momentum spadkowe")

    # Bollinger
    if bb_pct < 0.1:
        score += 2
        signals.append("🟢 Cena blisko dolnej Bollingera — potencjalne odbicie")
    elif bb_pct > 0.9:
        score -= 2
        signals.append("🔴 Cena blisko górnej Bollingera — ryzyko korekty")

    # Trend
    if change_5d > 0 and change_1d > 0:
        score += 1
        signals.append(f"🟢 Trend wzrostowy: +{change_5d:.1f}% (5d)")
    elif change_5d < -5:
        score -= 1
        signals.append(f"🔴 Trend spadkowy: {change_5d:.1f}% (5d)")

    # Rekomendacja
    if score >= 4:
        recommendation = "MOCNY KUP"
        color = "#00e676"
    elif score >= 2:
        recommendation = "KUP"
        color = "#69f0ae"
    elif score <= -4:
        recommendation = "MOCNA SPRZEDAŻ"
        color = "#ff5252"
    elif score <= -2:
        recommendation = "SPRZEDAJ"
        color = "#ff8a80"
    else:
        recommendation = "TRZYMAJ"
        color = "#ffeb3b"

    return {
        "score": score,
        "recommendation": recommendation,
        "color": color,
        "signals": signals,
        "rsi": rsi,
        "macd_diff": macd_diff,
        "bb_pct": bb_pct,
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_22d": last.get('Change_22d', 0),
        "price": last['Close'],
        "bb_upper": last.get('BB_upper', 0),
        "bb_lower": last.get('BB_lower', 0),
    }

def get_ai_recommendation(ticker: str, name: str, score_data: dict, api_key: str) -> str:
    if not api_key:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Jesteś analitykiem giełdowym. Przeanalizuj spółkę {name} ({ticker}) na GPW.

Dane techniczne:
- Cena: {score_data['price']:.2f} PLN
- RSI(14): {score_data['rsi']:.1f}
- MACD diff: {score_data['macd_diff']:.4f}
- Bollinger %B: {score_data['bb_pct']:.2f}
- Zmiana 1d: {score_data['change_1d']:.1f}%
- Zmiana 5d: {score_data['change_5d']:.1f}%
- Zmiana 22d: {score_data['change_22d']:.1f}%
- Score algorytmiczny: {score_data['score']}/10
- Rekomendacja algorytmu: {score_data['recommendation']}
- Sygnały: {', '.join(score_data['signals'])}

Napisz krótką (3-4 zdania) analizę inwestycyjną po polsku. Bądź konkretny. Zacznij od oceny sytuacji, podaj co obserwujesz i co zalecasz. Zakończ jednym zdaniem o ryzyku."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ Błąd AI: {str(e)[:50]}"

def plot_stock(df: pd.DataFrame, name: str, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{name} ({ticker})", "RSI", "MACD"),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Kurs", increasing_line_color='#00e676',
        decreasing_line_color='#ff5252'
    ), row=1, col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name="BB Upper",
        line=dict(color='rgba(100,100,255,0.4)', dash='dash'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name="BB Lower",
        fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
        line=dict(color='rgba(100,100,255,0.4)', dash='dash'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA20",
        line=dict(color='#ffeb3b', width=1)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
        line=dict(color='#ab47bc', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", row=2, col=1)

    # MACD
    colors = ['#00e676' if v >= 0 else '#ff5252' for v in df['MACD_diff'].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_diff'], name="MACD Histogram",
        marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
        line=dict(color='#29b6f6', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name="Signal",
        line=dict(color='#ff7043', width=1.5)), row=3, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#e0e0e0'),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_yaxes(gridcolor='#1e2130')
    fig.update_xaxes(gridcolor='#1e2130')

    return fig

# ── Główna aplikacja ─────────────────────────────────────────
st.title("📈 GPW + NewConnect Analyzer")
st.caption(f"Dane w czasie rzeczywistym via Yahoo Finance · {datetime.now().strftime('%d.%m.%Y %H:%M')}")

if not selected_names:
    st.warning("Wybierz co najmniej jedną spółkę z lewego panelu.")
    st.stop()

selected_tickers = {name: available[name] for name in selected_names}

# ── Pobieranie i analiza danych ───────────────────────────────
with st.spinner("⏳ Pobieranie danych giełdowych..."):
    all_results = []
    dfs = {}

    progress = st.progress(0)
    for i, (name, ticker) in enumerate(selected_tickers.items()):
        df = fetch_stock_data(ticker, period)
        if df is not None and len(df) > 20:
            df = compute_indicators(df, rsi_period)
            score_data = score_stock(df, rsi_oversold, rsi_overbought)
            score_data['name'] = name
            score_data['ticker'] = ticker
            all_results.append(score_data)
            dfs[ticker] = (name, df)
        progress.progress((i + 1) / len(selected_tickers))
    progress.empty()

if not all_results:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie internetowe.")
    st.stop()

# ── Ranking ───────────────────────────────────────────────────
st.header("🏆 Ranking spółek")

df_rank = pd.DataFrame(all_results).sort_values('score', ascending=False)

# Kolorowe karty KPI
cols = st.columns(min(len(all_results), 4))
for i, row in df_rank.head(4).iterrows():
    with cols[i % 4]:
        delta_color = "normal" if row['change_1d'] >= 0 else "inverse"
        st.metric(
            label=f"{row['recommendation']} | {row['name']}",
            value=f"{row['price']:.2f} PLN",
            delta=f"{row['change_1d']:.1f}% dziś"
        )

st.divider()

# Tabela rankingowa
df_display = df_rank[['name','ticker','price','rsi','change_1d','change_5d','change_22d','score','recommendation']].copy()
df_display.columns = ['Spółka', 'Ticker', 'Cena PLN', 'RSI', '1D %', '5D %', '22D %', 'Score', 'Rekomendacja']

def color_rec(val):
    if 'KUP' in str(val): return 'background-color: #1b5e20; color: #69f0ae'
    if 'SPRZEDAJ' in str(val): return 'background-color: #b71c1c; color: #ff8a80'
    return 'background-color: #333; color: #ccc'

def color_rsi(val):
    try:
        v = float(val)
        if v < 30: return 'color: #00e676; font-weight: bold'
        if v > 70: return 'color: #ff5252; font-weight: bold'
        return 'color: #ffeb3b'
    except: return ''

styled = df_display.style\
    .format({'Cena PLN': '{:.2f}', 'RSI': '{:.1f}', '1D %': '{:.1f}%', '5D %': '{:.1f}%', '22D %': '{:.1f}%', 'Score': '{:.0f}'})\
    .applymap(color_rec, subset=['Rekomendacja'])\
    .applymap(color_rsi, subset=['RSI'])\
    .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-6, vmax=6)

st.dataframe(styled, use_container_width=True, height=400)

# ── Szczegółowa analiza ───────────────────────────────────────
st.divider()
st.header("🔍 Szczegółowa analiza spółki")

col1, col2 = st.columns([2, 1])
with col1:
    selected_detail = st.selectbox("Wybierz spółkę:", list(dfs.keys()),
        format_func=lambda x: f"{dfs[x][0]} ({x})")

if selected_detail and selected_detail in dfs:
    name, df = dfs[selected_detail]
    score_data = next(r for r in all_results if r['ticker'] == selected_detail)

    # Metryki
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Cena", f"{score_data['price']:.2f} PLN", f"{score_data['change_1d']:.1f}%")
    m2.metric("📊 RSI", f"{score_data['rsi']:.1f}",
              "Wyprzedanie" if score_data['rsi'] < 30 else ("Wykupienie" if score_data['rsi'] > 70 else "Neutralny"))
    m3.metric("📈 5D", f"{score_data['change_5d']:.1f}%")
    m4.metric("📅 22D", f"{score_data['change_22d']:.1f}%")
    m5.metric("⭐ Score", f"{score_data['score']}/10")

    # Wykres
    fig = plot_stock(df, name, selected_detail)
    st.plotly_chart(fig, use_container_width=True)

    # Sygnały
    col_sig, col_ai = st.columns([1, 1])

    with col_sig:
        st.subheader("📡 Sygnały techniczne")
        for sig in score_data['signals']:
            st.markdown(f"- {sig}")

    with col_ai:
        st.subheader("🤖 Analiza AI (Claude)")
        if claude_key:
            if st.button(f"Generuj analizę AI dla {name}", type="primary"):
                with st.spinner("Claude analizuje..."):
                    ai_text = get_ai_recommendation(selected_detail, name, score_data, claude_key)
                    if ai_text:
                        st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Wpisz klucz Anthropic API w panelu bocznym aby aktywować analizę AI.")
            st.markdown("Klucz dostępny na: [console.anthropic.com](https://console.anthropic.com)")

# ── Sygnały alertowe ─────────────────────────────────────────
st.divider()
st.header("🚨 Sygnały alertowe")

buy_signals = [r for r in all_results if r['score'] >= 4]
sell_signals = [r for r in all_results if r['score'] <= -4]

col_buy, col_sell = st.columns(2)

with col_buy:
    st.subheader("🟢 Potencjalne kupna")
    if buy_signals:
        for r in sorted(buy_signals, key=lambda x: x['score'], reverse=True):
            st.success(f"**{r['name']}** ({r['ticker']}) — Score: {r['score']} | RSI: {r['rsi']:.1f} | {r['change_5d']:.1f}% (5d)")
    else:
        st.info("Brak silnych sygnałów kupna w wybranych spółkach.")

with col_sell:
    st.subheader("🔴 Potencjalne sprzedaże")
    if sell_signals:
        for r in sorted(sell_signals, key=lambda x: x['score']):
            st.error(f"**{r['name']}** ({r['ticker']}) — Score: {r['score']} | RSI: {r['rsi']:.1f} | {r['change_5d']:.1f}% (5d)")
    else:
        st.info("Brak silnych sygnałów sprzedaży.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Disclaimer: Ta aplikacja służy wyłącznie celom informacyjnym i edukacyjnym. Nie stanowi porady inwestycyjnej. Inwestowanie na giełdzie wiąże się z ryzykiem utraty kapitału.")
st.caption("📊 Dane: Yahoo Finance | 🤖 AI: Claude Sonnet | 🛠️ Built by Andrew Charnosh & Claude")
