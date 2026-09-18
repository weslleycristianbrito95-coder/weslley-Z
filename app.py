import datetime
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st
import yfinance as yf

# ==========================================
# 1. MODELO BLACK-SCHOLES E CÁLCULO DE IV
# ==========================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, option_type='call'):
    if T <= 0 or price <= 0:
        return np.nan
    def objective_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price
    try:
        return brentq(objective_function, 1e-4, 5.0)
    except (ValueError, RuntimeError):
        return np.nan

# ==========================================
# 2. COLETA E PROCESSAMENTO DA CADEIA
# ==========================================

def fetch_option_data(ticker_symbol='QQQ'):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='1d')
    if history.empty:
        raise ValueError("Não foi possível obter a cotação do ativo.")
    underlying_price = float(history['Close'].iloc[-1])
    expirations = ticker.options
    return underlying_price, tuple(expirations)

def get_skew_data(ticker_symbol, selected_expiration, underlying_price, risk_free_rate=0.045):
    ticker = yf.Ticker(ticker_symbol)
    opt_chain = ticker.option_chain(selected_expiration)
    calls, puts = opt_chain.calls.copy(), opt_chain.puts.copy()

    today = datetime.date.today()
    exp_date = datetime.datetime.strptime(selected_expiration, "%Y-%m-%d").date()
    days_to_exp = max((exp_date - today).days, 1)
    T = days_to_exp / 365.0

    calls['type'], puts['type'] = 'Call', 'Put'
    df = pd.concat([calls, puts], ignore_index=True)
    df = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['moneyness'] = df['strike'] / underlying_price

    calculated_ivs = []
    for _, row in df.iterrows():
        iv_yahoo = row.get('impliedVolatility', np.nan)
        if pd.notnull(iv_yahoo) and 0.01 < iv_yahoo < 3.0:
            calculated_ivs.append(iv_yahoo)
        else:
            calculated_ivs.append(
                implied_volatility(row['mid_price'], underlying_price, row['strike'], T, risk_free_rate, row['type'])
            )

    df['calculated_iv'] = calculated_ivs
    df['iv_percent'] = df['calculated_iv'] * 100
    df = df.dropna(subset=['iv_percent'])
    df = df[(df['iv_percent'] > 0.5) & (df['iv_percent'] < 200.0)]
    return df.sort_values(by='strike'), days_to_exp

# ==========================================
# 3. INTERFACE STREAMLIT E AUTO-REFRESH
# ==========================================

st.set_page_config(page_title="QQQ Intraday Skew Tracker", layout="wide")

st.title("⚡ Intraday Volatility Skew Monitor")

# Sidebar - Configurações de Auto-Refresh e Ativo
st.sidebar.header("Parâmetros do Monitor")
ticker_symbol = st.sidebar.text_input("Ticker", value="QQQ").upper()
risk_free_rate = st.sidebar.number_input("Taxa Livre de Risco (r)", value=0.045, step=0.005, format="%.3f")

auto_refresh = st.sidebar.checkbox("Ativar Auto-Refresh Intraday", value=True)
refresh_interval = st.sidebar.slider("Intervalo de Atualização (segundos)", min_value=10, max_value=300, value=30)

if st.sidebar.button("Limpar Histórico Intraday"):
    st.session_state['skew_history'] = pd.DataFrame()

# Inicializa sessão de histórico
if 'skew_history' not in st.session_state:
    st.session_state['skew_history'] = pd.DataFrame(columns=['timestamp', 'spot', 'put_2pct_iv', 'call_2pct_iv', 'skew_spread'])

try:
    current_price, expirations = fetch_option_data(ticker_symbol)
    selected_expiration = st.sidebar.selectbox("Data de Vencimento", expirations)
    df_skew, dte = get_skew_data(ticker_symbol, selected_expiration, current_price, risk_free_rate)

    # Cálculo da Skew Spread: Put OTM (98% do Spot) - Call OTM (102% do Spot)
    put_target = current_price * 0.98
    call_target = current_price * 1.02

    puts_df = df_skew[df_skew['type'] == 'Put']
    calls_df = df_skew[df_skew['type'] == 'Call']

    if not puts_df.empty and not calls_df.empty:
        closest_put_idx = (puts_df['strike'] - put_target).abs().idxmin()
        closest_call_idx = (calls_df['strike'] - call_target).abs().idxmin()

        put_iv = puts_df.loc[closest_put_idx, 'iv_percent']
        call_iv = calls_df.loc[closest_call_idx, 'iv_percent']
        skew_spread = put_iv - call_iv

        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        # Adiciona nova captura no histórico da sessão
        new_entry = pd.DataFrame([{
            'timestamp': now_str,
            'spot': current_price,
            'put_2pct_iv': put_iv,
            'call_2pct_iv': call_iv,
            'skew_spread': skew_spread
        }])
        
        # Evita duplicar se não mudou o timestamp segundo a segundo
        if st.session_state['skew_history'].empty or st.session_state['skew_history'].iloc[-1]['timestamp'] != now_str:
            st.session_state['skew_history'] = pd.concat([st.session_state['skew_history'], new_entry], ignore_index=True)

    # Métricas Superiores
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço Spot", f"${current_price:.2f}")
    col2.metric("2% OTM Put IV", f"{put_iv:.2f}%")
    col3.metric("2% OTM Call IV", f"{call_iv:.2f}%")
    
    # Delta de variação do Skew Spread em relação à primeira leitura do dia
    hist_df = st.session_state['skew_history']
    if len(hist_df) > 1:
        initial_skew = hist_df.iloc[0]['skew_spread']
        skew_delta = skew_spread - initial_skew
        col4.metric("Skew Spread (Put IV - Call IV)", f"{skew_spread:.2f}%", delta=f"{skew_delta:+.2f}% vs Abertura")
    else:
        col4.metric("Skew Spread (Put IV - Call IV)", f"{skew_spread:.2f}%")

    st.markdown("---")

    # Gráfico 1: Evolução Intraday do Skew Spread
    st.subheader("📉 Monitor Intraday da Inclinação do Skew (Put IV - Call IV Spread)")
    if len(hist_df) > 0:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=hist_df['timestamp'], y=hist_df['skew_spread'],
            mode='lines+markers', name='Skew Spread (%)',
            line=dict(color='yellow', width=3)
        ))
        fig_time.update_layout(
            template="plotly_dark",
            xaxis_title="Horário",
            yaxis_title="Diferença de IV (Put 98% - Call 102%) [%]",
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # Gráfico 2: Curva do Skew no momento atual
    st.subheader(f"Curva Atual de Skew (Vencimento: {selected_expiration})")
    puts_otm = df_skew[(df_skew['type'] == 'Put') & (df_skew['strike'] <= current_price)]
    calls_otm = df_skew[(df_skew['type'] == 'Call') & (df_skew['strike'] > current_price)]
    skew_otm = pd.concat([puts_otm, calls_otm]).sort_values(by='strike')

    fig_skew = go.Figure()
    fig_skew.add_trace(go.Scatter(
        x=skew_otm['strike'], y=skew_otm['iv_percent'],
        mode='lines+markers', name='OTM Skew Curve',
        line=dict(color='cyan', width=3)
    ))
    fig_skew.add_vline(x=current_price, line_dash="dash", line_color="orange", annotation_text="Spot")
    fig_skew.update_layout(template="plotly_dark", xaxis_title="Strike ($)", yaxis_title="IV (%)", height=380)
    st.plotly_chart(fig_skew, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao carregar dados intraday: {str(e)}")

# Loop de auto-refresh sem travar o app
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
