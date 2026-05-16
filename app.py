import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import base64
from datetime import datetime

# 1. Configuração de Tela Cheia do Dashboard
st.set_page_config(page_title="Monitor Quantitativo", page_icon="📊", layout="wide")

# 2. Injeção de Estilo Premium (Fundo Preto Absoluto e Temática Azul-Marinho / Shadcn UI)
st.markdown("""
    <style>
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #000000 !important;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        #MainMenu, footer {visibility: hidden;}
        
        .custom-header {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: rgba(0, 0, 0, 0.8) !important;
            backdrop-filter: blur(12px);
            border-b: 1px solid #002244;
            padding: 12px 0px;
            margin-bottom: 20px;
        }
        
        .stSlider [data-baseweb="slider"] {
            background-color: #002244 !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Lista de Ativos Alvo
ativos_futuros = {
    "ZM": "ZM=F", "ZN": "ZN=F", "CL": "CL=F", "GC": "GC=F", 
    "NQ": "NQ=F", "ES": "ES=F", "YM": "YM=F", "RTY": "RTY=F", 
    "6E": "6E=F", "6J": "6J=F", "6B": "6B=F", "ZS": "ZS=F"
}

# 4. Engine de Cálculo do Z-Score
def compute_zscore(close_prices, window):
    prices_series = pd.Series(close_prices).dropna()
    returns = prices_series.pct_change().dropna()
    
    if len(returns) < window:
        return np.nan, np.nan, np.nan
    
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    last_price = float(prices_series.iloc[-1].item() if hasattr(prices_series.iloc[-1], 'item') else prices_series.iloc[-1])
    last_return = float(returns.iloc[-1].item() if hasattr(returns.iloc[-1], 'item') else returns.iloc[-1])
    mean = float(rolling_mean.iloc[-1].item() if hasattr(rolling_mean.iloc[-1], 'item') else rolling_mean.iloc[-1])
    std = float(rolling_std.iloc[-1].item() if hasattr(rolling_std.iloc[-1], 'item') else rolling_std.iloc[-1])
    
    if std == 0 or np.isnan(std):
        zscore = 0.0
    else:
        zscore = (last_return - mean) / std
        
    return zscore, last_return, last_price

# 5. Engine para Alertas Sonoros
def emitir_alerta(audio_bytes=None):
    if audio_bytes is not None:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    else:
        audio_html = """
        <script>
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const playBeep = (time, freq) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = freq;
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.3, time);
                gain.gain.exponentialRampToValueAtTime(0.01, time + 0.15);
                osc.start(time);
                osc.stop(time + 0.15);
            };
            const now = ctx.currentTime;
            playBeep(now, 880);
            playBeep(now + 0.2, 1100);
            playBeep(now + 0.4, 880);
        </script>
        """
    st.markdown(audio_html, unsafe_allow_html=True)

# 6. Sidebar
st.sidebar.markdown('<p style="color:#0066cc; font-weight:800; font-size:16px; letter-spacing:0.5px; margin-bottom:5px;">⚙️ CONTROL PANEL</p>', unsafe_allow_html=True)
window = st.sidebar.slider("Janela de Cálculo (Dias)", min_value=5, max_value=100, value=20)
threshold = st.sidebar.slider("Gatilho de Alerta (±σ)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

st.sidebar.write("---")
st.sidebar.markdown('<p style="color:#0066cc; font-weight:800; font-size:14px; margin-bottom:5px;">🎵 SOUND CONFIG</p>', unsafe_allow_html=True)
arquivo_audio = st.sidebar.file_uploader("Upload de áudio personalizado (Opcional)", type=["mp3", "wav"])

# Lógica de Controle do Tempo/Contagem Regressiva
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = time.time()
if 'alertado_na_rodada' not in st.session_state:
    st.session_state.alertado_na_rodada = False

tempo_passado = time.time() - st.session_state.last_fetch_time
countdown = max(0, 60 - int(tempo_passado))

if countdown <= 0:
    st.session_state.last_fetch_time = time.time()
    st.session_state.alertado_na_rodada = False
    st.rerun()

# 7. Coleta robusta de dados compatível com yfinance atualizado (Ajuste Crítico Aqui)
@st.cache_data(ttl=55)
def carregar_dados_seguros(tickers_dict):
    dados_carregados = {}
    symbols = list(tickers_dict.values())
    
    try:
        # Desativa explicitamente o multi_level_index nativo para simplificar a estrutura de colunas
        df = yf.download(symbols, period="1y", group_by="ticker", progress=False, multi_level_index=False)
        
        for nome, ticker in tickers_dict.items():
            # Tenta encontrar no formato com MultiIndex plano ou agrupamento padrão
            if ticker in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else ticker in df.columns:
                try:
                    # Coleta limpa eliminando indexações truncadas do Pandas
                    dados_carregados[nome] = df[ticker]['Close'].dropna()
                except Exception:
                    pass
    except Exception:
        pass
        
    # BACKUP EXTRA: Se o download em lote falhar ou retornar vazio, busca ativo por ativo (Blindagem Máxima)
    for nome, ticker in tickers_dict.items():
        if nome not in dados_carregados or dados_carregados[nome].empty:
            try:
                single_df = yf.download(ticker, period="1y", progress=False, multi_level_index=False)
                if not single_df.empty:
                    if 'Close' in single_df.columns:
                        dados_carregados[nome] = single_df['Close'].dropna()
            except Exception:
                continue
                
    return dados_carregados

dados_finais = carregar_dados_seguros(ativos_futuros)

# 8. Processamento e Lógica de Ordenação do React (sortedAssets)
assets_data = []
alert_count = 0
disparar_som = False

if dados_finais:
    for nome, precos in dados_finais.items():
        try:
            if precos is not None and len(precos) > 0:
                ticker_cod = ativos_futuros[nome]
                zscore, ultimo_retorno, ultimo_preco = compute_zscore(precos, window)
                
                if not np.isnan(zscore):
                    em_alerta = abs(zscore) >= threshold
                    if em_alerta:
                        alert_count += 1
                        disparar_som = True
                        
                    assets_data.append({
                        "nome": nome,
                        "ticker": ticker_cod,
                        "z_score": zscore,
                        "retorno": ultimo_retorno,
                        "preco": ultimo_preco,
                        "abs_z": abs(zscore),
                        "alerta": em_alerta
                    })
        except Exception:
            continue

    # Ordenação decrescente por força absoluta do Z-Score
    assets_data = sorted(assets_data, key=lambda x: x['abs_z'], reverse=True)

# 9. Renderização do StatusHeader
hora_atual = datetime.now().strftime('%H:%M:%S')
progresso_barra = int((countdown / 60) * 100)

st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #002244; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <span style="color: #ffffff; font-size: 24px; font-weight: 800; letter-spacing: 0.5px;">MONITOR <span style="color:#0066cc;">Z-SCORE</span></span>
            <div style="color: #666666; font-size: 11px; font-family: monospace; margin-top: 2px;">SYNC: {hora_atual} • ATIVOS EM EXTREMO: <span style="color:{'#ff3333' if alert_count > 0 else '#00cc66'}; font-weight:bold;">{alert_count}</span></div>
        </div>
        <div style="text-align: right;">
            <div style="color: #0066cc; font-family: monospace; font-size: 13px; font-weight: bold;">PRÓXIMO REFRESH EM {countdown}s</div>
            <div style="width: 160px; background-color: #001122; height: 4px; border-radius: 4px; margin-top: 5px; overflow: hidden; display: inline-block;">
                <div style="background-color: #0066cc; width: {progresso_barra}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 10. Renderização da Grade Grid de Cards do Dashboard
if assets_data:
    colunas_grid = st.columns(2)
    
    for idx, asset in enumerate(assets_data):
        alvo_coluna = colunas_grid[idx % 2]
        
        if asset['alerta']:
            cor_fundo = "#1a1300"
            cor_borda = "#ffbb00"
            cor_badge = "#ffbb00"
            status_txt = "CRITICAL DESVIATION"
        else:
            if asset['z_score'] > 0:
                cor_fundo = "#020a14"
                cor_borda = "#002b5c"
                cor_badge = "#0088ff"
                status_txt = "BULLISH DISPLACEMENT"
            else:
                cor_fundo = "#050508"
                cor_borda = "#1c1d24"
                cor_badge = "#888899"
                status_txt = "BEARISH DISPLACEMENT"

        with alvo_coluna:
            st.markdown(f"""
                <div style="background-color: {cor_fundo}; border: 1px solid {cor_borda}; padding: 16px; border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #ffffff; font-size: 22px; font-weight: 800;">{asset['nome']}</span>
                            <span style="background-color: rgba(255,255,255,0.05); color: #666666; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-family: monospace;">{asset['ticker']}</span>
                        </div>
                        <div style="color: {cor_badge}; font-size: 10px; font-weight: bold; letter-spacing: 0.5px; margin-top: 4px; text-transform: uppercase;">{status_txt}</div>
                        <div style="color: #aaaaaa; font-size: 12px; font-family: monospace; margin-top: 8px;">
                            Ref: <span style="color:#ffffff; font-weight:bold;">{asset['preco']:.2f}</span> 
                            <span style="color: {'#00cc66' if asset['retorno'] >= 0 else '#ff3333'}; margin-left: 6px;">({asset['retorno']*100:+.2f}%)</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #ffffff; font-size: 10px; opacity: 0.4; margin-bottom: 2px; font-family: monospace;">Z-SCORE</div>
                        <div style="color: {cor_badge}; font-size: 32px; font-family: monospace; font-weight: 900; letter-spacing: -0.5px;">{asset['z_score']:+.2f}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 11. Legenda Inferior
    st.markdown(f"""
        <div style="margin-top: 25px; padding: 16px; background-color: #050505; border-radius: 12px; border: 1px solid #11111a;">
            <div style="color: #555555; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">Legenda do Sistema</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 11px; color: #888888;">
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #0088ff;"></div> <span>Z &gt; 0 (Desvio Positivo / Alta)</span></div>
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #888899;"></div> <span>Z &lt; 0 (Desvio Negativo / Baixa)</span></div>
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #ffbb00;"></div> <span>|Z| ≥ {threshold:.1f} (Gatilho de Alerta Ativo)</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if disparar_som and not st.session_state.alertado_na_rodada:
        audio_dados = arquivo_audio.read() if arquivo_audio is not None else None
        emitir_alerta(audio_dados)
        st.session_state.alertado_na_rodada = True

else:
    st.markdown(
        """
        <div style="text-align: center; padding: 60px; color: #666666;">
            <div style="border: 3px solid rgba(0,102,204,0.1); border-top: 3px solid #0066cc; width: 40px; height: 40px; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px auto;"></div>
            <p style="font-size: 14px; font-family: monospace;">Sincronizando com a API do Yahoo Finance...</p>
            <p style="font-size: 11px; color: #444444; margin-top: 5px;">Se demorar, verifique se o mercado de futuros está em horário de negociação ou reinicie o terminal.</p>
            <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
        </div>
        """, 
        unsafe_allow_html=True
    )

time.sleep(1)
st.rerun()
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import base64
from datetime import datetime

# 1. Configuração de Tela Cheia do Dashboard
st.set_page_config(page_title="Monitor Quantitativo", page_icon="📊", layout="wide")

# 2. Injeção de Estilo Premium (Fundo Preto Absoluto e Temática Azul-Marinho / Shadcn UI)
st.markdown("""
    <style>
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #000000 !important;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        #MainMenu, footer {visibility: hidden;}
        
        .custom-header {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: rgba(0, 0, 0, 0.8) !important;
            backdrop-filter: blur(12px);
            border-b: 1px solid #002244;
            padding: 12px 0px;
            margin-bottom: 20px;
        }
        
        .stSlider [data-baseweb="slider"] {
            background-color: #002244 !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Lista de Ativos Alvo
ativos_futuros = {
    "ZM": "ZM=F", "ZN": "ZN=F", "CL": "CL=F", "GC": "GC=F", 
    "NQ": "NQ=F", "ES": "ES=F", "YM": "YM=F", "RTY": "RTY=F", 
    "6E": "6E=F", "6J": "6J=F", "6B": "6B=F", "ZS": "ZS=F"
}

# 4. Engine de Cálculo do Z-Score
def compute_zscore(close_prices, window):
    prices_series = pd.Series(close_prices).dropna()
    returns = prices_series.pct_change().dropna()
    
    if len(returns) < window:
        return np.nan, np.nan, np.nan
    
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    last_price = float(prices_series.iloc[-1].item() if hasattr(prices_series.iloc[-1], 'item') else prices_series.iloc[-1])
    last_return = float(returns.iloc[-1].item() if hasattr(returns.iloc[-1], 'item') else returns.iloc[-1])
    mean = float(rolling_mean.iloc[-1].item() if hasattr(rolling_mean.iloc[-1], 'item') else rolling_mean.iloc[-1])
    std = float(rolling_std.iloc[-1].item() if hasattr(rolling_std.iloc[-1], 'item') else rolling_std.iloc[-1])
    
    if std == 0 or np.isnan(std):
        zscore = 0.0
    else:
        zscore = (last_return - mean) / std
        
    return zscore, last_return, last_price

# 5. Engine para Alertas Sonoros
def emitir_alerta(audio_bytes=None):
    if audio_bytes is not None:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    else:
        audio_html = """
        <script>
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const playBeep = (time, freq) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = freq;
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.3, time);
                gain.gain.exponentialRampToValueAtTime(0.01, time + 0.15);
                osc.start(time);
                osc.stop(time + 0.15);
            };
            const now = ctx.currentTime;
            playBeep(now, 880);
            playBeep(now + 0.2, 1100);
            playBeep(now + 0.4, 880);
        </script>
        """
    st.markdown(audio_html, unsafe_allow_html=True)

# 6. Sidebar
st.sidebar.markdown('<p style="color:#0066cc; font-weight:800; font-size:16px; letter-spacing:0.5px; margin-bottom:5px;">⚙️ CONTROL PANEL</p>', unsafe_allow_html=True)
window = st.sidebar.slider("Janela de Cálculo (Dias)", min_value=5, max_value=100, value=20)
threshold = st.sidebar.slider("Gatilho de Alerta (±σ)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

st.sidebar.write("---")
st.sidebar.markdown('<p style="color:#0066cc; font-weight:800; font-size:14px; margin-bottom:5px;">🎵 SOUND CONFIG</p>', unsafe_allow_html=True)
arquivo_audio = st.sidebar.file_uploader("Upload de áudio personalizado (Opcional)", type=["mp3", "wav"])

# Lógica de Controle do Tempo/Contagem Regressiva
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = time.time()
if 'alertado_na_rodada' not in st.session_state:
    st.session_state.alertado_na_rodada = False

tempo_passado = time.time() - st.session_state.last_fetch_time
countdown = max(0, 60 - int(tempo_passado))

if countdown <= 0:
    st.session_state.last_fetch_time = time.time()
    st.session_state.alertado_na_rodada = False
    st.rerun()

# 7. Coleta robusta de dados compatível com yfinance atualizado (Ajuste Crítico Aqui)
@st.cache_data(ttl=55)
def carregar_dados_seguros(tickers_dict):
    dados_carregados = {}
    symbols = list(tickers_dict.values())
    
    try:
        # Desativa explicitamente o multi_level_index nativo para simplificar a estrutura de colunas
        df = yf.download(symbols, period="1y", group_by="ticker", progress=False, multi_level_index=False)
        
        for nome, ticker in tickers_dict.items():
            # Tenta encontrar no formato com MultiIndex plano ou agrupamento padrão
            if ticker in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else ticker in df.columns:
                try:
                    # Coleta limpa eliminando indexações truncadas do Pandas
                    dados_carregados[nome] = df[ticker]['Close'].dropna()
                except Exception:
                    pass
    except Exception:
        pass
        
    # BACKUP EXTRA: Se o download em lote falhar ou retornar vazio, busca ativo por ativo (Blindagem Máxima)
    for nome, ticker in tickers_dict.items():
        if nome not in dados_carregados or dados_carregados[nome].empty:
            try:
                single_df = yf.download(ticker, period="1y", progress=False, multi_level_index=False)
                if not single_df.empty:
                    if 'Close' in single_df.columns:
                        dados_carregados[nome] = single_df['Close'].dropna()
            except Exception:
                continue
                
    return dados_carregados

dados_finais = carregar_dados_seguros(ativos_futuros)

# 8. Processamento e Lógica de Ordenação do React (sortedAssets)
assets_data = []
alert_count = 0
disparar_som = False

if dados_finais:
    for nome, precos in dados_finais.items():
        try:
            if precos is not None and len(precos) > 0:
                ticker_cod = ativos_futuros[nome]
                zscore, ultimo_retorno, ultimo_preco = compute_zscore(precos, window)
                
                if not np.isnan(zscore):
                    em_alerta = abs(zscore) >= threshold
                    if em_alerta:
                        alert_count += 1
                        disparar_som = True
                        
                    assets_data.append({
                        "nome": nome,
                        "ticker": ticker_cod,
                        "z_score": zscore,
                        "retorno": ultimo_retorno,
                        "preco": ultimo_preco,
                        "abs_z": abs(zscore),
                        "alerta": em_alerta
                    })
        except Exception:
            continue

    # Ordenação decrescente por força absoluta do Z-Score
    assets_data = sorted(assets_data, key=lambda x: x['abs_z'], reverse=True)

# 9. Renderização do StatusHeader
hora_atual = datetime.now().strftime('%H:%M:%S')
progresso_barra = int((countdown / 60) * 100)

st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #002244; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <span style="color: #ffffff; font-size: 24px; font-weight: 800; letter-spacing: 0.5px;">MONITOR <span style="color:#0066cc;">Z-SCORE</span></span>
            <div style="color: #666666; font-size: 11px; font-family: monospace; margin-top: 2px;">SYNC: {hora_atual} • ATIVOS EM EXTREMO: <span style="color:{'#ff3333' if alert_count > 0 else '#00cc66'}; font-weight:bold;">{alert_count}</span></div>
        </div>
        <div style="text-align: right;">
            <div style="color: #0066cc; font-family: monospace; font-size: 13px; font-weight: bold;">PRÓXIMO REFRESH EM {countdown}s</div>
            <div style="width: 160px; background-color: #001122; height: 4px; border-radius: 4px; margin-top: 5px; overflow: hidden; display: inline-block;">
                <div style="background-color: #0066cc; width: {progresso_barra}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 10. Renderização da Grade Grid de Cards do Dashboard
if assets_data:
    colunas_grid = st.columns(2)
    
    for idx, asset in enumerate(assets_data):
        alvo_coluna = colunas_grid[idx % 2]
        
        if asset['alerta']:
            cor_fundo = "#1a1300"
            cor_borda = "#ffbb00"
            cor_badge = "#ffbb00"
            status_txt = "CRITICAL DESVIATION"
        else:
            if asset['z_score'] > 0:
                cor_fundo = "#020a14"
                cor_borda = "#002b5c"
                cor_badge = "#0088ff"
                status_txt = "BULLISH DISPLACEMENT"
            else:
                cor_fundo = "#050508"
                cor_borda = "#1c1d24"
                cor_badge = "#888899"
                status_txt = "BEARISH DISPLACEMENT"

        with alvo_coluna:
            st.markdown(f"""
                <div style="background-color: {cor_fundo}; border: 1px solid {cor_borda}; padding: 16px; border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #ffffff; font-size: 22px; font-weight: 800;">{asset['nome']}</span>
                            <span style="background-color: rgba(255,255,255,0.05); color: #666666; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-family: monospace;">{asset['ticker']}</span>
                        </div>
                        <div style="color: {cor_badge}; font-size: 10px; font-weight: bold; letter-spacing: 0.5px; margin-top: 4px; text-transform: uppercase;">{status_txt}</div>
                        <div style="color: #aaaaaa; font-size: 12px; font-family: monospace; margin-top: 8px;">
                            Ref: <span style="color:#ffffff; font-weight:bold;">{asset['preco']:.2f}</span> 
                            <span style="color: {'#00cc66' if asset['retorno'] >= 0 else '#ff3333'}; margin-left: 6px;">({asset['retorno']*100:+.2f}%)</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #ffffff; font-size: 10px; opacity: 0.4; margin-bottom: 2px; font-family: monospace;">Z-SCORE</div>
                        <div style="color: {cor_badge}; font-size: 32px; font-family: monospace; font-weight: 900; letter-spacing: -0.5px;">{asset['z_score']:+.2f}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 11. Legenda Inferior
    st.markdown(f"""
        <div style="margin-top: 25px; padding: 16px; background-color: #050505; border-radius: 12px; border: 1px solid #11111a;">
            <div style="color: #555555; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">Legenda do Sistema</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 11px; color: #888888;">
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #0088ff;"></div> <span>Z &gt; 0 (Desvio Positivo / Alta)</span></div>
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #888899;"></div> <span>Z &lt; 0 (Desvio Negativo / Baixa)</span></div>
                <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 8px; height: 8px; border-radius: 50%; background-color: #ffbb00;"></div> <span>|Z| ≥ {threshold:.1f} (Gatilho de Alerta Ativo)</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if disparar_som and not st.session_state.alertado_na_rodada:
        audio_dados = arquivo_audio.read() if arquivo_audio is not None else None
        emitir_alerta(audio_dados)
        st.session_state.alertado_na_rodada = True

else:
    st.markdown(
        """
        <div style="text-align: center; padding: 60px; color: #666666;">
            <div style="border: 3px solid rgba(0,102,204,0.1); border-top: 3px solid #0066cc; width: 40px; height: 40px; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px auto;"></div>
            <p style="font-size: 14px; font-family: monospace;">Sincronizando com a API do Yahoo Finance...</p>
            <p style="font-size: 11px; color: #444444; margin-top: 5px;">Se demorar, verifique se o mercado de futuros está em horário de negociação ou reinicie o terminal.</p>
            <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
        </div>
        """, 
        unsafe_allow_html=True
    )

time.sleep(1)
st.rerun()
