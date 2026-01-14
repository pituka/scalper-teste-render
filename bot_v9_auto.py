import os
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client

# LER VARIÁVEIS DO RENDER
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)



# inicializar logs
if "logs" not in st.session_state:
    st.session_state.logs = []

# inicializar auto-scan
if "auto_scan_running" not in st.session_state:
    st.session_state.auto_scan_running = False
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = 0


# ================================================================
# 1. CONFIGURAÇÕES GERAIS E CONSTANTES
# ================================================================
load_dotenv()


st.set_page_config(page_title="Scalper v13.2 - Auto-Scan Fixed (120s)", layout="wide")


LOG_ABERTOS = "trades_abertos.csv"
LOG_FECHADOS = "trades_fechados.csv"


TOP_100_SYMBOLS = [
    "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TONUSDT", "TRXUSDT",
    "DOTUSDT", "UNIUSDT", "NEARUSDT", "INJUSDT", "ARBUSDT", "TURBOUSDT", "ICPUSDT",
    "SEIUSDT", "SUIUSDT", "RNDRUSDT", "TIAUSDT", "1000PEPEUSDT", "FETUSDT", "JUPUSDT", "WIFUSDT", "ZECUSDT", "SANDUSDT",
    "AAVEUSDT", "ATOMUSDT", "RUNEUSDT", "ETCUSDT", "FTMUSDT", "GRTUSDT", "STXUSDT", "CRVUSDT", "BLURUSDT", "IMXUSDT",
    "FLOWUSDT", "EGLDUSDT", "XLMUSDT", "FILUSDT", "HBARUSDT", "VETUSDT", "MKRUSDT", "PYTHUSDT", "IOTAUSDT", "ORDIUSDT",
    "ONDOUSDT", "MANAUSDT", "ENSUSDT", "PEOPLEUSDT", "NOTUSDT", "RENDERUSDT", "IOUSDT", "WUSDT", "LDOUSDT", "1000FLOKIUSDT",
    "THETAUSDT", "ALGOUSDT", "KASUSDT", "GALAUSDT", "KAVAUSDT", "JASMYUSDT", "BEAMXUSDT", "POPCATUSDT",
    "BRETTUSDT", "MEWUSDT", "NEIROUSDT", "1000RATSUSDT", "1000LUNCUSDT", "USTCUSDT",
    "POWRUSDT", "WAVESUSDT", "MTLUSDT", "MINAUSDT", "ZENUSDT", "CKBUSDT", "CHRUSDT", "GLMRUSDT", "ASTRUSDT", "METISUSDT",
    "ZROUSDT", "STRKUSDT", "DYMUSDT", "MANTAUSDT", "ALTUSDT", "XAIUSDT", "PORTALUSDT", "AEVOUSDT", "ETHFIUSDT", "ENAUSDT",
]


# ================================================================
# 2. CLIENTE BINANCE E MEMÓRIA PERSISTENTE
# ================================================================
@st.cache_resource
def obter_cliente(api_key: str, api_secret: str):
    try:
        if not api_key or not api_secret:
            return None
        return Client(api_key, api_secret)
    except Exception as e:
        st.error(f"Erro ao criar cliente Binance: {e}")
        return None


def carregar_csv(caminho: str) -> pd.DataFrame:
    if os.path.exists(caminho):
        try:
            return pd.read_csv(caminho)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def salvar_csv(df: pd.DataFrame, caminho: str):
    try:
        df.to_csv(caminho, index=False)
    except Exception as e:
        st.error(f"Erro ao salvar {caminho}: {e}")


def init_session_state():
    if "trades_abertos" not in st.session_state:
        st.session_state.trades_abertos = carregar_csv(LOG_ABERTOS)
    if "trades_fechados" not in st.session_state:
        st.session_state.trades_fechados = carregar_csv(LOG_FECHADOS)
    if "cooldown" not in st.session_state:
        st.session_state.cooldown = {}


init_session_state()


# ================================================================
# 3. INDICADORES: FLUXO (GAUSSIAN) + ATR + EMAs
# ================================================================
def n_pole_gaussian_filter(src: np.ndarray, period: int = 35, order: int = 5) -> np.ndarray:
    w = 2.0 * math.pi / period
    b = (1.0 - math.cos(w)) / (math.pow(1.414, 2.0 / order) - 1.0)
    alpha = -b + math.sqrt(b * b + 2.0 * b)

    filt = np.zeros(len(src))
    a_pow = math.pow(alpha, order)
    om_a = 1.0 - alpha

    for i in range(len(src)):
        if i < order:
            filt[i] = src[i]
            continue

        filt[i] = (
            src[i] * a_pow
            + 5 * om_a * filt[i - 1]
            - 10 * (om_a ** 2) * filt[i - 2]
            + 10 * (om_a ** 3) * filt[i - 3]
            - 5 * (om_a ** 4) * filt[i - 4]
            + (om_a ** 5) * filt[i - 5]
        )

    return filt


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    df["h-l"] = df["high"] - df["low"]
    df["h-pc"] = (df["high"] - df["close"].shift(1)).abs()
    df["l-pc"] = (df["low"] - df["close"].shift(1)).abs()
    tr = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


# ================================================================
# 4. DETECÇÃO DE SINAL (PRECISÃO + ATR + VELA FECHADA)
# ================================================================
def detectar_sinal_precisao(df: pd.DataFrame, symbol: str, adx_min: float = 20.0):

    # ================= CONFIGURAÇÕES =================
    ATRmult = 0.25
    ATRmax = 0.02
    SL_MIN = 0.003
    SL_MAX = 0.025
    RR_MIN = 1.1
    # =================================================

    df = df.copy()

    # ================= INDICADORES =================
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["fluxo"] = n_pole_gaussian_filter(df["close"].values, period=35, order=5)
    df["vol_ema"] = df["vol"].rolling(window=20).mean()
    df["atr"] = calc_atr(df, period=14)
    df["adx"] = calc_adx(df, period=14)

    if len(df) < 50:
        return None

    l = df.iloc[-2]
    p = df.iloc[-3]
    pp = df.iloc[-4]

    adx_val = l["adx"]
    if np.isnan(adx_val) or adx_val < adx_min:
        return None

    atr_val = l["atr"]
    close_val = l["close"]

    if atr_val <= 0 or atr_val / close_val > ATRmax:
        return None

    # ================= VOLUME =================
    if l["vol"] < df["vol_ema"].iloc[-2]:
        return None

    if l["vol"] > df["vol_ema"].iloc[-2] * 3:
        return None

    # ================= CONDIÇÕES =================
    cond_buy = (
        l["close"] > l["ema200"] and
        l["close"] > l["ema50"] and
        l["fluxo"] > p["fluxo"] > pp["fluxo"]
    )

    cond_sell = (
        l["close"] < l["ema200"] and
        l["close"] < l["ema50"] and
        l["fluxo"] < p["fluxo"] < pp["fluxo"]
    )

    lows = df["low"].iloc[-4:-1]
    highs = df["high"].iloc[-4:-1]

    # ================= BUY =================
    if cond_buy:
        entry = l["close"]
        sl = lows.min() - atr_val * ATRmult
        risk = entry - sl

        if risk <= 0:
            return None

        sl_dist_pct = risk / entry

        if sl_dist_pct < SL_MIN or sl_dist_pct > SL_MAX:
            log_event(symbol, "BUY bloqueado por SL inválido", {
                "sl_pct": round(sl_dist_pct * 100, 2)
            })
            return None

        tp = entry + risk * RR_MIN

        return {
            "side": "BUY",
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp)
        }

    # ================= SELL =================
    if cond_sell:
        entry = l["close"]
        sl = highs.max() + atr_val * ATRmult
        risk = sl - entry

        if risk <= 0:
            return None

        sl_dist_pct = risk / entry

        if sl_dist_pct < SL_MIN or sl_dist_pct > SL_MAX:
            log_event(symbol, "SELL bloqueado por SL inválido", {
                "sl_pct": round(sl_dist_pct * 100, 2)
            })
            return None

        tp = entry - risk * RR_MIN

        return {
            "side": "SELL",
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp)
        }

    return None


# ================================================================
# 5. UTILITÁRIOS DE PRECISÃO (QTY, PRICE, STEP, TICK, ID)
# ================================================================
def obter_step_tick(client: Client, symbol: str):
    """
    FIXED: Removida duplicação de chamada API
    """
    try:
        info = client.futures_exchange_info()
        step = 0.001
        tick = 0.0001
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                    if f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                break
        return step, tick
    except Exception as e:
        log_event(symbol, f"Erro ao obter step/tick: {str(e)}", {})
        return 0.001, 0.0001


def arredondar_qtd(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    precisao = max(0, int(-math.log10(step)))
    qty_corrigida = math.floor(qty / step) * step
    return round(qty_corrigida, precisao)


def arredondar_preco(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    precisao = max(0, int(-math.log10(tick)))
    return round(math.floor(price / tick) * tick, precisao)


def gerar_id_trade(symbol: str) -> str:
    agora = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{symbol}-{agora}"


def log_event(symbol, motivo, dados=None):
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol,
        "motivo": motivo,
        "dados": dados
    })


# ================================================================
# 6. EXECUÇÃO REAL NA BINANCE (ONE-WAY FIX)
# ================================================================
def calcular_SL_confluente(client: Client, symbol: str, lado: str, entry: float):
    try:
        klines = client.futures_klines(symbol=symbol, interval="5m", limit=50)
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        atr = np.mean([h - l for h, l in zip(highs, lows)])

        topo = max(highs[-10:-2])
        fundo = min(lows[-10:-2])
        margem = atr * 0.5

        if lado == "SELL":
            sl = topo + margem
        elif lado == "BUY":
            sl = fundo - margem
        else:
            sl = entry

        if abs(sl - entry) < atr * 0.3:
            sl = entry + atr if lado == "SELL" else entry - atr

        return round(sl, 4)

    except Exception as e:
        log_event(symbol, f"Erro no SL confluente: {str(e)}", {})
        return None


def abrir_trade_real(client: Client, symbol: str, side: str, leverage: int,
                    notional_total: float, entry: float, sl: float, tp: float,
                    cooldown_min: int = 5):
    """
    One-way:
    - NÃO usar positionSide.
    - SL/TP: STOP_MARKET e TAKE_PROFIT_MARKET + closePosition=True.
    
    FIXED:
    - Indentação corrigida em SL/TP
    - Buffer aumentado para 0.15%
    - Cooldown ativado automaticamente
    - Proteção contra divisão por zero
    """
    try:
        client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

        step, tick = obter_step_tick(client, symbol)

        qty_bruta = notional_total / entry
        qty = arredondar_qtd(qty_bruta, step)

        if qty <= 0:
            st.error(f"Quantidade inválida calculada para {symbol}.")
            return None

        sl_adj = arredondar_preco(sl, tick)
        tp_adj = arredondar_preco(tp, tick)
        
        # FIXED: Proteção contra divisão por zero
        if abs(entry - sl_adj) <= 0:
            log_event(symbol, "ERRO: SL igual ao entry", {"entry": entry, "sl": sl_adj})
            return None
        
        rr_final = abs(tp_adj - entry) / abs(entry - sl_adj)
        if rr_final < 1.2:
            st.warning(f"⛔ RR final abaixo do mínimo em {symbol}: {rr_final:.2f}")
            return None

        if side == "BUY":
            sl_stop = arredondar_preco(sl_adj * 0.999, tick)
            tp_stop = arredondar_preco(tp_adj * 1.001, tick)
            close_side = "SELL"
        else:
            sl_stop = arredondar_preco(sl_adj * 1.001, tick)
            tp_stop = arredondar_preco(tp_adj * 0.999, tick)
            close_side = "BUY"

        # 1) ENTRADA MARKET (sem positionSide)
        try:
            client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty,
            )
        except Exception as e:
            st.error(f"Erro ao criar ordem de entrada em {symbol}: {e}")
            return None

        time.sleep(0.8)

        # preço de entrada real aproximado
        try:
            ult_trade = client.futures_account_trades(symbol=symbol, limit=1)
            entry_real = float(ult_trade[0]["price"]) if ult_trade else entry
        except Exception:
            entry_real = entry

        trade_id = gerar_id_trade(symbol)
        data_abertura = datetime.now().strftime("%d/%m %H:%M")

        reg = {
            "ID": trade_id,
            "Data_Abertura": data_abertura,
            "Data_Fecho": "",
            "Moeda": symbol,
            "Lado": side,
            "Entry": round(entry_real, 6),
            "SL": round(sl_adj, 6),
            "TP": round(tp_adj, 6),
            "SL_Original": round(sl_adj, 6),
            "SL_Atual": round(sl_adj, 6),
            "TP_Parcial": "",
            "TP_Final": round(tp_adj, 6),
            "Qty_Total": qty,
            "Qty_Parcial": "",
            "Qty_Final": "",
            "BreakEven_Ativado": False,
            "Exit": "",
            "Resultado": "",
            "Lucro_USDT": "",
        }

        # registo imediato
        st.session_state.trades_abertos = pd.concat(
            [st.session_state.trades_abertos, pd.DataFrame([reg])],
            ignore_index=True,
        )
        salvar_csv(st.session_state.trades_abertos, LOG_ABERTOS)
        
        
        # --- Ajuste de SL/TP para evitar "Order would immediately trigger" ---
        entry = float(f"{reg['Entry']:.4f}")
        sl_stop = float(f"{sl_stop:.4f}")
        tp_stop = float(f"{tp_stop:.4f}")

        # FIXED: Buffer aumentado de 0.05% para 0.15%
        buffer = entry * 0.0015  # 0.15%

        if side == "BUY":
            if sl_stop >= entry - buffer:
                sl_stop = entry - buffer
            if tp_stop <= entry + buffer:
                tp_stop = entry + buffer

        elif side == "SELL":
            if sl_stop <= entry + buffer:
                sl_stop = entry + buffer
            if tp_stop >= entry - buffer:
                tp_stop = entry - buffer

        # 2) SL/TP compatíveis com One-way (FIXED: Indentação corrigida)
        try:
            client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=sl_stop,
                closePosition=True,
                workingType="MARK_PRICE",
            )

            client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_stop,
                closePosition=True,
                workingType="MARK_PRICE",
            )

        except Exception as e_sl_tp:
            st.warning(f"⚠️ Trade aberto em {symbol}, mas falhou ao criar SL/TP: {e_sl_tp}")

        # FIXED: Ativar cooldown automaticamente após trade aberto
        st.session_state.cooldown[symbol] = time.time() + cooldown_min * 60
        log_event(symbol, f"Cooldown ativado por {cooldown_min} min", {})

        return reg

    except Exception as e:
        st.error(f"Erro ao abrir trade em {symbol}: {e}")
        return None


# ================================================================
# 7. GESTÃO DE POSIÇÕES, ORDENS E TRADES FECHADOS
# ================================================================
def obter_posicoes_ativas(client: Client):
    try:
        acc = client.futures_position_information()
        pos_ativas = {}
        for p in acc:
            amt = float(p["positionAmt"])
            if amt != 0:
                pos_ativas[p["symbol"].upper()] = p
        return pos_ativas
    except Exception as e:
        st.error(f"Erro ao obter posições: {e}")
        return {}


def obter_simbolos_com_ordens_abertas(client: Client):
    try:
        ordens = client.futures_get_open_orders()
        symbols = set()
        for o in ordens:
            symbol = o.get("symbol") or o.get("s")
            if symbol:
                symbols.add(symbol.upper())
        return symbols
    except Exception as e:
        st.error(f"Erro ao obter ordens abertas: {e}")
        return set()


def cancelar_ordens_universais(client: Client):
    """
    Cancela SL/TP pendentes de símbolos sem posição ativa.
    """
    pos_ativas = obter_posicoes_ativas(client)
    ordens_pendentes = obter_simbolos_com_ordens_abertas(client)

    for symbol in ordens_pendentes:
        if symbol not in pos_ativas:
            try:
                client.futures_cancel_all_open_orders(symbol=symbol)
                log_event(symbol, "Ordens canceladas (sem posição)", {})
            except Exception as e:
                log_event(symbol, f"Erro ao cancelar ordens: {str(e)}", {})


def obter_ultimo_trade(client: Client, symbol: str):
    """
    Último trade para aproximar Exit e PnL realized.
    """
    try:
        trades = client.futures_account_trades(symbol=symbol, limit=1)
        if not trades:
            return None
        t = trades[0]
        price = float(t["price"])
        realized_pnl = float(t.get("realizedPnl", 0.0))
        return price, realized_pnl
    except Exception:
        return None


def classificar_resultado(lado: str, sl: float, tp: float, exit_price: float) -> str:
    """
    Usa SL e TP finais para classificar o resultado.
    """
    if lado == "BUY":
        if exit_price <= sl:
            return "SL"
        if exit_price >= tp:
            return "TP"
        return "MANUAL"
    else:
        if exit_price >= sl:
            return "SL"
        if exit_price <= tp:
            return "TP"
        return "MANUAL"


def atualizar_trades_fechados(client: Client, cooldown_min: int = 5):
    """
    - Cancela ordens pendentes quando não há posição.
    - Move trades para fechados.
    """
    pos_ativas = obter_posicoes_ativas(client)

    # 1) limpar ordens pendentes em símbolos flat
    cancelar_ordens_universais(client)

    if st.session_state.trades_abertos.empty:
        return

    ainda_abertos = []
    fechados_novos = []

    for _, row in st.session_state.trades_abertos.iterrows():
        symbol = str(row["Moeda"]).strip().upper()
        lado = row["Lado"]
        sl = float(row["SL_Atual"]) if "SL_Atual" in row and not pd.isna(row["SL_Atual"]) else float(row["SL"])
        tp_final = float(row["TP_Final"]) if "TP_Final" in row and not pd.isna(row["TP_Final"]) else float(row["TP"])

        if symbol in pos_ativas:
            ainda_abertos.append(row)
            continue

        # 2) acabou a posição -> cancela ordens desse símbolo imediatamente
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception:
            pass

        ultimo = obter_ultimo_trade(client, symbol)
        if ultimo is None:
            exit_price = float(row["Entry"])
            lucro = 0.0
        else:
            exit_price, realized_pnl = ultimo
            lucro = realized_pnl

        resultado = classificar_resultado(lado, sl, tp_final, exit_price)

        data_fecho = datetime.now().strftime("%d/%m %H:%M")
        novo_reg = row.copy()
        novo_reg["Data_Fecho"] = data_fecho
        novo_reg["Exit"] = round(exit_price, 6)
        novo_reg["Resultado"] = resultado
        novo_reg["Lucro_USDT"] = round(lucro, 4)
        fechados_novos.append(novo_reg)

    st.session_state.trades_abertos = pd.DataFrame(ainda_abertos)
    if fechados_novos:
        st.session_state.trades_fechados = pd.concat(
            [st.session_state.trades_fechados, pd.DataFrame(fechados_novos)],
            ignore_index=True,
        )

    salvar_csv(st.session_state.trades_abertos, LOG_ABERTOS)
    salvar_csv(st.session_state.trades_fechados, LOG_FECHADOS)


# ================================================================
# 8. SCANNER (MANUAL + AUTO)
# ================================================================
def iniciar_scan(lev_val: int, notional_ui: float, adx_min: float, cooldown_min: int = 5):
    """
    FIXED: adx_min agora é passado como parâmetro
    """
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = obter_cliente(api_key, api_secret)

    if not client:
        st.error("Cliente Binance não inicializado. Verifica API KEY e SECRET.")
        return

    # tempo para Binance atualizar após fecho manual
    time.sleep(0.6)

    atualizar_trades_fechados(client, cooldown_min=cooldown_min)

    symbols_com_ordens = obter_simbolos_com_ordens_abertas(client)
    pos_ativas = obter_posicoes_ativas(client)
    symbols_ocupados = set(pos_ativas.keys()).union(symbols_com_ordens)

    if len(symbols_ocupados) >= 4:
        st.sidebar.warning("Limite de 4 trades atingido (posição ou SL/TP pendente).")
        return

    msg = st.empty()

    for sym in TOP_100_SYMBOLS:
        # cooldown check
        until = st.session_state.cooldown.get(sym, 0)
        if time.time() < until:
            continue

        if sym in pos_ativas or sym in symbols_com_ordens:
            continue

        try:
            msg.text(f"🔍 Analisando {sym}")

            k = client.futures_klines(symbol=sym, interval="5m", limit=300)
            df = pd.DataFrame(
                k,
                columns=[
                    "ot", "open", "high", "low", "close", "vol",
                    "ct", "qav", "nt", "tbb", "tbq", "i",
                ],
            )

            df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)

            sinal = detectar_sinal_precisao(df, symbol=sym, adx_min=adx_min)
            if not sinal:
                continue

            st.success(f"🎯 SINAL {sinal['side']} EM {sym}!")
            entry = sinal["entry"]
            lado = sinal["side"]
            sl = sinal["sl"]
            tp = sinal["tp"]

            res = abrir_trade_real(
                client=client,
                symbol=sym,
                side=lado,
                leverage=lev_val,
                notional_total=notional_ui,
                entry=entry,
                sl=sl,
                tp=tp,
                cooldown_min=cooldown_min,
            )

            if res:
                st.success(f"Trade aberto em {sym} com ID {res['ID']}")
                msg.empty()
                return

            time.sleep(0.2)

        except Exception as e:
            log_event(sym, f"Erro no scan: {str(e)}", {})
            continue


# ================================================================
# 9. INTERFACE STREAMLIT
# ================================================================
st.sidebar.header("⚙️ Configuração Scalper v13.2 (Auto-Scan 300s)")

lev_val = st.sidebar.slider("Alavancagem (x)", 1, 20, 10)
adx_min = st.sidebar.slider("ADX mínimo (5m)", 0.0, 50.0, 20.0, 0.5)
notional_ui = st.sidebar.number_input(
    "Tamanho Total da Posição (USDT)",
    min_value=5.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
)

cooldown_min = st.sidebar.slider("Cooldown por símbolo (min)", 0, 30, 5)

custo_estimado = notional_ui / lev_val
st.sidebar.info(f"💰 Custo real estimado por trade: ~{custo_estimado:.2f} USDT")

# FIXED: Auto-scan correctamente implementado (SEM congelar UI)
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🚀 SCAN MANUAL"):
        iniciar_scan(lev_val, notional_ui, adx_min=adx_min, cooldown_min=cooldown_min)

with col2:
    if st.button("🤖 AUTO-SCAN ON/OFF"):
        st.session_state.auto_scan_running = not st.session_state.auto_scan_running
        if st.session_state.auto_scan_running:
            st.sidebar.success("✅ Auto-scan ACTIVADO")
        else:
            st.sidebar.info("⏸️ Auto-scan DESACTIVADO")

# Auto-scan a cada 300 segundos (CORRIGIDO - não congela)
if st.session_state.auto_scan_running:
    tempo_agora = time.time()
    tempo_restante = 300 - (tempo_agora - st.session_state.last_scan_time)
    
    if tempo_restante <= 0:
        st.session_state.last_scan_time = tempo_agora
        with st.spinner("🔄 Auto-scanning..."):
            iniciar_scan(lev_val, notional_ui, adx_min=adx_min, cooldown_min=cooldown_min)
    
    # CONTADOR VISÍVEL (DEBUG)
    st.sidebar.metric("⏱️ Próximo scan", f"{int(max(0, tempo_restante))}s")
    
    st.rerun()  # ← AQUI é que fica (roda sempre)


# Status do auto-scan
if st.session_state.auto_scan_running:
    st.sidebar.success("🟢 AUTO-SCAN ACTIVO (a cada 300s)")
else:
    st.sidebar.info("⚪ Auto-scan desactivado")

st.title("Fluxo Scalper v13.2 - Gestão de Banca (Auto-Scan 300s)")

st.subheader("📂 Trades Abertos")
if st.session_state.trades_abertos.empty:
    st.write("Nenhum trade aberto.")
else:
    st.dataframe(st.session_state.trades_abertos.tail(50), use_container_width=True)

st.subheader("📊 Trades Fechados (últimos 50)")
if st.session_state.trades_fechados.empty:
    st.write("Ainda não há trades fechados.")
else:
    st.dataframe(st.session_state.trades_fechados.tail(50), use_container_width=True)

# ================================================================
# 10. LOGS DE EVENTOS
# ================================================================
st.subheader("📋 Logs de Eventos")

if st.session_state.logs:
    st.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True)
else:
    st.info("Nenhum log registado até agora.")

# ================================================================
# 11. ESTATÍSTICAS DE PERFORMANCE
# ================================================================
st.subheader("📈 Estatísticas da Estratégia")
df = st.session_state.trades_fechados.copy()

if df.empty:
    st.info("Ainda não há trades suficientes para calcular estatísticas.")
else:
    try:
        df["Data"] = pd.to_datetime(df["Data_Fecho"], format="%d/%m %H:%M")
    except Exception:
        df["Data"] = pd.to_datetime(df["Data_Fecho"], errors="coerce")

    total = len(df)
    tp_count = (df["Resultado"] == "TP").sum()
    winrate = (tp_count / total) * 100 if total > 0 else 0

    rrs = []
    for _, row in df.iterrows():
        entry = float(row["Entry"])
        tp_col = row["TP_Final"] if "TP_Final" in row and not pd.isna(row["TP_Final"]) else row["TP"]
        sl_col = row["SL_Atual"] if "SL_Atual" in row and not pd.isna(row["SL_Atual"]) else row["SL"]
        sl_val = float(sl_col)
        tp_val = float(tp_col)
        risk = abs(entry - sl_val)
        reward = abs(tp_val - entry)
        if risk > 0:
            rrs.append(reward / risk)

    rr_medio = np.mean(rrs) if len(rrs) > 0 else 0

    lucro_series = pd.to_numeric(df["Lucro_USDT"], errors="coerce").fillna(0.0)
    lucro_total = float(lucro_series.sum())

    df = df.sort_values("Data")
    df["Equity"] = lucro_series.cumsum()
    max_equity = df["Equity"].cummax()
    drawdown = df["Equity"] - max_equity
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Winrate", f"{winrate:.1f}%")
    col2.metric("📐 RR Médio", f"{rr_medio:.2f}")
    col3.metric("💰 Lucro Acumulado", f"{lucro_total:.2f} USDT")
    col4.metric("📉 Máx. Drawdown", f"{max_dd:.2f} USDT")


    st.line_chart(df.set_index("Data")["Equity"])
