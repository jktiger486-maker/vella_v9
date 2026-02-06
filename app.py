# ============================================================
# VELLA_v9 — AUTO TRADING ENGINE
# ============================================================

import os
import sys
import time
from decimal import Decimal, ROUND_DOWN
import requests

# ============================================================
# CFG — USER CONTROL PANEL (TOP OF FILE / FINAL)
# ============================================================

CFG = {
    "01_TRADE_SYMBOL": "SUIUSDT",
    "02_CAPITAL_BASE_USDT": 30,
    "03_CAPITAL_MAX_LOSS_PCT": 100.0,

    # EXIT
    "10_SL_PCT": 5.00,
    "11_TP_PCT": 0.10,
    "12_TP_PARTIAL_PCT": 0.50,
    "13_EXIT_AVG_N": 2,

    # CANDIDATE
    "20_CAND_POOL_TTL_BARS": 1000,
    "21_CAND_POOL_MAX_SIZE": 50,
    "22_CAND_FAIL_MAX": 1000,

    # GATES
    "30_BTC_SESSION_BIAS": False,
    "31_EMA_SLOPE_LOOKBACK_BARS": 0,
    "32_EMA_SLOPE_MIN_PCT": 999.0,
    "33_VOLATILITY_MAX_PCT": 999.0,
    "34_EXECUTION_MIN_PRICE_MOVE_PCT": 0.0,
    "35_EXECUTION_ONLY_ON_NEW_LOW": False,
    "36_EMA_EPS_PCT": 100.0,
}

# ============================================================
# LOGGING
# ============================================================

import logging

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# BINANCE
# ============================================================

try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL, FUTURE_ORDER_TYPE_MARKET
except Exception:
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    FUTURE_ORDER_TYPE_MARKET = "MARKET"
    Client = None

def init_client():
    if Client is None:
        raise RuntimeError("binance client missing")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("credentials missing")
    return Client(api_key, api_secret)

# ============================================================
# CONSTANTS
# ============================================================

BINANCE_SPOT = "https://api.binance.com/api/v3/klines"
EMA_PERIOD = 9

# ============================================================
# DATA
# ============================================================

def fetch_klines(symbol, interval, limit):
    try:
        r = requests.get(
            BINANCE_SPOT,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"fetch_klines: {e}")
        return None

def calc_ema(closes, period):
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for p in closes[period:]:
        ema = (p * k) + (ema * (1 - k))
    return ema

def get_btc_daily_open():
    k = fetch_klines("BTCUSDT", "1d", 1)
    if k:
        return float(k[0][1])
    return None

def get_btc_current():
    k = fetch_klines("BTCUSDT", "5m", 2)
    if k:
        return float(k[-2][4])
    return None

def get_futures_lot_size(client, symbol):
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return {
                            "stepSize": Decimal(f["stepSize"]),
                            "minQty": Decimal(f["minQty"]),
                            "maxQty": Decimal(f["maxQty"])
                        }
        return None
    except Exception as e:
        logger.error(f"get_futures_lot_size: {e}")
        return None

def calculate_quantity(qty_raw, lot_size):
    if lot_size is None:
        return None
    qty_decimal = Decimal(str(qty_raw))
    step = lot_size["stepSize"]
    qty = (qty_decimal / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    if qty < lot_size["minQty"]:
        return None
    if qty > lot_size["maxQty"]:
        qty = lot_size["maxQty"]
    precision = abs(step.as_tuple().exponent)
    return float(qty.quantize(Decimal(10) ** -precision))

# ============================================================
# GATE 0 — BTC_SESSION_BIAS
# ============================================================

def gate_btc_session_bias(btc_current, btc_daily_open):
    if not CFG["30_BTC_SESSION_BIAS"]:
        return True
    if btc_daily_open is None or btc_current is None:
        return False
    return btc_current < btc_daily_open

# ============================================================
# GATE 1 — EMA_SLOPE_LOOKBACK_BARS
# ============================================================

def gate_ema_slope_lookback(ema_values):
    lookback = CFG["31_EMA_SLOPE_LOOKBACK_BARS"]
    if lookback < 0:
        return False
    if len(ema_values) < lookback + 1:
        return False
    return True

# ============================================================
# GATE 2 — EMA_SLOPE_MIN_PCT
# ============================================================

def gate_ema_slope_min_pct(ema_values):
    lookback = CFG["31_EMA_SLOPE_LOOKBACK_BARS"]
    min_pct = CFG["32_EMA_SLOPE_MIN_PCT"]
    if len(ema_values) < lookback + 1:
        return False
    if lookback == 0:
        if len(ema_values) < 2:
            return False
        slope = ((ema_values[-1] - ema_values[-2]) / ema_values[-2]) * 100
    else:
        old = ema_values[-(lookback + 1)]
        new = ema_values[-1]
        slope = ((new - old) / old) * 100
    return slope <= min_pct

# ============================================================
# GATE 3 — VOLATILITY_MAX_PCT
# ============================================================

def gate_volatility_max_pct(klines, lookback=10):
    max_pct = CFG["33_VOLATILITY_MAX_PCT"]
    if len(klines) < lookback:
        return False
    recent = klines[-lookback:]
    h = max([float(k[2]) for k in recent])
    l = min([float(k[3]) for k in recent])
    c = float(recent[-1][4])
    vol = ((h - l) / c) * 100
    return vol <= max_pct

# ============================================================
# GATE 4 — EXECUTION_MIN_PRICE_MOVE_PCT
# ============================================================

def gate_execution_min_price_move_pct(cand_price, current_price):
    min_move = CFG["34_EXECUTION_MIN_PRICE_MOVE_PCT"]
    move = ((cand_price - current_price) / cand_price) * 100
    return move >= min_move

# ============================================================
# GATE 5 — EXECUTION_ONLY_ON_NEW_LOW
# ============================================================

def gate_execution_only_on_new_low(current_price, klines, lookback=10):
    if not CFG["35_EXECUTION_ONLY_ON_NEW_LOW"]:
        return True
    if len(klines) < lookback:
        return False
    lows = [float(k[3]) for k in klines[-lookback:]]
    return current_price < min(lows)

# ============================================================
# GATE 6 — EMA_EPS_PCT
# ============================================================

def gate_ema_eps_pct(current_price, ema):
    max_eps = CFG["36_EMA_EPS_PCT"]
    if current_price >= ema:
        return False
    eps = ((ema - current_price) / ema) * 100
    return eps <= max_eps

# ============================================================
# STAGE 1 — CANDIDATE GENERATION
# ============================================================

def stage_candidate_generation(close, ema, bar):
    if close < ema:
        return {"bar": bar, "price": close, "fail_count": 0}
    return None

# ============================================================
# STAGE 2 — CANDIDATE MANAGEMENT
# ============================================================

def stage_candidate_management(candidates, current_bar):
    ttl = CFG["20_CAND_POOL_TTL_BARS"]
    max_size = CFG["21_CAND_POOL_MAX_SIZE"]
    fail_max = CFG["22_CAND_FAIL_MAX"]
    
    cleaned = []
    for c in candidates:
        if (current_bar - c["bar"]) >= ttl:
            continue
        if c["fail_count"] >= fail_max:
            continue
        cleaned.append(c)
    
    if len(cleaned) > max_size:
        cleaned = cleaned[-max_size:]
    
    return cleaned

# ============================================================
# STAGE 3 — GATES
# ============================================================

def stage_gates(candidate, close, ema, ema_values, klines, btc_current, btc_daily_open):
    gates = [
        lambda: gate_btc_session_bias(btc_current, btc_daily_open),
        lambda: gate_ema_slope_lookback(ema_values),
        lambda: gate_ema_slope_min_pct(ema_values),
        lambda: gate_volatility_max_pct(klines),
        lambda: gate_execution_min_price_move_pct(candidate["price"], close),
        lambda: gate_execution_only_on_new_low(close, klines),
        lambda: gate_ema_eps_pct(close, ema),
    ]
    
    for gate in gates:
        if not gate():
            return False
    
    return True

# ============================================================
# STAGE 4 — ENTRY
# ============================================================

def stage_entry(client, symbol, capital, lot_size):
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        
        qty_raw = capital / price
        qty = calculate_quantity(qty_raw, lot_size)
        if qty is None:
            logger.error(f"entry: qty calculation failed")
            return None
        
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=qty
        )
        
        return {
            "entry_price": price,
            "qty": qty,
            "qty_remaining": qty,
            "tp_triggered": False,
            "entry_bar": None,
            "tp_bar": None
        }
    except Exception as e:
        logger.error(f"entry: {e}")
        return None

# ============================================================
# STAGE 5 — EXIT
# ============================================================

def stage_exit_sl(client, symbol, position, close, lot_size):
    sl_pct = CFG["10_SL_PCT"] / 100
    if close >= position["entry_price"] * (1 + sl_pct):
        qty_remaining = position["qty_remaining"]
        min_qty = float(lot_size["minQty"])
        
        # FIX: minQty 미만 시 경고만 출력, position 종료 처리 금지
        if qty_remaining < min_qty:
            logger.error(f"sl: qty_remaining {qty_remaining} < minQty {min_qty}, cannot close")
            return False
        
        try:
            # FIX: 숏 청산 시 reduceOnly=True 강제
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=qty_remaining,
                reduceOnly=True
            )
            return True
        except Exception as e:
            logger.error(f"sl: {e}")
            return False
    return False

def stage_exit_tp_event(client, symbol, position, close, bar, lot_size):
    if position["tp_triggered"]:
        return position
    tp_pct = CFG["11_TP_PCT"] / 100
    if close <= position["entry_price"] * (1 - tp_pct):
        partial_pct = CFG["12_TP_PARTIAL_PCT"]
        partial_raw = position["qty_remaining"] * partial_pct
        
        partial_qty = calculate_quantity(partial_raw, lot_size)
        
        if partial_qty is None:
            logger.error(f"tp: partial_qty < minQty")
            return position
        
        try:
            # FIX: 숏 청산 시 reduceOnly=True 강제
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=partial_qty,
                reduceOnly=True
            )
            position["qty_remaining"] -= partial_qty
            position["tp_triggered"] = True
            position["tp_bar"] = bar
            
        except Exception as e:
            logger.error(f"tp: {e}")
    return position

def stage_exit_final(client, symbol, position, close_history, bar, lot_size):
    if position["tp_bar"] is not None and position["tp_bar"] == bar:
        return False
    
    if position["tp_triggered"]:
        n = CFG["13_EXIT_AVG_N"]
        if len(close_history) < n:
            return False
        avg = sum(close_history[-n:]) / n
        current = close_history[-1]
        if current > avg:
            qty_remaining = position["qty_remaining"]
            min_qty = float(lot_size["minQty"])
            
            # FIX: minQty 미만 시 경고만 출력, position 종료 처리 금지
            if qty_remaining < min_qty:
                logger.error(f"final_exit: qty_remaining {qty_remaining} < minQty {min_qty}, cannot close")
                return False
            
            try:
                # FIX: 숏 청산 시 reduceOnly=True 강제
                client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_MARKET,
                    quantity=qty_remaining,
                    reduceOnly=True
                )
                return True
            except Exception as e:
                logger.error(f"final_exit: {e}")
                return False
    
    n = CFG["13_EXIT_AVG_N"]
    if len(close_history) < n:
        return False
    avg = sum(close_history[-n:]) / n
    current = close_history[-1]
    if current > avg:
        qty_remaining = position["qty_remaining"]
        min_qty = float(lot_size["minQty"])
        
        # FIX: minQty 미만 시 경고만 출력, position 종료 처리 금지
        if qty_remaining < min_qty:
            logger.error(f"final_exit: qty_remaining {qty_remaining} < minQty {min_qty}, cannot close")
            return False
        
        try:
            # FIX: 숏 청산 시 reduceOnly=True 강제
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=qty_remaining,
                reduceOnly=True
            )
            return True
        except Exception as e:
            logger.error(f"final_exit: {e}")
            return False
    return False

# ============================================================
# ENGINE
# ============================================================

def engine():
    client = init_client()
    symbol = CFG["01_TRADE_SYMBOL"]
    capital = CFG["02_CAPITAL_BASE_USDT"]
    
    lot_size = get_futures_lot_size(client, symbol)
    if lot_size is None:
        logger.error("lot_size retrieval failed")
        return
    
    bar = 0
    last_open_time = None
    candidates = []
    position = None
    close_history = []
    ema_history = []
    
    while True:
        try:
            klines = fetch_klines(symbol, "5m", 100)
            if not klines:
                time.sleep(10)
                continue
            
            completed = klines[-2]
            open_time = completed[0]
            
            if last_open_time == open_time:
                time.sleep(10)
                continue
            
            last_open_time = open_time
            bar += 1
            
            close = float(completed[4])
            close_history.append(close)
            if len(close_history) > 50:
                close_history = close_history[-50:]
            
            closes = [float(k[4]) for k in klines[:-1]]
            ema = calc_ema(closes, EMA_PERIOD)
            if ema is None:
                time.sleep(10)
                continue
            
            ema_history.append(ema)
            if len(ema_history) > 20:
                ema_history = ema_history[-20:]
            
            btc_daily_open = get_btc_daily_open()
            btc_current = get_btc_current()
            
            if position is None:
                cand = stage_candidate_generation(close, ema, bar)
                if cand:
                    candidates.append(cand)
                
                candidates = stage_candidate_management(candidates, bar)
                
                for c in candidates:
                    gate_result = stage_gates(c, close, ema, ema_history, klines[:-1], btc_current, btc_daily_open)
                    if gate_result:
                        position = stage_entry(client, symbol, capital, lot_size)
                        if position:
                            position["entry_bar"] = bar
                            candidates = []
                            break
                    else:
                        c["fail_count"] += 1
            
            else:
                if position["entry_bar"] == bar:
                    time.sleep(10)
                    continue
                
                if stage_exit_sl(client, symbol, position, close, lot_size):
                    position = None
                    continue
                
                position = stage_exit_tp_event(client, symbol, position, close, bar, lot_size)
                
                if stage_exit_final(client, symbol, position, close_history, bar, lot_size):
                    position = None
                    continue
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"engine: {e}")
            time.sleep(10)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    engine()