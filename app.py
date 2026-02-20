# ============================================================
# VELLA_v9 — SHORT ENGINE (FINAL: 벨라 EXIT 정밀도 강화)
# ============================================================

import os
import sys
import time
import signal
import logging
import requests
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# ============================================================
# CFG
# ============================================================

CFG = {
    "01_TRADE_SYMBOL": "SEIUSDT",
    "02_INTERVAL": "5m",
    "03_CAPITAL_BASE_USDT": 10.0,
    "04_LEVERAGE": 1,
    
    "10_EMA_FAST": 7,
    "11_EMA_MID": 12,
    
    "17_SLOPE_PCT_BARS": 2,
    "18_SLOPE_PCT_MIN": 0.00,
    
    "19_EXEC_MIN_MOVE_PCT": 0.0,
    
    "20_ENTRY_COOLDOWN_BARS": 0,
    "21_MAX_ENTRY_PER_TREND": 2,
    
    "22_CONFIRM_BARS": 0,
    
    "23_ENTRY2_ENABLE": True,
    
    "30_EXIT_EMA": 5,
    
    "40_SL_ENABLE": False,
    "41_SL_PCT": 1.2,
    
    "50_TIMEOUT_EXIT_ENABLE": False,
    "51_TIMEOUT_BARS": 60,
    
    "90_KLINE_LIMIT": 1500,
    "91_POLL_SEC": 5,
    "92_LOG_LEVEL": "INFO",
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["92_LOG_LEVEL"], logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VELLA_v9_SHORT")

# ============================================================
# BINANCE
# ============================================================

try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
except Exception:
    Client = None
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    ORDER_TYPE_MARKET = "MARKET"

BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"

def init_client() -> "Client":
    if Client is None:
        raise RuntimeError("python-binance missing")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET")
    return Client(api_key, api_secret)

def set_leverage(client: "Client", symbol: str, leverage: int) -> None:
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        log.error(f"set_leverage failed: {e}")

def fetch_klines_futures(symbol: str, interval: str, limit: int) -> Optional[List[Any]]:
    try:
        r = requests.get(
            BINANCE_FUTURES_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_klines_futures: {e}")
        return None

def get_futures_lot_size(client: "Client", symbol: str) -> Optional[Dict[str, Decimal]]:
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return {
                            "stepSize": Decimal(f["stepSize"]),
                            "minQty": Decimal(f["minQty"]),
                            "maxQty": Decimal(f["maxQty"]),
                        }
        return None
    except Exception as e:
        log.error(f"get_futures_lot_size: {e}")
        return None

def calculate_quantity(qty_raw: float, lot: Dict[str, Decimal]) -> Optional[float]:
    if lot is None:
        return None
    qty_decimal = Decimal(str(qty_raw))
    step = lot["stepSize"]
    qty = (qty_decimal / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    if qty < lot["minQty"]:
        return None
    if qty > lot["maxQty"]:
        qty = lot["maxQty"]
    precision = abs(step.as_tuple().exponent)
    return float(qty.quantize(Decimal(10) ** -precision))

# ============================================================
# INDICATORS
# ============================================================

def ema_series(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    if len(values) < period:
        return [values[0]] * len(values)
    k = 2 / (period + 1)
    out = [values[0]] * len(values)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = (values[i] * k) + (prev * (1 - k))
        out[i] = prev
    for i in range(period - 1):
        out[i] = out[period - 1]
    return out

# ============================================================
# STATE
# ============================================================

@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    entry_bar: int
    entry_type: str = "E1"

@dataclass
class ShortEntryState:
    entry_count: int = 0
    signal_bar: int = -1

@dataclass
class EngineState:
    bar: int = 0
    last_open_time: Optional[int] = None
    cooldown_until_bar: int = 0
    entry_state: ShortEntryState = field(default_factory=ShortEntryState)
    position: Optional[Position] = None
    close_history: List[float] = None

    def __post_init__(self):
        self.close_history = []

# ============================================================
# ENTRY FILTERS
# ============================================================

def filter_slope_pct(ema_fast_s: List[float]) -> bool:
    min_pct = float(CFG["18_SLOPE_PCT_MIN"])
    if min_pct == 0:
        return True
    bars = int(CFG["17_SLOPE_PCT_BARS"])
    if len(ema_fast_s) < bars + 1:
        return False
    old = ema_fast_s[-(bars + 1)]
    new = ema_fast_s[-1]
    if old == 0:
        return False
    slope_pct = ((new - old) / old) * 100
    return slope_pct <= -min_pct

def filter_exec_min_move(close: float, ema_mid: float) -> bool:
    min_move = float(CFG["19_EXEC_MIN_MOVE_PCT"])
    if min_move == 0:
        return True
    if ema_mid == 0:
        return False
    dist_pct = abs((close - ema_mid) / ema_mid) * 100
    return dist_pct <= min_move

def filter_confirm_bars(st: ShortEntryState, current_bar: int, raw_signal: bool) -> bool:
    confirm_n = int(CFG["22_CONFIRM_BARS"])
    if confirm_n == 0:
        return raw_signal

    if raw_signal:
        if st.signal_bar < 0:
            st.signal_bar = current_bar
            return True
        bars_since = current_bar - st.signal_bar
        if bars_since <= confirm_n:
            return True
        else:
            st.signal_bar = -1
            return False
    else:
        if st.signal_bar >= 0:
            bars_since = current_bar - st.signal_bar
            if bars_since > confirm_n:
                st.signal_bar = -1
        return False

# ============================================================
# ENTRY 1 (Dead Cross)
# ============================================================

def short_entry1_signal(
    closes: List[float],
    ema_fast_s: List[float],
    ema_mid_s: List[float],
    st: ShortEntryState,
    current_bar: int
) -> bool:
    if len(closes) < max(CFG["11_EMA_MID"], 60):
        return False

    ema_mid = ema_mid_s[-1]
    close = closes[-1]

    max_entry = int(CFG["21_MAX_ENTRY_PER_TREND"])
    if max_entry <= 0:
        return False

    raw_signal = (
        (ema_fast_s[-2] >= ema_mid_s[-2]) and
        (ema_fast_s[-1] < ema_mid_s[-1])
    )
    if not raw_signal:
        return False

    if not filter_slope_pct(ema_fast_s):
        return False

    if not filter_exec_min_move(close, ema_mid):
        return False

    if not filter_confirm_bars(st, current_bar, raw_signal):
        return False

    if st.entry_count >= max_entry:
        log.info(f"[ENTRY_BLOCKED] E1 타점이나 최대 횟수({max_entry}) 도달로 패스")
        return False

    return True

# ============================================================
# ENTRY 2 (Re-Acceleration)
# ============================================================

def short_entry2_signal(
    closes: List[float],
    ema_fast_s: List[float],
    ema_mid_s: List[float],
    st: ShortEntryState
) -> bool:
    if not CFG["23_ENTRY2_ENABLE"]:
        return False
    
    if len(closes) < 3:
        return False
    
    max_entry = int(CFG["21_MAX_ENTRY_PER_TREND"])
    if max_entry <= 0:
        return False
    
    if ema_fast_s[-1] >= ema_mid_s[-1]:
        return False
    
    pullback = closes[-2] > ema_fast_s[-2]
    reentry = closes[-1] < ema_fast_s[-1]
    
    if pullback and reentry:
        if st.entry_count >= max_entry:
            log.info(f"[ENTRY_BLOCKED] E2(재가속) 타점이나 최대 횟수({max_entry}) 도달로 패스")
            return False
        return True
    
    return False

# ============================================================
# ENTRY 실행 후 처리
# ============================================================

def on_entry_executed(st: ShortEntryState) -> None:
    st.signal_bar = -1
    st.entry_count += 1
    log.info(f"[ENTRY_COUNT] now={st.entry_count}/{CFG['21_MAX_ENTRY_PER_TREND']}")

# ============================================================
# EXIT
# ============================================================

def exit_signal(state: EngineState, ema_fast_s: List[float], ema_mid_s: List[float]) -> bool:
    pos = state.position
    if pos is None:
        return False
    
    close = state.close_history[-1]
    
    # SL (CFG OFF 시 무시)
    if CFG["40_SL_ENABLE"]:
        sl = float(CFG["41_SL_PCT"]) / 100.0
        if close >= pos.entry_price * (1.0 + sl):
            log.info(f"[EXIT_SL] close={close} >= SL={pos.entry_price * (1.0 + sl)}")
            return True
    
    # Timeout (CFG OFF 시 무시)
    if CFG["50_TIMEOUT_EXIT_ENABLE"]:
        if (state.bar - pos.entry_bar) >= int(CFG["51_TIMEOUT_BARS"]):
            log.info(f"[EXIT_TIMEOUT] bars={state.bar - pos.entry_bar}")
            return True
    
    # EMA EXIT
    ema_exit_s = ema_series(state.close_history, CFG["30_EXIT_EMA"])
    ema_exit_now = ema_exit_s[-1]
    
    if close > ema_exit_now:
        log.info(f"[EXIT_EMA] close={close} > EMA6={ema_exit_now}")
        return True
    
    # === 벨라 FIX: 골든크로스 발생 시점만 EXIT (정밀도 강화) ===
    if len(ema_fast_s) >= 2 and len(ema_mid_s) >= 2:
        if ema_fast_s[-2] < ema_mid_s[-2] and ema_fast_s[-1] >= ema_mid_s[-1]:
            log.info(f"[EXIT_GOLDEN_CROSS] 골든크로스 발생 (FAST ↑ MID)")
            return True
    
    return False

# ============================================================
# EXECUTION
# ============================================================

def place_short_entry(client: "Client", symbol: str, capital_usdt: float, lot: Dict[str, Decimal]) -> Optional[Dict[str, Any]]:
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        leverage = int(CFG["04_LEVERAGE"])
        notional = float(capital_usdt) * float(leverage)
        qty_raw = notional / price
        qty = calculate_quantity(qty_raw, lot)
        if qty is None:
            log.error("entry: qty calculation failed")
            return None
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
        )
        return {"entry_price": price, "qty": qty}
    except Exception as e:
        log.error(f"place_short_entry: {e}")
        return None

def place_short_exit(client: "Client", symbol: str, qty: float, lot: Dict[str, Decimal]) -> bool:
    try:
        qty_rounded = calculate_quantity(qty, lot)
        if qty_rounded is None:
            log.error("exit: qty too small")
            return False
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty_rounded,
            reduceOnly=True
        )
        return True
    except Exception as e:
        log.error(f"place_short_exit: {e}")
        return False

# ============================================================
# ENGINE LOOP
# ============================================================

STOP = False
def _sig_handler(_sig, _frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def engine():
    client = init_client()
    symbol = CFG["01_TRADE_SYMBOL"]
    interval = CFG["02_INTERVAL"]
    capital = float(CFG["03_CAPITAL_BASE_USDT"])

    set_leverage(client, symbol, int(CFG["04_LEVERAGE"]))

    lot = get_futures_lot_size(client, symbol)
    if lot is None:
        raise RuntimeError("lot_size retrieval failed")

    st = EngineState()

    log.info(f"START v9 SHORT | symbol={symbol} interval={interval} capital={capital} lev={CFG['04_LEVERAGE']}")

    while not STOP:
        try:
            kl = fetch_klines_futures(symbol, interval, int(CFG["90_KLINE_LIMIT"]))
            if not kl:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            completed = kl[-2]
            open_time = int(completed[0])

            if st.last_open_time == open_time:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            if not st.close_history:
                for k in kl[:-1]:
                    st.close_history.append(float(k[4]))
                st.bar = len(st.close_history)
                st.last_open_time = int(kl[-2][0])
                continue

            st.last_open_time = open_time
            st.bar += 1

            close = float(completed[4])
            st.close_history.append(close)

            if len(st.close_history) > 2000:
                st.close_history = st.close_history[-2000:]

            # EMA 계산 (루프당 1회, 전역)
            ema_fast_s = ema_series(st.close_history, CFG["10_EMA_FAST"])
            ema_mid_s = ema_series(st.close_history, CFG["11_EMA_MID"])

            # === 트렌드 리셋 (상태 기준: FAST >= MID 유지 중) ===
            if ema_fast_s[-1] >= ema_mid_s[-1]:
                if st.entry_state.entry_count > 0:
                    log.info(f"[TREND_RESET] 하락 추세 종료 (FAST >= MID) → entry_count 리셋")
                    st.entry_state.entry_count = 0

            if st.position is None:
                if st.bar < st.cooldown_until_bar:
                    continue

                sig_entry1 = short_entry1_signal(
                    st.close_history,
                    ema_fast_s,
                    ema_mid_s,
                    st.entry_state,
                    st.bar
                )

                sig_entry2 = False
                entry_type = "E1"

                if not sig_entry1:
                    sig_entry2 = short_entry2_signal(
                        st.close_history,
                        ema_fast_s,
                        ema_mid_s,
                        st.entry_state
                    )
                    if sig_entry2:
                        entry_type = "E2"

                if sig_entry1 or sig_entry2:
                    order = place_short_entry(client, symbol, capital, lot)
                    if order:
                        st.position = Position(
                            side="SHORT",
                            entry_price=float(order["entry_price"]),
                            qty=float(order["qty"]),
                            entry_bar=st.bar,
                            entry_type=entry_type
                        )
                        on_entry_executed(st.entry_state)
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                        log.info(f"[ENTRY] SHORT type={entry_type} qty={st.position.qty} entry={st.position.entry_price} bar={st.bar}")
                    else:
                        log.error("[ENTRY_FAIL] order failed")
            else:
                if st.position.entry_bar == st.bar:
                    continue

                if exit_signal(st, ema_fast_s, ema_mid_s):
                    ok = place_short_exit(client, symbol, st.position.qty, lot)
                    if ok:
                        log.info(f"[EXIT] SHORT type={st.position.entry_type} close={close} entry={st.position.entry_price} bar={st.bar}")
                        st.position = None
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                    else:
                        log.error("[EXIT_FAIL] order failed")

        except Exception as e:
            log.error(f"engine loop error: {e}")
            time.sleep(CFG["91_POLL_SEC"])

    log.info("STOP v9 SHORT")

if __name__ == "__main__":
    engine()