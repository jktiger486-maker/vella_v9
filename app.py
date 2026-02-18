# ============================================================
# VELLA_v9 — SHORT ENGINE (Binance Futures)
# - EXECUTION CORE: based on v7 proven trade plumbing (lotSize/qty/order/reduceOnly/closed-bar loop)
# - ENTRY 1: EMA_FAST ↓ EMA_MID (dead cross)
# - ENTRY 2: EMA9 재이탈 (Re-Acceleration) — SLOT ONLY (기본 OFF)
# - EXIT: close > EMA_EXIT
# - TIME AXIS: REST closed-bar only (kline[-2])
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
# CFG (ALL CONTROL HERE)
# ============================================================
# 20260210_1730 : 클로드 엔트리 + 벨라 n봉 엑시트 기본 매매확인후 필터 추가
# 20260211      : Step1~3 이식 — slope PCT / exec min move / confirm bars
# 20260211      : 죽은 CFG 13~16 제거 / logger 이름 수정 / confirm_bars TTL 버그 수정
# 20260211_0930 : 매매 확인 우선, 이후 22번 0>1 19번 0>0.1
# 20260211_1210 : 벨라 세팅
# - 추세 시작 첫 음봉 거의 다 잡음
# - 눌림 후 재하락도 재진입
# - slope 완화로 초입 빠르게 탑승
# - exit 3봉 평균으로 수익 조금 더 끌고 감
# - SL 1.2%로 큰 역추세 방어))
# 20260215_BASE : 순수 EMA 교차 엣지 검증 1탄
# 20260217_PURE : EMA_SLOW 제거 / EXIT EMA4 / 필터 OFF 유지
# 20260217_FIX  : EXIT 후 entry_count 리셋 추가
# 20260218_E2   : ENTRY 2 슬롯 추가 (기본 OFF / 벨라팀 확정 구조)

CFG = {
    # -------------------------
    # BASIC
    # -------------------------
    "01_TRADE_SYMBOL": "SEIUSDT",
    "02_INTERVAL": "5m",
    "03_CAPITAL_BASE_USDT": 30.0,
    "04_LEVERAGE": 1,

    # -------------------------
    # ENTRY 1 (v9 SHORT — Dead Cross)
    # -------------------------
    "10_EMA_FAST": 9,
    "11_EMA_MID": 14,

    # -------------------------
    # ENTRY FILTER — STEP 1: EMA Slope PCT (횡보장 차단)
    # -------------------------
    "17_SLOPE_PCT_BARS": 2,  # 18번과 연계 18번 0이면 off 상태
    "18_SLOPE_PCT_MIN": 0.00,

    # -------------------------
    # ENTRY FILTER — STEP 2: Execution Min Move (추격 진입 차단)
    # -------------------------
    "19_EXEC_MIN_MOVE_PCT": 0.0,

    # -------------------------
    # ENTRY MANAGEMENT FILTERS (plug-in slots)
    # -------------------------
    "20_ENTRY_COOLDOWN_BARS": 0,
    "21_MAX_ENTRY_PER_TREND": 999,

    # -------------------------
    # ENTRY FILTER — STEP 3: Confirm Bars (시그널 후 N봉 확인)
    # -------------------------
    "22_CONFIRM_BARS": 0,

    # -------------------------
    # ENTRY 2 (Re-Acceleration Short) — SLOT ONLY (기본 OFF)
    # -------------------------
    "23_ENTRY2_ENABLE": True,

    # -------------------------
    # EXIT
    # -------------------------
    "30_EXIT_EMA": 4,

    # -------------------------
    # EXIT OPTIONS (plug-in slots; default OFF)
    # -------------------------
    "40_SL_ENABLE": False,
    "41_SL_PCT": 1.2,

    "50_TIMEOUT_EXIT_ENABLE": False,
    "51_TIMEOUT_BARS": 60,

    # -------------------------
    # ENGINE
    # -------------------------
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
# BINANCE (v7 style)
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
        raise RuntimeError("python-binance missing. pip install python-binance")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET env vars.")
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
# ENTRY FILTERS (독립 함수 — CFG만 읽음)
# ============================================================

def filter_slope_pct(ema_fast_s: List[float]) -> bool:
    """
    Step1: EMA slope PCT 필터 (횡보장 차단)
    숏 기준: EMA_FAST가 N봉 전 대비 이 % 이상 하락해야 통과
    CFG 18_SLOPE_PCT_MIN == 0 이면 OFF
    """
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
    """
    Step2: Execution Min Move 필터 (추격 숏 진입 차단)
    CFG 19_EXEC_MIN_MOVE_PCT == 0 이면 OFF
    현재가가 EMA_MID 대비 이 % 이내일 때만 진입 허용
    """
    min_move = float(CFG["19_EXEC_MIN_MOVE_PCT"])
    if min_move == 0:
        return True
    if ema_mid == 0:
        return False
    dist_pct = abs((close - ema_mid) / ema_mid) * 100
    return dist_pct <= min_move

def filter_confirm_bars(st: ShortEntryState, current_bar: int, raw_signal: bool) -> bool:
    """
    Step3: Confirm Bars 필터 (시그널 후 N봉 재확인)
    CFG 22_CONFIRM_BARS == 0 이면 OFF (즉시 진입)
    """
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
    """
    ENTRY 1: EMA9 ↓ EMA14 (Dead Cross)
    """
    if len(closes) < max(CFG["11_EMA_MID"], 60):
        return False

    ema_mid = ema_mid_s[-1]
    close   = closes[-1]

    max_entry = int(CFG["21_MAX_ENTRY_PER_TREND"])
    if max_entry <= 0:
        return False
    if st.entry_count >= max_entry:
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

    return True

# ============================================================
# ENTRY 2 (Re-Acceleration) — SLOT ONLY
# ============================================================

def short_entry2_signal(
    closes: List[float],
    ema_fast_s: List[float],
    ema_mid_s: List[float],
    st: ShortEntryState
) -> bool:
    """
    ENTRY 2: EMA9 재이탈 (Re-Acceleration Short)
    전제: EMA9 < EMA14
    조건: 직전 봉 close > EMA9 AND 현재 봉 close < EMA9
    """
    if not CFG["23_ENTRY2_ENABLE"]:
        return False
    
    if len(closes) < 3:
        return False
    
    max_entry = int(CFG["21_MAX_ENTRY_PER_TREND"])
    if max_entry <= 0:
        return False
    if st.entry_count >= max_entry:
        return False
    
    # 전제: 하방 정렬
    if ema_fast_s[-1] >= ema_mid_s[-1]:
        return False
    
    # 눌림 + 재이탈
    pullback = closes[-2] > ema_fast_s[-2]
    reentry  = closes[-1] < ema_fast_s[-1]
    
    return pullback and reentry

# ============================================================
# ENTRY 실행 후 처리
# ============================================================

def on_entry_executed(st: ShortEntryState) -> None:
    st.signal_bar = -1
    st.entry_count += 1

# ============================================================
# EXIT (EMA_EXIT based)
# ============================================================

def exit_option_sl(close: float, entry_price: float) -> bool:
    if not CFG["40_SL_ENABLE"]:
        return False
    sl = float(CFG["41_SL_PCT"]) / 100.0
    return close >= entry_price * (1.0 + sl)

def exit_option_timeout(current_bar: int, entry_bar: int) -> bool:
    if not CFG["50_TIMEOUT_EXIT_ENABLE"]:
        return False
    return (current_bar - entry_bar) >= int(CFG["51_TIMEOUT_BARS"])

def exit_signal(state: EngineState) -> bool:
    pos = state.position
    if pos is None:
        return False
    close = state.close_history[-1]
    if exit_option_sl(close, pos.entry_price):
        return True
    if exit_option_timeout(state.bar, pos.entry_bar):
        return True
    ema_exit_s = ema_series(state.close_history, CFG["30_EXIT_EMA"])
    ema_exit_now = ema_exit_s[-1]
    close_now = state.close_history[-1]
    if close_now > ema_exit_now:
        return True
    return False

# ============================================================
# EXECUTION (v7-style order plumbing)
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
            log.error("entry: qty calculation failed (minQty/stepSize)")
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
            log.error("exit: qty too small (minQty) — cannot close")
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
# ENGINE LOOP (closed bar only)
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

            if st.position is None:
                if st.bar < st.cooldown_until_bar:
                    continue

                # EMA 계산 (루프당 1회, E1/E2 공유)
                ema_fast_s = ema_series(st.close_history, CFG["10_EMA_FAST"])
                ema_mid_s  = ema_series(st.close_history, CFG["11_EMA_MID"])

                # ENTRY 1 판정
                sig_entry1 = short_entry1_signal(
                    st.close_history,
                    ema_fast_s,
                    ema_mid_s,
                    st.entry_state,
                    st.bar
                )

                # ENTRY 2 판정 (E1 우선)
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

                if exit_signal(st):
                    ok = place_short_exit(client, symbol, st.position.qty, lot)
                    if ok:
                        log.info(f"[EXIT] SHORT type={st.position.entry_type} close={close} entry={st.position.entry_price} bar={st.bar}")
                        st.position = None
                        st.entry_state.entry_count = 0
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                    else:
                        log.error("[EXIT_FAIL] order failed (kept position)")

        except Exception as e:
            log.error(f"engine loop error: {e}")
            time.sleep(CFG["91_POLL_SEC"])

    log.info("STOP v9 SHORT")

if __name__ == "__main__":
    engine()