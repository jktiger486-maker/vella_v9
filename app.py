# ============================================================
# VELLA_BR9 — SHORT ENGINE
# BASE: v9 기준선
# 전략: 1H·15M 하락 허가 → E1 선행(30%) → E2 증폭(70%)
#       TP1 부분익절 → 본절이동 → ema6>ema10 전량청산
# ============================================================

import os
import sys
import time
import signal
import logging
import requests
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque, Tuple
from collections import deque

# ============================================================
# CFG
# ============================================================

CFG = {
    "01_TRADE_SYMBOL":              "SUIUSDT",
    "02_INTERVAL":                  "5m",
    "03_CAPITAL_BASE_USDT":         10.0,
    "04_LEVERAGE":                  1,

    # ---- HTF EMA ----
    "05_HTF1_INTERVAL":             "1h",
    "06_HTF2_INTERVAL":             "15m",
    "07_HTF_FAST":                  9,
    "08_HTF_MID":                   21,

    # ---- LTF ENTRY EMA ----
    "10_EMA_FAST":                  5,
    "11_EMA_MID":                   8,

    # ---- LTF EXIT EMA ----
    "30_EXIT_FAST_EMA":             5,
    "31_EXIT_MID_EMA":              8,

    # ---- E1 / E2 ----
    "20_ENTRY1_ENABLE":             True,
    "21_ENTRY2_ENABLE":             True,
    "22_E1_ALLOC_PCT":              0.30,
    "23_E2_ALLOC_PCT":              0.70,

    # ---- 체결가 가드 ----
    "24_ENTRY_SLIPPAGE_GUARD_PCT":  0.004,   # 0.4%

    # ---- 저점 근접 금지 ----
    "25_RECENT_LOW_LOOKBACK":       20,
    "26_LOW_PROXIMITY_BLOCK_PCT":   0.005,   # 0.5%

    # ---- E2 단독 강화 차단 ----
    "27_E2_ONLY_LOW_BLOCK_PCT":     0.010,   # 1.0%

    # ---- TP ----
    "35_TP1_ENABLE":                True,
    "36_TP1_PCT":                   0.004,   # 0.4%
    "37_TP1_CLOSE_RATIO":           0.50,

    # ---- SL ----
    "40_SL_ENABLE":                 True,
    "41_SL_PCT":                    0.8,

    # ---- TIMEOUT ----
    "50_TIMEOUT_EXIT_ENABLE":       True,
    "51_TIMEOUT_BARS":              10,

    # ---- ENGINE ----
    "90_KLINE_LIMIT":               1500,
    "91_POLL_SEC":                  5,
    "92_LOG_LEVEL":                 "INFO",
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["92_LOG_LEVEL"], logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VELLA_BR9_SHORT")

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
    api_key    = os.getenv("BINANCE_API_KEY")
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
        log.error(f"fetch_klines_futures [{interval}]: {e}")
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
                            "minQty":   Decimal(f["minQty"]),
                            "maxQty":   Decimal(f["maxQty"]),
                        }
        return None
    except Exception as e:
        log.error(f"get_futures_lot_size: {e}")
        return None

# ============================================================
# QTY
# ============================================================

def calculate_quantity(qty_raw, lot: Dict[str, Decimal]) -> Optional[str]:
    if lot is None:
        return None
    qty_decimal = Decimal(str(qty_raw))
    step = lot["stepSize"]
    qty  = (qty_decimal / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    if qty < lot["minQty"]:
        return None
    if qty > lot["maxQty"]:
        qty = lot["maxQty"]
    precision = abs(step.as_tuple().exponent)
    return f"{qty:.{precision}f}"

def normalize_qty_str(qty_str: str, lot: Dict[str, Decimal]) -> Optional[str]:
    if lot is None:
        return None
    qty_decimal = Decimal(qty_str)
    step = lot["stepSize"]
    qty  = (qty_decimal / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    if qty < lot["minQty"]:
        return None
    if qty > lot["maxQty"]:
        qty = lot["maxQty"]
    precision = abs(step.as_tuple().exponent)
    return f"{qty:.{precision}f}"

# ============================================================
# IncrementalEMA
# ============================================================

class IncrementalEMA:
    def __init__(self, period: int):
        self.period  = period
        self.k       = 2.0 / (period + 1)
        self.value   = None
        self.ready   = False
        self._buf: List[float] = []
        self._history: Deque[float] = deque()

    def update(self, price: float) -> None:
        if not self.ready:
            self._buf.append(price)
            if len(self._buf) >= self.period:
                self.value = sum(self._buf) / len(self._buf)
                self.ready = True
                self._buf  = []
        else:
            self.value = price * self.k + self.value * (1.0 - self.k)
        if self.ready:
            self._history.append(self.value)

    def get(self) -> Optional[float]:
        return self.value if self.ready else None

    def get_prev(self) -> Optional[float]:
        if len(self._history) >= 2:
            return self._history[-2]
        return None

    def trim_history(self, maxlen: int = 2100) -> None:
        while len(self._history) > maxlen:
            self._history.popleft()

# ============================================================
# HTF EMA 계산 (매 루프 단순 재계산 방식)
# ============================================================

def _calc_htf_ema(closes: List[float], fast_period: int, mid_period: int) -> Tuple[Optional[float], Optional[float]]:
    """closes 리스트로 fast/mid EMA 계산, 마지막 값 반환"""
    def ema_series(data, period):
        if len(data) < period:
            return None
        k = 2.0 / (period + 1)
        val = sum(data[:period]) / period
        for price in data[period:]:
            val = price * k + val * (1 - k)
        return val

    fast = ema_series(closes, fast_period)
    mid  = ema_series(closes, mid_period)
    return fast, mid

def get_htf_short_ok(symbol: str) -> bool:
    """1H + 15M 하락 레짐 동시 True면 숏 허가"""
    fast_p = int(CFG["07_HTF_FAST"])
    mid_p  = int(CFG["08_HTF_MID"])
    limit  = mid_p + 10  # 충분한 데이터

    # ---- 1H ----
    kl_1h = fetch_klines_futures(symbol, CFG["05_HTF1_INTERVAL"], limit)
    if not kl_1h or len(kl_1h) < (mid_p + 1):
        log.warning("[HTF_BLOCK] 1H kline fetch 실패 또는 부족")
        return False
    closes_1h = [float(k[4]) for k in kl_1h[:-1]]  # 완료봉만
    last_close_1h = closes_1h[-1]
    fast_1h, mid_1h = _calc_htf_ema(closes_1h, fast_p, mid_p)
    if fast_1h is None or mid_1h is None:
        log.warning("[HTF_BLOCK] 1H EMA 계산 실패")
        return False
    h1_ok = (fast_1h < mid_1h) and (last_close_1h < mid_1h)

    # ---- 15M ----
    kl_15m = fetch_klines_futures(symbol, CFG["06_HTF2_INTERVAL"], limit)
    if not kl_15m or len(kl_15m) < (mid_p + 1):
        log.warning("[HTF_BLOCK] 15M kline fetch 실패 또는 부족")
        return False
    closes_15m = [float(k[4]) for k in kl_15m[:-1]]
    last_close_15m = closes_15m[-1]
    fast_15m, mid_15m = _calc_htf_ema(closes_15m, fast_p, mid_p)
    if fast_15m is None or mid_15m is None:
        log.warning("[HTF_BLOCK] 15M EMA 계산 실패")
        return False
    h15_ok = (fast_15m < mid_15m) and (last_close_15m < mid_15m)

    if not (h1_ok and h15_ok):
        log.info(
            f"[HTF_BLOCK] 1H_ok={h1_ok}(f={fast_1h:.6f} m={mid_1h:.6f} c={last_close_1h:.6f}) "
            f"15M_ok={h15_ok}(f={fast_15m:.6f} m={mid_15m:.6f} c={last_close_15m:.6f})"
        )
        return False
    return True

# ============================================================
# STATE
# ============================================================

@dataclass
class Position:
    side:             str
    avg_entry_price:  float
    qty_remaining:    str        # 현재 남은 수량 (주문용 str)
    qty_remaining_f:  float      # 계산용 float
    entry_bar:        int
    entry_type:       str   = "E1"
    e1_filled:        bool  = False
    e2_filled:        bool  = False
    tp1_done:         bool  = False
    breakeven_armed:  bool  = False

@dataclass
class EngineState:
    bar:            int           = 0
    last_open_time: Optional[int] = None
    position:       Optional[Position] = None

    close_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    high_history:  Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    low_history:   Deque[float] = field(default_factory=lambda: deque(maxlen=2000))

    ema_fast:      IncrementalEMA = field(default_factory=lambda: IncrementalEMA(CFG["10_EMA_FAST"]))
    ema_mid:       IncrementalEMA = field(default_factory=lambda: IncrementalEMA(CFG["11_EMA_MID"]))
    ema_exit_fast: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(CFG["30_EXIT_FAST_EMA"]))
    ema_exit_mid:  IncrementalEMA = field(default_factory=lambda: IncrementalEMA(CFG["31_EXIT_MID_EMA"]))

# ============================================================
# WARMUP
# ============================================================

def _warmup_done(st: EngineState) -> bool:
    needed = max(
        CFG["10_EMA_FAST"],
        CFG["11_EMA_MID"],
        CFG["30_EXIT_FAST_EMA"],
        CFG["31_EXIT_MID_EMA"],
        30,
    )
    return st.bar >= needed

# ============================================================
# ENTRY GUARD 공통
# ============================================================

def _entry_guards_ok(st: EngineState, signal_close: float, current_price: float, e2_only: bool) -> bool:
    # 체결가 가드
    guard_pct = float(CFG["24_ENTRY_SLIPPAGE_GUARD_PCT"])
    if current_price > signal_close * (1.0 + guard_pct):
        log.info(
            f"[ENTRY_GUARD_BLOCK] current={current_price:.6f} > signal_close*(1+{guard_pct})={signal_close*(1+guard_pct):.6f}"
        )
        return False

    # 저점 근접 금지
    lookback = int(CFG["25_RECENT_LOW_LOOKBACK"])
    if len(st.low_history) < lookback:
        return True
    recent_low = min(list(st.low_history)[-lookback:])
    block_pct = float(CFG["27_E2_ONLY_LOW_BLOCK_PCT"] if e2_only else CFG["26_LOW_PROXIMITY_BLOCK_PCT"])
    if current_price <= recent_low * (1.0 + block_pct):
        log.info(
            f"[LOW_BLOCK] current={current_price:.6f} <= recent_low*{1+block_pct:.3f}={recent_low*(1+block_pct):.6f} e2_only={e2_only}"
        )
        return False

    return True

# ============================================================
# ENTRY SIGNALS
# ============================================================

def short_entry_signals(st: EngineState, htf_ok: bool) -> str:
    """
    반환: "" | "E1" | "E2_ONLY" | "E2_ADD"
    E2_ADD: 포지션 있고 E1 진입 후 E2 추가 조건
    """
    if not htf_ok:
        return ""
    if not _warmup_done(st):
        return ""

    fast = st.ema_fast
    mid  = st.ema_mid
    if not (fast.ready and mid.ready):
        return ""

    fast_now  = fast.get()
    fast_prev = fast.get_prev()
    mid_now   = mid.get()
    mid_prev  = mid.get_prev()

    if fast_prev is None or mid_prev is None:
        return ""
    if len(st.high_history) < 3 or len(st.close_history) < 2:
        return ""

    close_now = st.close_history[-1]

    # ---- E1: 5M 빠른 데드크로스 ----
    e1_cross = (fast_prev >= mid_prev) and (fast_now < mid_now)

    # ---- E2: 반등 존재 + 반등 실패 ----
    # bounce_exists: 최근 3봉 각각의 high를 해당 시점 ema8 히스토리와 비교
    # ema_mid._history[-4:-1] = 최근 완료봉 3개 각각의 ema8 값
    e2_signal = False
    if len(mid._history) >= 3 and len(st.high_history) >= 3:
        mid_hist = list(mid._history)
        high_hist = list(st.high_history)
        # 최근 3봉(인덱스 -3, -2, -1)의 high vs 해당 봉 시점 ema8
        bounce_exists = (
            high_hist[-3] > mid_hist[-3] or
            high_hist[-2] > mid_hist[-2] or
            high_hist[-1] > mid_hist[-1]
        )
        bounce_failed = (close_now < fast_now) and (fast_now < mid_now)
        e2_signal = bounce_exists and bounce_failed

    pos = st.position

    # 포지션 없음 → E1 or E2_ONLY
    if pos is None:
        if CFG["20_ENTRY1_ENABLE"] and e1_cross:
            return "E1"
        if CFG["21_ENTRY2_ENABLE"] and e2_signal:
            return "E2_ONLY"
        return ""

    # 포지션 있음 → E2_ADD (E1 진입 후 E2 추가)
    if pos.e1_filled and not pos.e2_filled and not pos.tp1_done:
        if CFG["21_ENTRY2_ENABLE"] and e2_signal:
            return "E2_ADD"

    return ""

# ============================================================
# EXECUTION — 비중 기반 진입
# ============================================================

def place_short_entry_alloc(
    client: "Client",
    symbol: str,
    capital_usdt: float,
    alloc_pct: float,
    lot: Dict[str, Decimal],
) -> Optional[Dict[str, Any]]:
    try:
        ticker   = client.futures_symbol_ticker(symbol=symbol)
        price    = float(ticker["price"])
        leverage = int(CFG["04_LEVERAGE"])
        notional = float(capital_usdt) * float(leverage) * alloc_pct
        qty_str  = calculate_quantity(notional / price, lot)
        if qty_str is None:
            log.error(f"place_short_entry_alloc: qty calculation failed alloc={alloc_pct}")
            return None
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty_str,
        )
        qty_f = float(qty_str)
        return {"entry_price": price, "qty": qty_str, "qty_f": qty_f}
    except Exception as e:
        log.error(f"place_short_entry_alloc: {e}")
        return None

def place_short_exit_qty(
    client: "Client",
    symbol: str,
    qty_str: str,
    lot: Dict[str, Decimal],
) -> bool:
    try:
        qty2 = normalize_qty_str(qty_str, lot)
        if qty2 is None:
            log.error("place_short_exit_qty: qty too small")
            return False
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty2,
            reduceOnly=True,
        )
        return True
    except Exception as e:
        log.error(f"place_short_exit_qty: {e}")
        return False

# ============================================================
# EXIT ACTION
# ============================================================

def get_exit_action(st: EngineState) -> str:
    """
    반환: "" | "SL" | "TP1" | "BREAKEVEN" | "TIMEOUT" | "EMA"
    """
    pos = st.position
    if pos is None:
        return ""

    close = st.close_history[-1]
    avg   = pos.avg_entry_price

    # [1] SL
    if CFG["40_SL_ENABLE"]:
        sl_pct = float(CFG["41_SL_PCT"]) / 100.0
        if close >= avg * (1.0 + sl_pct):
            return "SL"

    # [2] TP1
    if CFG["35_TP1_ENABLE"] and not pos.tp1_done and pos.entry_type != "SYNC":
        tp1_pct = float(CFG["36_TP1_PCT"])
        if close <= avg * (1.0 - tp1_pct):
            return "TP1"

    # [3] BREAKEVEN (TP1 이후)
    if pos.tp1_done and pos.breakeven_armed:
        if close >= avg:
            return "BREAKEVEN"

    # [4] TIMEOUT
    if CFG["50_TIMEOUT_EXIT_ENABLE"] and pos.entry_type != "SYNC":
        if (st.bar - pos.entry_bar) >= int(CFG["51_TIMEOUT_BARS"]):
            return "TIMEOUT"

    # [5] EMA EXIT
    ef = st.ema_exit_fast.get()
    em = st.ema_exit_mid.get()
    if ef is not None and em is not None:
        if ef > em:
            return "EMA"

    return ""

# ============================================================
# _apply_bar
# ============================================================

def _apply_bar(st: EngineState, close: float, high: float, low: float) -> None:
    st.close_history.append(close)
    st.high_history.append(high)
    st.low_history.append(low)

    st.ema_fast.update(close)
    st.ema_mid.update(close)
    st.ema_exit_fast.update(close)
    st.ema_exit_mid.update(close)

    st.ema_fast.trim_history()
    st.ema_mid.trim_history()
    st.ema_exit_fast.trim_history()
    st.ema_exit_mid.trim_history()

# ============================================================
# ENGINE LOOP
# ============================================================

STOP = False

def _sig_handler(_sig, _frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT,  _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def engine():
    client   = init_client()
    symbol   = CFG["01_TRADE_SYMBOL"]
    interval = CFG["02_INTERVAL"]
    capital  = float(CFG["03_CAPITAL_BASE_USDT"])

    set_leverage(client, symbol, int(CFG["04_LEVERAGE"]))

    lot = get_futures_lot_size(client, symbol)
    if lot is None:
        raise RuntimeError("lot_size retrieval failed")

    st = EngineState()

    # ---- SYNC ----
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos["symbol"] == symbol:
                amt = float(pos["positionAmt"])
                if amt < 0:
                    sync_qty_str = calculate_quantity(abs(amt), lot)
                    if sync_qty_str is None:
                        log.error("[SYNC] qty calculation failed, skipping sync")
                    else:
                        ep = float(pos["entryPrice"])
                        st.position = Position(
                            side="SHORT",
                            avg_entry_price=ep,
                            qty_remaining=sync_qty_str,
                            qty_remaining_f=float(sync_qty_str),
                            entry_bar=st.bar,
                            entry_type="SYNC",
                            e1_filled=True,
                            e2_filled=True,
                            tp1_done=False,
                            breakeven_armed=False,
                        )
                        log.info(f"[SYNC] SHORT qty={sync_qty_str} entry={ep}")
                break
    except Exception as e:
        log.error(f"position sync failed: {e}")

    log.info(
        f"START VELLA_BR9_SHORT | symbol={symbol} interval={interval} capital={capital} lev={CFG['04_LEVERAGE']} "
        f"| LTF_EMA=({CFG['10_EMA_FAST']},{CFG['11_EMA_MID']}) "
        f"| EXIT_EMA=({CFG['30_EXIT_FAST_EMA']},{CFG['31_EXIT_MID_EMA']}) "
        f"| HTF=({CFG['05_HTF1_INTERVAL']}/{CFG['06_HTF2_INTERVAL']} ema{CFG['07_HTF_FAST']}/{CFG['08_HTF_MID']}) "
        f"| E1={CFG['22_E1_ALLOC_PCT']*100:.0f}% E2={CFG['23_E2_ALLOC_PCT']*100:.0f}% "
        f"| TP1={CFG['36_TP1_PCT']*100:.1f}% SL={CFG['41_SL_PCT']}%"
    )

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

            # ---- COLD START ----
            if not st.close_history:
                for k in kl[:-1]:
                    _apply_bar(st, float(k[4]), float(k[2]), float(k[3]))
                st.bar = len(st.close_history)
                st.last_open_time = int(kl[-2][0])
                log.info(f"[BOOT] {st.bar} bars loaded warmup_done={_warmup_done(st)}")
                continue

            st.last_open_time = open_time
            st.bar += 1
            _apply_bar(st, float(completed[4]), float(completed[2]), float(completed[3]))

            close_now     = st.close_history[-1]
            signal_close  = close_now  # 완료봉 종가 = 신호봉 종가

            # ---- HTF ----
            htf_ok = get_htf_short_ok(symbol)

            # ======================================
            # 포지션 없음 → 신규 진입
            # ======================================
            if st.position is None:
                sig = short_entry_signals(st, htf_ok)

                if sig in ("E1", "E2_ONLY"):
                    e2_only = (sig == "E2_ONLY")
                    alloc   = float(CFG["22_E1_ALLOC_PCT"] if sig == "E1" else CFG["23_E2_ALLOC_PCT"])

                    # 현재가 조회 (체결가 가드용)
                    try:
                        ticker       = client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker["price"])
                    except Exception as e:
                        log.error(f"ticker fetch failed: {e}")
                        time.sleep(CFG["91_POLL_SEC"])
                        continue

                    if not _entry_guards_ok(st, signal_close, current_price, e2_only):
                        time.sleep(CFG["91_POLL_SEC"])
                        continue

                    order = place_short_entry_alloc(client, symbol, capital, alloc, lot)
                    if order:
                        st.position = Position(
                            side="SHORT",
                            avg_entry_price=float(order["entry_price"]),
                            qty_remaining=order["qty"],
                            qty_remaining_f=order["qty_f"],
                            entry_bar=st.bar,
                            entry_type=sig,
                            e1_filled=(sig == "E1"),
                            e2_filled=(sig == "E2_ONLY"),
                        )
                        log.info(
                            f"[ENTRY_{sig}] qty={order['qty']} price={order['entry_price']:.6f} "
                            f"alloc={alloc*100:.0f}% bar={st.bar}"
                        )
                    else:
                        log.error(f"[ENTRY_FAIL] {sig} order failed")

            # ======================================
            # 포지션 있음
            # ======================================
            else:
                pos = st.position

                # 같은 봉 보호
                if pos.entry_bar == st.bar:
                    time.sleep(CFG["91_POLL_SEC"])
                    continue

                # ---- E2_ADD: E1 후 E2 추가진입 ----
                if pos.e1_filled and not pos.e2_filled and not pos.tp1_done:
                    sig = short_entry_signals(st, htf_ok)
                    if sig == "E2_ADD":
                        try:
                            ticker        = client.futures_symbol_ticker(symbol=symbol)
                            current_price = float(ticker["price"])
                        except Exception as e:
                            log.error(f"ticker fetch failed: {e}")
                            current_price = close_now

                        if _entry_guards_ok(st, signal_close, current_price, False):
                            alloc = float(CFG["23_E2_ALLOC_PCT"])
                            order = place_short_entry_alloc(client, symbol, capital, alloc, lot)
                            if order:
                                # 평균단가 갱신
                                old_qty = pos.qty_remaining_f
                                add_qty = order["qty_f"]
                                old_ep  = pos.avg_entry_price
                                new_ep  = order["entry_price"]
                                total_qty = old_qty + add_qty
                                avg_ep    = (old_ep * old_qty + new_ep * add_qty) / total_qty

                                new_qty_str = calculate_quantity(total_qty, lot)
                                if new_qty_str is None:
                                    new_qty_str = pos.qty_remaining

                                pos.avg_entry_price = avg_ep
                                pos.qty_remaining   = new_qty_str
                                pos.qty_remaining_f = total_qty
                                pos.e2_filled       = True
                                # TIMEOUT은 최초 진입 기준 유지, E2_ADD로 entry_bar 리셋하지 않음
                                log.info(
                                    f"[ENTRY_E2_ADD] qty={order['qty']} price={new_ep:.6f} "
                                    f"avg_entry={avg_ep:.6f} total_qty={new_qty_str} bar={st.bar}"
                                )
                                # E2_ADD 성공 → 같은 루프에서 exit 평가 금지
                                continue
                            else:
                                log.error("[ENTRY_FAIL] E2_ADD order failed")

                # ---- EXIT 판단 ----
                action = get_exit_action(st)

                if action == "TP1":
                    # 50% 부분익절
                    ratio     = float(CFG["37_TP1_CLOSE_RATIO"])
                    close_qty = calculate_quantity(pos.qty_remaining_f * ratio, lot)
                    if close_qty is None:
                        # dust → 전량
                        close_qty = pos.qty_remaining
                        full_exit = True
                    else:
                        full_exit = False

                    ok = place_short_exit_qty(client, symbol, close_qty, lot)
                    if ok:
                        log.info(
                            f"[TP1] qty={close_qty} close={close_now:.6f} avg_entry={pos.avg_entry_price:.6f} bar={st.bar}"
                        )
                        if full_exit:
                            st.position = None
                        else:
                            remain_f   = pos.qty_remaining_f - float(close_qty)
                            remain_str = calculate_quantity(remain_f, lot)
                            if remain_str is None:
                                # 남은 수량 너무 작음 → 포지션 종료
                                st.position = None
                                log.info("[TP1] remaining dust → position closed")
                            else:
                                pos.tp1_done        = True
                                pos.breakeven_armed = True
                                pos.qty_remaining   = remain_str
                                pos.qty_remaining_f = float(remain_str)
                    else:
                        log.error("[TP1_FAIL] order failed")

                elif action in ("SL", "BREAKEVEN", "TIMEOUT", "EMA"):
                    ok = place_short_exit_qty(client, symbol, pos.qty_remaining, lot)
                    if ok:
                        log.info(
                            f"[EXIT_{action}] qty={pos.qty_remaining} close={close_now:.6f} "
                            f"avg_entry={pos.avg_entry_price:.6f} tp1_done={pos.tp1_done} bar={st.bar}"
                        )
                        st.position = None
                    else:
                        log.error(f"[EXIT_FAIL] {action} order failed (position kept)")

        except Exception as e:
            log.error(f"engine loop error: {e}")
            time.sleep(CFG["91_POLL_SEC"])

    log.info("STOP VELLA_BR9_SHORT")

if __name__ == "__main__":
    engine()