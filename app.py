"""
============================================================
VELLA RANGE SHORT LADDER v9_거미줄 작전
============================================================
[버그 수정 패치 — 공식 확정 5개]

BUG 1 — INIT 순서 오류
  __init__에서 set_margin_type/set_leverage 호출 제거
  → run() 진입 후 _sync_on_start() 먼저, 그 다음 margin/leverage 설정

BUG 2 — SL 누적 구조
  _sync_on_start() 포지션 복구 시 sl_orders[0] 외 나머지 전부 cancel

BUG 3 — 예외 처리 문자열 취약성
  set_margin_type() except 문자열 폭넓게 처리 (변형 방어)

BUG 4 — 1차 시장가 실패 후 later fill 시 SL 미배치
  POSITION_HOLD 진입 경로에서 sl_price 존재 + sl_order_id 없으면 즉시 SL 배치

BUG 5 — LADDER_ACTIVE 복구 시 avg_full/sl_price 미복구
  _sync_on_start() LADDER_ACTIVE 경로에서 prices/qtys 재계산 후 avg_full/sl_price 복구

============================================================
"""

import time
import logging
import os
from decimal import Decimal, ROUND_DOWN

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except Exception:
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception

ClientError = (BinanceAPIException, BinanceOrderException)

# ============================================================
# CFG
# ============================================================
CFG = {
    "SYMBOL":              "SEIUSDT",
    "INTERVAL_TRIGGER":    "5m",
    "INTERVAL_EXEC":       "5m",
    "INTERVAL_FILTER_HTF": "4h",

    "EMA_TRIGGER_LEN":    15,
    "HTF_FILTER_EMA_LEN": 15,
    "HTF_FILTER_ENABLE":  True,

    "TOTAL_CAPITAL_USDT": 5000.0,
    "LEVERAGE":           3,
    "MARGIN_TYPE":        "CROSS",
    "MAX_CAPITAL_RATIO":  0.95,

    "LADDER_COUNT":   10,
    "LADDER_GAP_PCT": 0.025,
    "SIZE_WEIGHTS": [
        0.6, 0.8, 1.1, 1.5, 2.0,
        1.2, 1.0, 0.8, 0.7, 0.6
    ],
    "LADDER_INVALIDATION_MULT":    2.0,
    "LADDER_NO_FILL_TIMEOUT_BARS": 12,

    "TP1_PROFIT_PCT":       0.01,
    "TP1_PARTIAL_RATIO":    0.5,
    "TRAILING_REBOUND_PCT": 0.005,

    "FEE_PCT_ONEWAY":            0.0004,
    "TARGET_PROFIT_STAGE_1_3":   0.012,
    "TARGET_PROFIT_STAGE_4_5":   0.005,
    "TARGET_PROFIT_STAGE_6_7":   0.003,
    "TARGET_PROFIT_STAGE_8_9":   0.001,
    "TARGET_PROFIT_STAGE_10":   -0.0008,
    "EXIT_REPRICE_THRESHOLD_PCT": 0.003,

    "HARD_SL_PCT":             0.05,
    "SL_TICK_BUFFER":          0.003,
    "CAPITAL_CHECK_MIN_RATIO": 0.80,
    "CAPITAL_CHECK_MAX_RATIO": 1.10,
    "DEEP_FILL_STAGE":         8,
    "TIMEOUT_BARS_AFTER_DEEP": 12,

    "REENTRY_COOLDOWN_BARS":      8,
    "POLL_INTERVAL_SEC":          10,
    "BAR_CHECK_MIN_INTERVAL_SEC": 40,
    "LOG_LEVEL": "INFO",
}

# ============================================================
# 로거
# ============================================================
logging.basicConfig(
    level=getattr(logging, CFG["LOG_LEVEL"]),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vella_range_short_v9.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("VELLA_BR9")

# ============================================================
# 클라이언트
# ============================================================
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

if Client is None:
    raise RuntimeError("python-binance missing")


class BinanceFuturesCompat:
    def __init__(self, key: str, secret: str):
        self._client = Client(key, secret)

    def exchange_info(self):
        return self._client.futures_exchange_info()

    def klines(self, symbol: str, interval: str, limit: int = 500):
        return self._client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    def get_position_risk(self, symbol: str):
        return self._client.futures_position_information(symbol=symbol)

    def get_orders(self, symbol: str):
        return self._client.futures_get_open_orders(symbol=symbol)

    def cancel_order(self, symbol: str, orderId: int):
        return self._client.futures_cancel_order(symbol=symbol, orderId=orderId)

    def cancel_open_orders(self, symbol: str):
        return self._client.futures_cancel_all_open_orders(symbol=symbol)

    def query_order(self, symbol: str, orderId: int):
        return self._client.futures_get_order(symbol=symbol, orderId=orderId)

    def new_order(self, **kwargs):
        if "reduceOnly" in kwargs and isinstance(kwargs["reduceOnly"], str):
            kwargs["reduceOnly"] = kwargs["reduceOnly"].lower() == "true"
        return self._client.futures_create_order(**kwargs)

    def change_leverage(self, symbol: str, leverage: int):
        return self._client.futures_change_leverage(symbol=symbol, leverage=leverage)

    def change_margin_type(self, symbol: str, marginType: str):
        return self._client.futures_change_margin_type(symbol=symbol, marginType=marginType)

    def ticker_price(self, symbol: str):
        return self._client.futures_symbol_ticker(symbol=symbol)


client = BinanceFuturesCompat(API_KEY, API_SECRET)

# ============================================================
# 심볼 필터 캐시
# ============================================================
_SYM_FILTERS: dict = {}


def load_symbol_filters(symbol: str) -> dict:
    global _SYM_FILTERS
    if symbol in _SYM_FILTERS:
        return _SYM_FILTERS[symbol]
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] != symbol:
            continue
        result = {
            "price_prec":   s["pricePrecision"],
            "qty_prec":     s["quantityPrecision"],
            "tick_size":    None,
            "step_size":    None,
            "min_qty":      None,
            "min_notional": None,
        }
        for f in s["filters"]:
            ft = f["filterType"]
            if ft == "PRICE_FILTER":
                result["tick_size"] = f["tickSize"]
            elif ft == "LOT_SIZE":
                result["step_size"] = f["stepSize"]
                result["min_qty"]   = float(f["minQty"])
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                result["min_notional"] = float(f.get("notional", f.get("minNotional", 5.0)))
        _SYM_FILTERS[symbol] = result
        log.info(
            f"필터 로드: tick={result['tick_size']} step={result['step_size']} "
            f"minQty={result['min_qty']} minNotional={result['min_notional']}"
        )
        return result
    raise RuntimeError(f"심볼 {symbol} 필터 없음")


# ============================================================
# 수치 유틸
# ============================================================

def _quantize(value: float, unit_str: str, prec: int) -> str:
    d_val   = Decimal(str(value))
    d_unit  = Decimal(unit_str)
    floored = (d_val / d_unit).to_integral_value(rounding=ROUND_DOWN) * d_unit
    quant   = Decimal("0." + "0" * prec) if prec > 0 else Decimal("1")
    return str(floored.quantize(quant))


def fmt_price(price: float, sym: str) -> str:
    f = _SYM_FILTERS[sym]
    if f["tick_size"]:
        return _quantize(price, f["tick_size"], f["price_prec"])
    return f"{round(price, f['price_prec']):.{f['price_prec']}f}"


def fmt_qty(qty: float, sym: str) -> str:
    f = _SYM_FILTERS[sym]
    if f["step_size"]:
        return _quantize(qty, f["step_size"], f["qty_prec"])
    return f"{round(qty, f['qty_prec']):.{f['qty_prec']}f}"


def is_order_valid(price: float, qty: float, sym: str) -> bool:
    f = _SYM_FILTERS[sym]
    if f["min_qty"] and qty < f["min_qty"]:
        log.warning(f"주문 스킵: qty {qty} < minQty {f['min_qty']}")
        return False
    if f["min_notional"] and price * qty < f["min_notional"]:
        log.warning(f"주문 스킵: notional {price*qty:.2f} < minNotional {f['min_notional']}")
        return False
    return True


# ============================================================
# EMA
# ============================================================

def calc_ema(values: list, period: int) -> list:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    series = [e]
    for v in values[period:]:
        e = float(v) * k + e * (1 - k)
        series.append(e)
    return series


# ============================================================
# 캔들 조회
# ============================================================

def get_closed_bar_ts_with_closes(symbol: str, interval: str, limit: int = 60):
    raw    = client.klines(symbol, interval, limit=limit + 1)
    closed = raw[:-1]
    closes = [float(k[4]) for k in closed]
    ts     = int(closed[-1][0]) if closed else 0
    return closes, ts


def get_closed_bar_open_ts(symbol: str, interval: str) -> int:
    raw = client.klines(symbol, interval, limit=2)
    return int(raw[-2][0])


# ============================================================
# BarCache
# ============================================================

class BarCache:
    def __init__(self, min_interval_sec: float = 0):
        self._last_ts: int         = 0
        self._cached_result        = None
        self._last_api_time: float = 0.0
        self._min_interval         = min_interval_sec

    def query(self, fetch_fn, compute_fn):
        now = time.time()
        if self._cached_result is not None and \
                (now - self._last_api_time) < self._min_interval:
            return self._cached_result, self._last_ts
        closes, ts          = fetch_fn()
        self._last_api_time = now
        if ts != self._last_ts or self._cached_result is None:
            self._cached_result = compute_fn(closes)
            self._last_ts       = ts
        return self._cached_result, ts


# ============================================================
# 4시간 필터
# ============================================================

def _compute_4h_filter(closes: list) -> bool:
    period = CFG["HTF_FILTER_EMA_LEN"]
    if len(closes) < period + 1:
        log.warning("HTF 데이터 부족 → 필터 차단")
        return False
    ema_s = calc_ema(closes, period)
    ok    = closes[-1] < ema_s[-1]
    label = "PASS" if ok else "BLOCK"
    log.info(f"[HTF FILTER {label}] 4H close {closes[-1]:.4f} {'<' if ok else '>='} EMA{period} {ema_s[-1]:.4f}")
    return ok


def check_4h_short_filter(symbol: str, cache: BarCache) -> bool:
    if not CFG["HTF_FILTER_ENABLE"]:
        return True
    period = CFG["HTF_FILTER_EMA_LEN"]
    result, _ = cache.query(
        fetch_fn=lambda: get_closed_bar_ts_with_closes(
            symbol, CFG["INTERVAL_FILTER_HTF"], limit=period + 10
        ),
        compute_fn=_compute_4h_filter,
    )
    return result


# ============================================================
# 5M EMA15 역전 트리거 v9.2
# ============================================================

def _compute_5m_trigger(closes: list, highs: list) -> bool:
    period = CFG["EMA_TRIGGER_LEN"]
    if len(closes) < period + 2 or len(highs) < period + 2:
        return False

    ema_s = calc_ema(closes, period)

    cond1   = closes[-1] < ema_s[-1]
    cond2   = highs[-2]  > ema_s[-2]
    cond2_b = highs[-1]  < ema_s[-1] * 1.003
    cond3   = closes[-1] < closes[-2]

    triggered = cond1 and cond2 and cond2_b and cond3

    if triggered:
        log.info(
            f"[5M TRIGGER V8.2] EMA 이탈 + 고가 억제(0.3%) + 1봉 하락 | "
            f"close={closes[-1]:.4f}<ema={ema_s[-1]:.4f} | "
            f"high[-2]={highs[-2]:.4f}>ema[-2]={ema_s[-2]:.4f} | "
            f"high[-1]={highs[-1]:.4f}<ema[-1]*1.003={(ema_s[-1]*1.003):.4f} | "
            f"closes={closes[-2]:.4f}->{closes[-1]:.4f}"
        )
    return triggered


def _fetch_5m_trigger_inputs(symbol: str, limit: int):
    raw    = client.klines(symbol, CFG["INTERVAL_TRIGGER"], limit=limit + 1)
    closed = raw[:-1]
    closes = [float(k[4]) for k in closed]
    highs  = [float(k[2]) for k in closed]
    ts     = int(closed[-1][0]) if closed else 0
    return closes, highs, ts


def calc_ema15_trigger(symbol: str, cache: BarCache) -> tuple[bool, int]:
    period = CFG["EMA_TRIGGER_LEN"]
    limit  = period + 10

    def fetch():
        closes, highs, ts = _fetch_5m_trigger_inputs(symbol, limit)
        return (closes, highs), ts

    def compute(data):
        closes, highs = data
        return _compute_5m_trigger(closes, highs)

    result, ts = cache.query(fetch_fn=fetch, compute_fn=compute)
    return result, ts


# ============================================================
# 포지션
# ============================================================

def get_position(symbol: str) -> dict:
    for p in client.get_position_risk(symbol=symbol):
        if p["symbol"] == symbol:
            return {"amt": float(p["positionAmt"]), "avg_price": float(p["entryPrice"])}
    return {"amt": 0.0, "avg_price": 0.0}


def has_short_position(pos: dict) -> bool:
    return pos["amt"] < -0.0001


# ============================================================
# 주문 유틸
# ============================================================

def get_open_orders(symbol: str) -> list:
    try:
        return client.get_orders(symbol=symbol)
    except ClientError as e:
        log.error(f"주문 조회 실패: {e}")
        return []


def cancel_order(symbol: str, order_id: int) -> bool:
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        log.info(f"주문 취소: {order_id}")
        return True
    except ClientError as e:
        log.warning(f"주문 취소 실패 ({order_id}): {e}")
        return False


def cancel_all_orders(symbol: str):
    try:
        client.cancel_open_orders(symbol=symbol)
        log.info("미체결 전체 취소")
    except ClientError as e:
        log.warning(f"전체 취소 실패: {e}")


def query_order_status(symbol: str, order_id: int) -> str:
    try:
        return client.query_order(symbol=symbol, orderId=order_id).get("status", "UNKNOWN")
    except ClientError as e:
        log.warning(f"query_order 실패 ({order_id}): {e}")
        return "UNKNOWN"


def place_limit_short(symbol: str, price: float, qty: float) -> dict | None:
    if not is_order_valid(price, qty, symbol):
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="SELL", type="LIMIT", timeInForce="GTC",
            price=fmt_price(price, symbol), quantity=fmt_qty(qty, symbol),
        )
        log.info(f"[ENTRY LADDER] SELL LIMIT price={fmt_price(price, symbol)} qty={fmt_qty(qty, symbol)}")
        return order
    except ClientError as e:
        log.error(f"숏 주문 실패: {e}")
        return None


def place_market_short(symbol: str, qty: float) -> dict | None:
    q_str = fmt_qty(abs(qty), symbol)
    if float(q_str) <= 0:
        log.warning(f"시장가 숏 스킵: qty={q_str}")
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="SELL", type="MARKET",
            quantity=q_str,
        )
        log.info(f"[ENTRY LADDER] SELL MARKET qty={q_str}")
        return order
    except ClientError as e:
        log.error(f"시장가 숏 실패: {e}")
        return None


def place_limit_exit(symbol: str, price: float, qty: float) -> dict | None:
    if not is_order_valid(price, qty, symbol):
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="BUY", type="LIMIT", timeInForce="GTC",
            price=fmt_price(price, symbol), quantity=fmt_qty(qty, symbol),
            reduceOnly="true",
        )
        log.info(f"[EXIT/SL] BUY EXIT LIMIT price={fmt_price(price, symbol)} qty={fmt_qty(qty, symbol)}")
        return order
    except ClientError as e:
        log.error(f"청산 주문 실패: {e}")
        return None


def place_stop_limit_sl(symbol: str, stop_price: float, limit_price: float, qty: float) -> dict | None:
    if not is_order_valid(stop_price, qty, symbol):
        return None
    try:
        order = client.new_order(
            symbol=symbol,
            side="BUY",
            type="STOP",
            timeInForce="GTC",
            stopPrice=fmt_price(stop_price, symbol),
            price=fmt_price(limit_price, symbol),
            quantity=fmt_qty(qty, symbol),
            reduceOnly="true",
        )
        log.info(
            f"[EXIT/SL] BUY SL STOP_LIMIT stopPrice={fmt_price(stop_price, symbol)} "
            f"price={fmt_price(limit_price, symbol)} qty={fmt_qty(qty, symbol)} "
            f"reduceOnly=True mode=FULL_LADDER_AVG_BASED_STATIC"
        )
        return order
    except ClientError as e:
        log.error(f"SL 주문 실패: {e}")
        return None


def market_close_short(symbol: str, qty: float) -> bool:
    q_str = fmt_qty(abs(qty), symbol)
    if float(q_str) <= 0:
        log.warning(f"시장가 청산 스킵: qty={q_str}")
        return False
    try:
        client.new_order(
            symbol=symbol, side="BUY", type="MARKET",
            quantity=q_str, reduceOnly="true",
        )
        log.info(f"[EXIT/SL] BUY MARKET 시장가 청산 qty={q_str}")
        return True
    except ClientError as e:
        log.error(f"시장가 청산 실패: {e}")
        return False


def set_leverage(symbol: str, leverage: int):
    try:
        client.change_leverage(symbol=symbol, leverage=leverage)
        log.info(f"레버리지 {leverage}x 설정")
    except ClientError as e:
        log.warning(f"레버리지 설정 오류: {e}")


# [BUG 3 수정] 예외 문자열 폭넓게 처리
def set_margin_type(symbol: str, margin_type: str):
    try:
        mt = "CROSSED" if margin_type.upper() == "CROSS" else "ISOLATED"
        client.change_margin_type(symbol=symbol, marginType=mt)
        log.info(f"마진 타입 {mt} 설정 완료")
    except ClientError as e:
        msg = str(e).lower()
        # 이미 설정된 상태 — 정상
        if "no need to change" in msg:
            log.info(f"마진 타입 이미 설정된 상태 ({mt})")
        # 포지션 또는 주문 존재로 변경 불가 — 경고 후 계속
        elif (
            "cannot be changed" in msg
            or "open orders" in msg
            or "position" in msg
            or "4067" in msg
            or "4046" in msg
        ):
            log.warning(f"마진 타입 변경 불가 (포지션/주문 존재 추정) | 요청={mt} | {e}")
        else:
            log.error(f"마진 타입 설정 실패: 요청={mt} | {e}")
            raise


# ============================================================
# 사이즈 / 가격 계산
# ============================================================

def normalize_weights(weights: list, count: int) -> list:
    w = weights[:count]
    t = sum(w)
    return [x / t for x in w]


def build_ladder_prices(entry_price: float, count: int, gap_pct: float) -> list:
    return [entry_price * (1 + gap_pct * i) for i in range(count)]


def calc_ladder_quantities_per_stage(
    total_capital: float,
    leverage: float,
    weights: list,
    prices: list,
    current_price: float,
) -> list:
    effective = total_capital * CFG["MAX_CAPITAL_RATIO"] * leverage
    qtys = []
    for i, w in enumerate(weights):
        capital_i = effective * w
        price_i   = current_price if i == 0 else prices[i]
        qtys.append(capital_i / price_i)
    return qtys


def calc_avg_full(prices: list, qtys: list) -> float:
    total_notional = sum(p * q for p, q in zip(prices, qtys))
    total_qty      = sum(qtys)
    return total_notional / total_qty if total_qty > 0 else 0.0


def get_stage_target_pct(stage: int) -> float:
    if stage <= 3: return CFG["TARGET_PROFIT_STAGE_1_3"]
    if stage <= 5: return CFG["TARGET_PROFIT_STAGE_4_5"]
    if stage <= 7: return CFG["TARGET_PROFIT_STAGE_6_7"]
    if stage <= 9: return CFG["TARGET_PROFIT_STAGE_8_9"]
    return CFG["TARGET_PROFIT_STAGE_10"]


def calc_exit_price(avg_price: float, stage: int) -> float:
    return avg_price * (1 - CFG["FEE_PCT_ONEWAY"] * 2 - get_stage_target_pct(stage))


# ============================================================
# 5분 완료봉 감지
# ============================================================

class BarTracker:
    def __init__(self, symbol: str, interval: str):
        self.symbol        = symbol
        self.interval      = interval
        self.last_ts       = None
        self._cached_ts    = None
        self._last_checked = 0.0

    def new_bar_closed(self) -> bool:
        now = time.time()
        if now - self._last_checked >= CFG["BAR_CHECK_MIN_INTERVAL_SEC"]:
            self._cached_ts    = get_closed_bar_open_ts(self.symbol, self.interval)
            self._last_checked = now
        ts = self._cached_ts
        if ts is None:
            return False
        if self.last_ts is None:
            self.last_ts = ts
            return False
        if ts > self.last_ts:
            self.last_ts = ts
            return True
        return False


# ============================================================
# 상태 머신
# ============================================================

class RangeShortEngine:
    def __init__(self):
        self.state  = "WATCHING"
        self.symbol = CFG["SYMBOL"]

        self.ladder_orders: list[dict] = []
        self.entry_price_base = None

        self.max_filled_stage = 0
        self.exit_order_ids: list[int] = []
        self.last_exit_qty   = 0.0
        self.last_exit_price = 0.0
        self.last_stage      = 0

        self.tp1_done:  bool         = False
        self.trail_low: float | None = None

        self._filled_order_ids:   set[int] = set()
        self._canceled_order_ids: set[int] = set()
        self._last_position_amt            = 0.0

        self._closing_in_progress: bool = False
        self._last_filled_check_ts: int  = 0

        self.bars_after_deep  = 0
        self.cooldown_bars    = 0
        self.no_fill_bars     = 0

        self.last_trigger_bar_ts: int = 0

        self.avg_full:    float | None = None
        self.sl_price:    float | None = None
        self.sl_order_id: int   | None = None

        self.bar_tracker = BarTracker(self.symbol, CFG["INTERVAL_EXEC"])

        min_iv = CFG["BAR_CHECK_MIN_INTERVAL_SEC"]
        self._htf_cache     = BarCache(min_interval_sec=min_iv)
        self._trigger_cache = BarCache(min_interval_sec=min_iv)

        # [BUG 1 수정] __init__에서 심볼 필터만 로드
        # margin/leverage 설정은 run()에서 _sync_on_start() 이후 호출
        load_symbol_filters(self.symbol)

    # --------------------------------------------------------
    # 안전 취소
    # --------------------------------------------------------
    def _safe_cancel(self, order_id: int) -> bool:
        if order_id in self._filled_order_ids:
            return True   # 체결된 주문은 이미 없음 → 성공으로 간주
        if order_id in self._canceled_order_ids:
            return True   # 이미 취소됨 → 성공으로 간주
        success = cancel_order(self.symbol, order_id)
        if success:
            self._canceled_order_ids.add(order_id)
        return success

    def _cancel_ladder_orders(self):
        still_alive = []
        for o in self.ladder_orders:
            success = self._safe_cancel(o["order_id"])
            if not success:
                log.warning(f"[CANCEL LADDER] cancel 실패 — orphan 가능 | orderId={o['order_id']} stage={o['stage']}")
                still_alive.append(o)
        if still_alive:
            log.warning(f"[CANCEL LADDER] cancel 실패 {len(still_alive)}건 → ladder_orders 잔존 유지")
            self.ladder_orders = still_alive
        else:
            self.ladder_orders = []

    def cancel_buy_exit_orders(self, exit_order_ids: list):
        for oid in exit_order_ids:
            self._safe_cancel(oid)

    # --------------------------------------------------------
    # _reset_sl_order 헬퍼
    # --------------------------------------------------------
    def _reset_sl_order(self, new_qty: float):
        if self.sl_price is None:
            log.error("[SL RESET] sl_price 없음 → 재설정 불가")
            return

        if self.sl_order_id is not None:
            cancelled = self._safe_cancel(self.sl_order_id)
            if not cancelled:
                log.warning(
                    f"[SL RESET] 기존 SL cancel 실패 | orderId={self.sl_order_id} "
                    f"→ 새 SL로 교체 진행 (거래소에서 이미 체결/만료됐을 가능성)"
                )
            self.sl_order_id = None  # 성공/실패 무관하게 참조 초기화 후 새 SL 배치

        time.sleep(0.05)

        stop_price  = self.sl_price
        limit_price = self.sl_price * (1 + CFG["SL_TICK_BUFFER"])

        order = place_stop_limit_sl(self.symbol, stop_price, limit_price, abs(new_qty))

        if order is None:
            log.warning("[SL RESET] 1차 실패 → 0.1초 후 재시도")
            time.sleep(0.1)
            order = place_stop_limit_sl(self.symbol, stop_price, limit_price, abs(new_qty))

        if order:
            self.sl_order_id = int(order["orderId"])
            log.info(
                f"[SL ORDER] mode=FULL_LADDER_AVG_BASED_STATIC "
                f"stopPrice={fmt_price(stop_price, self.symbol)} "
                f"price={fmt_price(limit_price, self.symbol)} "
                f"qty={fmt_qty(abs(new_qty), self.symbol)} reduceOnly=True"
            )
        else:
            log.critical("[SL RESET FAIL] 재시도 실패 → SL 없는 상태, 엔진 중단")
            raise RuntimeError("SL NOT PLACED")

    # --------------------------------------------------------
    # FILLED 캐시 기반 체결 단계 카운트
    # --------------------------------------------------------
    def _count_filled_stages(self) -> int:
        for o in self.ladder_orders:
            oid = o["order_id"]
            if oid in self._filled_order_ids:
                continue
            if query_order_status(self.symbol, oid) == "FILLED":
                self._filled_order_ids.add(oid)
        return sum(1 for o in self.ladder_orders
                   if o["order_id"] in self._filled_order_ids)

    # --------------------------------------------------------
    # pending SELL 잔존 조회
    # --------------------------------------------------------
    def _get_pending_sell(self) -> list:
        return [
            o for o in self.ladder_orders
            if o["order_id"] not in self._filled_order_ids
            and o["order_id"] not in self._canceled_order_ids
        ]

    # --------------------------------------------------------
    # [BUG 1 수정] run()에서 _sync_on_start() 먼저, 그 후 margin/leverage
    # [BUG 2 수정] SL 누적 — 나머지 sl_orders 전부 cancel
    # [BUG 5 수정] LADDER_ACTIVE 복구 시 avg_full/sl_price 재계산
    # --------------------------------------------------------
    def _sync_on_start(self):
        pos         = get_position(self.symbol)
        open_orders = get_open_orders(self.symbol)

        sell_orders = [o for o in open_orders if o["side"] == "SELL" and o["status"] == "NEW"]
        sell_sorted = sorted(sell_orders, key=lambda x: float(x["price"]))

        buy_normal  = [
            o for o in open_orders
            if o["side"] == "BUY"
            and o["status"] == "NEW"
            and o.get("type") not in ("STOP", "STOP_MARKET", "STOP_LIMIT")
        ]

        sl_orders   = [
            o for o in open_orders
            if o["side"] == "BUY"
            and o.get("reduceOnly")
            and o.get("type") in ("STOP", "STOP_MARKET", "STOP_LIMIT")
        ]

        log.info(f"[SYNC] 전체 주문 목록:")
        for o in open_orders:
            log.info(
                f"  orderId={o['orderId']} side={o['side']} type={o.get('type')} "
                f"price={o.get('price')} qty={o.get('origQty')} status={o.get('status')} "
                f"reduceOnly={o.get('reduceOnly')}"
            )

        if has_short_position(pos):
            log.info("[SYNC] 포지션 감지 → POSITION_HOLD 복구")
            self.state = "POSITION_HOLD"

            for i, o in enumerate(sell_sorted):
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(o["orderId"]),
                    "price":    float(o["price"]),
                    "qty":      float(o["origQty"]),
                })
            self.entry_price_base   = pos["avg_price"]
            self._last_position_amt = pos["amt"]

            self.exit_order_ids = [int(o["orderId"]) for o in buy_normal]

            self.max_filled_stage = self._count_filled_stages()
            self.last_stage       = self.max_filled_stage

            self.tp1_done  = True
            self.trail_low = None

            # [BUG 2 수정] SL 복구 — 0번째만 채택, 나머지 전부 cancel
            if sl_orders:
                # 유효한 SL 1개 선택 (stopPrice 기준 가장 낮은 것 = 최초 배치)
                sl_orders_sorted = sorted(
                    sl_orders,
                    key=lambda x: float(x.get("stopPrice", x.get("price", 0)))
                )
                sl_o = sl_orders_sorted[0]
                self.sl_order_id = int(sl_o["orderId"])
                self.sl_price    = float(sl_o.get("stopPrice", sl_o.get("price", 0)))
                log.info(
                    f"[SYNC] SL 복구 | orderId={self.sl_order_id} "
                    f"stopPrice={self.sl_price}"
                )

                # 나머지 SL 전부 즉시 취소
                for extra_sl in sl_orders_sorted[1:]:
                    eid = int(extra_sl["orderId"])
                    log.warning(
                        f"[SYNC] 잉여 SL 취소 | orderId={eid} "
                        f"stopPrice={extra_sl.get('stopPrice')} qty={extra_sl.get('origQty')}"
                    )
                    cancel_order(self.symbol, eid)
                    self._canceled_order_ids.add(eid)
            else:
                log.critical("[SYNC FAIL] 포지션 있음 + SL 주문 없음 → 엔진 중단")
                raise RuntimeError("SL BASE LOST")

            log.info(
                f"[SYNC] 복구 완료 | avg={pos['avg_price']} | "
                f"SELL {len(sell_sorted)}개 | BUY exit {len(buy_normal)}개 | "
                f"SL 채택 1개 / 취소 {len(sl_orders)-1}개 | "
                f"max_filled_stage={self.max_filled_stage} | tp1_done=True trail_low=None"
            )

        elif sell_sorted:
            log.info("[SYNC] 포지션 없음 + SELL 주문 존재 → LADDER_ACTIVE 복구")
            self.state = "LADDER_ACTIVE"
            for i, o in enumerate(sell_sorted):
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(o["orderId"]),
                    "price":    float(o["price"]),
                    "qty":      float(o["origQty"]),
                })
            self.entry_price_base = float(sell_sorted[0]["price"])
            log.info(f"[SYNC] entry_price_base = {self.entry_price_base:.4f} (min SELL price)")

            # [BUG 5 수정] avg_full / sl_price 재계산
            recovered_prices = [float(o["price"]) for o in sell_sorted]
            recovered_qtys   = [float(o["origQty"]) for o in sell_sorted]
            if recovered_prices and recovered_qtys:
                self.avg_full = calc_avg_full(recovered_prices, recovered_qtys)
                self.sl_price = self.avg_full * (1 + CFG["HARD_SL_PCT"])
                log.info(
                    f"[SYNC] avg_full 재계산 | avg_full={self.avg_full:.6f} "
                    f"sl_price={self.sl_price:.6f}"
                )

            # 고아 SL 처리
            if sl_orders:
                for sl_o in sl_orders:
                    log.warning(
                        f"[ORPHAN SL] 포지션 없음 + SL 잔존 → 취소 | "
                        f"orderId={sl_o['orderId']} stopPrice={sl_o.get('stopPrice')} "
                        f"price={sl_o.get('price')} qty={sl_o.get('origQty')}"
                    )
                    cancel_order(self.symbol, int(sl_o["orderId"]))

        else:
            log.info("[SYNC] 포지션 없음 + 주문 없음 → WATCHING 시작")
            self.state = "WATCHING"

            if sl_orders:
                for sl_o in sl_orders:
                    log.warning(
                        f"[ORPHAN SL] 포지션 없음 + SL 잔존 → 취소 | "
                        f"orderId={sl_o['orderId']} stopPrice={sl_o.get('stopPrice')} "
                        f"price={sl_o.get('price')} qty={sl_o.get('origQty')}"
                    )
                    cancel_order(self.symbol, int(sl_o["orderId"]))

    # --------------------------------------------------------
    # 메인 루프 — [BUG 1 수정] _sync_on_start() 먼저, margin/leverage 후순위
    # --------------------------------------------------------
    def run(self):
        log.info("=" * 60)
        log.info("VELLA RANGE SHORT LADDER v9 (BUG PATCH 5) 시작")
        log.info(f"심볼: {self.symbol} | 자본: {CFG['TOTAL_CAPITAL_USDT']} USDT | 레버: {CFG['LEVERAGE']}x")
        log.info(f"[HARD SL MODE] engine-side backup + exchange STOP_LIMIT reduceOnly")
        log.info("=" * 60)

        # [BUG 1 수정] 포지션/주문 상태 먼저 확인 후 설정 변경
        self._sync_on_start()
        set_margin_type(self.symbol, CFG["MARGIN_TYPE"])
        set_leverage(self.symbol, CFG["LEVERAGE"])

        while True:
            try:
                self._tick()
            except Exception as e:
                log.error(f"루프 오류: {e}", exc_info=True)
            time.sleep(CFG["POLL_INTERVAL_SEC"])

    # --------------------------------------------------------
    # 틱
    # --------------------------------------------------------
    def _tick(self):
        symbol = self.symbol
        ticker = client.ticker_price(symbol=symbol)
        current_price = float(ticker["price"])

        pos     = get_position(symbol)
        has_pos = has_short_position(pos)
        new_bar = self.bar_tracker.new_bar_closed()

        # ── COOLDOWN ──
        if self.state == "COOLDOWN":
            if new_bar:
                self.cooldown_bars -= 1
                log.info(f"쿨다운: 남은 봉 {self.cooldown_bars}")
            if self.cooldown_bars <= 0:
                self.state = "WATCHING"
                log.info("쿨다운 종료 → WATCHING")
            return

        # ── WATCHING ──
        if self.state == "WATCHING":
            if has_pos:
                log.warning("외부 포지션 감지 → POSITION_HOLD")
                self.state = "POSITION_HOLD"
                return

            if not check_4h_short_filter(symbol, self._htf_cache):
                return

            triggered, bar_ts = calc_ema15_trigger(symbol, self._trigger_cache)

            if triggered and bar_ts == self.last_trigger_bar_ts:
                log.debug(f"동일 5M 봉 재트리거 차단: ts={bar_ts}")
                return

            if triggered:
                self.last_trigger_bar_ts = bar_ts
                self._deploy_ladder(current_price)
            return

        # ── LADDER_ACTIVE ──
        if self.state == "LADDER_ACTIVE":
            if has_pos:
                log.info("포지션 체결 감지 → POSITION_HOLD")
                self.state              = "POSITION_HOLD"
                self.bars_after_deep    = 0
                self.no_fill_bars       = 0
                self._last_position_amt = pos["amt"]

                # [BUG 4 수정] LADDER_ACTIVE→POSITION_HOLD 전환 시 SL 미배치 체크
                # sl_price는 있으나 sl_order_id가 없는 경우 즉시 배치
                if self.sl_price is not None and self.sl_order_id is None:
                    log.warning(
                        "[BUG4 GUARD] POSITION_HOLD 진입 시 SL 미배치 감지 → 즉시 배치"
                    )
                    self._reset_sl_order(new_qty=pos["amt"])
                return

            if new_bar:
                self.no_fill_bars += 1
                log.info(f"거미줄 미체결 대기: {self.no_fill_bars}/{CFG['LADDER_NO_FILL_TIMEOUT_BARS']}봉")
            if self.no_fill_bars >= CFG["LADDER_NO_FILL_TIMEOUT_BARS"]:
                log.warning(f"거미줄 미체결 타임아웃 ({self.no_fill_bars}봉) → 철거 후 WATCHING")
                self._cancel_ladder_orders()
                self._reset_ladder()
                self.state = "WATCHING"
                return

            if self._is_ladder_invalid(current_price):
                log.warning("거미줄 무효화: 상단 이탈 → SELL 취소 후 WATCHING")
                self._cancel_ladder_orders()
                self._reset_ladder()
                self.state = "WATCHING"
                return

            log.info(f"거미줄 대기 | 현재가: {current_price:.4f}")
            return

        # ── POSITION_HOLD ──
        if self.state == "POSITION_HOLD":
            if not has_pos:
                log.info("포지션 청산 감지 → 쿨다운")
                self.cancel_buy_exit_orders(self.exit_order_ids)
                self.exit_order_ids = []
                self._cancel_ladder_orders()
                if self.sl_order_id is not None:
                    cancelled = self._safe_cancel(self.sl_order_id)
                    if not cancelled:
                        log.warning(f"[SL CANCEL] 포지션 청산 후 SL cancel 실패 | orderId={self.sl_order_id} → 무시 후 쿨다운")
                    self.sl_order_id = None
                self._start_cooldown()
                return

            avg_price    = pos["avg_price"]
            position_qty = pos["amt"]

            amt_changed = abs(position_qty - self._last_position_amt) > 0.0001
            cur_bar_ts  = self.bar_tracker.last_ts or 0
            need_check  = (
                amt_changed
                or (new_bar and cur_bar_ts != self._last_filled_check_ts)
                or self.max_filled_stage == 0
            )
            if need_check:
                filled = self._count_filled_stages()
                if filled > self.max_filled_stage:
                    log.info(f"체결 단계 갱신: {self.max_filled_stage} → {filled}")
                    self.max_filled_stage = filled
                self._last_position_amt    = position_qty
                self._last_filled_check_ts = cur_bar_ts

                pending_sell = self._get_pending_sell()
                stage1_only  = (self.max_filled_stage == 1 and len(pending_sell) > 0)

                log.info(
                    f"[POSITION STATUS] "
                    f"stage1_only={stage1_only} | "
                    f"pending_sell_count={len(pending_sell)} | "
                    f"max_filled_stage={self.max_filled_stage} | "
                    f"sl_order_id={self.sl_order_id} | "
                    f"sl_price={self.sl_price}"
                )

                if amt_changed and self.sl_price is not None:
                    self._reset_sl_order(new_qty=position_qty)

                if pending_sell and self._is_ladder_invalid(current_price):
                    log.warning("[POSITION_HOLD] 거미줄 상단 이탈 → pending SELL 취소")
                    for o in pending_sell:
                        self._safe_cancel(o["order_id"])

            log.info(
                f"HOLD | avg={avg_price:.4f} | price={current_price:.4f} | "
                f"stage={self.max_filled_stage} | qty={position_qty:.4f} | "
                f"tp1={self.tp1_done} | trail_low={self.trail_low} | "
                f"closing={self._closing_in_progress} | "
                f"sl_price={self.sl_price} | sl_order_id={self.sl_order_id}"
            )

            pnl_pct = (avg_price - current_price) / avg_price

            # 1. HARD SL (엔진 내부 백업)
            if pnl_pct < -CFG["HARD_SL_PCT"]:
                log.warning(
                    f"[HARD SL] engine-side 발동 | 손실 {pnl_pct*100:.2f}% | "
                    f"거래소 SL이 미작동한 경우"
                )
                self._final_close(symbol, position_qty, "HARD_SL")
                return

            # 2. TIMEOUT
            if self.max_filled_stage >= CFG["DEEP_FILL_STAGE"]:
                if new_bar:
                    self.bars_after_deep += 1
                if self.bars_after_deep >= CFG["TIMEOUT_BARS_AFTER_DEEP"]:
                    log.warning(f"TIMEOUT 발동 | {self.bars_after_deep}봉")
                    self._final_close(symbol, position_qty, "TIMEOUT")
                    return

            # 3. TP1
            if not self.tp1_done and pnl_pct >= CFG["TP1_PROFIT_PCT"]:
                self._handle_tp1(symbol, position_qty, current_price)
                return

            # 4. 트레일링
            if self.tp1_done:
                if self.trail_low is None:
                    self.trail_low = current_price
                    log.info(f"trail_low 초기화: {self.trail_low:.4f}")

                self.trail_low = min(self.trail_low, current_price)

                if current_price >= self.trail_low * (1 + CFG["TRAILING_REBOUND_PCT"]):
                    log.info(
                        f"[TRAIL EXIT] 저점={self.trail_low:.4f} 대비 +0.5% 반등 "
                        f"(current={current_price:.4f})"
                    )
                    self._final_close(symbol, position_qty, "TRAIL")
                return

            # 5. 지정가 EXIT 동기화
            if not self._closing_in_progress:
                self._sync_exit_order(symbol, avg_price, position_qty)

    # --------------------------------------------------------
    # TP1 처리
    # --------------------------------------------------------
    def _handle_tp1(self, symbol: str, position_qty: float, current_price: float):
        partial_qty = abs(position_qty) * CFG["TP1_PARTIAL_RATIO"]
        log.info(f"[EXIT/SL] BUY TP1 MARKET 50% 부분청산 시도 qty={partial_qty:.4f}")

        success = market_close_short(symbol, partial_qty)

        if success:
            time.sleep(0.2)
            pos = get_position(symbol)

            self.cancel_buy_exit_orders(self.exit_order_ids)
            self.exit_order_ids = []

            self._cancel_ladder_orders()
            self.ladder_orders     = []
            self._filled_order_ids = set()
            self.max_filled_stage  = 0

            self._last_position_amt = pos["amt"]
            self.tp1_done  = True
            self.trail_low = None

            self._reset_sl_order(new_qty=pos["amt"])

            log.info(
                f"[TP1] 부분청산 성공 → tp1_done=True | "
                f"잔량={pos['amt']:.4f} | trail_low=None(다음 tick 세팅)"
            )
        else:
            log.error("[TP1] 부분청산 실패 → 기존 주문 유지, 다음 tick 재시도")

    # --------------------------------------------------------
    # 공용 종료 헬퍼
    # --------------------------------------------------------
    def _final_close(self, symbol: str, position_qty: float, reason: str):
        log.info(f"[FINAL CLOSE] 사유={reason} | qty={position_qty:.4f}")
        self._closing_in_progress = True

        self.cancel_buy_exit_orders(self.exit_order_ids)
        self.exit_order_ids = []

        self._cancel_ladder_orders()

        if self.sl_order_id is not None:
            cancelled = self._safe_cancel(self.sl_order_id)
            if not cancelled:
                log.warning(f"[SL CANCEL] FINAL CLOSE 중 SL cancel 실패 | orderId={self.sl_order_id} → 시장가 청산으로 진행")
            self.sl_order_id = None

        success = market_close_short(symbol, abs(position_qty))

        if success:
            self._closing_in_progress = False
            self._start_cooldown()
        else:
            log.error(
                f"[FINAL CLOSE] 청산 실패 → POSITION_HOLD 유지, 다음 tick 재시도 "
                f"(사유={reason})"
            )

    # --------------------------------------------------------
    # 거미줄 배치
    # --------------------------------------------------------
    def _deploy_ladder(self, current_price: float):
        symbol  = self.symbol
        count   = CFG["LADDER_COUNT"]
        gap     = CFG["LADDER_GAP_PCT"]
        weights = normalize_weights(CFG["SIZE_WEIGHTS"], count)
        prices  = build_ladder_prices(current_price, count, gap)

        qtys = calc_ladder_quantities_per_stage(
            CFG["TOTAL_CAPITAL_USDT"], CFG["LEVERAGE"], weights, prices, current_price
        )

        effective_capital    = CFG["TOTAL_CAPITAL_USDT"] * CFG["MAX_CAPITAL_RATIO"] * CFG["LEVERAGE"]
        market_1_notional    = current_price * qtys[0]
        limit_notional_sum   = sum(prices[i] * qtys[i] for i in range(1, count))
        total_planned        = market_1_notional + limit_notional_sum
        ratio                = total_planned / effective_capital

        log.info(
            f"[CAPITAL CHECK] planned={total_planned:.2f} effective={effective_capital:.2f} "
            f"ratio={ratio:.3f} min={CFG['CAPITAL_CHECK_MIN_RATIO']} max={CFG['CAPITAL_CHECK_MAX_RATIO']}"
        )

        if not (CFG["CAPITAL_CHECK_MIN_RATIO"] <= ratio <= CFG["CAPITAL_CHECK_MAX_RATIO"]):
            log.error(f"[CAPITAL CHECK] 범위 이탈 ratio={ratio:.3f} → 거미줄 배치 중단")
            return

        for i in range(1, count):
            if prices[i] < current_price * 0.999:
                log.error(
                    f"[SHORT SAFETY] stage={i+1} price={prices[i]:.4f} < "
                    f"current*0.999={current_price*0.999:.4f} → 거미줄 배치 중단"
                )
                return

        cancel_all_orders(symbol)
        self._reset_ladder()
        self.entry_price_base = current_price

        all_prices    = [current_price] + prices[1:]
        self.avg_full = calc_avg_full(all_prices, qtys)
        self.sl_price = self.avg_full * (1 + CFG["HARD_SL_PCT"])

        log.info(
            f"[EXPECTED FULL AVG] avg_full={self.avg_full:.6f} "
            f"total_qty_full={sum(qtys):.2f} "
            f"total_notional_full={sum(p*q for p,q in zip(all_prices,qtys)):.2f}"
        )
        log.info(f"거미줄 배치 | 기준가: {current_price:.4f} | {count}단계")

        success   = 0
        order_1st = None

        order_1st = place_market_short(symbol, qtys[0])
        if order_1st:
            self.ladder_orders.append({
                "stage":    1,
                "order_id": int(order_1st["orderId"]),
                "price":    current_price,
                "qty":      qtys[0],
            })
            self._filled_order_ids.add(int(order_1st["orderId"]))
            self.max_filled_stage = 1
            success += 1
            log.info(f"[ENTRY LADDER] SELL stage=1 MARKET qty={fmt_qty(qtys[0], symbol)}")
        else:
            log.error("[ENTRY LADDER] 1차 시장가 진입 실패")

        for i in range(1, count):
            order = place_limit_short(symbol, prices[i], qtys[i])
            if order:
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(order["orderId"]),
                    "price":    prices[i],
                    "qty":      qtys[i],
                })
                success += 1
            time.sleep(0.15)

        if success == 0:
            log.error("거미줄 주문 0개 성공 → WATCHING 복귀")
            self.state = "WATCHING"
            return

        log.info(f"거미줄 배치 완료: {success}/{count}개")
        self.no_fill_bars = 0
        self.state = "POSITION_HOLD" if order_1st else "LADDER_ACTIVE"

        if order_1st:
            pos_now = get_position(symbol)
            if self.avg_full is not None and pos_now["avg_price"] > 0:
                log.info(
                    f"[AVG CHECK] calc_avg_full={self.avg_full:.6f} "
                    f"vs real_avg={pos_now['avg_price']:.6f}"
                )
            self._reset_sl_order(new_qty=pos_now["amt"])
        # [BUG 4 수정] 1차 실패 → LADDER_ACTIVE 진입 시 sl_price는 이미 세팅됨
        # LADDER_ACTIVE→POSITION_HOLD 전환 시 _tick() 내 BUG4 GUARD가 처리

    # --------------------------------------------------------
    # 거미줄 무효화
    # --------------------------------------------------------
    def _is_ladder_invalid(self, current_price: float) -> bool:
        if not self.entry_price_base or not self.ladder_orders:
            return False
        top_price  = self.ladder_orders[-1]["price"]
        buffer_pct = CFG["LADDER_GAP_PCT"] * CFG["LADDER_INVALIDATION_MULT"]
        return current_price > top_price * (1 + buffer_pct)

    # --------------------------------------------------------
    # 지정가 EXIT 동기화
    # --------------------------------------------------------
    def _sync_exit_order(self, symbol: str, avg_price: float, position_qty: float):
        stage      = max(self.max_filled_stage, 1)
        exit_price = calc_exit_price(avg_price, stage)
        exit_qty   = abs(position_qty)
        threshold  = CFG["EXIT_REPRICE_THRESHOLD_PCT"]

        need_replace = (
            not self.exit_order_ids
            or stage != self.last_stage
            or abs(exit_price - self.last_exit_price) > exit_price * threshold
            or abs(exit_qty   - self.last_exit_qty)   > exit_qty   * 0.05
        )

        if not need_replace:
            return

        self.cancel_buy_exit_orders(self.exit_order_ids)
        self.exit_order_ids = []
        self.last_stage     = -1

        order = place_limit_exit(symbol, exit_price, exit_qty)
        if order:
            self.exit_order_ids  = [int(order["orderId"])]
            self.last_exit_price = exit_price
            self.last_exit_qty   = exit_qty
            self.last_stage      = stage
            log.info(
                f"[EXIT/SL] BUY EXIT LIMIT 동기화 | stage={stage} | "
                f"청산가={exit_price:.4f} | qty={exit_qty:.4f}"
            )

    # --------------------------------------------------------
    # 내부 리셋
    # --------------------------------------------------------
    def _reset_ladder(self):
        self.ladder_orders          = []
        self.entry_price_base       = None
        self.max_filled_stage       = 0
        self.exit_order_ids         = []
        self.last_exit_qty          = 0.0
        self.last_exit_price        = 0.0
        self.bars_after_deep        = 0
        self.no_fill_bars           = 0
        self.last_stage             = 0
        self._filled_order_ids      = set()
        self._canceled_order_ids    = set()
        self._last_position_amt     = 0.0
        self._closing_in_progress   = False
        self._last_filled_check_ts  = 0
        self.tp1_done               = False
        self.trail_low              = None
        self.avg_full               = None
        self.sl_price               = None
        self.sl_order_id            = None

    def _start_cooldown(self):
        self._reset_ladder()
        self.state         = "COOLDOWN"
        self.cooldown_bars = CFG["REENTRY_COOLDOWN_BARS"]
        log.info(f"쿨다운 시작: {self.cooldown_bars}봉 (5m 기준)")


# ============================================================
# 엔트리포인트
# ============================================================
if __name__ == "__main__":
    engine = RangeShortEngine()
    engine.run()