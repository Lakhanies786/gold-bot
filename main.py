import asyncio
import io
import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import ta
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# ── PostgreSQL (optional — falls back to JSON if DATABASE_URL not set) ─
try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _get_db():
    if not _PG_AVAILABLE or not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        return None


def _init_db():
    conn = _get_db()
    if not conn:
        print("[DB] No DATABASE_URL — using JSON file storage")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id                 TEXT PRIMARY KEY,
                    mode               TEXT,
                    logged_at          TIMESTAMPTZ,
                    date               TEXT,
                    time_utc           TEXT,
                    symbol             TEXT,
                    signal             TEXT,
                    grade              TEXT,
                    trade_allowed      TEXT,
                    blocked_by         TEXT,
                    confidence         TEXT,
                    score              TEXT,
                    adx                REAL,
                    rsi                REAL,
                    vol_ratio          REAL,
                    spread             REAL,
                    session            TEXT,
                    daily_bias         TEXT,
                    market_regime      TEXT,
                    entry_price        REAL,
                    stop_loss          REAL,
                    take_profit        REAL,
                    risk_reward        TEXT,
                    nearest_support    REAL,
                    nearest_resistance REAL,
                    status             TEXT DEFAULT 'OPEN',
                    outcome            TEXT DEFAULT '-',
                    exit_price         REAL,
                    pnl_pct            REAL,
                    closed_at          TIMESTAMPTZ,
                    bars_held          INTEGER,
                    partial_closed     BOOLEAN DEFAULT FALSE,
                    partial_exit_price REAL,
                    partial_pnl_pct    REAL,
                    partial_closed_at  TIMESTAMPTZ,
                    sl_at_breakeven    BOOLEAN DEFAULT FALSE,
                    original_stop_loss REAL
                );
            """)
            conn.commit()
            # Migration-safe: add new columns if table already existed from
            # before partial-TP/break-even feature was introduced
            for col_def in [
                "partial_closed     BOOLEAN DEFAULT FALSE",
                "partial_exit_price REAL",
                "partial_pnl_pct    REAL",
                "partial_closed_at  TIMESTAMPTZ",
                "sl_at_breakeven    BOOLEAN DEFAULT FALSE",
                "original_stop_loss REAL",
            ]:
                col_name = col_def.split()[0]
                try:
                    cur.execute(f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col_def};")
                    conn.commit()
                except Exception as mig_err:
                    print(f"[DB] Migration note for {col_name}: {mig_err}")
                    conn.rollback()
        print("[DB] Tables ready ✅")
    except Exception as e:
        print(f"[DB] Init error: {e}")
    finally:
        conn.close()


def _db_insert_trade(entry: dict):
    conn = _get_db()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (
                    id, mode, logged_at, date, time_utc, symbol, signal, grade,
                    trade_allowed, blocked_by, confidence, score, adx, rsi,
                    vol_ratio, spread, session, daily_bias, market_regime,
                    entry_price, stop_loss, take_profit, risk_reward,
                    nearest_support, nearest_resistance,
                    status, outcome, exit_price, pnl_pct, original_stop_loss
                ) VALUES (
                    %(id)s, %(mode)s, %(logged_at)s, %(date)s, %(time_utc)s,
                    %(symbol)s, %(signal)s, %(grade)s, %(trade_allowed)s,
                    %(blocked_by)s, %(confidence)s, %(score)s, %(adx)s, %(rsi)s,
                    %(vol_ratio)s, %(spread)s, %(session)s, %(daily_bias)s,
                    %(market_regime)s, %(entry_price)s, %(stop_loss)s,
                    %(take_profit)s, %(risk_reward)s, %(nearest_support)s,
                    %(nearest_resistance)s, %(status)s, %(outcome)s,
                    %(exit_price)s, %(pnl_pct)s, %(stop_loss)s
                ) ON CONFLICT (id) DO NOTHING;
            """, {**entry, "logged_at": entry.get("logged_at")})
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] Insert error: {e}")
        return False
    finally:
        conn.close()


def _db_close_trade(trade_id, outcome, exit_price, pnl_pct, closed_at, bars_held):
    conn = _get_db()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades SET status='CLOSED', outcome=%s, exit_price=%s,
                    pnl_pct=%s, closed_at=%s, bars_held=%s
                WHERE id=%s AND status='OPEN';
            """, (outcome, exit_price, pnl_pct, closed_at, bars_held, trade_id))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] Update error: {e}")
        return False
    finally:
        conn.close()


def _db_partial_close_trade(trade_id, partial_exit_price, partial_pnl_pct,
                             partial_closed_at, new_stop_loss):
    """Record a 50% partial take-profit + move remaining SL to break-even.
    Trade stays OPEN — only stop_loss and the partial_* fields update.
    """
    conn = _get_db()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades SET
                    partial_closed = TRUE,
                    partial_exit_price = %s,
                    partial_pnl_pct = %s,
                    partial_closed_at = %s,
                    stop_loss = %s,
                    sl_at_breakeven = TRUE
                WHERE id=%s AND status='OPEN' AND partial_closed = FALSE;
            """, (partial_exit_price, partial_pnl_pct, partial_closed_at,
                  new_stop_loss, trade_id))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] Partial close error: {e}")
        return False
    finally:
        conn.close()


def _db_get_open_trades():
    conn = _get_db()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY logged_at;")
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[DB] Fetch error: {e}")
        return []
    finally:
        conn.close()


def _db_get_all_trades(mode=None, limit=500):
    conn = _get_db()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if mode:
                cur.execute(
                    "SELECT * FROM trades WHERE mode=%s ORDER BY logged_at DESC LIMIT %s;",
                    (mode, limit))
            else:
                cur.execute(
                    "SELECT * FROM trades ORDER BY logged_at DESC LIMIT %s;", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return rows
    except Exception as e:
        print(f"[DB] Fetch error: {e}")
        return []
    finally:
        conn.close()


app = FastAPI(title="Gold Trading Bot — XAUUSD", version="1.0.0")

# ── Log file paths — absolute, never reset on redeploy ───────────────
_BASE_DIR = Path(__file__).parent.resolve()
SIGNAL_LOG_FILE    = str(_BASE_DIR / "gold_signal_log.json")
BLOCKED_LOG_FILE   = str(_BASE_DIR / "gold_blocked_log.json")
SCALP_LOG_FILE     = str(_BASE_DIR / "gold_scalp_log.json")
SCALP_BLOCKED_FILE = str(_BASE_DIR / "gold_scalp_blocked.json")
NEWS_LOG_FILE      = str(_BASE_DIR / "gold_news_blocked_log.json")  # separate news event log

# ── Data Provider Config ───────────────────────────────────────────────
# Primary: AllTick (1000 req/day free, 10/min, WebSocket support, gold-specific)
# Fallback: Twelve Data (800 req/day) — used automatically if AllTick fails
ALLTICK_TOKEN    = os.getenv("ALLTICK_TOKEN", "")
ALLTICK_URL      = "https://quote.alltick.co/quote-b-api/kline"
ALLTICK_TICK_URL = "https://quote.alltick.co/quote-b-api/trade-tick"
ALLTICK_SYMBOL   = "GOLD"
# AllTick free tier: 1000 req/day, 10/min — gives headroom for faster scans
# Twelve Data free tier: 800 req/day, 8/min — needs slower scans to avoid 429

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_URL     = "https://api.twelvedata.com"
SYMBOL              = "XAU/USD"   # Gold spot price (Twelve Data fallback)

USE_ALLTICK = bool(ALLTICK_TOKEN)  # auto-detect which provider to use

# Dynamic scan cycle — faster if AllTick available, safer if only Twelve Data
SCAN_CYCLE_SECONDS = 120 if USE_ALLTICK else 600  # 2 min vs 10 min

# ── Twelve Data timeframe map ─────────────────────────────────────────
TF_MAP = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D":   "1day",
}

# ── AllTick kline_type map ─────────────────────────────────────────────
# 1=1min 2=5min 3=15min 4=30min 5=1h 6=2h 7=4h 8=daily
ALLTICK_TF_MAP = {
    "M1":  1,
    "M5":  2,
    "M15": 3,
    "H1":  5,
    "H4":  7,
    "D":   8,
}

# ── Filter Thresholds ─────────────────────────────────────────────────
MIN_CONFIDENCE      = 50
MIN_SCORE           = 9
MIN_VOL_RATIO       = 0.7
ADX_MIN             = 20
MIN_RSI_BUY         = 40
MAX_RSI_BUY         = 70
MIN_RSI_SELL        = 30
MAX_RSI_SELL        = 60

# ── Scalp Thresholds ──────────────────────────────────────────────────
SCALP_MIN_CONFIDENCE = 55
SCALP_MIN_SCORE      = 9
SCALP_MIN_VOL_RATIO  = 0.6
SCALP_ADX_MIN        = 20

# ── Grade Thresholds ─────────────────────────────────────────────────
GRADE_A = {"min_confidence": 70, "min_score": 11, "min_adx": 25, "min_volume": 1.0, "min_tf": 3}
GRADE_B = {"min_confidence": 55, "min_score":  9, "min_adx": 20, "min_volume": 0.7, "min_tf": 2}

# ── Gold Session Filter (UTC) ─────────────────────────────────────────
# Gold moves best during London and New York sessions
# Gold forex sessions (UTC) — covers full 22:00 Sun – 21:00 Fri market
SESSIONS = [
    (0,  8,  "ASIA"),       # Hong Kong / Tokyo
    (7,  12, "LONDON"),     # London morning
    (12, 16, "OVERLAP"),    # London/NY overlap — highest volume
    (16, 21, "NEW_YORK"),   # New York afternoon
    (22, 24, "ASIA_OPEN"),  # Sydney/Asia open
]

# ── In-memory logs ────────────────────────────────────────────────────
def _load_json(path: str) -> list:
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return []

def _save_json(path: str, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

signal_log:    list = _load_json(SIGNAL_LOG_FILE)
blocked_log:   list = _load_json(BLOCKED_LOG_FILE)
scalp_log:     list = _load_json(SCALP_LOG_FILE)
scalp_blocked: list = _load_json(SCALP_BLOCKED_FILE)
news_log:      list = _load_json(NEWS_LOG_FILE)


# ══════════════════════════════════════════════════════════════════════
# TWELVE DATA FETCHER
# ══════════════════════════════════════════════════════════════════════
# ── API rate-limit cache ─────────────────────────────────────────────
# Twelve Data free tier: 8 req/min, 800 req/day
# Price is derived from the M1 candle close — no separate /price API call needed
# This halves daily quota usage and makes the app resilient to rate limits
_candle_cache: dict = {}   # key: (granularity, count) → (df, timestamp)
_price_cache:  dict = {"price": None, "spread": 0.3, "ts": 0.0}

CANDLE_CACHE_TTL = 90 if USE_ALLTICK else 600  # 90s with AllTick, 10min with Twelve Data only
PRICE_CACHE_TTL  = 20 if USE_ALLTICK else 60  # 20s AllTick, 60s Twelve Data only
PRICE_STALE_TTL  = 180   # seconds — after this, price is dangerously stale
CANDLE_DANGEROUS_STALE_TTL = 300  # seconds — beyond this, refuse to log signals (was causing 10hr frozen data)

# ── Per-timeframe candle cache TTL ──────────────────────────────────────
# A flat 90s TTL for every timeframe was the real cause of hitting
# AllTick's 1000/day limit: H4/Daily candles barely change within 90s,
# yet were being re-fetched every cycle just like M1. Background scanning
# alone (swing + scalp + resolver running every 120s) was generating
# ~5,760 calls/day against AllTick — 5.7x over the free-tier limit.
#
# Fix: tier TTLs by how often each timeframe genuinely needs refreshing.
# M1 stays fast (it's scalp's primary trigger); M5/M15/H1/H4/D scale up
# since they're confirmation/trend-filter inputs that don't need to be
# second-fresh. This keeps total background usage to ~800/day, leaving
# real headroom for your own app/dashboard checks.
if USE_ALLTICK:
    CANDLE_CACHE_TTL_BY_TF = {
        "M1":  180,    # 3 min  — scalp's primary trigger, needs to stay fresh
        "M5":  600,    # 10 min — scalp confirmation timeframe
        "M15": 900,    # 15 min — shared by scalp (filter) and swing
        "H1":  1800,   # 30 min — swing timeframe, slow-moving
        "H4":  3600,   # 60 min — swing timeframe, very slow-moving
        "D":   7200,   # 120 min — daily candle, changes once a day anyway
    }
else:
    CANDLE_CACHE_TTL_BY_TF = {
        "M1":  600, "M5": 600, "M15": 900,
        "H1":  1800, "H4": 3600, "D": 7200,
    }

def get_fresh_price() -> tuple:
    """Force fresh price at signal log time for accurate entry price.
    Tries M1 candle first (free), then API, then falls back to cache.
    Uses count=100 to match other M1 fetches so they share one cache
    entry instead of each (granularity, count) pair being a separate
    cache key that independently expires and triggers its own API call.
    """
    try:
        df_m1 = get_oanda_candles("M1", 100)
        price = float(df_m1["close"].iloc[-1])
        _price_cache.update({"price": price, "spread": 0.3, "ts": time.time()})
        return price, 0.3
    except Exception:
        pass
    try:
        params = {"symbol": SYMBOL, "apikey": TWELVE_DATA_API_KEY}
        resp   = requests.get(f"{TWELVE_DATA_URL}/price", params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if "price" in data:
                price = float(data["price"])
                _price_cache.update({"price": price, "spread": 0.3, "ts": time.time()})
                return price, 0.3
    except Exception:
        pass
    return get_current_price()


# ── AllTick global rate limiter (1 req/sec per token, documented limit) ─
# A single shared timestamp + lock ensures every AllTick call anywhere in
# the app — whether from compute_swing_signal()'s 4 sequential candle
# fetches, compute_scalp_signal()'s 3, the resolver, or get_fresh_price()
# — waits long enough since the last AllTick call before firing the next
# one. Without this, calls landing in the same second get silently
# rejected with 429 regardless of remaining daily quota.
_alltick_last_call_ts = 0.0
_alltick_lock = threading.Lock()
ALLTICK_MIN_INTERVAL = 1.05  # seconds — slightly over 1.0 for safety margin

# ── AllTick usage tracking ──────────────────────────────────────────────
# Tracks calls/429s per UTC calendar day so we can tell apart "burst
# rate-limit" (now fixed) from "daily quota genuinely exhausted" (a
# different problem the rate limiter above can't solve) — without needing
# to find AllTick's own dashboard usage page.
_alltick_usage = {"date": None, "success": 0, "rate_limited": 0, "other_error": 0}


def _alltick_track_call(outcome: str):
    """outcome: 'success' | 'rate_limited' | 'other_error'"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _alltick_usage["date"] != today:
        _alltick_usage.update({"date": today, "success": 0, "rate_limited": 0, "other_error": 0})
    _alltick_usage[outcome] = _alltick_usage.get(outcome, 0) + 1


def _alltick_rate_limit_wait():
    """Block just long enough to respect AllTick's 1 req/sec limit."""
    global _alltick_last_call_ts
    with _alltick_lock:
        now = time.time()
        elapsed = now - _alltick_last_call_ts
        if elapsed < ALLTICK_MIN_INTERVAL:
            time.sleep(ALLTICK_MIN_INTERVAL - elapsed)
        _alltick_last_call_ts = time.time()


def _fetch_alltick_kline(granularity: str, count: int = 200) -> pd.DataFrame:
    """Fetch gold candles from AllTick (1000 req/day free, no PC needed).

    AllTick's documented limit is 1 request/second PER TOKEN — not just a
    daily cap. If two calls land in the same second, the second is
    rejected with 429 regardless of remaining daily quota. This is why
    429s persisted even after per-timeframe caching: compute_swing_signal()
    fires 4 sequential candle calls with zero delay between them, all
    landing in the same second. _alltick_rate_limit_wait() below enforces
    a minimum gap before every AllTick request so this can't happen.
    """
    _alltick_rate_limit_wait()
    kline_type = ALLTICK_TF_MAP.get(granularity, 5)
    query_obj = {
        "trace": f"goldbot-{int(time.time()*1000)}",
        "data": {
            "code": ALLTICK_SYMBOL,
            "kline_type": kline_type,
            "kline_timestamp_end": 0,
            "query_kline_num": min(count, 500),
            "adjust_type": 0,
        },
    }
    query_str = json.dumps(query_obj)
    params = {"token": ALLTICK_TOKEN, "query": query_str}
    resp = requests.get(ALLTICK_URL, params=params, timeout=15)
    if resp.status_code == 429:
        _alltick_track_call("rate_limited")
        resp.raise_for_status()  # raises HTTPError, caught by caller's fallback
    resp.raise_for_status()
    data = resp.json()

    if data.get("ret") != 200:
        _alltick_track_call("other_error")
        raise ValueError(f"AllTick error: {data.get('msg', 'unknown error')}")

    kline_list = data.get("data", {}).get("kline_list", [])
    if not kline_list:
        _alltick_track_call("other_error")
        raise ValueError("AllTick returned no kline data")

    _alltick_track_call("success")
    rows = []
    for c in kline_list:
        rows.append({
            "time":   pd.to_datetime(int(c["timestamp"]), unit="s"),
            "open":   float(c["open_price"]),
            "high":   float(c["high_price"]),
            "low":    float(c["low_price"]),
            "close":  float(c["close_price"]),
            "volume": float(c.get("volume", 1000) or 1000),
        })
    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df


def _fetch_alltick_price() -> tuple:
    """Fetch latest gold price from AllTick trade-tick endpoint."""
    _alltick_rate_limit_wait()
    query_obj = {
        "trace": f"goldbot-{int(time.time()*1000)}",
        "data": {"symbol_list": [{"code": ALLTICK_SYMBOL}]},
    }
    query_str = json.dumps(query_obj)
    params = {"token": ALLTICK_TOKEN, "query": query_str}
    resp = requests.get(ALLTICK_TICK_URL, params=params, timeout=10)
    if resp.status_code == 429:
        _alltick_track_call("rate_limited")
        resp.raise_for_status()
    resp.raise_for_status()
    data = resp.json()

    if data.get("ret") != 200:
        _alltick_track_call("other_error")
        raise ValueError(f"AllTick error: {data.get('msg', 'unknown error')}")

    tick_list = data.get("data", {}).get("tick_list", [])
    if not tick_list:
        _alltick_track_call("other_error")
        raise ValueError("AllTick returned no tick data")

    _alltick_track_call("success")
    price = float(tick_list[0]["price"])
    return price, 0.3


def get_oanda_candles(granularity: str, count: int = 200) -> pd.DataFrame:
    """
    Fetch XAU/USD candles — tries AllTick first (if token set), falls back
    to Twelve Data automatically. Returns cached data if fresh.
    granularity: M1, M5, M15, H1, H4, D

    Cache TTL is per-timeframe, not flat — a Daily candle doesn't need
    refreshing every 90s like an M1 candle does. This is what actually
    keeps daily API usage under the free-tier limits; a flat 90s TTL for
    ALL timeframes (including H4/Daily) was burning quota for data that
    hadn't changed, causing 429s and "refusing stale data" blank screens.
    """
    cache_key = (granularity, count)
    cached = _candle_cache.get(cache_key)
    ttl = CANDLE_CACHE_TTL_BY_TF.get(granularity, CANDLE_CACHE_TTL)
    if cached and (time.time() - cached[1]) < ttl:
        return cached[0]

    # ── Try AllTick first ───────────────────────────────────────────
    if USE_ALLTICK:
        try:
            df = _fetch_alltick_kline(granularity, count)
            _candle_cache[cache_key] = (df, time.time())
            return df
        except Exception as e:
            print(f"[Candles] AllTick failed ({e}) — falling back to Twelve Data")

    # ── Fallback: Twelve Data ────────────────────────────────────────
    interval = TF_MAP.get(granularity, "1h")
    params   = {
        "symbol":     SYMBOL,
        "interval":   interval,
        "outputsize": min(count, 5000),
        "apikey":     TWELVE_DATA_API_KEY,
        "format":     "JSON",
        "order":      "ASC",
    }
    resp = requests.get(f"{TWELVE_DATA_URL}/time_series", params=params, timeout=15)
    if resp.status_code == 429:
        if cached:
            stale_age = time.time() - cached[1]
            print(f"[Candles] 429 rate limit for {granularity} — cache is {int(stale_age)}s old")
            # Dangerous-stale threshold scales per timeframe: a 2-hour-old
            # Daily candle is still completely valid, but a 2-hour-old M1
            # candle is corrupted data. Using one flat threshold for all
            # timeframes was incorrectly refusing perfectly good H4/Daily
            # cache. Threshold = 3x the timeframe's own normal TTL.
            danger_threshold = max(
                CANDLE_CACHE_TTL_BY_TF.get(granularity, CANDLE_CACHE_TTL) * 3,
                CANDLE_DANGEROUS_STALE_TTL,
            )
            if stale_age > danger_threshold:
                # Cache too old to trust — refuse to silently serve corrupted data.
                # This is what was causing hours-long frozen price/volume in logs.
                raise ValueError(
                    f"Rate limited and cache for {granularity} is {int(stale_age)}s old "
                    f"(> {danger_threshold}s limit) — refusing stale data."
                )
            return cached[0]
        raise ValueError("Rate limited by Twelve Data. Wait 1 minute and retry.")
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', 'No data returned')}")

    rows = []
    for c in data["values"]:
        rows.append({
            "time":   pd.to_datetime(c["datetime"]),
            "open":   float(c["open"]),
            "high":   float(c["high"]),
            "low":    float(c["low"]),
            "close":  float(c["close"]),
            "volume": float(c.get("volume", 1000)),
        })

    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    _candle_cache[cache_key] = (df, time.time())
    return df


def get_current_price() -> tuple:
    """Get current XAU/USD price.

    Priority:
    1. Cache if fresh (< 60s)
    2. AllTick live tick (if token set) — most accurate, free
    3. Fresh M1 candle close (< 90s old) — free, no API call
    4. Fresh M15 candle close (< 120s old) — free
    5. Twelve Data /price API call (fallback)
    6. Stale cache if < 3 min old
    7. Error if cache > 3 min old — prevents stuck price display
    """
    now       = time.time()
    cache_age = now - _price_cache["ts"] if _price_cache["ts"] else 9999

    # 1. Return cache if fresh
    if _price_cache["price"] is not None and cache_age < PRICE_CACHE_TTL:
        return _price_cache["price"], _price_cache["spread"]

    # 2. Try AllTick live tick — most accurate, near real-time
    if USE_ALLTICK:
        try:
            price, spread = _fetch_alltick_price()
            _price_cache.update({"price": price, "spread": spread, "ts": now})
            return price, spread
        except Exception as e:
            print(f"[Price] AllTick failed ({e}) — trying candle cache")

    # 3. Derive from M1 candle cache if candle itself is fresh
    m1_cached = _candle_cache.get(("M1", 100))
    if m1_cached and (now - m1_cached[1]) < 90:
        try:
            price = float(m1_cached[0]["close"].iloc[-1])
            _price_cache.update({"price": price, "spread": 0.3, "ts": now})
            return price, 0.3
        except Exception:
            pass

    # 4. Try M15 candle cache if fresh
    m15_cached = _candle_cache.get(("M15", 200))
    if m15_cached and (now - m15_cached[1]) < 120:
        try:
            price = float(m15_cached[0]["close"].iloc[-1])
            _price_cache.update({"price": price, "spread": 0.3, "ts": now})
            return price, 0.3
        except Exception:
            pass

    # 5. Call Twelve Data API (fallback)
    try:
        params = {"symbol": SYMBOL, "apikey": TWELVE_DATA_API_KEY}
        resp   = requests.get(f"{TWELVE_DATA_URL}/price", params=params, timeout=10)

        if resp.status_code == 429:
            if _price_cache["price"] is not None and cache_age < PRICE_STALE_TTL:
                print(f"[Price] 429 — using cache ({int(cache_age)}s old)")
                return _price_cache["price"], _price_cache["spread"]
            _price_cache["price"] = None
            raise ValueError(f"Rate limited. Cache {int(cache_age)}s old — too stale.")

        resp.raise_for_status()
        data = resp.json()
        if "price" not in data:
            raise ValueError(f"Twelve Data error: {data.get('message', 'No price data')}")

        price  = float(data["price"])
        spread = 0.3
        _price_cache.update({"price": price, "spread": spread, "ts": now})
        print(f"[Price] Updated: ${price:.2f}")
        return price, spread

    except requests.exceptions.HTTPError as e:
        if _price_cache["price"] is not None and cache_age < PRICE_STALE_TTL:
            print(f"[Price] API error ({e}) — cache ({int(cache_age)}s old)")
            return _price_cache["price"], _price_cache["spread"]
        raise


# ══════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # Trend
    df["ema9"]   = ta.trend.ema_indicator(close, window=9)
    df["ema21"]  = ta.trend.ema_indicator(close, window=21)
    df["ema50"]  = ta.trend.ema_indicator(close, window=50)
    df["ema200"] = ta.trend.ema_indicator(close, window=200)

    # ADX
    adx_ind      = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"]    = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()
    df["adx_neg"] = adx_ind.adx_neg()

    # MACD
    macd_ind       = ta.trend.MACD(close)
    df["macd"]     = macd_ind.macd()
    df["macd_sig"] = macd_ind.macd_signal()
    df["macd_hist"]= macd_ind.macd_diff()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR — critical for gold position sizing
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume SMA — NOTE: Twelve Data and AllTick both return synthetic/static
    # volume for XAU/USD (it's an OTC forex/spot instrument with no central
    # exchange tape, unlike stocks). Relying on it as a real "volume_ratio"
    # was producing meaningless/stuck values (e.g. 0.08 frozen for 10 hours).
    # Instead we use a range-based activity proxy: how wide is the current
    # candle's true range vs its recent average — a real, live-updating
    # measure of market activity that works for any instrument including gold.
    true_range = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["range_sma"]  = true_range.rolling(20).mean()
    df["vol_ratio"]  = (true_range / df["range_sma"].replace(0, np.nan)).clip(upper=5.0)
    # Keep raw volume column too (for display/debug) but it is NOT used for grading
    df["vol_sma"]    = vol.rolling(20).mean()

    # Stochastic
    stoch          = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"]  = stoch.stoch()
    df["stoch_d"]  = stoch.stoch_signal()

    # Support & Resistance (pivot-based)
    df["pivot"]     = (high + low + close) / 3
    df["resist1"]   = (2 * df["pivot"]) - low
    df["support1"]  = (2 * df["pivot"]) - high

    return df


def detect_daily_bias(df_daily: pd.DataFrame) -> dict:
    """Determine daily trend direction for gold."""
    df = add_indicators(df_daily.copy())
    last = df.iloc[-1]

    ema21  = float(last["ema21"])
    ema50  = float(last["ema50"])
    ema200 = float(last["ema200"])
    adx    = float(last["adx"]) if pd.notna(last["adx"]) else 0
    close  = float(last["close"])
    macd_h = float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0

    bullish_count = sum([
        close > ema21,
        ema21 > ema50,
        ema50 > ema200,
        macd_h > 0,
        adx > 20,
    ])

    if bullish_count >= 4:
        bias = "BULLISH"
    elif bullish_count <= 1:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bias":   bias,
        "adx":    round(adx, 1),
        "ema21":  round(ema21, 2),
        "ema50":  round(ema50, 2),
        "ema200": round(ema200, 2),
        "close":  round(close, 2),
    }


def detect_market_regime(df: pd.DataFrame) -> dict:
    """Detect gold market regime."""
    last    = df.iloc[-1]
    adx     = float(last["adx"])     if pd.notna(last["adx"])     else 0
    atr     = float(last["atr"])     if pd.notna(last["atr"])     else 0
    atr_avg = df["atr"].rolling(20).mean().iloc[-1]
    atr_avg = float(atr_avg)         if pd.notna(atr_avg)         else atr
    close   = float(last["close"])
    ema21   = float(last["ema21"])   if pd.notna(last["ema21"])   else close
    ema50   = float(last["ema50"])   if pd.notna(last["ema50"])   else close

    volatile = atr > atr_avg * 2.0

    if volatile:
        regime = "VOLATILE"
    elif adx >= 25 and close > ema21 and ema21 > ema50:
        regime = "TRENDING_BULL"
    elif adx >= 25 and close < ema21 and ema21 < ema50:
        regime = "TRENDING_BEAR"
    elif adx < 20:
        regime = "RANGING"
    else:
        regime = "WEAK_TREND"

    safe = regime in ("TRENDING_BULL", "TRENDING_BEAR", "WEAK_TREND")
    return {"regime": regime, "safe": safe, "adx": round(adx, 1), "atr": round(atr, 2)}


def find_support_resistance(df: pd.DataFrame) -> dict:
    """Find nearest support and resistance levels for gold."""
    price   = float(df["close"].iloc[-1])
    highs   = df["high"].rolling(10).max().dropna()
    lows    = df["low"].rolling(10).min().dropna()

    resistances = sorted([h for h in highs.tail(50) if h > price])
    supports    = sorted([l for l in lows.tail(50) if l < price], reverse=True)

    nearest_res = resistances[0] if resistances else price * 1.005
    nearest_sup = supports[0]    if supports    else price * 0.995

    dist_res = ((nearest_res - price) / price) * 100
    dist_sup = ((price - nearest_sup) / price) * 100

    return {
        "nearest_resistance": round(nearest_res, 2),
        "nearest_support":    round(nearest_sup, 2),
        "dist_to_resistance_pct": round(dist_res, 2),
        "dist_to_support_pct":    round(dist_sup, 2),
    }


def calculate_smart_sltp(price: float, signal: str, atr: float,
                          nearest_support: float, nearest_resistance: float,
                          regime: str, mode: str = "swing") -> dict:
    """
    Smart SL/TP calculation using structure + ATR + regime.

    Logic:
    1. Place SL just beyond nearest support/resistance (structure-aware)
    2. Apply ATR-based minimum distance so SL is never too tight
    3. Adjust multipliers based on market regime (volatile = wider)
    4. Enforce minimum $3 SL distance for gold
    5. Calculate TP to maintain minimum 1:2 RR
    6. Scalp uses tighter multipliers than swing
    """
    MIN_SL_DISTANCE = 3.0   # gold minimum $3 SL
    MAX_SL_DISTANCE = 8.0 if mode == "scalp" else 35.0  # scalp max $8, swing max $35
    BUFFER          = 0.30  # $0.30 buffer beyond S/R level

    # ── Regime-based ATR multipliers ─────────────────────────────────
    if mode == "scalp":
        # Scalp needs tight SL/TP — gold should hit these within 30-60 min
        regime_multipliers = {
            "VOLATILE":      (1.2, 2.4),
            "TRENDING_BULL": (0.8, 1.6),
            "TRENDING_BEAR": (0.8, 1.6),
            "WEAK_TREND":    (0.8, 1.6),
            "RANGING":       (0.7, 1.4),
        }
    else:  # swing
        regime_multipliers = {
            "VOLATILE":      (2.2, 4.4),
            "TRENDING_BULL": (1.5, 3.0),
            "TRENDING_BEAR": (1.5, 3.0),
            "WEAK_TREND":    (1.5, 3.0),
            "RANGING":       (1.2, 2.4),
        }

    sl_mult, tp_mult = regime_multipliers.get(regime, (1.5, 3.0))
    atr_sl_dist = atr * sl_mult
    atr_sl_dist = max(atr_sl_dist, MIN_SL_DISTANCE)

    if signal == "BUY":
        # SL: just below nearest support, but at least ATR-based distance
        if nearest_support and nearest_support < price:
            structure_sl = nearest_support - BUFFER
            sl_distance  = max(price - structure_sl, atr_sl_dist)
        else:
            sl_distance  = atr_sl_dist
        # Cap scalp SL so it's never too wide
        sl_distance = min(sl_distance, MAX_SL_DISTANCE)
        stop_loss   = round(price - sl_distance, 2)
        take_profit = round(price + sl_distance * (tp_mult / sl_mult), 2)

    elif signal == "SELL":
        # SL: just above nearest resistance, but at least ATR-based distance
        if nearest_resistance and nearest_resistance > price:
            structure_sl = nearest_resistance + BUFFER
            sl_distance  = max(structure_sl - price, atr_sl_dist)
        else:
            sl_distance  = atr_sl_dist
        # Cap scalp SL so it's never too wide
        sl_distance = min(sl_distance, MAX_SL_DISTANCE)
        stop_loss   = round(price + sl_distance, 2)
        take_profit = round(price - sl_distance * (tp_mult / sl_mult), 2)

    else:
        return {"stop_loss": None, "take_profit": None, "risk_reward": "N/A",
                "sl_method": "none", "sl_distance": 0}

    # ── RR calculation ────────────────────────────────────────────────
    risk   = abs(price - stop_loss)
    reward = abs(take_profit - price)
    rr     = round(reward / risk, 1) if risk > 0 else 2.0

    # ── Label which method set the SL ────────────────────────────────
    if signal == "BUY" and nearest_support:
        used_structure = (price - stop_loss) > atr_sl_dist
    elif signal == "SELL" and nearest_resistance:
        used_structure = (stop_loss - price) > atr_sl_dist
    else:
        used_structure = False

    sl_method = f"structure+ATR({regime})" if used_structure else f"ATR({regime})"

    return {
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "risk_reward": f"1:{rr}",
        "sl_method":   sl_method,
        "sl_distance": round(risk, 2),
    }


def calculate_fibonacci(df: pd.DataFrame) -> dict:
    """
    Calculate Fibonacci retracement levels from recent swing high/low.
    Gold respects Fibonacci levels extremely well — especially 0.382, 0.5, 0.618.
    Uses last 50 candles to find swing high and swing low.
    """
    recent   = df.tail(50)
    swing_high = float(recent["high"].max())
    swing_low  = float(recent["low"].min())
    diff       = swing_high - swing_low
    price      = float(df["close"].iloc[-1])

    if diff == 0:
        return {"fib_valid": False}

    # Standard Fibonacci retracement levels
    levels = {
        "fib_0":     round(swing_high, 2),
        "fib_236":   round(swing_high - 0.236 * diff, 2),
        "fib_382":   round(swing_high - 0.382 * diff, 2),
        "fib_500":   round(swing_high - 0.500 * diff, 2),
        "fib_618":   round(swing_high - 0.618 * diff, 2),
        "fib_786":   round(swing_high - 0.786 * diff, 2),
        "fib_100":   round(swing_low, 2),
        "swing_high": round(swing_high, 2),
        "swing_low":  round(swing_low, 2),
        "fib_valid":  True,
    }

    # Find nearest Fibonacci level to current price
    fib_vals = [levels["fib_236"], levels["fib_382"], levels["fib_500"],
                levels["fib_618"], levels["fib_786"]]
    nearest_fib    = min(fib_vals, key=lambda x: abs(x - price))
    dist_to_fib    = round(abs(price - nearest_fib), 2)
    dist_to_fib_pct = round((dist_to_fib / price) * 100, 3)

    # Determine if price is AT a key Fibonacci level (within 0.3%)
    at_fib_support    = False
    at_fib_resistance = False

    # In an uptrend: fib levels below price = support (BUY zone)
    # In a downtrend: fib levels above price = resistance (SELL zone)
    for fib in fib_vals:
        dist_pct = abs(price - fib) / price * 100
        if dist_pct < 0.3:
            if fib < price:
                at_fib_support    = True   # price just above fib = potential bounce up
            else:
                at_fib_resistance = True   # price just below fib = potential rejection

    levels.update({
        "nearest_fib":        nearest_fib,
        "dist_to_fib":        dist_to_fib,
        "dist_to_fib_pct":    dist_to_fib_pct,
        "at_fib_support":     at_fib_support,
        "at_fib_resistance":  at_fib_resistance,
    })

    return levels



def detect_market_structure(df: pd.DataFrame) -> dict:
    """
    SMC / BOS Analysis for Gold.
    Detects:
    - Market structure: UPTREND / DOWNTREND / RANGING
    - Break of Structure (BOS) bullish or bearish
    - Demand and Supply zones
    - Retest of zone after BOS
    - Liquidity sweep detection
    Uses last 50 candles of the provided dataframe (1H recommended).
    """
    df   = df.copy().tail(100).reset_index(drop=True)
    n    = len(df)
    if n < 20:
        return {'structure': 'UNKNOWN', 'bos_bullish': False, 'bos_bearish': False}

    close  = df['close'].values
    high   = df['high'].values
    low    = df['low'].values
    volume = df['volume'].values
    price  = float(close[-1])

    # ── Step 1: Find swing highs and swing lows (pivot points) ───────
    # A swing high: higher than 3 candles before and after
    # A swing low:  lower  than 3 candles before and after
    swing_highs = []
    swing_lows  = []
    pivot       = 3

    for i in range(pivot, n - pivot):
        if all(high[i] > high[i-j] for j in range(1, pivot+1)) and \
           all(high[i] > high[i+j] for j in range(1, pivot+1)):
            swing_highs.append((i, float(high[i])))
        if all(low[i]  < low[i-j]  for j in range(1, pivot+1)) and \
           all(low[i]  < low[i+j]  for j in range(1, pivot+1)):
            swing_lows.append((i, float(low[i])))

    # ── Step 2: Market structure — label HH/HL/LH/LL ─────────────────
    structure = 'RANGING'
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_hh = swing_highs[-1][1]
        prev_hh = swing_highs[-2][1]
        last_hl = swing_lows[-1][1]
        prev_hl = swing_lows[-2][1]

        higher_highs = last_hh > prev_hh
        higher_lows  = last_hl > prev_hl
        lower_highs  = last_hh < prev_hh
        lower_lows   = last_hl < prev_hl

        if higher_highs and higher_lows:
            structure = 'UPTREND'
        elif lower_highs and lower_lows:
            structure = 'DOWNTREND'
        else:
            structure = 'RANGING'

    # ── Step 3: Break of Structure (BOS) ─────────────────────────────
    bos_bullish = False
    bos_bearish = False
    bos_level   = None
    bos_index   = None
    vol_avg     = float(np.mean(volume[-20:])) if len(volume) >= 20 else 1.0

    if swing_highs and len(close) > swing_highs[-1][0]:
        prev_swing_high     = swing_highs[-1][1]
        # BOS Bullish: price closes above previous swing high with volume
        for i in range(swing_highs[-1][0] + 1, n):
            if close[i] > prev_swing_high and float(volume[i]) > vol_avg * 1.2:
                bos_bullish = True
                bos_level   = prev_swing_high
                bos_index   = i
                break

    if swing_lows and len(close) > swing_lows[-1][0]:
        prev_swing_low      = swing_lows[-1][1]
        # BOS Bearish: price closes below previous swing low with volume
        for i in range(swing_lows[-1][0] + 1, n):
            if close[i] < prev_swing_low and float(volume[i]) > vol_avg * 1.2:
                bos_bearish = True
                bos_level   = prev_swing_low
                bos_index   = i
                break

    # ── Step 4: Demand & Supply Zone Detection ────────────────────────
    # Demand zone: last bearish candle before a strong bullish impulse
    # Supply zone: last bullish candle before a strong bearish impulse
    atr_val     = float(np.mean(np.abs(np.diff(close[-20:]))))  # simple ATR proxy
    zone_width  = atr_val * 0.5

    demand_zones = []
    supply_zones = []

    for i in range(3, n - 1):
        candle_body = abs(close[i] - close[i-1])
        # Strong bullish impulse candle
        if close[i] > close[i-1] and candle_body > atr_val * 1.5:
            # Find last bearish candle before this impulse
            for j in range(i-1, max(i-5, 0), -1):
                if close[j] < close[j-1]:  # bearish candle
                    zone_top    = float(high[j])
                    zone_bottom = float(low[j])
                    demand_zones.append({
                        'top':    round(zone_top, 2),
                        'bottom': round(zone_bottom, 2),
                        'mid':    round((zone_top + zone_bottom) / 2, 2),
                        'index':  j,
                    })
                    break

        # Strong bearish impulse candle
        if close[i] < close[i-1] and candle_body > atr_val * 1.5:
            for j in range(i-1, max(i-5, 0), -1):
                if close[j] > close[j-1]:  # bullish candle
                    zone_top    = float(high[j])
                    zone_bottom = float(low[j])
                    supply_zones.append({
                        'top':    round(zone_top, 2),
                        'bottom': round(zone_bottom, 2),
                        'mid':    round((zone_top + zone_bottom) / 2, 2),
                        'index':  j,
                    })
                    break

    # Keep only recent zones (last 5 of each)
    demand_zones = demand_zones[-5:]
    supply_zones = supply_zones[-5:]

    # ── Step 5: Retest Detection ──────────────────────────────────────
    # Price returned to demand zone after BOS bullish = retest
    in_demand_zone  = False
    in_supply_zone  = False
    nearest_demand  = None
    nearest_supply  = None

    for zone in demand_zones:
        if zone['bottom'] <= price <= zone['top'] * 1.002:
            in_demand_zone = True
            nearest_demand = zone
            break
        if nearest_demand is None or abs(price - zone['mid']) < abs(price - nearest_demand['mid']):
            nearest_demand = zone

    for zone in supply_zones:
        if zone['bottom'] * 0.998 <= price <= zone['top']:
            in_supply_zone = True
            nearest_supply = zone
            break
        if nearest_supply is None or abs(price - zone['mid']) < abs(price - nearest_supply['mid']):
            nearest_supply = zone

    # Retest confirmation: BOS happened + price now in zone + bullish rejection
    bullish_rejection = close[-1] > close[-2] and low[-1] < low[-2]  # hammer-like
    bearish_rejection = close[-1] < close[-2] and high[-1] > high[-2]

    retest_buy  = bos_bullish and in_demand_zone and bullish_rejection
    retest_sell = bos_bearish and in_supply_zone and bearish_rejection

    # ── Step 6: Liquidity Sweep Detection ────────────────────────────
    # Price briefly breaks a swing low then immediately recovers (bull liq sweep)
    # Price briefly breaks a swing high then immediately drops  (bear liq sweep)
    liq_sweep_bull = False
    liq_sweep_bear = False

    if swing_lows and n >= 3:
        recent_low = swing_lows[-1][1]
        # Last 3 candles: broke below swing low then recovered
        if low[-2] < recent_low and close[-1] > recent_low and float(volume[-1]) > vol_avg * 1.3:
            liq_sweep_bull = True

    if swing_highs and n >= 3:
        recent_high = swing_highs[-1][1]
        # Last 3 candles: broke above swing high then dropped
        if high[-2] > recent_high and close[-1] < recent_high and float(volume[-1]) > vol_avg * 1.3:
            liq_sweep_bear = True

    # ── Step 7: Confidence boost from SMC ─────────────────────────────
    smc_buy_boost  = 0
    smc_sell_boost = 0

    if structure == 'UPTREND':   smc_buy_boost  += 15
    if structure == 'DOWNTREND': smc_sell_boost += 15
    if bos_bullish:              smc_buy_boost  += 10
    if bos_bearish:              smc_sell_boost += 10
    if in_demand_zone:           smc_buy_boost  += 10
    if in_supply_zone:           smc_sell_boost += 10
    if retest_buy:               smc_buy_boost  += 15
    if retest_sell:              smc_sell_boost += 15
    if liq_sweep_bull:           smc_buy_boost  += 10
    if liq_sweep_bear:           smc_sell_boost += 10

    return {
        'structure':         structure,          # UPTREND / DOWNTREND / RANGING
        'bos_bullish':       bos_bullish,        # Break of Structure up
        'bos_bearish':       bos_bearish,        # Break of Structure down
        'bos_level':         round(bos_level, 2) if bos_level else None,
        'in_demand_zone':    in_demand_zone,     # price inside demand zone
        'in_supply_zone':    in_supply_zone,     # price inside supply zone
        'nearest_demand':    nearest_demand,     # closest demand zone dict
        'nearest_supply':    nearest_supply,     # closest supply zone dict
        'retest_buy':        retest_buy,         # BOS + demand zone + rejection
        'retest_sell':       retest_sell,        # BOS + supply zone + rejection
        'liq_sweep_bull':    liq_sweep_bull,     # liquidity sweep bullish
        'liq_sweep_bear':    liq_sweep_bear,     # liquidity sweep bearish
        'demand_zones':      demand_zones,       # all detected demand zones
        'supply_zones':      supply_zones,       # all detected supply zones
        'smc_buy_boost':     smc_buy_boost,      # confidence boost for BUY
        'smc_sell_boost':    smc_sell_boost,     # confidence boost for SELL
        'swing_highs':       [(i, round(v,2)) for i,v in swing_highs[-3:]],
        'swing_lows':        [(i, round(v,2)) for i,v in swing_lows[-3:]],
    }



def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    Detect high-probability candlestick patterns at key levels.
    These are used as CONFIRMATION signals — only valid when price
    is at a demand/supply zone, Fibonacci level, or after BOS retest.

    Patterns detected:
    - Hammer / Inverted Hammer (bullish reversal)
    - Shooting Star / Hanging Man (bearish reversal)
    - Bullish Engulfing (strong bullish reversal)
    - Bearish Engulfing (strong bearish reversal)
    - Pin Bar / Rejection Candle (both directions)
    - Doji (indecision — reduces confidence)
    - Morning Star / Evening Star (3-candle reversal)
    - Inside Bar (consolidation breakout setup)
    """
    if len(df) < 3:
        return {'pattern': 'NONE', 'direction': 'NEUTRAL', 'strength': 0}

    c  = df['close'].values
    o  = df['open'].values
    h  = df['high'].values
    l  = df['low'].values

    # Last 3 candles
    c0, c1, c2 = float(c[-3]), float(c[-2]), float(c[-1])  # c2 = most recent
    o0, o1, o2 = float(o[-3]), float(o[-2]), float(o[-1])
    h0, h1, h2 = float(h[-3]), float(h[-2]), float(h[-1])
    l0, l1, l2 = float(l[-3]), float(l[-2]), float(l[-1])

    # Candle measurements
    body2       = abs(c2 - o2)
    body1       = abs(c1 - o1)
    body0       = abs(c0 - o0)
    range2      = h2 - l2 if h2 != l2 else 0.0001
    range1      = h1 - l1 if h1 != l1 else 0.0001
    upper_wick2 = h2 - max(c2, o2)
    lower_wick2 = min(c2, o2) - l2
    upper_wick1 = h1 - max(c1, o1)
    lower_wick1 = min(c1, o1) - l1
    is_bull2    = c2 > o2
    is_bear2    = c2 < o2
    is_bull1    = c1 > o1
    is_bear1    = c1 < o1

    pattern   = 'NONE'
    direction = 'NEUTRAL'
    strength  = 0  # 1-100
    description = ''

    # ── Hammer (Bullish) ──────────────────────────────────────────────
    # Long lower wick (>2x body), small body at top, tiny upper wick
    if (lower_wick2 >= body2 * 2.0 and
            upper_wick2 <= body2 * 0.5 and
            body2 > 0 and
            lower_wick2 / range2 > 0.55):
        pattern     = 'HAMMER'
        direction   = 'BULLISH'
        strength    = 70
        description = 'Hammer — buyers rejected lower prices strongly'

    # ── Shooting Star (Bearish) ───────────────────────────────────────
    # Long upper wick (>2x body), small body at bottom, tiny lower wick
    elif (upper_wick2 >= body2 * 2.0 and
            lower_wick2 <= body2 * 0.5 and
            body2 > 0 and
            upper_wick2 / range2 > 0.55):
        pattern     = 'SHOOTING_STAR'
        direction   = 'BEARISH'
        strength    = 70
        description = 'Shooting Star — sellers rejected higher prices strongly'

    # ── Bullish Engulfing ─────────────────────────────────────────────
    # Current candle is bullish AND fully engulfs previous bearish candle
    elif (is_bull2 and is_bear1 and
            c2 > o1 and o2 < c1 and
            body2 > body1 * 1.1):
        pattern     = 'BULLISH_ENGULFING'
        direction   = 'BULLISH'
        strength    = 85
        description = 'Bullish Engulfing — strong buyer takeover'

    # ── Bearish Engulfing ─────────────────────────────────────────────
    elif (is_bear2 and is_bull1 and
            o2 > c1 and c2 < o1 and
            body2 > body1 * 1.1):
        pattern     = 'BEARISH_ENGULFING'
        direction   = 'BEARISH'
        strength    = 85
        description = 'Bearish Engulfing — strong seller takeover'

    # ── Pin Bar Bullish ───────────────────────────────────────────────
    # Very long lower wick (>60% of range), closes in upper 30%
    elif (lower_wick2 / range2 > 0.60 and
            (c2 - l2) / range2 > 0.70 and
            body2 / range2 < 0.35):
        pattern     = 'PIN_BAR_BULL'
        direction   = 'BULLISH'
        strength    = 80
        description = 'Bullish Pin Bar — strong rejection of lows'

    # ── Pin Bar Bearish ───────────────────────────────────────────────
    elif (upper_wick2 / range2 > 0.60 and
            (h2 - c2) / range2 > 0.70 and
            body2 / range2 < 0.35):
        pattern     = 'PIN_BAR_BEAR'
        direction   = 'BEARISH'
        strength    = 80
        description = 'Bearish Pin Bar — strong rejection of highs'

    # ── Morning Star (3-candle Bullish) ──────────────────────────────
    # Large bearish → small doji/body → large bullish
    elif (is_bear0 := c0 < o0) and body0 > range1 * 0.5 and body1 < body0 * 0.4 and is_bull2 and body2 > body0 * 0.5 and c2 > (o0 + c0) / 2:
        pattern     = 'MORNING_STAR'
        direction   = 'BULLISH'
        strength    = 90
        description = 'Morning Star — 3-candle bullish reversal'

    # ── Evening Star (3-candle Bearish) ──────────────────────────────
    elif (is_bull0 := c0 > o0) and body0 > range1 * 0.5 and body1 < body0 * 0.4 and is_bear2 and body2 > body0 * 0.5 and c2 < (o0 + c0) / 2:
        pattern     = 'EVENING_STAR'
        direction   = 'BEARISH'
        strength    = 90
        description = 'Evening Star — 3-candle bearish reversal'

    # ── Inside Bar ───────────────────────────────────────────────────
    # Current candle completely inside previous candle range
    elif h2 < h1 and l2 > l1:
        pattern     = 'INSIDE_BAR'
        direction   = 'NEUTRAL'  # direction determined by breakout
        strength    = 50
        description = 'Inside Bar — consolidation, watch for breakout'

    # ── Doji (Indecision) ─────────────────────────────────────────────
    elif body2 / range2 < 0.1:
        pattern     = 'DOJI'
        direction   = 'NEUTRAL'
        strength    = 20
        description = 'Doji — market indecision, avoid entry'

    # ── Inverted Hammer (Bullish after downtrend) ─────────────────────
    elif (upper_wick2 >= body2 * 2.0 and
            lower_wick2 <= body2 * 0.3 and
            is_bull2):
        pattern     = 'INVERTED_HAMMER'
        direction   = 'BULLISH'
        strength    = 60
        description = 'Inverted Hammer — potential bullish reversal'

    # ── Hanging Man (Bearish after uptrend) ───────────────────────────
    elif (lower_wick2 >= body2 * 2.0 and
            upper_wick2 <= body2 * 0.3 and
            is_bear2):
        pattern     = 'HANGING_MAN'
        direction   = 'BEARISH'
        strength    = 60
        description = 'Hanging Man — potential bearish reversal'

    # ── Strong Momentum Candle ────────────────────────────────────────
    # Large body (>70% of range), small wicks — trend continuation
    elif body2 / range2 > 0.70:
        if is_bull2:
            pattern     = 'STRONG_BULL_CANDLE'
            direction   = 'BULLISH'
            strength    = 65
            description = 'Strong bullish momentum candle'
        else:
            pattern     = 'STRONG_BEAR_CANDLE'
            direction   = 'BEARISH'
            strength    = 65
            description = 'Strong bearish momentum candle'

    # ── Confidence impact ─────────────────────────────────────────────
    # How much this pattern adds/subtracts from signal confidence
    candle_boost = 0
    candle_penalty = 0

    if direction == 'BULLISH':
        candle_boost = round(strength * 0.25)   # max +22 boost
    elif direction == 'BEARISH':
        candle_boost = round(strength * 0.25)
    elif pattern == 'DOJI':
        candle_penalty = 15  # doji reduces confidence
    elif pattern == 'INSIDE_BAR':
        candle_penalty = 5

    return {
        'pattern':        pattern,
        'direction':      direction,
        'strength':       strength,
        'description':    description,
        'candle_boost':   candle_boost,
        'candle_penalty': candle_penalty,
        'confirms_buy':   direction == 'BULLISH',
        'confirms_sell':  direction == 'BEARISH',
        'is_doji':        pattern == 'DOJI',
        'is_inside_bar':  pattern == 'INSIDE_BAR',
    }


def check_gold_news() -> dict:
    """
    Check ForexFactory for high-impact gold news events.
    Blocks trading 30 min before and 30 min after high-impact events:
    - Fed interest rate decision
    - CPI / inflation data
    - NFP (Non-Farm Payroll)
    - FOMC minutes
    - USD events (gold is priced in USD — USD news moves gold)
    Falls back to safe=True if API unavailable.
    """
    try:
        now_utc = datetime.now(timezone.utc)

        # High impact keywords that move gold
        HIGH_IMPACT_KEYWORDS = [
            "fed", "federal reserve", "fomc", "interest rate",
            "cpi", "inflation", "nfp", "non-farm", "payroll",
            "gdp", "unemployment", "pce", "powell",
            "gold", "xau", "geopolit",
        ]

        # Try ForexFactory calendar API (free, no key needed)
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        events = resp.json()

        risky_events = []
        for ev in events:
            impact  = str(ev.get("impact", "")).lower()
            title   = str(ev.get("title", "")).lower()
            country = str(ev.get("country", "")).upper()

            # Only USD high-impact events (gold priced in USD)
            if country != "USD" or impact != "high":
                continue

            # Check if any keyword matches
            if not any(kw in title for kw in HIGH_IMPACT_KEYWORDS):
                # Still block all high-impact USD events as they move gold
                pass

            # Parse event time
            try:
                ev_time_str = ev.get("date", "") + " " + ev.get("time", "")
                ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %I:%M%p")
                ev_time = ev_time.replace(tzinfo=timezone.utc)
            except:
                continue

            # Block window: 30 min before to 30 min after
            window_start = ev_time - timedelta(minutes=30)
            window_end   = ev_time + timedelta(minutes=30)

            if window_start <= now_utc <= window_end:
                risky_events.append({
                    "title":   ev.get("title", ""),
                    "time":    ev.get("time", ""),
                    "country": country,
                    "impact":  impact,
                    "minutes_to_event": int((ev_time - now_utc).total_seconds() / 60),
                })

        if risky_events:
            return {
                "safe":        False,
                "risk_level":  "HIGH",
                "reason":      f"High-impact event: {risky_events[0]['title']}",
                "events":      risky_events,
                "checked_at":  now_utc.isoformat(),
            }

        return {
            "safe":       True,
            "risk_level": "CLEAR",
            "reason":     "",
            "events":     [],
            "checked_at": now_utc.isoformat(),
        }

    except Exception as e:
        # If news API fails — don't block trading, just log
        return {
            "safe":       True,
            "risk_level": "UNKNOWN",
            "reason":     f"News API unavailable: {str(e)[:50]}",
            "events":     [],
        }


def is_trading_session() -> dict:
    """Check if current time is within gold trading sessions.
    Gold forex trades 22:00 Sun – 21:00 Fri UTC.
    Only closed: 21:00–22:00 UTC daily gap, and full weekend.
    """
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun

    # Saturday — fully closed
    if weekday == 5:
        return {"in_session": False, "session": "WEEKEND", "hour_utc": hour}
    # Sunday before 22:00 UTC — closed
    if weekday == 6 and hour < 22:
        return {"in_session": False, "session": "WEEKEND", "hour_utc": hour}
    # Friday after 21:00 UTC — market closing
    if weekday == 4 and hour >= 21:
        return {"in_session": False, "session": "MARKET_CLOSE", "hour_utc": hour}
    # Daily close gap 21:00–22:00 UTC Mon–Thu
    if hour == 21:
        return {"in_session": False, "session": "DAILY_CLOSE", "hour_utc": hour}

    # Find named session — SESSIONS is list of (start, end, name)
    for start, end, name in SESSIONS:
        if start <= hour < end:
            return {"in_session": True, "session": name, "hour_utc": hour}

    # Fallback — should rarely hit
    return {"in_session": True, "session": "OPEN", "hour_utc": hour}


def generate_gold_signal(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                          df_daily: pd.DataFrame, df_15m: pd.DataFrame) -> dict:
    """
    Core gold signal generator.
    Uses 4-timeframe analysis: Daily bias → 4H → 1H → 15m entry.
    """
    df_1h    = add_indicators(df_1h.copy())
    df_4h    = add_indicators(df_4h.copy())
    df_15m   = add_indicators(df_15m.copy())

    last_1h  = df_1h.iloc[-1]
    last_4h  = df_4h.iloc[-1]
    last_15m = df_15m.iloc[-1]

    price    = float(last_1h["close"])
    atr      = float(last_1h["atr"]) if pd.notna(last_1h["atr"]) else price * 0.005

    # ── Daily Bias ────────────────────────────────────────────────────
    daily_bias_info = detect_daily_bias(df_daily)
    daily_bias      = daily_bias_info["bias"]

    # ── 1H Signal ────────────────────────────────────────────────────
    rsi_1h    = float(last_1h["rsi"])      if pd.notna(last_1h["rsi"])      else 50
    macd_1h   = float(last_1h["macd_hist"])if pd.notna(last_1h["macd_hist"])else 0
    adx_1h    = float(last_1h["adx"])      if pd.notna(last_1h["adx"])      else 0
    ema21_1h  = float(last_1h["ema21"])    if pd.notna(last_1h["ema21"])    else price
    ema50_1h  = float(last_1h["ema50"])    if pd.notna(last_1h["ema50"])    else price
    vol_r_1h  = float(last_1h["vol_ratio"])if pd.notna(last_1h["vol_ratio"])else 1.0

    # ── 4H Signal ────────────────────────────────────────────────────
    rsi_4h    = float(last_4h["rsi"])      if pd.notna(last_4h["rsi"])      else 50
    macd_4h   = float(last_4h["macd_hist"])if pd.notna(last_4h["macd_hist"])else 0
    ema21_4h  = float(last_4h["ema21"])    if pd.notna(last_4h["ema21"])    else price
    ema50_4h  = float(last_4h["ema50"])    if pd.notna(last_4h["ema50"])    else price

    # ── 15m Signal ───────────────────────────────────────────────────
    rsi_15m   = float(last_15m["rsi"])       if pd.notna(last_15m["rsi"])       else 50
    macd_15m  = float(last_15m["macd_hist"]) if pd.notna(last_15m["macd_hist"]) else 0
    ema9_15m  = float(last_15m["ema9"])      if pd.notna(last_15m["ema9"])      else price

    # ── Scoring ───────────────────────────────────────────────────────
    buy_score  = 0
    sell_score = 0

    # Daily bias (strong weight)
    if daily_bias == "BULLISH":  buy_score  += 3
    if daily_bias == "BEARISH":  sell_score += 3
    if daily_bias == "NEUTRAL":  pass

    # 4H trend
    if price > ema21_4h > ema50_4h:  buy_score  += 2
    if price < ema21_4h < ema50_4h:  sell_score += 2
    if macd_4h > 0:                  buy_score  += 1
    if macd_4h < 0:                  sell_score += 1
    if rsi_4h > 50:                  buy_score  += 1
    if rsi_4h < 50:                  sell_score += 1

    # 1H trend
    if price > ema21_1h:  buy_score  += 1
    if price < ema21_1h:  sell_score += 1
    if macd_1h > 0:       buy_score  += 1
    if macd_1h < 0:       sell_score += 1
    if adx_1h > 25:       buy_score  += 1; sell_score += 1  # trend strength helps both

    # RSI zones
    if MIN_RSI_BUY  <= rsi_1h <= MAX_RSI_BUY:   buy_score  += 1
    if MIN_RSI_SELL <= rsi_1h <= MAX_RSI_SELL:   sell_score += 1

    # 15m entry trigger
    if price > ema9_15m and macd_15m > 0:  buy_score  += 1
    if price < ema9_15m and macd_15m < 0:  sell_score += 1

    total_score = buy_score + sell_score
    if total_score == 0:
        raw_signal = "HOLD"
        confidence = 0
    elif buy_score > sell_score:
        raw_signal = "BUY"
        confidence = round((buy_score / max(total_score, 1)) * 100)
    else:
        raw_signal = "SELL"
        confidence = round((sell_score / max(total_score, 1)) * 100)

    # Normalise score to /16 for consistency with crypto bot display
    score_16 = min(round((max(buy_score, sell_score) / 12) * 16), 16)

    # ── S/R Levels ───────────────────────────────────────────────────
    sr = find_support_resistance(df_1h)

    # ── Fibonacci Retracement ─────────────────────────────────────────
    fib = calculate_fibonacci(df_1h)

    # ── SMC / Market Structure Analysis ──────────────────────────────
    smc = detect_market_structure(df_1h)

    # ── Candlestick Pattern Detection (1H confirmation) ───────────────
    candle_1h  = detect_candlestick_patterns(df_1h)
    candle_15m = detect_candlestick_patterns(df_15m)

    # ── Apply SMC + Candlestick confidence boost ──────────────────────
    if raw_signal == "BUY":
        confidence = min(confidence + smc["smc_buy_boost"], 95)
        if candle_1h["confirms_buy"]:
            confidence = min(confidence + candle_1h["candle_boost"], 95)
        elif candle_1h["is_doji"]:
            confidence = max(confidence - candle_1h["candle_penalty"], 0)
    elif raw_signal == "SELL":
        confidence = min(confidence + smc["smc_sell_boost"], 95)
        if candle_1h["confirms_sell"]:
            confidence = min(confidence + candle_1h["candle_boost"], 95)
        elif candle_1h["is_doji"]:
            confidence = max(confidence - candle_1h["candle_penalty"], 0)

    # SMC structure gate — block signals against structure
    # Only block in strong trends, allow RANGING
    if smc["structure"] == "DOWNTREND" and raw_signal == "BUY":
        confidence = max(confidence - 20, 0)  # penalise, don't hard block
    elif smc["structure"] == "UPTREND" and raw_signal == "SELL":
        confidence = max(confidence - 20, 0)

    # ── Timeframe agreement ──────────────────────────────────────────
    tf_1h_signal  = "BUY" if price > ema21_1h  and macd_1h  > 0 else "SELL" if price < ema21_1h  and macd_1h  < 0 else "HOLD"
    tf_4h_signal  = "BUY" if price > ema21_4h  and macd_4h  > 0 else "SELL" if price < ema21_4h  and macd_4h  < 0 else "HOLD"
    tf_15m_signal = "BUY" if price > ema9_15m  and macd_15m > 0 else "SELL" if price < ema9_15m  and macd_15m < 0 else "HOLD"
    tf_daily      = "BUY" if daily_bias == "BULLISH" else "SELL" if daily_bias == "BEARISH" else "HOLD"

    # ── Regime ───────────────────────────────────────────────────────
    regime_info   = detect_market_regime(df_1h)

    return {
        "symbol":            "XAUUSD",
        "price":             round(price, 2),
        "signal":            raw_signal,
        "pre_block_signal":  raw_signal,
        "confidence":        f"{confidence}%",
        "score":             f"{score_16}/16",
        "adx":               round(adx_1h, 1),
        "rsi":               round(rsi_1h, 1),
        "macd_hist":         round(macd_1h, 4),
        "vol_ratio":         round(vol_r_1h, 2),
        "atr":               round(atr, 2),
        "daily_bias":        daily_bias,
        "timeframes": {
            "15m":   tf_15m_signal,
            "1h":    tf_1h_signal,
            "4h":    tf_4h_signal,
            "daily": tf_daily,
        },
        "nearest_support":    sr["nearest_support"],
        "nearest_resistance": sr["nearest_resistance"],
        "dist_to_resistance_pct": sr["dist_to_resistance_pct"],
        "dist_to_support_pct":    sr["dist_to_support_pct"],
        "market_regime":     regime_info["regime"],
        "regime_safe":       regime_info["safe"],
        "ema21_1h":          round(ema21_1h, 2),
        "ema50_1h":          round(ema50_1h, 2),
        "ema21_4h":          round(ema21_4h, 2),
        "bb_upper":          round(float(last_1h["bb_upper"]) if pd.notna(last_1h["bb_upper"]) else price, 2),
        "bb_lower":          round(float(last_1h["bb_lower"]) if pd.notna(last_1h["bb_lower"]) else price, 2),
        "stoch_k":           round(float(last_1h["stoch_k"])  if pd.notna(last_1h["stoch_k"])  else 50, 1),
        # Fibonacci
        "fib_236":           fib.get("fib_236"),
        "fib_382":           fib.get("fib_382"),
        "fib_500":           fib.get("fib_500"),
        "fib_618":           fib.get("fib_618"),
        "fib_786":           fib.get("fib_786"),
        "swing_high":        fib.get("swing_high"),
        "swing_low":         fib.get("swing_low"),
        "nearest_fib":       fib.get("nearest_fib"),
        "dist_to_fib_pct":   fib.get("dist_to_fib_pct"),
        "at_fib_support":    fib.get("at_fib_support", False),
        "at_fib_resistance": fib.get("at_fib_resistance", False),
        # SMC / Market Structure
        "smc_structure":     smc["structure"],
        "bos_bullish":       smc["bos_bullish"],
        "bos_bearish":       smc["bos_bearish"],
        "bos_level":         smc["bos_level"],
        "in_demand_zone":    smc["in_demand_zone"],
        "in_supply_zone":    smc["in_supply_zone"],
        "retest_buy":        smc["retest_buy"],
        "retest_sell":       smc["retest_sell"],
        "liq_sweep_bull":    smc["liq_sweep_bull"],
        "liq_sweep_bear":    smc["liq_sweep_bear"],
        "demand_zones":      smc["demand_zones"],
        "supply_zones":      smc["supply_zones"],
        "smc_buy_boost":     smc["smc_buy_boost"],
        "smc_sell_boost":    smc["smc_sell_boost"],
        # Candlestick patterns
        "candle_pattern":    candle_1h["pattern"],
        "candle_direction":  candle_1h["direction"],
        "candle_strength":   candle_1h["strength"],
        "candle_desc":       candle_1h["description"],
        "candle_confirms":   candle_1h["confirms_buy"] if raw_signal == "BUY" else candle_1h["confirms_sell"],
        "candle_15m_pattern": candle_15m["pattern"],
        "candle_15m_direction": candle_15m["direction"],
    }


# ══════════════════════════════════════════════════════════════════════
# COMPUTE SWING SIGNAL
# ══════════════════════════════════════════════════════════════════════
def compute_swing_signal() -> dict:
    """Full swing signal with all gates applied."""

    # Fetch data
    df_15m   = get_oanda_candles("M15", 200)
    df_1h    = get_oanda_candles("H1",  200)
    df_4h    = get_oanda_candles("H4",  200)
    df_daily = get_oanda_candles("D",   100)

    price, spread = get_current_price()

    # Core signal
    main = generate_gold_signal(df_1h, df_4h, df_daily, df_15m)
    main["price"]  = price
    main["spread"] = round(spread, 1)

    final_signal   = main["signal"]
    raw_signal     = final_signal

    # ── Spread Gate — gold spread > 5 pips is dangerous ─────────────
    spread_blocked = False
    if spread > 5.0:
        spread_blocked = True
        final_signal   = "HOLD"

    # ── Session Gate ──────────────────────────────────────────────────
    session_info    = is_trading_session()
    session_blocked = False
    if not session_info["in_session"] and not spread_blocked:
        session_blocked = True
        final_signal    = "HOLD"

    # ── Regime Gate ──────────────────────────────────────────────────
    regime_blocked = False
    if not main["regime_safe"] and not spread_blocked and not session_blocked:
        regime_blocked = True
        final_signal   = "HOLD"

    # ── Daily Bias Gate ───────────────────────────────────────────────
    bias_blocked = False
    daily_bias   = main["daily_bias"]
    if not regime_blocked and not spread_blocked and not session_blocked:
        if daily_bias == "BEARISH" and final_signal == "BUY":
            bias_blocked = True
            final_signal = "HOLD"
        elif daily_bias == "BULLISH" and final_signal == "SELL":
            bias_blocked = True
            final_signal = "HOLD"

    # ── S/R Distance Gate (1.0% threshold) ───────────────────────────
    sr_blocked = False
    if not any([spread_blocked, session_blocked, regime_blocked, bias_blocked]) and final_signal != "HOLD":
        nr = main.get("nearest_resistance")
        ns = main.get("nearest_support")
        if final_signal == "BUY" and nr:
            dist = ((nr - price) / price) * 100
            if 0 < dist < 1.0:
                sr_blocked   = True
                final_signal = "HOLD"
        elif final_signal == "SELL" and ns:
            dist = ((price - ns) / price) * 100
            if 0 < dist < 1.0:
                sr_blocked   = True
                final_signal = "HOLD"

    # ── News Gate — block during high-impact USD/gold events ──────────
    news_blocked = False
    news_info    = {"safe": True, "risk_level": "CLEAR", "reason": "", "events": []}
    if not any([spread_blocked, session_blocked, regime_blocked, bias_blocked, sr_blocked]) and final_signal != "HOLD":
        news_info = check_gold_news()
        if not news_info["safe"]:
            news_blocked = True
            final_signal = "HOLD"

    # ── Fibonacci Gate — block BUY at fib resistance, SELL at fib support
    fib_blocked = False
    if not any([spread_blocked, session_blocked, regime_blocked, bias_blocked,
                sr_blocked, news_blocked]) and final_signal != "HOLD":
        at_fib_res = main.get("at_fib_resistance", False)
        at_fib_sup = main.get("at_fib_support", False)
        if final_signal == "BUY" and at_fib_res:
            fib_blocked  = True
            final_signal = "HOLD"
        elif final_signal == "SELL" and at_fib_sup:
            fib_blocked  = True
            final_signal = "HOLD"

    # ── SMC Structure Gate ────────────────────────────────────────────
    # Hard block: BUY in DOWNTREND structure, SELL in UPTREND structure
    # Only applies when structure is clearly defined (not RANGING)
    smc_blocked   = False
    smc_structure = main.get("smc_structure", "RANGING")
    if not any([spread_blocked, session_blocked, regime_blocked, bias_blocked,
                sr_blocked, news_blocked, fib_blocked]) and final_signal != "HOLD":
        if smc_structure == "DOWNTREND" and final_signal == "BUY":
            smc_blocked  = True
            final_signal = "HOLD"
        elif smc_structure == "UPTREND" and final_signal == "SELL":
            smc_blocked  = True
            final_signal = "HOLD"

    # ── Candlestick Confirmation Gate ────────────────────────────────
    # At demand/supply zones or after BOS retest, require candle confirmation.
    # Doji or opposing candle at key level = avoid entry.
    candle_blocked = False
    in_key_zone    = (main.get("in_demand_zone") or main.get("in_supply_zone") or
                      main.get("retest_buy") or main.get("retest_sell"))
    if not any([spread_blocked, session_blocked, regime_blocked, bias_blocked,
                sr_blocked, news_blocked, fib_blocked, smc_blocked]) and final_signal != "HOLD":
        candle_dir = main.get("candle_direction", "NEUTRAL")
        is_doji    = main.get("candle_pattern") == "DOJI"
        if in_key_zone:
            if final_signal == "BUY" and candle_dir == "BEARISH":
                candle_blocked = True
                final_signal   = "HOLD"
            elif final_signal == "SELL" and candle_dir == "BULLISH":
                candle_blocked = True
                final_signal   = "HOLD"
        if is_doji and in_key_zone and not candle_blocked:
            candle_blocked = True
            final_signal   = "HOLD"

    retest_confirmed = main.get("retest_buy") if final_signal == "BUY" else main.get("retest_sell")

    # ── SL/TP (structure-aware + regime-adjusted) ────────────────────
    atr       = main["atr"]
    regime    = main.get("market_regime", "WEAK_TREND")
    ns        = main.get("nearest_support")    or 0
    nr        = main.get("nearest_resistance") or 0
    sl_tp_dir = final_signal if final_signal != "HOLD" else raw_signal

    sltp = calculate_smart_sltp(
        price            = price,
        signal           = sl_tp_dir,
        atr              = atr,
        nearest_support  = ns,
        nearest_resistance = nr,
        regime           = regime,
        mode             = "swing",
    )
    stop_loss   = sltp["stop_loss"]
    take_profit = sltp["take_profit"]
    risk_reward = sltp["risk_reward"]

    # ── Strength label ────────────────────────────────────────────────
    conf = int(str(main["confidence"]).replace("%", ""))
    if news_blocked:
        strength = f"🚫 BLOCKED — {news_info.get('reason', 'High-impact news event')}"
    elif fib_blocked:
        strength = f"🚫 BLOCKED — Price at Fibonacci {'resistance' if main.get('at_fib_resistance') else 'support'} ({main.get('nearest_fib')})"
    elif smc_blocked:
        strength = f"🚫 BLOCKED — SMC structure is {smc_structure} — signal opposes trend"
    elif sr_blocked:
        strength = f"🚫 BLOCKED — S/R too close"
    elif bias_blocked:
        strength = f"🚫 BLOCKED — Daily bias {daily_bias} opposes {raw_signal}"
    elif session_blocked:
        strength = f"🚫 BLOCKED — {session_info['session']} (off hours)"
    elif spread_blocked:
        strength = f"🚫 BLOCKED — Spread {spread:.1f} pips (too wide)"
    elif regime_blocked:
        strength = f"🚫 BLOCKED — {main['market_regime']} regime"
    elif final_signal != "HOLD":
        smc_note    = ""
        if main.get("retest_buy") and final_signal == "BUY":
            smc_note = " | ✅ Retest confirmed"
        elif main.get("retest_sell") and final_signal == "SELL":
            smc_note = " | ✅ Retest confirmed"
        elif main.get("liq_sweep_bull") and final_signal == "BUY":
            smc_note = " | ⚡ Liq sweep bullish"
        elif main.get("liq_sweep_bear") and final_signal == "SELL":
            smc_note = " | ⚡ Liq sweep bearish"
        elif main.get("bos_bullish") and final_signal == "BUY":
            smc_note = " | 📈 BOS bullish"
        elif main.get("bos_bearish") and final_signal == "SELL":
            smc_note = " | 📉 BOS bearish"
        candle_note = f" | 🕯 {main.get('candle_pattern','')}" if main.get("candle_pattern") not in ("NONE", None) else ""
        fib_note  = f" | Near Fib {main.get('nearest_fib')}" if main.get("dist_to_fib_pct", 1) < 0.5 else ""
        strength  = f"{'🟢' if final_signal=='BUY' else '🔴'} {final_signal} — Conf {conf}% | {smc_structure}{fib_note}{smc_note}{candle_note}"
    else:
        strength = "⏸ HOLD — No clear setup"

    return {
        **main,
        "mode":             "SWING",
        "signal":           final_signal,
        "pre_block_signal": raw_signal,
        "agreement":        strength,
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "risk_reward":      risk_reward,
        "session":          session_info["session"],
        "session_active":   session_info["in_session"],
        "spread_blocked":   spread_blocked,
        "session_blocked":  session_blocked,
        "regime_blocked":   regime_blocked,
        "bias_blocked":     bias_blocked,
        "sr_blocked":       sr_blocked,
        "news_blocked":     news_blocked,
        "fib_blocked":      fib_blocked,
        "smc_blocked":      smc_blocked,
        "candle_blocked":   candle_blocked,
        "news_risk_level":  news_info.get("risk_level", "CLEAR"),
        "news_reason":      news_info.get("reason", ""),
        "news_events":      news_info.get("events", []),
    }


# ══════════════════════════════════════════════════════════════════════
# COMPUTE SCALP SIGNAL
# ══════════════════════════════════════════════════════════════════════
def compute_scalp_signal() -> dict:
    """Scalp signal using 1m + 5m + 15m timeframes."""

    df_1m  = get_oanda_candles("M1",  100)
    df_5m  = get_oanda_candles("M5",  150)
    df_15m = get_oanda_candles("M15", 200)

    price, spread = get_current_price()

    df_1m  = add_indicators(df_1m.copy())
    df_5m  = add_indicators(df_5m.copy())
    df_15m = add_indicators(df_15m.copy())

    last_1m  = df_1m.iloc[-1]
    last_5m  = df_5m.iloc[-1]
    last_15m = df_15m.iloc[-1]

    atr   = float(last_5m["atr"])      if pd.notna(last_5m["atr"])      else price * 0.002
    rsi   = float(last_5m["rsi"])      if pd.notna(last_5m["rsi"])      else 50
    macd  = float(last_5m["macd_hist"])if pd.notna(last_5m["macd_hist"])else 0
    adx   = float(last_5m["adx"])      if pd.notna(last_5m["adx"])      else 0
    vol_r = float(last_5m["vol_ratio"])if pd.notna(last_5m["vol_ratio"])else 1.0
    ema9  = float(last_5m["ema9"])     if pd.notna(last_5m["ema9"])     else price

    # 1m trigger
    macd_1m = float(last_1m["macd_hist"]) if pd.notna(last_1m["macd_hist"]) else 0
    ema9_1m = float(last_1m["ema9"])      if pd.notna(last_1m["ema9"])      else price

    # 15m context
    macd_15m = float(last_15m["macd_hist"]) if pd.notna(last_15m["macd_hist"]) else 0
    ema9_15m = float(last_15m["ema9"])      if pd.notna(last_15m["ema9"])      else price

    # Scalp signal logic
    tf_1m_sig  = "BUY" if price > ema9_1m  and macd_1m  > 0 else "SELL" if price < ema9_1m  and macd_1m  < 0 else "HOLD"
    tf_5m_sig  = "BUY" if price > ema9     and macd      > 0 else "SELL" if price < ema9     and macd      < 0 else "HOLD"
    tf_15m_sig = "BUY" if price > ema9_15m and macd_15m > 0 else "SELL" if price < ema9_15m and macd_15m < 0 else "HOLD"

    signals   = [tf_1m_sig, tf_5m_sig, tf_15m_sig]
    buy_count  = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count >= 2:
        raw_signal = "BUY"
        confidence = round(50 + (adx / 2) + (buy_count * 5))
    elif sell_count >= 2:
        raw_signal = "SELL"
        confidence = round(50 + (adx / 2) + (sell_count * 5))
    else:
        raw_signal = "HOLD"
        confidence = 0

    confidence = min(confidence, 95)

    # ── Score (0-16) — additive multi-factor, NOT just timeframe count ──
    # PREVIOUS BUG: score_16 = round((max(buy_count,sell_count)/3)*12) could
    # ONLY ever equal 8 (2-of-3 agreement) or 12 (3-of-3 agreement) — meaning
    # Grade A (needs score>=11) was unreachable without unanimous timeframes,
    # even when confidence/ADX/volume all showed a genuinely strong setup.
    # Fixed: build score from independent factors like swing scoring does,
    # so it can land anywhere 0-16 based on real signal quality.
    agree_count = max(buy_count, sell_count)
    raw_score = 0
    # Timeframe agreement (up to 6 pts: 2-of-3=4, 3-of-3=6)
    if agree_count == 3:   raw_score += 6
    elif agree_count == 2: raw_score += 4
    # Trend strength via ADX (up to 4 pts)
    if adx >= 30:   raw_score += 4
    elif adx >= 20: raw_score += 2
    elif adx >= 15: raw_score += 1
    # Momentum confirmation via 5m MACD magnitude (up to 3 pts)
    if abs(macd) > 0.5: raw_score += 3
    elif abs(macd) > 0.2: raw_score += 2
    elif macd != 0: raw_score += 1
    # Market activity via range-based vol_ratio (up to 3 pts)
    if vol_r >= 1.3:   raw_score += 3
    elif vol_r >= 1.0: raw_score += 2
    elif vol_r >= 0.7: raw_score += 1

    score_16 = min(raw_score, 16)

    final_signal = raw_signal

    # ── Spread gate (stricter for scalp — max 3 pips) ────────────────
    spread_blocked = False
    if spread > 3.0:
        spread_blocked = True
        final_signal   = "HOLD"

    # ── Session gate ─────────────────────────────────────────────────
    session_info    = is_trading_session()
    session_blocked = not session_info["in_session"] and not spread_blocked
    if session_blocked:
        final_signal = "HOLD"

    # ── S/R gate (0.3% for scalp) ────────────────────────────────────
    sr = find_support_resistance(df_15m)
    sr_blocked = False
    if not spread_blocked and not session_blocked and final_signal != "HOLD":
        if final_signal == "BUY" and sr["nearest_resistance"]:
            if 0 < sr["dist_to_resistance_pct"] < 0.3:
                sr_blocked   = True
                final_signal = "HOLD"
        elif final_signal == "SELL" and sr["nearest_support"]:
            if 0 < sr["dist_to_support_pct"] < 0.3:
                sr_blocked   = True
                final_signal = "HOLD"

    # ── News gate (scalp — block 45 min around event) ─────────────────
    news_blocked = False
    news_info    = {"safe": True, "risk_level": "CLEAR", "reason": ""}
    if not spread_blocked and not session_blocked and not sr_blocked and final_signal != "HOLD":
        news_info = check_gold_news()
        if not news_info["safe"]:
            news_blocked = True
            final_signal = "HOLD"

    # ── Fibonacci gate for scalp (tighter — 0.2%) ────────────────────
    fib_blocked = False
    fib_scalp   = calculate_fibonacci(df_15m)
    if not any([spread_blocked, session_blocked, sr_blocked, news_blocked]) and final_signal != "HOLD":
        if final_signal == "BUY" and fib_scalp.get("at_fib_resistance"):
            fib_blocked  = True
            final_signal = "HOLD"
        elif final_signal == "SELL" and fib_scalp.get("at_fib_support"):
            fib_blocked  = True
            final_signal = "HOLD"

    # ── SMC gate for scalp (15m structure) ───────────────────────────
    smc_scalp   = detect_market_structure(df_15m)
    smc_blocked = False
    smc_structure = smc_scalp["structure"]
    if not any([spread_blocked, session_blocked, sr_blocked, news_blocked, fib_blocked]) and final_signal != "HOLD":
        if smc_structure == "DOWNTREND" and final_signal == "BUY":
            smc_blocked  = True
            final_signal = "HOLD"
        elif smc_structure == "UPTREND" and final_signal == "SELL":
            smc_blocked  = True
            final_signal = "HOLD"

    # ── Candlestick gate for scalp ────────────────────────────────────
    candle_scalp   = detect_candlestick_patterns(df_5m)
    candle_blocked = False
    in_key_zone    = smc_scalp.get("in_demand_zone") or smc_scalp.get("in_supply_zone")
    if not any([spread_blocked, session_blocked, sr_blocked, news_blocked, fib_blocked, smc_blocked]) and final_signal != "HOLD":
        if in_key_zone:
            if final_signal == "BUY" and candle_scalp["direction"] == "BEARISH":
                candle_blocked = True
                final_signal   = "HOLD"
            elif final_signal == "SELL" and candle_scalp["direction"] == "BULLISH":
                candle_blocked = True
                final_signal   = "HOLD"
        if candle_scalp["is_doji"] and in_key_zone and not candle_blocked:
            candle_blocked = True
            final_signal   = "HOLD"

    # ── SL/TP (structure-aware + regime-adjusted, tight for scalp) ──
    sr_scalp  = find_support_resistance(df_15m)
    regime_sc = detect_market_regime(df_5m)["regime"]
    sl_tp_dir = final_signal if final_signal != "HOLD" else raw_signal

    sltp = calculate_smart_sltp(
        price              = price,
        signal             = sl_tp_dir,
        atr                = atr,
        nearest_support    = sr_scalp["nearest_support"],
        nearest_resistance = sr_scalp["nearest_resistance"],
        regime             = regime_sc,
        mode               = "scalp",
    )
    stop_loss   = sltp["stop_loss"]
    take_profit = sltp["take_profit"]
    risk_reward = sltp.get("risk_reward", "1:2.0")

    return {
        "mode":             "SCALP",
        "symbol":           "XAUUSD",
        "price":            round(price, 2),
        "signal":           final_signal,
        "pre_block_signal": raw_signal,
        "confidence":       f"{confidence}%",
        "score":            f"{score_16}/16",
        "adx":              round(adx, 1),
        "rsi":              round(rsi, 1),
        "macd_hist":        round(macd, 4),
        "vol_ratio":        round(vol_r, 2),
        "atr":              round(atr, 2),
        "spread":           round(spread, 1),
        "timeframes": {
            "1m":  tf_1m_sig,
            "5m":  tf_5m_sig,
            "15m": tf_15m_sig,
        },
        "nearest_support":    sr["nearest_support"],
        "nearest_resistance": sr["nearest_resistance"],
        "fib_382":            fib_scalp.get("fib_382"),
        "fib_500":            fib_scalp.get("fib_500"),
        "fib_618":            fib_scalp.get("fib_618"),
        "nearest_fib":        fib_scalp.get("nearest_fib"),
        "at_fib_support":     fib_scalp.get("at_fib_support", False),
        "at_fib_resistance":  fib_scalp.get("at_fib_resistance", False),
        "stop_loss":          stop_loss,
        "take_profit":        take_profit,
        "risk_reward":        risk_reward,
        "sl_method":          sltp.get("sl_method", "ATR"),
        "sl_distance":        sltp.get("sl_distance", 0),
        "spread_blocked":     spread_blocked,
        "session_blocked":    session_blocked,
        "sr_blocked":         sr_blocked,
        "news_blocked":       news_blocked,
        "fib_blocked":        fib_blocked,
        "smc_blocked":        smc_blocked,
        "candle_blocked":     candle_blocked,
        "smc_structure":      smc_structure,
        "candle_pattern":     candle_scalp["pattern"],
        "candle_direction":   candle_scalp["direction"],
        "candle_strength":    candle_scalp["strength"],
        "candle_desc":        candle_scalp["description"],
        "bos_bullish":        smc_scalp["bos_bullish"],
        "bos_bearish":        smc_scalp["bos_bearish"],
        "in_demand_zone":     smc_scalp["in_demand_zone"],
        "in_supply_zone":     smc_scalp["in_supply_zone"],
        "retest_buy":         smc_scalp["retest_buy"],
        "retest_sell":        smc_scalp["retest_sell"],
        "liq_sweep_bull":     smc_scalp["liq_sweep_bull"],
        "liq_sweep_bear":     smc_scalp["liq_sweep_bear"],
        "news_risk_level":    news_info.get("risk_level", "CLEAR"),
        "session":            session_info["session"],
    }


# ══════════════════════════════════════════════════════════════════════
# GRADING
# ══════════════════════════════════════════════════════════════════════
def grade_signal(sig: dict) -> str:
    if sig.get("signal", "HOLD") == "HOLD":
        return "NONE"
    signal    = sig["signal"]
    conf      = int(str(sig.get("confidence", "0%")).replace("%", ""))
    score_str = str(sig.get("score", "0/16"))
    score     = int(score_str.split("/")[0]) if "/" in score_str else 0
    adx       = sig.get("adx") or 0
    vol       = sig.get("vol_ratio") or 0
    tf        = sig.get("timeframes", {})
    tf_vals   = list(tf.values())
    tf_agree  = tf_vals.count(signal)

    g = GRADE_A
    if (conf >= g["min_confidence"] and score >= g["min_score"]
            and adx >= g["min_adx"] and vol >= g["min_volume"]
            and tf_agree >= g["min_tf"]):
        return "A"
    g = GRADE_B
    if (conf >= g["min_confidence"] and score >= g["min_score"]
            and adx >= g["min_adx"] and vol >= g["min_volume"]
            and tf_agree >= g["min_tf"]):
        return "B"
    return "C"


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════
def log_swing_signal(sig: dict, passed: bool, reasons: list):
    global signal_log, blocked_log
    signal = sig.get("signal") or sig.get("pre_block_signal", "HOLD")
    if signal == "HOLD":
        return
    now = datetime.now(timezone.utc)

    entry = {
        "id":           f"GOLD_{now.strftime('%Y%m%d_%H%M%S')}",
        "mode":         "SWING",
        "logged_at":    now.isoformat(),
        "date":         now.strftime("%Y-%m-%d"),
        "time_utc":     now.strftime("%H:%M"),
        "symbol":       "XAUUSD",
        "signal":       signal,
        "grade":        grade_signal({**sig, "signal": signal}),
        "trade_allowed": "YES" if passed else "NO",
        "blocked_by":   reasons[0] if not passed and reasons else "none",
        "confidence":   sig.get("confidence", "0%"),
        "score":        sig.get("score", "0/16"),
        "adx":          sig.get("adx", 0),
        "rsi":          sig.get("rsi", 0),
        "vol_ratio":    sig.get("vol_ratio", 0),
        "session":      sig.get("session", "UNKNOWN"),
        "spread":       sig.get("spread", 0),
        "daily_bias":   sig.get("daily_bias", "NEUTRAL"),
        "market_regime":sig.get("market_regime", "UNKNOWN"),
        "entry_price":  round(sig.get("price", 0), 2),
        "stop_loss":    round(sig.get("stop_loss") or 0, 2),
        "take_profit":  round(sig.get("take_profit") or 0, 2),
        "risk_reward":  sig.get("risk_reward", "N/A"),
        "nearest_support":    sig.get("nearest_support", 0),
        "nearest_resistance": sig.get("nearest_resistance", 0),
        "status":       "OPEN",
        "outcome":      "-",
        "exit_price":   None,
        "pnl_pct":      None,
    }

    # Deduplicate — 30 min window for swing
    thirty_ago = (now - timedelta(minutes=30)).isoformat()
    for s in signal_log[-50:]:
        if s["signal"] == signal and s["logged_at"] > thirty_ago:
            return

    signal_log.append(entry)
    if len(signal_log) > 500:
        signal_log[:] = signal_log[-500:]
    _save_json(SIGNAL_LOG_FILE, signal_log)
    _db_insert_trade(entry)

    if not passed:
        blocked_log.append(entry)
        if len(blocked_log) > 500:
            blocked_log[:] = blocked_log[-500:]
        _save_json(BLOCKED_LOG_FILE, blocked_log)


def log_scalp_signal(sig: dict, passed: bool, reasons: list):
    global scalp_log, scalp_blocked
    signal = sig.get("signal") or sig.get("pre_block_signal", "HOLD")
    if signal == "HOLD":
        return
    now = datetime.now(timezone.utc)

    entry = {
        "id":           f"GOLD_SCALP_{now.strftime('%Y%m%d_%H%M%S')}",
        "mode":         "SCALP",
        "logged_at":    now.isoformat(),
        "date":         now.strftime("%Y-%m-%d"),
        "time_utc":     now.strftime("%H:%M"),
        "symbol":       "XAUUSD",
        "signal":       signal,
        "grade":        grade_signal({**sig, "signal": signal}),
        "trade_allowed": "YES" if passed else "NO",
        "blocked_by":   reasons[0] if not passed and reasons else "none",
        "confidence":   sig.get("confidence", "0%"),
        "score":        sig.get("score", "0/16"),
        "adx":          sig.get("adx", 0),
        "rsi":          sig.get("rsi", 0),
        "vol_ratio":    sig.get("vol_ratio", 0),
        "spread":       sig.get("spread", 0),
        "session":      sig.get("session", "UNKNOWN"),
        "entry_price":  round(sig.get("price", 0), 2),
        "stop_loss":    round(sig.get("stop_loss") or 0, 2),
        "take_profit":  round(sig.get("take_profit") or 0, 2),
        "nearest_support":    sig.get("nearest_support", 0),
        "nearest_resistance": sig.get("nearest_resistance", 0),
        "status":       "OPEN",
        "outcome":      "-",
        "exit_price":   None,
        "pnl_pct":      None,
    }

    # Deduplicate — 15 min window for scalp
    fifteen_ago = (now - timedelta(minutes=15)).isoformat()
    for s in scalp_log[-50:]:
        if s["signal"] == signal and s["logged_at"] > fifteen_ago:
            return

    scalp_log.append(entry)
    if len(scalp_log) > 500:
        scalp_log[:] = scalp_log[-500:]
    _save_json(SCALP_LOG_FILE, scalp_log)
    _db_insert_trade(entry)

    if not passed:
        scalp_blocked.append(entry)
        if len(scalp_blocked) > 500:
            scalp_blocked[:] = scalp_blocked[-500:]
        _save_json(SCALP_BLOCKED_FILE, scalp_blocked)



def log_news_signal(sig: dict, news_info: dict):
    """
    Separately log signals blocked specifically by news gate.
    Stored in gold_news_blocked_log.json — independent from main blocked_log.
    Purpose: later analysis of whether news filter actually helped or blocked winners.
    The would_have_won field gets filled after the news event passes.
    """
    global news_log
    signal = sig.get("pre_block_signal") or sig.get("signal", "HOLD")
    if signal == "HOLD":
        return
    now = datetime.now(timezone.utc)

    entry = {
        "id":              f"GOLD_NEWS_{now.strftime('%Y%m%d_%H%M%S')}",
        "mode":            sig.get("mode", "SWING"),
        "logged_at":       now.isoformat(),
        "date":            now.strftime("%Y-%m-%d"),
        "time_utc":        now.strftime("%H:%M"),
        "symbol":          "XAUUSD",
        "signal":          signal,
        "blocked_by":      "NEWS",
        "news_reason":     news_info.get("reason", "High-impact event"),
        "news_risk_level": news_info.get("risk_level", "HIGH"),
        "news_events":     news_info.get("events", []),
        "confidence":      sig.get("confidence", "0%"),
        "score":           sig.get("score", "0/16"),
        "adx":             sig.get("adx", 0),
        "rsi":             sig.get("rsi", 0),
        "vol_ratio":       sig.get("vol_ratio", 0),
        "spread":          sig.get("spread", 0),
        "session":         sig.get("session", "UNKNOWN"),
        "daily_bias":      sig.get("daily_bias", "NEUTRAL"),
        "market_regime":   sig.get("market_regime", "UNKNOWN"),
        "entry_price":     round(sig.get("price", 0), 2),
        "stop_loss":       round(sig.get("stop_loss") or 0, 2),
        "take_profit":     round(sig.get("take_profit") or 0, 2),
        "nearest_fib":     sig.get("nearest_fib"),
        "fib_618":         sig.get("fib_618"),
        "status":          "OPEN",
        "outcome":         "-",
        "exit_price":      None,
        "pnl_pct":         None,
        "would_have_won":  None,
    }

    thirty_ago = (now - timedelta(minutes=30)).isoformat()
    for s in news_log[-50:]:
        if s["signal"] == signal and s["logged_at"] > thirty_ago:
            return

    news_log.append(entry)
    if len(news_log) > 300:
        news_log[:] = news_log[-300:]
    _save_json(NEWS_LOG_FILE, news_log)


def log_news_signal(sig: dict, news_info: dict):
    """Log signals blocked by news gate to separate gold_news_blocked_log.json."""
    global news_log
    signal = sig.get('pre_block_signal') or sig.get('signal', 'HOLD')
    if signal == 'HOLD':
        return
    now = datetime.now(timezone.utc)
    entry = {
        'id':              f"GOLD_NEWS_{now.strftime('%Y%m%d_%H%M%S')}",
        'mode':            sig.get('mode', 'SWING'),
        'logged_at':       now.isoformat(),
        'date':            now.strftime('%Y-%m-%d'),
        'time_utc':        now.strftime('%H:%M'),
        'symbol':          'XAUUSD',
        'signal':          signal,
        'blocked_by':      'NEWS',
        'news_reason':     news_info.get('reason', 'High-impact event'),
        'news_risk_level': news_info.get('risk_level', 'HIGH'),
        'news_events':     news_info.get('events', []),
        'confidence':      sig.get('confidence', '0%'),
        'score':           sig.get('score', '0/16'),
        'adx':             sig.get('adx', 0),
        'rsi':             sig.get('rsi', 0),
        'vol_ratio':       sig.get('vol_ratio', 0),
        'spread':          sig.get('spread', 0),
        'session':         sig.get('session', 'UNKNOWN'),
        'daily_bias':      sig.get('daily_bias', 'NEUTRAL'),
        'market_regime':   sig.get('market_regime', 'UNKNOWN'),
        'entry_price':     round(sig.get('price', 0), 2),
        'stop_loss':       round(sig.get('stop_loss') or 0, 2),
        'take_profit':     round(sig.get('take_profit') or 0, 2),
        'nearest_fib':     sig.get('nearest_fib'),
        'fib_618':         sig.get('fib_618'),
        'status':          'OPEN',
        'outcome':         '-',
        'exit_price':      None,
        'pnl_pct':         None,
        'would_have_won':  None,
    }
    thirty_ago = (now - timedelta(minutes=30)).isoformat()
    for s in news_log[-50:]:
        if s['signal'] == signal and s['logged_at'] > thirty_ago:
            return
    news_log.append(entry)
    if len(news_log) > 300:
        news_log[:] = news_log[-300:]
    _save_json(NEWS_LOG_FILE, news_log)


def passes_swing_filters(sig: dict) -> tuple:
    reasons = []
    conf  = int(str(sig.get("confidence", "0%")).replace("%", ""))
    score = int(str(sig.get("score", "0/16")).split("/")[0])
    vol   = sig.get("vol_ratio") or 0
    adx   = sig.get("adx") or 0
    tf    = sig.get("timeframes", {})
    signal = sig.get("signal", "HOLD")
    tf_agree = list(tf.values()).count(signal)

    if conf  < MIN_CONFIDENCE:   reasons.append(f"Confidence {conf}% < {MIN_CONFIDENCE}%")
    if score < MIN_SCORE:        reasons.append(f"Score {score}/16 < {MIN_SCORE}")
    if vol   < MIN_VOL_RATIO:    reasons.append(f"Volume {vol:.2f}x < {MIN_VOL_RATIO}x")
    if adx   < ADX_MIN:          reasons.append(f"ADX {adx:.1f} < {ADX_MIN}")
    if tf_agree < 2:             reasons.append(f"Only {tf_agree}/4 TF agree")
    if sig.get("spread_blocked"):  reasons.append("Spread too wide")
    if sig.get("session_blocked"): reasons.append("Off-hours session")
    if sig.get("regime_blocked"):  reasons.append("Unfavorable regime")
    if sig.get("bias_blocked"):    reasons.append("Daily bias opposes")
    if sig.get("sr_blocked"):      reasons.append("S/R too close")
    if sig.get("news_blocked"):    reasons.append(f"News: {sig.get('news_reason','high-impact event')}")
    if sig.get("fib_blocked"):     reasons.append("Fibonacci level blocking entry")
    if sig.get("smc_blocked"):     reasons.append(f"SMC structure {sig.get('smc_structure','?')} opposes signal")
    if sig.get("candle_blocked"):  reasons.append(f"Candle {sig.get('candle_pattern','?')} opposes signal at key zone")
    return len(reasons) == 0, reasons


def passes_scalp_filters(sig: dict) -> tuple:
    reasons = []
    conf  = int(str(sig.get("confidence", "0%")).replace("%", ""))
    score = int(str(sig.get("score", "0/16")).split("/")[0])
    vol   = sig.get("vol_ratio") or 0
    adx   = sig.get("adx") or 0
    tf    = sig.get("timeframes", {})
    signal = sig.get("signal", "HOLD")
    tf_agree = list(tf.values()).count(signal)

    if conf  < SCALP_MIN_CONFIDENCE: reasons.append(f"Confidence {conf}% < {SCALP_MIN_CONFIDENCE}%")
    if score < SCALP_MIN_SCORE:      reasons.append(f"Score {score}/16 < {SCALP_MIN_SCORE}")
    if vol   < SCALP_MIN_VOL_RATIO:  reasons.append(f"Volume {vol:.2f}x < {SCALP_MIN_VOL_RATIO}x")
    if adx   < SCALP_ADX_MIN:        reasons.append(f"ADX {adx:.1f} < {SCALP_ADX_MIN}")
    if tf_agree < 2:                 reasons.append(f"Only {tf_agree}/3 scalp TF agree")
    if sig.get("spread_blocked"):    reasons.append("Spread too wide (>3 pips)")
    if sig.get("session_blocked"):   reasons.append("Off-hours session")
    if sig.get("sr_blocked"):        reasons.append("S/R too close (0.3%)")
    if sig.get("news_blocked"):      reasons.append(f"News: {sig.get('news_reason','high-impact event')}")
    if sig.get("fib_blocked"):       reasons.append("Fibonacci level blocking entry")
    return len(reasons) == 0, reasons


# ══════════════════════════════════════════════════════════════════════
# GRADE STATS
# ══════════════════════════════════════════════════════════════════════
def _grade_stats(signals: list, grade: str) -> dict:
    bucket = [s for s in signals if s.get("grade") == grade]
    open_  = [s for s in bucket if s.get("status") == "OPEN"]
    closed = [s for s in bucket if s.get("status") == "CLOSED"]
    wins   = [s for s in closed if s.get("outcome") == "WIN"]
    losses = [s for s in closed if s.get("outcome") == "LOSS"]
    pnls   = [s["pnl_pct"] for s in closed if s.get("pnl_pct") is not None]

    has_closed = len(wins) + len(losses) > 0
    win_rate   = round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1) if has_closed else None
    avg_profit = round(sum(pnls) / len(pnls), 2) if pnls else None

    return {
        "grade":      grade,
        "total":      len(bucket),      # all signals including open
        "open":       len(open_),       # currently open
        "closed":     len(closed),      # resolved trades
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   f"{win_rate}%" if win_rate is not None else "--",
        "avg_profit": f"{avg_profit:+.2f}%" if avg_profit is not None else "--",
    }


# ══════════════════════════════════════════════════════════════════════
# BACKGROUND SCANNER
# ══════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    _init_db()
    asyncio.create_task(background_scanner())
    asyncio.create_task(paper_trade_resolver())


async def background_scanner():
    """Runs every 5 minutes — scans swing + scalp signals and logs them."""
    await asyncio.sleep(90)  # wait for server + rate limit recovery

    while True:
        try:
            # ── Skip entire scan when market is closed — saves API quota ──
            # is_trading_session() costs 0 API calls (pure UTC time check),
            # so checking it BEFORE fetching candles avoids wasting AllTick/
            # Twelve Data requests on a day when no real signal can fire.
            session_check = is_trading_session()
            if not session_check["in_session"]:
                print(f"[Scanner] Market closed ({session_check['session']}) — skipping scan, saving API quota")
                await asyncio.sleep(SCAN_CYCLE_SECONDS)
                continue

            # ── Swing scan ────────────────────────────────────────────
            try:
                sig    = compute_swing_signal()
                signal = sig.get("signal", "HOLD")
                pre    = sig.get("pre_block_signal", "HOLD")

                if signal != "HOLD":
                    passed, reasons = passes_swing_filters(sig)
                    log_swing_signal(sig, passed, reasons)
                elif pre != "HOLD":
                    reasons = []
                    for key, label in [
                        ("spread_blocked",  "Spread too wide"),
                        ("session_blocked", "Off-hours session"),
                        ("regime_blocked",  "Unfavorable regime"),
                        ("bias_blocked",    "Daily bias opposes"),
                        ("sr_blocked",      "S/R too close"),
                        ("news_blocked",    f"News: {sig.get('news_reason','high-impact event')}"),
                        ("fib_blocked",     "Fibonacci level blocking entry"),
                        ("smc_blocked",     f"SMC structure {sig.get('smc_structure','?')} opposes signal"),
                        ("candle_blocked",   f"Candle {sig.get('candle_pattern','?')} opposes at zone"),
                    ]:
                        if sig.get(key):
                            reasons.append(label)
                    if reasons:
                        pseudo = {**sig, "signal": pre}
                        log_swing_signal(pseudo, False, reasons)
                        # Separately log news-blocked signals for analysis
                        if sig.get("news_blocked"):
                            log_news_signal(pseudo, {
                                "reason":     sig.get("news_reason", ""),
                                "risk_level": sig.get("news_risk_level", "HIGH"),
                                "events":     sig.get("news_events", []),
                            })
            except Exception as e:
                print(f"[Gold Swing Scanner] {e}")

            await asyncio.sleep(5)

            # ── Scalp scan ────────────────────────────────────────────
            try:
                scalp  = compute_scalp_signal()
                signal = scalp.get("signal", "HOLD")
                pre    = scalp.get("pre_block_signal", "HOLD")

                if signal != "HOLD":
                    grade  = grade_signal(scalp)
                    passed, reasons = passes_scalp_filters(scalp)
                    log_scalp_signal(scalp, passed, reasons)
                elif pre != "HOLD":
                    reasons = []
                    for key, label in [
                        ("spread_blocked",  "Spread too wide"),
                        ("session_blocked", "Off-hours session"),
                        ("sr_blocked",      "S/R too close"),
                        ("news_blocked",    f"News: {scalp.get('news_reason','high-impact event')}"),
                        ("fib_blocked",     "Fibonacci level blocking entry"),
                        ("smc_blocked",     f"SMC structure {scalp.get('smc_structure','?')} opposes signal"),
                        ("candle_blocked",   f"Candle {scalp.get('candle_pattern','?')} opposes at zone"),
                    ]:
                        if scalp.get(key):
                            reasons.append(label)
                    if reasons:
                        pseudo = {**scalp, "signal": pre}
                        log_scalp_signal(pseudo, False, reasons)
                        # Separately log news-blocked scalp signals for analysis
                        if scalp.get("news_blocked"):
                            log_news_signal({**pseudo, "mode": "SCALP"}, {
                                "reason":     scalp.get("news_reason", ""),
                                "risk_level": scalp.get("news_risk_level", "HIGH"),
                                "events":     [],
                            })
            except Exception as e:
                print(f"[Gold Scalp Scanner] {e}")

        except Exception as e:
            print(f"[Gold Scanner] {e}")

        await asyncio.sleep(SCAN_CYCLE_SECONDS)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "Gold Bot running ✅", "instrument": "XAUUSD", "version": "1.0.0"}

@app.get("/health")
def health():
    try:
        price, spread = get_current_price()
        return {
            "status":        "ok ✅",
            "symbol":        SYMBOL,
            "xauusd_price":  price,
            "spread_pips":   spread,
            "data_provider": "AllTick" if USE_ALLTICK else "Twelve Data",
            "scan_cycle":    f"{SCAN_CYCLE_SECONDS}s",
        }
    except Exception as e:
        return {
            "status": "error ❌",
            "symbol": SYMBOL,
            "error":  str(e),
            "data_provider": "AllTick" if USE_ALLTICK else "Twelve Data",
        }


@app.get("/data-source")
def data_source_status():
    """Debug endpoint — confirms which data provider is active and tests it."""
    result = {
        "alltick_token_set": bool(ALLTICK_TOKEN),
        "active_provider":   "AllTick" if USE_ALLTICK else "Twelve Data",
        "scan_cycle_seconds": SCAN_CYCLE_SECONDS,
        "candle_cache_ttl":  CANDLE_CACHE_TTL,
        "price_cache_ttl":   PRICE_CACHE_TTL,
    }
    if USE_ALLTICK:
        try:
            price, spread = _fetch_alltick_price()
            result["alltick_test"] = {"status": "✅ working", "price": price}
        except Exception as e:
            result["alltick_test"] = {"status": "❌ failed", "error": str(e)}
        try:
            df = _fetch_alltick_kline("M5", 5)
            result["alltick_kline_test"] = {"status": "✅ working", "last_close": float(df["close"].iloc[-1])}
        except Exception as e:
            result["alltick_kline_test"] = {"status": "❌ failed", "error": str(e)}
    return result


@app.get("/alltick-usage")
def alltick_usage():
    """
    Read-only — shows today's AllTick call counts WITHOUT making any new
    API calls (unlike /data-source which does test fetches). Use this to
    tell apart 'burst rate-limit' (now fixed by the 1 req/sec limiter)
    from 'daily quota genuinely exhausted' (a separate problem — only
    fixable by waiting for reset or upgrading the AllTick plan).
    """
    total = _alltick_usage["success"] + _alltick_usage["rate_limited"] + _alltick_usage["other_error"]
    rate_limited_pct = round(_alltick_usage["rate_limited"] / total * 100, 1) if total else 0
    diagnosis = "no calls made yet today"
    if total > 0:
        if rate_limited_pct > 30:
            diagnosis = "HIGH 429 rate — likely daily quota exhausted, not a burst issue"
        elif _alltick_usage["rate_limited"] > 0:
            diagnosis = "some 429s — could be transient, monitor over next hour"
        else:
            diagnosis = "clean — no 429s today"
    return {
        "date_utc":        _alltick_usage["date"],
        "successful_calls": _alltick_usage["success"],
        "rate_limited_429": _alltick_usage["rate_limited"],
        "other_errors":     _alltick_usage["other_error"],
        "total_attempts":   total,
        "rate_limited_pct": f"{rate_limited_pct}%",
        "diagnosis":        diagnosis,
        "note": "Counts reset when Railway restarts the container — not a persistent log, "
                "just a live snapshot since this deploy started.",
    }

@app.get("/instruments")
def list_instruments():
    """
    Twelve Data API status and symbol info.
    """
    try:
        params = {"symbol": SYMBOL, "apikey": TWELVE_DATA_API_KEY}
        resp   = requests.get(f"{TWELVE_DATA_URL}/quote", params=params, timeout=10)
        data   = resp.json()
        return {
            "symbol":      SYMBOL,
            "api_status":  "ok" if "close" in data else "error",
            "last_price":  data.get("close", "N/A"),
            "provider":    "Twelve Data",
            "tip":         "Set TWELVE_DATA_API_KEY env var with your API key from twelvedata.com"
        }
    except Exception as e:
        return {"error": str(e), "tip": "Check TWELVE_DATA_API_KEY"}

@app.get("/price")
def get_price():
    price, spread = get_current_price()
    session = is_trading_session()
    return {
        "instrument": "XAUUSD",
        "price":      round(price, 2),
        "spread":     round(spread, 1),
        "session":    session["session"],
        "in_session": session["in_session"],
    }

@app.get("/signal")
def get_swing_signal():
    try:
        return compute_swing_signal()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scalp/signal")
def get_scalp_signal():
    try:
        return compute_scalp_signal()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/stats")
def get_signal_stats():
    directional = [s for s in signal_log if s.get("signal") in ("BUY", "SELL")]
    closed      = [s for s in directional if s.get("status") == "CLOSED"]
    wins        = [s for s in closed if s.get("outcome") == "WIN"]
    losses      = [s for s in closed if s.get("outcome") == "LOSS"]
    pnls        = [s["pnl_pct"] for s in closed if s.get("pnl_pct") is not None]
    win_rate    = round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1)
    total_pnl   = round(sum(pnls), 2) if pnls else 0
    open_count  = sum(1 for s in directional if s.get("status") == "OPEN")

    return {
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  f"{win_rate}%",
        "pnl":       f"{total_pnl:+.2f}%",
        "open":      open_count,
        "total":     len(directional),
    }

@app.get("/scalp/stats")
def get_scalp_stats():
    directional = [s for s in scalp_log if s.get("signal") in ("BUY", "SELL")]
    closed      = [s for s in directional if s.get("status") == "CLOSED"]
    wins        = [s for s in closed if s.get("outcome") == "WIN"]
    losses      = [s for s in closed if s.get("outcome") == "LOSS"]
    pnls        = [s["pnl_pct"] for s in closed if s.get("pnl_pct") is not None]
    win_rate    = round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1)
    total_pnl   = round(sum(pnls), 2) if pnls else 0
    open_count  = sum(1 for s in directional if s.get("status") == "OPEN")

    return {
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  f"{win_rate}%",
        "pnl":       f"{total_pnl:+.2f}%",
        "open":      open_count,
        "total":     len(directional),
    }

@app.get("/journal/grade-stats")
def swing_grade_stats():
    directional = [s for s in signal_log if s.get("signal") in ("BUY", "SELL")]
    return {
        "total_signals": len(directional),
        "grades": {
            "A": _grade_stats(directional, "A"),
            "B": _grade_stats(directional, "B"),
            "C": _grade_stats(directional, "C"),
        }
    }

@app.get("/scalp/grade-stats")
def scalp_grade_stats():
    directional = [s for s in scalp_log if s.get("signal") in ("BUY", "SELL")]
    return {
        "total_signals": len(directional),
        "grades": {
            "A": _grade_stats(directional, "A"),
            "B": _grade_stats(directional, "B"),
            "C": _grade_stats(directional, "C"),
        }
    }

@app.get("/session")
def session_status():
    return is_trading_session()

@app.get("/news")
def news_status():
    """Check current gold news risk level."""
    return check_gold_news()

@app.get("/news/log")
def get_news_log(limit: int = 50):
    """
    Returns all signals that were blocked by the news gate.
    Use this to analyse whether news filter helped or blocked winners.
    Key field: would_have_won — True/False filled after event passes.
    """
    return {
        "news_blocked_signals": news_log[-limit:],
        "total":                len(news_log),
        "summary": {
            "total_blocked":    len(news_log),
            "would_have_won":   sum(1 for s in news_log if s.get("would_have_won") is True),
            "would_have_lost":  sum(1 for s in news_log if s.get("would_have_won") is False),
            "unresolved":       sum(1 for s in news_log if s.get("would_have_won") is None),
        }
    }

@app.get("/fibonacci")
def fibonacci_levels():
    """Get current Fibonacci retracement levels for XAUUSD."""
    try:
        df_1h = get_oanda_candles("H1", 200)
        fib   = calculate_fibonacci(df_1h)
        price, _ = get_current_price()
        return {**fib, "current_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-structure")
def market_structure():
    """Get current SMC market structure analysis for XAUUSD."""
    try:
        df_1h    = get_oanda_candles("H1", 200)
        price, _ = get_current_price()
        smc      = detect_market_structure(df_1h)
        return {**smc, "current_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/log")
def get_signals_log(limit: int = 50):
    return {"signals": signal_log[-limit:], "total": len(signal_log)}

@app.get("/scalp/log")
def get_scalp_log(limit: int = 50):
    return {"signals": scalp_log[-limit:], "total": len(scalp_log)}

# ══════════════════════════════════════════════════════════════════════
# DASHBOARD — single endpoint that replaces 5 separate Android calls
# Android app calls this once per minute instead of 5 endpoints
# Saves ~4 API calls per minute = ~5,760 Twelve Data calls/day saved
# ══════════════════════════════════════════════════════════════════════
@app.get("/dashboard")
def get_dashboard(mode: str = "swing"):
    """
    Returns everything the Android app needs in one request:
    price, signal, stats, grade-stats, news log.
    mode: swing | scalp
    """
    result = {}

    # 1. Price (uses cache — no extra API call if already fetched)
    try:
        price, spread = get_current_price()
        session       = is_trading_session()
        result["price"] = {
            "instrument": "XAUUSD",
            "price":      round(price, 2),
            "spread":     round(spread, 1),
            "session":    session["session"],
            "in_session": session["in_session"],
        }
    except Exception as e:
        result["price"] = {"error": str(e)}

    # 2. Signal (uses candle cache — no extra API call after first fetch)
    try:
        if mode.lower() == "scalp":
            result["signal"] = compute_scalp_signal()
        else:
            result["signal"] = compute_swing_signal()
    except Exception as e:
        result["signal"] = {"error": str(e), "signal": "HOLD"}

    # 3. Stats
    try:
        log = scalp_log if mode.lower() == "scalp" else signal_log
        directional = [s for s in log if s.get("signal") in ("BUY", "SELL")]
        closed      = [s for s in directional if s.get("status") == "CLOSED"]
        wins        = [s for s in closed if s.get("outcome") == "WIN"]
        losses      = [s for s in closed if s.get("outcome") == "LOSS"]
        pnls        = [s["pnl_pct"] for s in closed if s.get("pnl_pct") is not None]
        win_rate    = round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1)
        result["stats"] = {
            "wins":     len(wins),
            "losses":   len(losses),
            "win_rate": f"{win_rate}%",
            "pnl":      f"{round(sum(pnls), 2):+.2f}%" if pnls else "+0.00%",
            "open":     sum(1 for s in directional if s.get("status") == "OPEN"),
            "total":    len(directional),
        }
    except Exception as e:
        result["stats"] = {"error": str(e)}

    # 4. Grade stats
    try:
        log = scalp_log if mode.lower() == "scalp" else signal_log
        directional = [s for s in log if s.get("signal") in ("BUY", "SELL")]
        result["grade_stats"] = {
            "total_signals": len(directional),
            "grades": {
                "A": _grade_stats(directional, "A"),
                "B": _grade_stats(directional, "B"),
                "C": _grade_stats(directional, "C"),
            }
        }
    except Exception as e:
        result["grade_stats"] = {"error": str(e)}

    # 5. News log (swing only)
    try:
        if mode.lower() == "swing":
            result["news_log"] = {
                "news_blocked_signals": news_log[-20:],
                "total": len(news_log),
                "summary": {
                    "total_blocked":   len(news_log),
                    "would_have_won":  sum(1 for s in news_log if s.get("would_have_won") is True),
                    "would_have_lost": sum(1 for s in news_log if s.get("would_have_won") is False),
                    "unresolved":      sum(1 for s in news_log if s.get("would_have_won") is None),
                }
            }
    except Exception as e:
        result["news_log"] = {"error": str(e)}

    result["mode"]         = mode.lower()
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/signals/download")
def download_swing_report():
    """Download swing signal log as Excel file."""
    try:
        import openpyxl  # noqa
        COL_ORDER = [
            "id", "mode", "date", "time_utc", "symbol", "signal", "grade",
            "trade_allowed", "blocked_by", "confidence", "score",
            "adx", "rsi", "vol_ratio", "spread", "session", "daily_bias", "market_regime",
            "entry_price", "stop_loss", "take_profit", "risk_reward",
            "nearest_support", "nearest_resistance",
            "status", "outcome", "exit_price", "pnl_pct", "logged_at",
        ]
        src = _db_get_all_trades(mode="SWING") if (DATABASE_URL and _PG_AVAILABLE) else signal_log
        if src:
            flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in src]
            df = pd.DataFrame(flat)
            ordered = [c for c in COL_ORDER if c in df.columns]
            df = df[ordered + [c for c in df.columns if c not in COL_ORDER]]
        else:
            df = pd.DataFrame(columns=COL_ORDER)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Swing Signals")
        output.seek(0)
        today = datetime.now().strftime("%Y-%m-%d")
        return StreamingResponse(output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=gold_swing_{today}.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {e}")


@app.get("/scalp/download")
def download_scalp_report():
    """Download scalp signal log as Excel file."""
    try:
        import openpyxl  # noqa
        COL_ORDER = [
            "id", "mode", "date", "time_utc", "symbol", "signal", "grade",
            "trade_allowed", "blocked_by", "confidence", "score",
            "adx", "rsi", "vol_ratio", "spread", "session",
            "entry_price", "stop_loss", "take_profit",
            "nearest_support", "nearest_resistance",
            "status", "outcome", "exit_price", "pnl_pct", "logged_at",
        ]
        src = _db_get_all_trades(mode="SCALP") if (DATABASE_URL and _PG_AVAILABLE) else scalp_log
        if src:
            flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in src]
            df = pd.DataFrame(flat)
            ordered = [c for c in COL_ORDER if c in df.columns]
            df = df[ordered + [c for c in df.columns if c not in COL_ORDER]]
        else:
            df = pd.DataFrame(columns=COL_ORDER)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Scalp Signals")
        output.seek(0)
        today = datetime.now().strftime("%Y-%m-%d")
        return StreamingResponse(output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=gold_scalp_{today}.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# PAPER TRADE RESOLVER
# ══════════════════════════════════════════════════════════════════════
async def paper_trade_resolver():
    """Every 5 min: auto-close open paper trades vs live price."""
    await asyncio.sleep(180)  # wait 3 min before first resolve
    while True:
        try:
            # Skip the API-hungry candle fetch entirely if nothing is open —
            # no point burning AllTick/Twelve Data quota checking trades
            # that don't exist (e.g. quiet weekend with all trades closed).
            has_open_trades = any(t.get("status") == "OPEN" for t in signal_log + scalp_log)
            if not has_open_trades:
                await asyncio.sleep(SCAN_CYCLE_SECONDS)
                continue

            # Try fresh M1 candles — but don't fail if rate limited
            # count=100 matches the scalp scanner's M1 fetch so they share
            # one cache entry instead of doubling M1 API calls
            try:
                get_oanda_candles("M1", 100)
            except Exception:
                pass

            # Use cached price if available — never skip resolver due to rate limit
            now = datetime.now(timezone.utc)
            try:
                price, _ = get_current_price()
            except Exception:
                # Fall back to whatever is in cache
                if _price_cache["price"] is not None:
                    price = _price_cache["price"]
                    print(f"[Resolver] Using cached price ${price:.2f}")
                else:
                    print("[Resolver] No price available — skipping this cycle")
                    await asyncio.sleep(600)
                    continue

            # Use M1 high/low to catch sharp wicks between scan cycles
            recent_high = price
            recent_low  = price
            try:
                m1 = _candle_cache.get(("M1", 100))
                if m1 is not None:
                    df_m1 = m1[0].tail(10)  # last 10 min of M1 candles
                    recent_high  = float(df_m1["high"].max())
                    recent_low   = float(df_m1["low"].min())
                    latest_close = float(m1[0]["close"].iloc[-1])
                    if abs(latest_close - price) < 50:  # sanity check
                        price = latest_close
            except Exception:
                pass

            def _resolve(log_list, log_file, max_hours):
                changed = False
                for trade in log_list:
                    if trade.get("status") != "OPEN":
                        continue
                    sig   = trade.get("signal", "HOLD")
                    entry = trade.get("entry_price") or 0
                    sl    = trade.get("stop_loss") or 0
                    tp    = trade.get("take_profit") or 0
                    if not entry or not sl or not tp:
                        continue
                    try:
                        logged_dt = datetime.fromisoformat(
                            trade.get("logged_at", "").replace("Z", "+00:00"))
                        age_hrs = (now - logged_dt).total_seconds() / 3600
                    except Exception:
                        age_hrs = 0

                    # ── Partial TP (50%) + move SL to break-even ──────────
                    # Once price reaches the halfway point to TP, simulate
                    # closing 50% of the position there and moving the SL
                    # for the remaining 50% to entry — so the trade can no
                    # longer turn into a full loss after this point.
                    if not trade.get("partial_closed"):
                        halfway = entry + (tp - entry) / 2 if sig == "BUY" \
                                  else entry - (entry - tp) / 2
                        hit_halfway = False
                        if sig == "BUY" and max(price, recent_high) >= halfway:
                            hit_halfway = True
                        elif sig == "SELL" and min(price, recent_low) <= halfway:
                            hit_halfway = True

                        if hit_halfway:
                            partial_pnl = round(
                                ((halfway - entry) / entry * 100) if sig == "BUY"
                                else ((entry - halfway) / entry * 100), 3)
                            trade["partial_closed"]      = True
                            trade["partial_exit_price"]  = round(halfway, 2)
                            trade["partial_pnl_pct"]      = partial_pnl
                            trade["partial_closed_at"]    = now.isoformat()
                            trade["original_stop_loss"]   = trade.get("original_stop_loss", sl)
                            trade["sl_at_breakeven"]      = True
                            trade["stop_loss"] = sl = round(entry, 2)  # SL -> break-even
                            _db_partial_close_trade(
                                trade["id"], round(halfway, 2), partial_pnl,
                                now.isoformat(), round(entry, 2))
                            changed = True
                            print(f"[Resolver] {trade['id']} -> PARTIAL TP 50% @ {halfway:.2f}  "
                                  f"(+{partial_pnl:.3f}%)  SL moved to break-even ${entry:.2f}")

                    outcome = exit_price = None
                    if sig == "BUY":
                        # Use candle low to catch sharp drops within the scan window
                        if min(price, recent_low) <= sl:
                            outcome, exit_price = ("BREAKEVEN" if trade.get("sl_at_breakeven") else "LOSS"), sl
                        elif max(price, recent_high) >= tp:
                            outcome, exit_price = "WIN",  tp
                    elif sig == "SELL":
                        # Use candle high to catch sharp spikes within the scan window
                        if max(price, recent_high) >= sl:
                            outcome, exit_price = ("BREAKEVEN" if trade.get("sl_at_breakeven") else "LOSS"), sl
                        elif min(price, recent_low) <= tp:
                            outcome, exit_price = "WIN",  tp

                    if outcome is None and age_hrs >= max_hours:
                        outcome, exit_price = "EXPIRED", price

                    if outcome:
                        # Final leg PnL — for the remaining 50% if a partial
                        # close already happened, otherwise the full position
                        final_leg_pnl = 0.0
                        if entry and exit_price:
                            final_leg_pnl = round(
                                ((exit_price - entry) / entry * 100) if sig == "BUY"
                                else ((entry - exit_price) / entry * 100), 3)

                        if trade.get("partial_closed"):
                            # Blend: 50% weight on partial leg + 50% weight on final leg
                            pnl = round((trade.get("partial_pnl_pct", 0) * 0.5)
                                        + (final_leg_pnl * 0.5), 3)
                        else:
                            pnl = final_leg_pnl

                        bars = int(age_hrs * 12)
                        trade.update({
                            "status": "CLOSED", "outcome": outcome,
                            "exit_price": round(exit_price, 2),
                            "pnl_pct": pnl, "closed_at": now.isoformat(),
                            "bars_held": bars,
                        })
                        _db_close_trade(trade["id"], outcome, round(exit_price, 2),
                                        pnl, now.isoformat(), bars)
                        changed = True
                        tag = " (after partial TP)" if trade.get("partial_closed") else ""
                        print(f"[Resolver] {trade['id']} -> {outcome}{tag} @ {exit_price:.2f}  PnL {pnl:+.3f}%")
                if changed:
                    _save_json(log_file, log_list)

            _resolve(signal_log, SIGNAL_LOG_FILE, max_hours=48)
            _resolve(scalp_log,  SCALP_LOG_FILE,  max_hours=1 if USE_ALLTICK else 2)
        except Exception as e:
            print(f"[Resolver] Error: {e}")
        await asyncio.sleep(SCAN_CYCLE_SECONDS)


# ══════════════════════════════════════════════════════════════════════
# PAPER TRADING ANALYSIS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/paper/stats")
def paper_stats():
    """Full paper-trading performance: win rate, PnL, expectancy, by grade/session/direction."""
    def _analyse(trades, label):
        allowed = [t for t in trades if t.get("trade_allowed") == "YES"
                   and t.get("signal") in ("BUY", "SELL")]
        closed    = [t for t in allowed if t.get("status") == "CLOSED"]
        wins      = [t for t in closed  if t.get("outcome") == "WIN"]
        losses    = [t for t in closed  if t.get("outcome") == "LOSS"]
        breakeven = [t for t in closed  if t.get("outcome") == "BREAKEVEN"]
        expired   = [t for t in closed  if t.get("outcome") == "EXPIRED"]
        open_     = [t for t in allowed if t.get("status") == "OPEN"]
        pnls      = [t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None]
        partial_hits = [t for t in closed if t.get("partial_closed")]
        # Win rate counts breakeven separately — it's neither a loss nor a full win
        wr    = round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1)
        avg_w = round(sum(t["pnl_pct"] for t in wins)   / max(len(wins),   1), 3)
        avg_l = round(sum(t["pnl_pct"] for t in losses) / max(len(losses), 1), 3)
        grade_rows = {}
        for g in ("A", "B", "C"):
            gc = [t for t in closed if t.get("grade") == g]
            gw = [t for t in gc if t.get("outcome") == "WIN"]
            gl = [t for t in gc if t.get("outcome") == "LOSS"]
            gp = [t["pnl_pct"] for t in gc if t.get("pnl_pct") is not None]
            grade_rows[g] = {
                "total": len(gc), "wins": len(gw), "losses": len(gl),
                "win_rate": f"{round(len(gw)/max(len(gw)+len(gl),1)*100,1)}%",
                "total_pnl": f"{round(sum(gp),3):+.3f}%" if gp else "+0.000%",
            }
        sessions = {}
        for t in closed:
            s = t.get("session", "UNKNOWN")
            sessions.setdefault(s, {"wins": 0, "losses": 0, "pnl": 0.0})
            if t.get("outcome") == "WIN":    sessions[s]["wins"]   += 1
            elif t.get("outcome") == "LOSS": sessions[s]["losses"] += 1
            if t.get("pnl_pct") is not None:
                sessions[s]["pnl"] = round(sessions[s]["pnl"] + t["pnl_pct"], 3)
        buys  = [t for t in closed if t.get("signal") == "BUY"]
        sells = [t for t in closed if t.get("signal") == "SELL"]
        bw = len([t for t in buys  if t.get("outcome") == "WIN"])
        sw = len([t for t in sells if t.get("outcome") == "WIN"])
        return {
            "mode": label, "open_trades": len(open_),
            "total_closed": len(closed), "wins": len(wins),
            "losses": len(losses), "breakeven": len(breakeven),
            "expired": len(expired),
            "win_rate": f"{wr}%",
            "total_pnl": f"{round(sum(pnls),3):+.3f}%" if pnls else "+0.000%",
            "avg_pnl_per_trade": f"{round(sum(pnls)/len(pnls),3):+.3f}%" if pnls else "+0.000%",
            "avg_win": f"{avg_w:+.3f}%", "avg_loss": f"{avg_l:+.3f}%",
            "expectancy": f"{round((wr/100*avg_w)+((1-wr/100)*avg_l),3):+.3f}%",
            "by_grade": grade_rows, "by_session": sessions,
            "by_direction": {
                "BUY":  {"total": len(buys),  "wins": bw,
                         "win_rate": f"{round(bw/max(len(buys),1)*100,1)}%"},
                "SELL": {"total": len(sells), "wins": sw,
                         "win_rate": f"{round(sw/max(len(sells),1)*100,1)}%"},
            },
            "partial_tp": {
                "description": "Trades that hit 50% of TP, took partial profit, "
                                "and moved SL to break-even for the remaining position",
                "total_hit_halfway": len(partial_hits),
                "saved_from_full_loss": len(breakeven),
            },
        }
    sw = _db_get_all_trades(mode="SWING") if (DATABASE_URL and _PG_AVAILABLE) else signal_log
    sc = _db_get_all_trades(mode="SCALP") if (DATABASE_URL and _PG_AVAILABLE) else scalp_log
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "swing": _analyse(sw, "SWING"),
        "scalp": _analyse(sc, "SCALP"),
    }


@app.get("/paper/trades")
def paper_trades(mode: str = None, status: str = None, limit: int = 100):
    """List paper trades. Filters: mode=SWING|SCALP, status=OPEN|CLOSED."""
    if DATABASE_URL and _PG_AVAILABLE:
        trades = _db_get_all_trades(mode=mode.upper() if mode else None, limit=limit)
    else:
        trades = signal_log + scalp_log
        if mode:
            trades = [t for t in trades if t.get("mode","").upper() == mode.upper()]
    if status:
        trades = [t for t in trades if t.get("status","").upper() == status.upper()]
    return {"total": len(trades[:limit]), "trades": trades[:limit]}


@app.get("/paper/open")
def paper_open_trades():
    """All currently open paper trades."""
    if DATABASE_URL and _PG_AVAILABLE:
        open_trades = _db_get_open_trades()
    else:
        open_trades = [t for t in signal_log + scalp_log if t.get("status") == "OPEN"]
    return {"total": len(open_trades), "trades": open_trades}


@app.get("/paper/live")
def paper_live_trades(mode: str = "swing", grade: str = None):
    """
    Live-tracking view for open trades — used by the grade journal drill-down.
    Returns full detail per trade: live PnL%, distance to SL/TP, time held.

    mode: swing | scalp
    grade: A | B | C | None (all grades)
    """
    try:
        price, _ = get_current_price()
    except Exception:
        price = None

    # Always merge DB + in-memory so trades logged before DB was connected
    # (or during a DB outage) are still visible in the live screen.
    if DATABASE_URL and _PG_AVAILABLE:
        db_trades = _db_get_all_trades(mode=mode.upper(), limit=500)
        db_ids    = {t.get("id") for t in db_trades}
        mem_trades = signal_log if mode.lower() == "swing" else scalp_log
        # Include in-memory trades that aren't already in the DB result
        extra = [t for t in mem_trades
                 if t.get("id") not in db_ids
                 and t.get("mode", "").upper() == mode.upper()]
        all_trades = db_trades + extra
    else:
        all_trades = signal_log if mode.lower() == "swing" else scalp_log

    # Match the same filter the grade journal uses — include all BUY/SELL
    # regardless of trade_allowed so counts stay consistent across screens
    open_trades = [t for t in all_trades
                   if t.get("status") == "OPEN"
                   and t.get("signal") in ("BUY", "SELL")]

    if grade:
        open_trades = [t for t in open_trades if t.get("grade") == grade.upper()]

    now = datetime.now(timezone.utc)
    result = []

    for t in open_trades:
        entry = t.get("entry_price") or 0
        sl    = t.get("stop_loss") or 0
        tp    = t.get("take_profit") or 0
        sig   = t.get("signal", "")

        # Time held
        time_held_str = "unknown"
        hours_held     = 0
        try:
            logged_dt  = datetime.fromisoformat(t.get("logged_at", "").replace("Z", "+00:00"))
            delta      = now - logged_dt
            hours_held = delta.total_seconds() / 3600
            if hours_held < 1:
                time_held_str = f"{int(delta.total_seconds() / 60)}m"
            else:
                h = int(hours_held)
                m = int((hours_held - h) * 60)
                time_held_str = f"{h}h {m}m"
        except Exception:
            pass

        # Live PnL % and distance to SL/TP
        live_pnl_pct   = None
        dist_to_sl     = None
        dist_to_tp     = None
        floating_state = "unknown"

        if price and entry:
            if sig == "BUY":
                live_pnl_pct = round((price - entry) / entry * 100, 3)
                dist_to_sl   = round(price - sl, 2) if sl else None
                dist_to_tp   = round(tp - price, 2) if tp else None
            elif sig == "SELL":
                live_pnl_pct = round((entry - price) / entry * 100, 3)
                dist_to_sl   = round(sl - price, 2) if sl else None
                dist_to_tp   = round(price - tp, 2) if tp else None

            if live_pnl_pct is not None:
                floating_state = "profit" if live_pnl_pct > 0 else ("loss" if live_pnl_pct < 0 else "breakeven")

        result.append({
            "id":              t.get("id"),
            "mode":            t.get("mode"),
            "grade":           t.get("grade"),
            "signal":          sig,
            "confidence":      t.get("confidence"),
            "session":         t.get("session"),
            "entry_price":     entry,
            "current_price":   price,
            "stop_loss":       sl,
            "take_profit":     tp,
            "risk_reward":     t.get("risk_reward"),
            "sl_method":       t.get("sl_method"),
            "live_pnl_pct":    f"{live_pnl_pct:+.3f}%" if live_pnl_pct is not None else "--",
            "floating_state":  floating_state,
            "dist_to_sl":      dist_to_sl,
            "dist_to_tp":      dist_to_tp,
            "time_held":       time_held_str,
            "hours_held":      round(hours_held, 2),
            "logged_at":       t.get("logged_at"),
            "market_regime":   t.get("market_regime"),
            "partial_closed":      bool(t.get("partial_closed")),
            "partial_exit_price":  t.get("partial_exit_price"),
            "partial_pnl_pct":     t.get("partial_pnl_pct"),
            "sl_at_breakeven":     bool(t.get("sl_at_breakeven")),
            "original_stop_loss":  t.get("original_stop_loss"),
        })

    # Sort by most recently opened first
    result.sort(key=lambda x: x.get("logged_at", ""), reverse=True)

    wins_floating   = sum(1 for r in result if r["floating_state"] == "profit")
    losses_floating = sum(1 for r in result if r["floating_state"] == "loss")

    return {
        "mode":             mode.lower(),
        "grade_filter":     grade.upper() if grade else "ALL",
        "current_price":    price,
        "total_open":       len(result),
        "currently_winning": wins_floating,
        "currently_losing":  losses_floating,
        "trades":           result,
        "generated_at":     now.isoformat(),
    }


@app.get("/paper/download")
def paper_download():
    """Download ALL paper trades as Excel with Summary, Swing, and Scalp sheets."""
    try:
        import openpyxl  # noqa
        all_trades = (_db_get_all_trades(limit=5000)
                      if (DATABASE_URL and _PG_AVAILABLE)
                      else signal_log + scalp_log)
        COL = [
            "id", "mode", "date", "time_utc", "signal", "grade", "trade_allowed",
            "blocked_by", "confidence", "score", "adx", "rsi", "vol_ratio",
            "session", "daily_bias", "market_regime",
            "entry_price", "stop_loss", "take_profit", "risk_reward",
            "status", "outcome", "exit_price", "pnl_pct", "bars_held", "logged_at",
        ]
        def _to_df(rows):
            if not rows:
                return pd.DataFrame(columns=COL)
            flat = [{k: (json.dumps(v) if isinstance(v,(dict,list)) else v)
                     for k, v in r.items()} for r in rows]
            df = pd.DataFrame(flat)
            ordered = [c for c in COL if c in df.columns]
            return df[ordered + [c for c in df.columns if c not in COL]]
        swing_df = _to_df([t for t in all_trades if t.get("mode") == "SWING"])
        scalp_df = _to_df([t for t in all_trades if t.get("mode") == "SCALP"])
        stats = paper_stats()
        summary_rows = []
        for mk in ("swing", "scalp"):
            s = stats[mk]
            for label, val in [
                ("Win Rate", s["win_rate"]), ("Total PnL", s["total_pnl"]),
                ("Avg PnL/Trade", s["avg_pnl_per_trade"]), ("Expectancy", s["expectancy"]),
                ("Wins", s["wins"]), ("Losses", s["losses"]),
                ("Expired", s["expired"]), ("Open Trades", s["open_trades"]),
                ("Grade A Win Rate", s["by_grade"]["A"]["win_rate"]),
                ("Grade B Win Rate", s["by_grade"]["B"]["win_rate"]),
                ("Grade C Win Rate", s["by_grade"]["C"]["win_rate"]),
            ]:
                summary_rows.append({"Metric": f"[{mk.upper()}] {label}", "Value": val})
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")
            swing_df.to_excel(writer, index=False, sheet_name="Swing Trades")
            scalp_df.to_excel(writer, index=False, sheet_name="Scalp Trades")
        output.seek(0)
        today = datetime.now().strftime("%Y-%m-%d")
        return StreamingResponse(output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=gold_paper_{today}.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {e}")
