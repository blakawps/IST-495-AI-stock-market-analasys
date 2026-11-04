"""
rl_state.py

Builds the neural-network state from every market signal already collected by
this project. News gets a deliberately larger feature branch in the network.

Raw article text continues to be read by the existing FinBERT system. This RL
state consumes FinBERT sentiment/confidence plus news timing/count/trend
features; it does not replace FinBERT.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from db import db, headlines, market_data, stocktwits_posts

from rl_config import (
    LAST_ENTRY_HOUR,
    LAST_ENTRY_MINUTE,
    MAX_MARKET_DATA_AGE_MINUTES,
    MAX_POSITIONS,
    NEWS_WINDOW_HOURS,
)


EASTERN = ZoneInfo("America/New_York")

agent_state = db["agent_state"]
agent_positions = db["agent_positions"]
rl_market_history = db["rl_market_history"]
analytics_snapshots = db["analytics_snapshots"]


# =========================================================
# FEATURE GROUPS
# =========================================================

NEWS_FEATURES = [
    "news_sentiment_15m",
    "news_sentiment_60m",
    "news_sentiment_6h",
    "news_confidence_60m",
    "news_count_15m",
    "news_count_60m",
    "news_count_6h",
    "news_bullish_count_6h",
    "news_bearish_count_6h",
    "news_neutral_count_6h",
    "news_recency",
    "news_sentiment_trend",
]

MARKET_FEATURES = [
    "daily_change",
    "return_3m",
    "return_6m",
    "return_15m",
    "return_30m",
    "return_60m",
    "momentum_15m",
    "volatility_15m",
    "volatility_30m",
    "day_range_position",
    "volume_log",
    "market_cap_log",
    "price_log",
]

SOCIAL_FEATURES = [
    "st_sentiment",
    "st_density_log",
    "st_messages_log",
    "st_bullish_fraction",
    "st_bearish_fraction",
    "st_density_change_15m",
    "st_sentiment_change_15m",
    "st_density_acceleration",
]

POSITION_FEATURES = [
    "owns_position",
    "unrealized_return",
    "position_fraction",
    "cash_fraction",
    "num_positions_norm",
    "time_in_position",
    "time_of_day",
    "minutes_to_close",
]

FEATURE_NAMES = (
    NEWS_FEATURES
    + MARKET_FEATURES
    + SOCIAL_FEATURES
    + POSITION_FEATURES
)

FEATURE_COUNT = len(FEATURE_NAMES)

NEWS_SLICE = slice(0, len(NEWS_FEATURES))
MARKET_SLICE = slice(
    len(NEWS_FEATURES),
    len(NEWS_FEATURES) + len(MARKET_FEATURES),
)
SOCIAL_SLICE = slice(
    len(NEWS_FEATURES) + len(MARKET_FEATURES),
    len(NEWS_FEATURES) + len(MARKET_FEATURES) + len(SOCIAL_FEATURES),
)
POSITION_SLICE = slice(
    len(NEWS_FEATURES) + len(MARKET_FEATURES) + len(SOCIAL_FEATURES),
    FEATURE_COUNT,
)


# =========================================================
# BASIC HELPERS
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def aware_utc(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None

    if not isinstance(value, datetime):
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_change(new_value: float, old_value: float) -> float:
    if old_value is None or old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def _sentiment_value(doc: dict) -> tuple[float, float]:
    sentiment = doc.get("news_sentiment") or {}
    score = safe_float(sentiment.get("score"), 0.0)
    confidence = clamp(
        safe_float(sentiment.get("confidence"), 1.0),
        0.0,
        1.0,
    )
    return score, confidence


def _weighted_news_sentiment(
    docs: list[dict],
    start_time: datetime,
    end_time: datetime,
) -> tuple[float, float, int]:

    weighted_sum = 0.0
    total_weight = 0.0
    confidence_sum = 0.0
    count = 0

    for doc in docs:
        published = aware_utc(doc.get("published_at"))
        if published is None:
            continue

        if not (start_time <= published <= end_time):
            continue

        score, confidence = _sentiment_value(doc)

        # Confidence is used as weight, but never let an article disappear
        # completely if FinBERT confidence is very low.
        weight = max(0.05, confidence)

        weighted_sum += score * weight
        total_weight += weight
        confidence_sum += confidence
        count += 1

    if total_weight:
        sentiment = weighted_sum / total_weight
    else:
        sentiment = 0.0

    if count:
        avg_confidence = confidence_sum / count
    else:
        avg_confidence = 0.0

    return sentiment, avg_confidence, count


# =========================================================
# MARKET HOURS / PORTFOLIO CONTEXT
# =========================================================

def market_is_open_at(now_utc: datetime) -> bool:
    now_et = now_utc.astimezone(EASTERN)

    if now_et.weekday() >= 5:
        return False

    market_open = now_et.replace(
        hour=9,
        minute=30,
        second=0,
        microsecond=0,
    )

    market_close = now_et.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    return market_open <= now_et < market_close


def can_open_new_positions_at(now_utc: datetime) -> bool:
    now_et = now_utc.astimezone(EASTERN)

    if not market_is_open_at(now_utc):
        return False

    cutoff = now_et.replace(
        hour=LAST_ENTRY_HOUR,
        minute=LAST_ENTRY_MINUTE,
        second=0,
        microsecond=0,
    )

    return now_et < cutoff


def minutes_to_close_raw(now_utc: datetime) -> float:
    now_et = now_utc.astimezone(EASTERN)

    close = now_et.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    return max(
        0.0,
        (close - now_et).total_seconds() / 60.0,
    )


def build_portfolio_context() -> dict:
    state = agent_state.find_one({"_id": "paper_agent"}) or {}

    cash = safe_float(
        state.get("cash"),
        safe_float(state.get("starting_cash"), 100_000.0),
    )

    positions = list(agent_positions.find({}))

    position_map = {}
    positions_value = 0.0

    for position in positions:
        symbol = str(position.get("symbol", "")).strip().upper()
        if not symbol:
            continue

        shares = int(position.get("shares") or 0)
        avg_price = safe_float(position.get("avg_price"))

        quote = market_data.find_one(
            {"symbol": symbol},
            {"_id": 0, "price": 1},
        ) or {}

        current_price = safe_float(
            quote.get("price"),
            safe_float(position.get("last_price"), avg_price),
        )

        market_value = shares * current_price
        positions_value += market_value

        position_map[symbol] = {
            **position,
            "shares": shares,
            "avg_price": avg_price,
            "current_price": current_price,
            "market_value": market_value,
        }

    portfolio_value = cash + positions_value

    return {
        "cash": cash,
        "positions": position_map,
        "position_count": len(position_map),
        "positions_value": positions_value,
        "portfolio_value": portfolio_value,
    }


# =========================================================
# NEWS FEATURES
# =========================================================

def get_news_features(symbol: str, now: datetime) -> dict:
    six_hours_ago = now - timedelta(hours=NEWS_WINDOW_HOURS)

    docs = list(
        headlines.find(
            {
                "securities.symbol": symbol,
                "published_at": {
                    "$gte": six_hours_ago,
                    "$lte": now,
                },
            },
            {
                "_id": 0,
                "published_at": 1,
                "news_sentiment": 1,
            },
        )
    )

    sentiment_15m, _, count_15m = _weighted_news_sentiment(
        docs,
        now - timedelta(minutes=15),
        now,
    )

    sentiment_60m, confidence_60m, count_60m = _weighted_news_sentiment(
        docs,
        now - timedelta(minutes=60),
        now,
    )

    sentiment_6h, _, count_6h = _weighted_news_sentiment(
        docs,
        six_hours_ago,
        now,
    )

    previous_sentiment, _, _ = _weighted_news_sentiment(
        docs,
        now - timedelta(minutes=60),
        now - timedelta(minutes=15),
    )

    bullish = 0
    bearish = 0
    neutral = 0
    newest = None

    for doc in docs:
        published = aware_utc(doc.get("published_at"))
        if published is not None:
            if newest is None or published > newest:
                newest = published

        score, _ = _sentiment_value(doc)

        if score > 0:
            bullish += 1
        elif score < 0:
            bearish += 1
        else:
            neutral += 1

    if newest is None:
        age_minutes = 360.0
    else:
        age_minutes = max(
            0.0,
            (now - newest).total_seconds() / 60.0,
        )

    # 1 = very recent article, 0 = no article within the six-hour window.
    recency = 1.0 - clamp(age_minutes / 360.0, 0.0, 1.0)

    return {
        "news_sentiment_15m": sentiment_15m,
        "news_sentiment_60m": sentiment_60m,
        "news_sentiment_6h": sentiment_6h,
        "news_confidence_60m": confidence_60m,
        "news_count_15m_raw": count_15m,
        "news_count_60m_raw": count_60m,
        "news_count_6h_raw": count_6h,
        "news_bullish_count_6h_raw": bullish,
        "news_bearish_count_6h_raw": bearish,
        "news_neutral_count_6h_raw": neutral,
        "news_recency": recency,
        "news_sentiment_trend": (
            sentiment_15m - previous_sentiment
        ),
    }


# =========================================================
# STOCKTWITS FEATURES
# =========================================================

def _social_sentiment(posts: list[dict]) -> tuple[float, int, int]:
    bullish = 0
    bearish = 0

    for post in posts:
        sentiment = post.get("sentiment")

        if sentiment == "Bullish":
            bullish += 1
        elif sentiment == "Bearish":
            bearish += 1

    labeled = bullish + bearish

    if labeled:
        score = (bullish - bearish) / labeled
    else:
        score = 0.0

    return score, bullish, bearish


def get_social_features(symbol: str, now: datetime) -> dict:
    cutoff = now - timedelta(minutes=60)

    posts = list(
        stocktwits_posts.find(
            {
                "symbol": symbol,
                "created_at": {
                    "$gte": cutoff,
                    "$lte": now,
                },
            },
            {
                "_id": 0,
                "created_at": 1,
                "sentiment": 1,
            },
        )
    )

    recent_15 = []
    previous_15 = []
    previous_45 = []

    recent_cutoff = now - timedelta(minutes=15)
    prev_15_cutoff = now - timedelta(minutes=30)

    for post in posts:
        created = aware_utc(post.get("created_at"))
        if created is None:
            continue

        if created >= recent_cutoff:
            recent_15.append(post)
        elif created >= prev_15_cutoff:
            previous_15.append(post)

        if cutoff <= created < recent_cutoff:
            previous_45.append(post)

    sentiment, bullish, bearish = _social_sentiment(posts)
    recent_sentiment, _, _ = _social_sentiment(recent_15)
    previous_sentiment, _, _ = _social_sentiment(previous_45)

    message_count = len(posts)

    if message_count:
        bullish_fraction = bullish / message_count
        bearish_fraction = bearish / message_count
    else:
        bullish_fraction = 0.0
        bearish_fraction = 0.0

    # Compare the most recent 15 minutes with the equivalent 15-minute
    # rate from the prior 45 minutes.
    expected_previous_15 = len(previous_45) / 3.0

    density_change_15m = (
        len(recent_15) - expected_previous_15
    )

    density_acceleration = (
        len(recent_15) - len(previous_15)
    )

    return {
        "stocktwits_sentiment_raw": sentiment,
        "stocktwits_density_raw": message_count,
        "stocktwits_messages_raw": message_count,
        "stocktwits_bullish_raw": bullish,
        "stocktwits_bearish_raw": bearish,
        "stocktwits_bullish_fraction": bullish_fraction,
        "stocktwits_bearish_fraction": bearish_fraction,
        "stocktwits_density_change_15m_raw": density_change_15m,
        "stocktwits_sentiment_change_15m": (
            recent_sentiment - previous_sentiment
        ),
        "stocktwits_density_acceleration_raw": density_acceleration,
    }


# =========================================================
# PRICE-HISTORY FEATURES
# =========================================================

def _history_rows(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[dict]:

    rows = list(
        rl_market_history.find(
            {
                "symbol": symbol,
                "timestamp": {
                    "$gte": start,
                    "$lte": end,
                },
                "price": {
                    "$ne": None,
                },
            },
            {
                "_id": 0,
                "timestamp": 1,
                "price": 1,
            },
        ).sort("timestamp", 1)
    )

    # During the very first run of the RL system, the permanent history may
    # be empty. Fall back to the existing analytics snapshots if available.
    if not rows:
        rows = list(
            analytics_snapshots.find(
                {
                    "symbol": symbol,
                    "timestamp": {
                        "$gte": start,
                        "$lte": end,
                    },
                    "price": {
                        "$ne": None,
                    },
                },
                {
                    "_id": 0,
                    "timestamp": 1,
                    "price": 1,
                },
            ).sort("timestamp", 1)
        )

    return rows


def _price_at_or_before(
    symbol: str,
    target: datetime,
    max_lookback_minutes: int = 10,
) -> float | None:

    row = rl_market_history.find_one(
        {
            "symbol": symbol,
            "timestamp": {
                "$lte": target,
                "$gte": target - timedelta(minutes=max_lookback_minutes),
            },
            "price": {
                "$ne": None,
            },
        },
        {
            "_id": 0,
            "price": 1,
        },
        sort=[("timestamp", -1)],
    )

    if row is None:
        row = analytics_snapshots.find_one(
            {
                "symbol": symbol,
                "timestamp": {
                    "$lte": target,
                    "$gte": target - timedelta(minutes=max_lookback_minutes),
                },
                "price": {
                    "$ne": None,
                },
            },
            {
                "_id": 0,
                "price": 1,
            },
            sort=[("timestamp", -1)],
        )

    if row is None:
        return None

    return safe_float(row.get("price"), None)


def _volatility_from_rows(rows: list[dict]) -> float:
    prices = [
        safe_float(row.get("price"))
        for row in rows
        if safe_float(row.get("price")) > 0
    ]

    if len(prices) < 3:
        return 0.0

    returns = []

    for old, new in zip(prices[:-1], prices[1:]):
        if old > 0:
            returns.append(
                ((new - old) / old) * 100.0
            )

    if len(returns) < 2:
        return 0.0

    return statistics.pstdev(returns)


def get_market_features(
    symbol: str,
    now: datetime,
    price: float,
    quote: dict,
) -> dict:

    returns = {}

    for minutes in (3, 6, 15, 30, 60):
        historical_price = _price_at_or_before(
            symbol,
            now - timedelta(minutes=minutes),
            max_lookback_minutes=9,
        )

        if historical_price:
            returns[minutes] = pct_change(
                price,
                historical_price,
            )
        else:
            returns[minutes] = 0.0

    rows_15m = _history_rows(
        symbol,
        now - timedelta(minutes=15),
        now,
    )

    rows_30m = _history_rows(
        symbol,
        now - timedelta(minutes=30),
        now,
    )

    now_et = now.astimezone(EASTERN)

    day_start_et = now_et.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    day_rows = _history_rows(
        symbol,
        day_start_et.astimezone(timezone.utc),
        now,
    )

    day_prices = [
        safe_float(row.get("price"))
        for row in day_rows
        if safe_float(row.get("price")) > 0
    ]

    day_prices.append(price)

    low = min(day_prices)
    high = max(day_prices)

    if high > low:
        range_position = (price - low) / (high - low)
    else:
        range_position = 0.5

    volume = max(
        0.0,
        safe_float(quote.get("volume")),
    )

    market_cap = max(
        0.0,
        safe_float(quote.get("market_cap_millions")),
    )

    daily_change = safe_float(
        quote.get("change_percent")
    )

    return {
        "daily_change_raw": daily_change,
        "return_3m_raw": returns[3],
        "return_6m_raw": returns[6],
        "return_15m_raw": returns[15],
        "return_30m_raw": returns[30],
        "return_60m_raw": returns[60],
        "momentum_15m_raw": returns[15],
        "volatility_15m_raw": _volatility_from_rows(rows_15m),
        "volatility_30m_raw": _volatility_from_rows(rows_30m),
        "day_range_position": clamp(range_position, 0.0, 1.0),
        "volume_raw": volume,
        "market_cap_raw": market_cap,
        "price_raw": price,
    }


# =========================================================
# POSITION FEATURES
# =========================================================

def get_position_features(
    symbol: str,
    now: datetime,
    price: float,
    portfolio: dict,
) -> dict:

    position = portfolio["positions"].get(symbol)

    owns = position is not None

    portfolio_value = max(
        safe_float(portfolio.get("portfolio_value")),
        1.0,
    )

    cash = safe_float(portfolio.get("cash"))

    if position:
        avg_price = safe_float(position.get("avg_price"))

        if avg_price > 0:
            unrealized_return = pct_change(
                price,
                avg_price,
            )
        else:
            unrealized_return = 0.0

        position_fraction = (
            safe_float(position.get("market_value"))
            / portfolio_value
        )

        opened_at = aware_utc(
            position.get("opened_at")
        )

        if opened_at:
            time_in_position_minutes = max(
                0.0,
                (now - opened_at).total_seconds() / 60.0,
            )
        else:
            time_in_position_minutes = 0.0

    else:
        avg_price = 0.0
        unrealized_return = 0.0
        position_fraction = 0.0
        time_in_position_minutes = 0.0

    now_et = now.astimezone(EASTERN)

    minutes_since_open = (
        (now_et.hour * 60 + now_et.minute)
        - (9 * 60 + 30)
    )

    market_day_minutes = 390

    time_of_day = clamp(
        minutes_since_open / market_day_minutes,
        0.0,
        1.0,
    )

    return {
        "owns_position_raw": owns,
        "avg_price_raw": avg_price,
        "unrealized_return_raw": unrealized_return,
        "position_fraction_raw": position_fraction,
        "cash_fraction_raw": cash / portfolio_value,
        "num_positions_raw": portfolio.get("position_count", 0),
        "time_in_position_minutes_raw": time_in_position_minutes,
        "time_of_day": time_of_day,
        "minutes_to_close_raw": minutes_to_close_raw(now),
        "can_open_new_position_raw": can_open_new_positions_at(now),
    }


# =========================================================
# NORMALIZATION / VECTOR
# =========================================================

def _log_scale(value: float, max_value: float) -> float:
    value = max(0.0, value)
    return clamp(
        math.log1p(value) / math.log1p(max_value),
        0.0,
        1.0,
    )


def normalize_features(raw: dict) -> dict:
    return {
        # News branch
        "news_sentiment_15m": clamp(
            safe_float(raw.get("news_sentiment_15m")), -1.0, 1.0
        ),
        "news_sentiment_60m": clamp(
            safe_float(raw.get("news_sentiment_60m")), -1.0, 1.0
        ),
        "news_sentiment_6h": clamp(
            safe_float(raw.get("news_sentiment_6h")), -1.0, 1.0
        ),
        "news_confidence_60m": clamp(
            safe_float(raw.get("news_confidence_60m")), 0.0, 1.0
        ),
        "news_count_15m": _log_scale(
            safe_float(raw.get("news_count_15m_raw")), 10.0
        ),
        "news_count_60m": _log_scale(
            safe_float(raw.get("news_count_60m_raw")), 30.0
        ),
        "news_count_6h": _log_scale(
            safe_float(raw.get("news_count_6h_raw")), 100.0
        ),
        "news_bullish_count_6h": _log_scale(
            safe_float(raw.get("news_bullish_count_6h_raw")), 50.0
        ),
        "news_bearish_count_6h": _log_scale(
            safe_float(raw.get("news_bearish_count_6h_raw")), 50.0
        ),
        "news_neutral_count_6h": _log_scale(
            safe_float(raw.get("news_neutral_count_6h_raw")), 50.0
        ),
        "news_recency": clamp(
            safe_float(raw.get("news_recency")), 0.0, 1.0
        ),
        "news_sentiment_trend": clamp(
            safe_float(raw.get("news_sentiment_trend")), -1.0, 1.0
        ),

        # Market branch
        "daily_change": clamp(
            safe_float(raw.get("daily_change_raw")) / 20.0, -1.0, 1.0
        ),
        "return_3m": clamp(
            safe_float(raw.get("return_3m_raw")) / 5.0, -1.0, 1.0
        ),
        "return_6m": clamp(
            safe_float(raw.get("return_6m_raw")) / 7.0, -1.0, 1.0
        ),
        "return_15m": clamp(
            safe_float(raw.get("return_15m_raw")) / 10.0, -1.0, 1.0
        ),
        "return_30m": clamp(
            safe_float(raw.get("return_30m_raw")) / 15.0, -1.0, 1.0
        ),
        "return_60m": clamp(
            safe_float(raw.get("return_60m_raw")) / 20.0, -1.0, 1.0
        ),
        "momentum_15m": clamp(
            safe_float(raw.get("momentum_15m_raw")) / 10.0, -1.0, 1.0
        ),
        "volatility_15m": clamp(
            safe_float(raw.get("volatility_15m_raw")) / 5.0, 0.0, 1.0
        ),
        "volatility_30m": clamp(
            safe_float(raw.get("volatility_30m_raw")) / 7.0, 0.0, 1.0
        ),
        "day_range_position": clamp(
            safe_float(raw.get("day_range_position"), 0.5), 0.0, 1.0
        ),
        "volume_log": _log_scale(
            safe_float(raw.get("volume_raw")), 1_000_000_000.0
        ),
        "market_cap_log": _log_scale(
            safe_float(raw.get("market_cap_raw")), 5_000_000.0
        ),
        "price_log": _log_scale(
            safe_float(raw.get("price_raw")), 10_000.0
        ),

        # Social branch
        "st_sentiment": clamp(
            safe_float(raw.get("stocktwits_sentiment_raw")), -1.0, 1.0
        ),
        "st_density_log": _log_scale(
            safe_float(raw.get("stocktwits_density_raw")), 500.0
        ),
        "st_messages_log": _log_scale(
            safe_float(raw.get("stocktwits_messages_raw")), 500.0
        ),
        "st_bullish_fraction": clamp(
            safe_float(raw.get("stocktwits_bullish_fraction")), 0.0, 1.0
        ),
        "st_bearish_fraction": clamp(
            safe_float(raw.get("stocktwits_bearish_fraction")), 0.0, 1.0
        ),
        "st_density_change_15m": clamp(
            safe_float(raw.get("stocktwits_density_change_15m_raw")) / 50.0,
            -1.0,
            1.0,
        ),
        "st_sentiment_change_15m": clamp(
            safe_float(raw.get("stocktwits_sentiment_change_15m")),
            -1.0,
            1.0,
        ),
        "st_density_acceleration": clamp(
            safe_float(raw.get("stocktwits_density_acceleration_raw")) / 50.0,
            -1.0,
            1.0,
        ),

        # Portfolio / position branch
        "owns_position": 1.0 if raw.get("owns_position_raw") else 0.0,
        "unrealized_return": clamp(
            safe_float(raw.get("unrealized_return_raw")) / 10.0,
            -1.0,
            1.0,
        ),
        "position_fraction": clamp(
            safe_float(raw.get("position_fraction_raw")) / 0.10,
            0.0,
            1.0,
        ),
        "cash_fraction": clamp(
            safe_float(raw.get("cash_fraction_raw")), 0.0, 1.0
        ),
        "num_positions_norm": clamp(
            safe_float(raw.get("num_positions_raw")) / MAX_POSITIONS,
            0.0,
            1.0,
        ),
        "time_in_position": clamp(
            safe_float(raw.get("time_in_position_minutes_raw")) / 390.0,
            0.0,
            1.0,
        ),
        "time_of_day": clamp(
            safe_float(raw.get("time_of_day")), 0.0, 1.0
        ),
        "minutes_to_close": clamp(
            safe_float(raw.get("minutes_to_close_raw")) / 390.0,
            0.0,
            1.0,
        ),
    }


def feature_vector_from_normalized(normalized: dict) -> list[float]:
    return [
        safe_float(normalized.get(name))
        for name in FEATURE_NAMES
    ]


# =========================================================
# FULL STATE
# =========================================================

def build_state(
    symbol: str,
    now: datetime | None = None,
    portfolio: dict | None = None,
) -> dict | None:

    if now is None:
        now = utc_now()

    symbol = symbol.strip().upper()

    quote = market_data.find_one(
        {"symbol": symbol},
        {
            "_id": 0,
            "symbol": 1,
            "price": 1,
            "change_percent": 1,
            "volume": 1,
            "market_cap_millions": 1,
            "updated_at": 1,
        },
    ) or {}

    price = safe_float(quote.get("price"))

    updated_at = aware_utc(
        quote.get("updated_at")
    )

    if price <= 0 or updated_at is None:
        return None

    if (
        now - updated_at
        > timedelta(minutes=MAX_MARKET_DATA_AGE_MINUTES)
    ):
        return None

    if portfolio is None:
        portfolio = build_portfolio_context()

    raw = {
        **get_news_features(symbol, now),
        **get_social_features(symbol, now),
        **get_market_features(symbol, now, price, quote),
        **get_position_features(symbol, now, price, portfolio),
    }

    normalized = normalize_features(raw)
    vector = feature_vector_from_normalized(normalized)

    return {
        "symbol": symbol,
        "timestamp": now,
        "price": price,
        "market_updated_at": updated_at,
        "volume": safe_float(quote.get("volume")),
        "market_cap_millions": safe_float(
            quote.get("market_cap_millions")
        ),
        "raw": raw,
        "features": normalized,
        "state_vector": vector,
        "owns_position": bool(raw.get("owns_position_raw")),
        "can_open_new_position": bool(
            raw.get("can_open_new_position_raw")
        ),
    }


def record_market_state(state: dict) -> None:
    rl_market_history.update_one(
        {
            "symbol": state["symbol"],
            "timestamp": state["timestamp"],
        },
        {
            "$set": {
                "symbol": state["symbol"],
                "timestamp": state["timestamp"],
                "price": state["price"],
                "volume": state.get("volume"),
                "market_cap_millions": state.get("market_cap_millions"),
                "raw": state.get("raw", {}),
                "features": state.get("features", {}),
                "state_vector": state.get("state_vector", []),
                "owns_position": state.get("owns_position", False),
                "can_open_new_position": state.get(
                    "can_open_new_position",
                    False,
                ),
            }
        },
        upsert=True,
    )


def state_from_history_doc(doc: dict) -> dict:
    return {
        "symbol": doc.get("symbol"),
        "timestamp": doc.get("timestamp"),
        "price": safe_float(doc.get("price")),
        "volume": safe_float(doc.get("volume")),
        "market_cap_millions": safe_float(
            doc.get("market_cap_millions")
        ),
        "raw": doc.get("raw") or {},
        "features": doc.get("features") or {},
        "state_vector": doc.get("state_vector") or [],
        "owns_position": bool(doc.get("owns_position")),
        "can_open_new_position": bool(
            doc.get("can_open_new_position")
        ),
    }


# =========================================================
# LEGACY SIGNAL COMPATIBILITY
#
# Existing rule-based trades do not contain the new 41-feature state. This
# maps the fields they DO have into the new input vector so the RL bootstrap
# can learn from them rather than discarding the old history.
# =========================================================

def legacy_signal_to_state(
    signal: dict,
    owns_position: bool = False,
) -> dict:

    signal = signal or {}

    news_sentiment = safe_float(
        signal.get("news_sentiment")
    )

    news_count = max(
        0.0,
        safe_float(signal.get("news_count"))
    )

    st_sentiment = safe_float(
        signal.get("stocktwits_sentiment")
    )

    st_density = max(
        0.0,
        safe_float(signal.get("stocktwits_density"))
    )

    st_messages = max(
        0.0,
        safe_float(signal.get("stocktwits_messages"))
    )

    bullish = max(
        0.0,
        safe_float(signal.get("stocktwits_bullish"))
    )

    bearish = max(
        0.0,
        safe_float(signal.get("stocktwits_bearish"))
    )

    if st_messages:
        bullish_fraction = bullish / st_messages
        bearish_fraction = bearish / st_messages
    else:
        bullish_fraction = 0.0
        bearish_fraction = 0.0

    timestamp = aware_utc(
        signal.get("timestamp")
    ) or utc_now()

    raw = {
        "news_sentiment_15m": news_sentiment,
        "news_sentiment_60m": news_sentiment,
        "news_sentiment_6h": news_sentiment,
        "news_confidence_60m": 0.75 if news_count else 0.0,
        "news_count_15m_raw": news_count,
        "news_count_60m_raw": news_count,
        "news_count_6h_raw": news_count,
        "news_bullish_count_6h_raw": (
            news_count if news_sentiment > 0 else 0
        ),
        "news_bearish_count_6h_raw": (
            news_count if news_sentiment < 0 else 0
        ),
        "news_neutral_count_6h_raw": (
            news_count if news_sentiment == 0 else 0
        ),
        "news_recency": 1.0 if news_count else 0.0,
        "news_sentiment_trend": 0.0,

        "daily_change_raw": safe_float(
            signal.get("daily_change_percent")
        ),
        "return_3m_raw": 0.0,
        "return_6m_raw": 0.0,
        "return_15m_raw": safe_float(
            signal.get("momentum_15m_percent")
        ),
        "return_30m_raw": 0.0,
        "return_60m_raw": 0.0,
        "momentum_15m_raw": safe_float(
            signal.get("momentum_15m_percent")
        ),
        "volatility_15m_raw": 0.0,
        "volatility_30m_raw": 0.0,
        "day_range_position": 0.5,
        "volume_raw": safe_float(signal.get("volume")),
        "market_cap_raw": safe_float(
            signal.get("market_cap_millions")
        ),
        "price_raw": safe_float(signal.get("price")),

        "stocktwits_sentiment_raw": st_sentiment,
        "stocktwits_density_raw": st_density,
        "stocktwits_messages_raw": st_messages,
        "stocktwits_bullish_fraction": bullish_fraction,
        "stocktwits_bearish_fraction": bearish_fraction,
        "stocktwits_density_change_15m_raw": safe_float(
            signal.get("density_change_15m")
        ),
        "stocktwits_sentiment_change_15m": 0.0,
        "stocktwits_density_acceleration_raw": 0.0,

        "owns_position_raw": owns_position,
        "unrealized_return_raw": 0.0,
        "position_fraction_raw": 0.05 if owns_position else 0.0,
        "cash_fraction_raw": 0.75,
        "num_positions_raw": 1 if owns_position else 0,
        "time_in_position_minutes_raw": 0.0,
        "time_of_day": clamp(
            (
                timestamp.astimezone(EASTERN).hour * 60
                + timestamp.astimezone(EASTERN).minute
                - (9 * 60 + 30)
            ) / 390.0,
            0.0,
            1.0,
        ),
        "minutes_to_close_raw": minutes_to_close_raw(timestamp),
    }

    normalized = normalize_features(raw)

    return {
        "symbol": signal.get("symbol"),
        "timestamp": timestamp,
        "price": safe_float(signal.get("price")),
        "raw": raw,
        "features": normalized,
        "state_vector": feature_vector_from_normalized(normalized),
        "owns_position": owns_position,
        "can_open_new_position": can_open_new_positions_at(timestamp),
    }
