"""
rl_reward.py

Reward / punishment engine.

The agent is judged AFTER the market has had time to reveal whether its action
was good:
- BUY: did price rise, and could it have bought meaningfully lower?
- WAIT: did waiting avoid a drop or miss a strong move?
- SELL: did price fall afterward, or did it sell too early?
- Completed trade: profit, entry regret, exit regret, drawdown and opportunity
  capture are combined into a final trade reward.

No future information is exposed to the decision at action time. Future prices
are used only later for training.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from db import db

from rl_config import (
    ACTION_BUY,
    ACTION_NAMES,
    ACTION_SELL,
    ACTION_WAIT,
    BUY_REGRET_WEIGHT,
    BUY_REGRET_WINDOW_MINUTES,
    BUY_REWARD_WINDOW_MINUTES,
    CAPTURE_EFFICIENCY_WEIGHT,
    DRAWDOWN_WEIGHT,
    MAX_REWARD,
    MIN_REWARD,
    POST_SELL_PROTECTION_WEIGHT,
    POST_SELL_REGRET_WINDOW_MINUTES,
    PROFIT_REWARD_WEIGHT,
    SELL_REGRET_WEIGHT,
    SELL_REWARD_WINDOW_MINUTES,
    WAIT_REWARD_WINDOW_MINUTES,
)
from rl_state import (
    aware_utc,
    legacy_signal_to_state,
    safe_float,
    state_from_history_doc,
)
from rl_prediction import record_prediction


EASTERN = ZoneInfo("America/New_York")

agent_trades = db["agent_trades"]
rl_market_history = db["rl_market_history"]
rl_pending_actions = db["rl_pending_actions"]
rl_experiences = db["rl_experiences"]
rl_trade_reviews = db["rl_trade_reviews"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scaled_tanh(percent_value: float, scale_percent: float) -> float:
    if scale_percent <= 0:
        return 0.0
    return math.tanh(percent_value / scale_percent)


def pct_change(new_value: float, old_value: float) -> float:
    if old_value is None or old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def _diagnose_reward_components(
    components: dict[str, float],
    labels: dict[str, str],
    reward: float,
) -> dict:
    """Turn signed reward contributions into an inspectable primary lesson."""

    if not components:
        return {
            "primary_reason": "NO_DIAGNOSIS",
            "primary_component": None,
            "primary_component_value": 0.0,
        }

    if reward < -0.25:
        key = min(
            components,
            key=lambda name: components[name],
        )
    elif reward > 0.25:
        key = max(
            components,
            key=lambda name: components[name],
        )
    else:
        key = max(
            components,
            key=lambda name: abs(components[name]),
        )

    return {
        "primary_reason": labels.get(
            key,
            key.upper(),
        ),
        "primary_component": key,
        "primary_component_value": float(
            components.get(key, 0.0)
        ),
    }


def _market_close_after(timestamp: datetime) -> datetime:
    timestamp = aware_utc(timestamp) or utc_now()
    local = timestamp.astimezone(EASTERN)

    close_local = local.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    return close_local.astimezone(timezone.utc)


def evaluation_deadline(
    timestamp: datetime,
    minutes: int,
) -> datetime:

    timestamp = aware_utc(timestamp) or utc_now()

    normal_deadline = timestamp + timedelta(
        minutes=minutes
    )

    market_close = _market_close_after(
        timestamp
    )

    # If action occurred after market close for some reason, do not create a
    # deadline earlier than the action itself.
    if market_close <= timestamp:
        return normal_deadline

    return min(
        normal_deadline,
        market_close,
    )


def _price_path(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[dict]:

    return list(
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
                "state_vector": 1,
                "owns_position": 1,
                "can_open_new_position": 1,
                "features": 1,
                "raw": 1,
                "symbol": 1,
            },
        ).sort("timestamp", 1)
    )


def _next_valid_actions_from_history(
    doc: dict,
) -> list[int]:

    owns = bool(
        doc.get("owns_position")
    )

    can_open = bool(
        doc.get("can_open_new_position")
    )

    valid = [ACTION_WAIT]

    if can_open:
        valid.append(ACTION_BUY)

    if owns:
        valid.append(ACTION_SELL)

    return sorted(set(valid))


def _insert_experience(
    *,
    experience_id: str,
    symbol: str,
    timestamp: datetime,
    state_vector: list[float],
    action_index: int,
    reward: float,
    next_state_vector: list[float],
    next_valid_actions: list[int],
    done: bool,
    source: str,
    metadata: dict | None = None,
) -> bool:

    reward = clamp(
        float(reward),
        MIN_REWARD,
        MAX_REWARD,
    )

    result = rl_experiences.update_one(
        {
            "experience_id":
                experience_id
        },
        {
            "$setOnInsert": {
                "experience_id":
                    experience_id,
                "symbol":
                    symbol,
                "timestamp":
                    timestamp,
                "state_vector":
                    state_vector,
                "action_index":
                    int(action_index),
                "action":
                    ACTION_NAMES[
                        int(action_index)
                    ],
                "reward":
                    reward,
                "next_state_vector":
                    next_state_vector,
                "next_valid_actions":
                    next_valid_actions,
                "done":
                    bool(done),
                "source":
                    source,
                "metadata":
                    metadata or {},
                "created_at":
                    utc_now(),
            }
        },
        upsert=True,
    )

    return (
        result.upserted_id
        is not None
    )


# =========================================================
# PENDING ACTIONS
# =========================================================

def record_pending_action(
    *,
    state: dict,
    action_index: int,
    decision: dict,
    trade_id: str | None = None,
    trainable: bool = True,
    extra: dict | None = None,
) -> str:

    timestamp = aware_utc(
        state.get("timestamp")
    ) or utc_now()

    if action_index == ACTION_SELL:
        minutes = SELL_REWARD_WINDOW_MINUTES
    elif action_index == ACTION_BUY:
        minutes = BUY_REWARD_WINDOW_MINUTES
    else:
        minutes = WAIT_REWARD_WINDOW_MINUTES

    action_id = str(uuid4())

    pending_doc = {
            "action_id":
                action_id,
            "symbol":
                state["symbol"],
            "timestamp":
                timestamp,
            "due_at":
                evaluation_deadline(
                    timestamp,
                    minutes,
                ),
            "action_index":
                action_index,
            "action":
                ACTION_NAMES[
                    action_index
                ],
            "state_vector":
                state.get(
                    "state_vector",
                    [],
                ),
            "state_snapshot":
                state,
            "price":
                safe_float(
                    state.get("price")
                ),
            "owns_position":
                bool(
                    state.get(
                        "owns_position"
                    )
                ),
            "trade_id":
                trade_id,
            "decision":
                decision,
            "extra":
                extra or {},
            "trainable":
                bool(trainable),
            "status":
                "pending",
            "created_at":
                utc_now(),
        }

    rl_pending_actions.insert_one(
        pending_doc
    )

    prediction_id = record_prediction(
        state=state,
        action_id=action_id,
        action_index=action_index,
        action=ACTION_NAMES[action_index],
        prediction=decision.get("prediction"),
        trainable=trainable,
        trade_id=trade_id,
    )

    if prediction_id:
        rl_pending_actions.update_one(
            {"action_id": action_id},
            {
                "$set": {
                    "prediction_id": prediction_id
                }
            },
        )

    return action_id


def _score_buy_action(
    start_price: float,
    path_prices: list[float],
) -> tuple[float, dict]:

    if not path_prices or start_price <= 0:
        return 0.0, {}

    end_price = path_prices[-1]
    low = min(path_prices)
    high = max(path_prices)

    end_return = pct_change(
        end_price,
        start_price,
    )

    buy_regret = max(
        0.0,
        (
            (start_price - low)
            / start_price
            * 100.0
        ),
    )

    # pct_change(start, low) is positive when low < start.
    # The helper's argument order makes this directly the percent by which the
    # agent could have entered lower.
    upside = max(
        0.0,
        pct_change(
            high,
            start_price,
        ),
    )

    components = {
        "future_return": (
            3.0 * scaled_tanh(
                end_return,
                1.5,
            )
        ),
        "available_upside": (
            1.0 * scaled_tanh(
                upside,
                2.0,
            )
        ),
        "buy_regret_penalty": (
            -1.5 * scaled_tanh(
                buy_regret,
                1.0,
            )
        ),
    }

    reward = sum(components.values())

    diagnosis = _diagnose_reward_components(
        components,
        {
            "future_return": "PRICE_MOVED_AFTER_ENTRY",
            "available_upside": "ENTRY_LEFT_UPSIDE_AVAILABLE",
            "buy_regret_penalty": "BOUGHT_TOO_EARLY_OR_TOO_HIGH",
        },
        reward,
    )

    return reward, {
        "end_return_percent": end_return,
        "buy_regret_percent": buy_regret,
        "best_upside_percent": upside,
        "future_low": low,
        "future_high": high,
        "future_end": end_price,
        "reward_components": components,
        **diagnosis,
    }


def _score_wait_action(
    start_price: float,
    path_prices: list[float],
    owned_at_action: bool,
) -> tuple[float, dict]:

    if not path_prices or start_price <= 0:
        return 0.0, {}

    end_price = path_prices[-1]
    low = min(path_prices)
    high = max(path_prices)

    end_return = pct_change(
        end_price,
        start_price,
    )

    best_upside = max(
        0.0,
        pct_change(
            high,
            start_price,
        ),
    )

    downside = max(
        0.0,
        (
            (start_price - low)
            / start_price
            * 100.0
        ),
    )

    if owned_at_action:
        # WAIT means HOLD when already in a position.
        components = {
            "hold_return": (
                3.0 * scaled_tanh(
                    end_return,
                    1.5,
                )
            ),
            "drawdown_penalty": (
                -0.75 * scaled_tanh(
                    downside,
                    1.5,
                )
            ),
        }
        labels = {
            "hold_return": "HOLD_PRICE_OUTCOME",
            "drawdown_penalty": "HELD_THROUGH_DRAWDOWN",
        }
    else:
        # WAIT means "stay out". Reward avoiding a drop; punish missing a
        # strong upward move.
        components = {
            "downside_avoided": (
                2.0 * scaled_tanh(
                    downside,
                    1.5,
                )
            ),
            "missed_upside_penalty": (
                -2.5 * scaled_tanh(
                    best_upside,
                    1.5,
                )
            ),
        }
        labels = {
            "downside_avoided": "WAIT_AVOIDED_DOWNSIDE",
            "missed_upside_penalty": "WAIT_MISSED_UPSIDE",
        }

    reward = sum(components.values())
    diagnosis = _diagnose_reward_components(
        components,
        labels,
        reward,
    )

    return reward, {
        "owned_at_action": owned_at_action,
        "end_return_percent": end_return,
        "best_upside_percent": best_upside,
        "downside_percent": downside,
        "future_low": low,
        "future_high": high,
        "future_end": end_price,
        "reward_components": components,
        **diagnosis,
    }


def _score_sell_action(
    start_price: float,
    path_prices: list[float],
    realized_return_percent: float | None,
) -> tuple[float, dict]:

    if not path_prices or start_price <= 0:
        return 0.0, {}

    high = max(path_prices)
    low = min(path_prices)
    end_price = path_prices[-1]

    sold_too_early = max(
        0.0,
        pct_change(
            high,
            start_price,
        ),
    )

    protected_downside = max(
        0.0,
        (
            (start_price - low)
            / start_price
            * 100.0
        ),
    )

    components = {
        "downside_protection": (
            2.5 * scaled_tanh(
                protected_downside,
                1.5,
            )
        ),
        "sell_regret_penalty": (
            -3.0 * scaled_tanh(
                sold_too_early,
                1.5,
            )
        ),
        "realized_return": 0.0,
    }

    if realized_return_percent is not None:
        components["realized_return"] = (
            2.0 * scaled_tanh(
                realized_return_percent,
                1.5,
            )
        )

    reward = sum(components.values())

    diagnosis = _diagnose_reward_components(
        components,
        {
            "downside_protection": "SELL_PROTECTED_DOWNSIDE",
            "sell_regret_penalty": "SOLD_TOO_EARLY",
            "realized_return": "REALIZED_TRADE_OUTCOME",
        },
        reward,
    )

    return reward, {
        "realized_return_percent": realized_return_percent,
        "sell_regret_percent": sold_too_early,
        "protected_downside_percent": protected_downside,
        "future_low": low,
        "future_high": high,
        "future_end": end_price,
        "reward_components": components,
        **diagnosis,
    }


def process_due_action_rewards(
    now: datetime | None = None,
) -> dict:

    if now is None:
        now = utc_now()

    pending = list(
        rl_pending_actions.find(
            {
                "status":
                    "pending",
                "due_at": {
                    "$lte":
                        now
                },
            }
        ).sort(
            "due_at",
            1
        )
    )

    processed = 0
    experiences_added = 0
    skipped = 0

    for item in pending:
        start = aware_utc(
            item.get("timestamp")
        )

        end = aware_utc(
            item.get("due_at")
        )

        if start is None or end is None:
            rl_pending_actions.update_one(
                {"_id": item["_id"]},
                {
                    "$set": {
                        "status":
                            "invalid",
                        "processed_at":
                            now,
                    }
                },
            )
            skipped += 1
            continue

        path = _price_path(
            item["symbol"],
            start,
            end,
        )

        # Need a later observation to score the action fairly.
        if len(path) < 2:
            rl_pending_actions.update_one(
                {"_id": item["_id"]},
                {
                    "$set": {
                        "due_at":
                            now
                            + timedelta(
                                minutes=3
                            ),
                        "last_retry_at":
                            now,
                    }
                },
            )
            skipped += 1
            continue

        prices = [
            safe_float(row.get("price"))
            for row in path
            if safe_float(row.get("price")) > 0
        ]

        if len(prices) < 2:
            skipped += 1
            continue

        action_index = int(
            item["action_index"]
        )

        start_price = safe_float(
            item.get("price")
        )

        if action_index == ACTION_BUY:
            reward, metrics = _score_buy_action(
                start_price,
                prices,
            )
            done = False

        elif action_index == ACTION_SELL:
            reward, metrics = _score_sell_action(
                start_price,
                prices,
                item.get(
                    "extra",
                    {},
                ).get(
                    "realized_return_percent"
                ),
            )
            done = True

        else:
            reward, metrics = _score_wait_action(
                start_price,
                prices,
                bool(
                    item.get(
                        "owns_position"
                    )
                ),
            )
            done = False

        next_doc = path[-1]
        next_state = state_from_history_doc(
            next_doc
        )

        trainable = bool(
            item.get(
                "trainable",
                True,
            )
        )

        added = False

        if trainable:
            added = _insert_experience(
                experience_id=(
                    f"action:{item['action_id']}"
                ),
                symbol=item["symbol"],
                timestamp=start,
                state_vector=item.get(
                    "state_vector",
                    [],
                ),
                action_index=action_index,
                reward=reward,
                next_state_vector=next_state.get(
                    "state_vector",
                    [],
                ),
                next_valid_actions=(
                    _next_valid_actions_from_history(
                        next_doc
                    )
                ),
                done=done,
                source="live_action_reward",
                metadata={
                    "action_id":
                        item["action_id"],
                    "trade_id":
                        item.get("trade_id"),
                    "metrics":
                        metrics,
                },
            )

        rl_pending_actions.update_one(
            {"_id": item["_id"]},
            {
                "$set": {
                    "status":
                        "scored",
                    "reward":
                        clamp(
                            reward,
                            MIN_REWARD,
                            MAX_REWARD,
                        ),
                    "metrics":
                        metrics,
                    "processed_at":
                        now,
                    "experience_added":
                        added,
                }
            },
        )

        processed += 1

        if added:
            experiences_added += 1

    return {
        "processed":
            processed,
        "experiences_added":
            experiences_added,
        "skipped":
            skipped,
    }


# =========================================================
# FULL TRADE REVIEW
# =========================================================

def create_trade_review(
    sell_trade: dict,
) -> str | None:

    symbol = sell_trade.get("symbol")
    sell_timestamp = aware_utc(
        sell_trade.get("timestamp")
    )

    if not symbol or sell_timestamp is None:
        return None

    # Locate the prior SELL. BUYs after it belong to the position that was just
    # closed. This supports multiple BUY/add-on entries.
    previous_sell = agent_trades.find_one(
        {
            "symbol":
                symbol,
            "side":
                "SELL",
            "timestamp": {
                "$lt":
                    sell_timestamp
            },
        },
        {
            "_id": 0,
            "timestamp": 1,
        },
        sort=[
            ("timestamp", -1)
        ],
    )

    buy_query = {
        "symbol":
            symbol,
        "side":
            "BUY",
        "timestamp": {
            "$lte":
                sell_timestamp
        },
    }

    if previous_sell:
        buy_query[
            "timestamp"
        ][
            "$gt"
        ] = previous_sell[
            "timestamp"
        ]

    buy_trades = list(
        agent_trades.find(
            buy_query,
            {
                "_id": 0,
            },
        ).sort(
            "timestamp",
            1
        )
    )

    if not buy_trades:
        return None

    opened_at = aware_utc(
        buy_trades[0].get(
            "timestamp"
        )
    )

    if opened_at is None:
        return None

    review_id = str(uuid4())

    due_at = evaluation_deadline(
        sell_timestamp,
        POST_SELL_REGRET_WINDOW_MINUTES,
    )

    rl_trade_reviews.insert_one(
        {
            "review_id":
                review_id,
            "symbol":
                symbol,
            "sell_trade_id":
                sell_trade.get(
                    "trade_id"
                ),
            "sell_timestamp":
                sell_timestamp,
            "sell_price":
                safe_float(
                    sell_trade.get(
                        "price"
                    )
                ),
            "entry_price":
                safe_float(
                    sell_trade.get(
                        "entry_price"
                    )
                ),
            "shares":
                int(
                    sell_trade.get(
                        "shares"
                    )
                    or 0
                ),
            "realized_pnl":
                safe_float(
                    sell_trade.get(
                        "realized_pnl"
                    )
                ),
            "opened_at":
                opened_at,
            "buy_trade_ids": [
                row.get("trade_id")
                for row in buy_trades
                if row.get("trade_id")
            ],
            "due_at":
                due_at,
            "status":
                "pending",
            "created_at":
                utc_now(),
        }
    )

    return review_id


def _trade_reward(
    *,
    entry_price: float,
    sell_price: float,
    entry_window_low: float,
    held_low: float,
    held_high: float,
    future_high: float,
    future_low: float,
) -> tuple[float, int, dict]:

    realized_return = pct_change(
        sell_price,
        entry_price,
    )

    buy_regret = max(
        0.0,
        (
            (entry_price - entry_window_low)
            / entry_price
            * 100.0
        ),
    )

    sell_regret = max(
        0.0,
        pct_change(
            future_high,
            sell_price,
        ),
    )

    drawdown = max(
        0.0,
        (
            (entry_price - held_low)
            / entry_price
            * 100.0
        ),
    )

    post_sell_protection = max(
        0.0,
        (
            (sell_price - future_low)
            / sell_price
            * 100.0
        ),
    )

    max_possible_gain = max(
        0.0,
        held_high - entry_price,
    )

    realized_gain = (
        sell_price - entry_price
    )

    if max_possible_gain > 0:
        capture_efficiency = clamp(
            realized_gain
            / max_possible_gain,
            -1.0,
            1.0,
        )
    elif realized_gain >= 0:
        capture_efficiency = 1.0
    else:
        capture_efficiency = -1.0

    profit_score = scaled_tanh(
        realized_return,
        1.5,
    )

    components = {
        "profit_component": (
            PROFIT_REWARD_WEIGHT
            * profit_score
        ),
        "buy_regret_penalty": (
            -BUY_REGRET_WEIGHT
            * scaled_tanh(
                buy_regret,
                1.0,
            )
        ),
        "sell_regret_penalty": (
            -SELL_REGRET_WEIGHT
            * scaled_tanh(
                sell_regret,
                1.5,
            )
        ),
        "drawdown_penalty": (
            -DRAWDOWN_WEIGHT
            * scaled_tanh(
                drawdown,
                1.5,
            )
        ),
        "capture_efficiency_component": (
            CAPTURE_EFFICIENCY_WEIGHT
            * capture_efficiency
        ),
        "post_sell_protection_component": (
            POST_SELL_PROTECTION_WEIGHT
            * scaled_tanh(
                post_sell_protection,
                1.5,
            )
        ),
    }

    unclamped_reward = sum(
        components.values()
    )

    reward = clamp(
        unclamped_reward,
        MIN_REWARD,
        MAX_REWARD,
    )

    diagnosis = _diagnose_reward_components(
        components,
        {
            "profit_component": "REALIZED_PROFIT_OR_LOSS",
            "buy_regret_penalty": "BOUGHT_TOO_EARLY_OR_TOO_HIGH",
            "sell_regret_penalty": "SOLD_TOO_EARLY",
            "drawdown_penalty": "EXCESSIVE_DRAWDOWN",
            "capture_efficiency_component": "MOVE_CAPTURE_QUALITY",
            "post_sell_protection_component": "EXIT_PROTECTED_DOWNSIDE",
        },
        reward,
    )

    trade_score = int(
        round(
            clamp(
                (reward - MIN_REWARD)
                / (MAX_REWARD - MIN_REWARD)
                * 100.0,
                0.0,
                100.0,
            )
        )
    )

    metrics = {
        "realized_return_percent":
            realized_return,
        "buy_regret_percent":
            buy_regret,
        "sell_regret_percent":
            sell_regret,
        "max_drawdown_percent":
            drawdown,
        "post_sell_protection_percent":
            post_sell_protection,
        "capture_efficiency":
            capture_efficiency,
        "entry_window_low":
            entry_window_low,
        "held_low":
            held_low,
        "held_high":
            held_high,
        "future_high":
            future_high,
        "future_low":
            future_low,
        "reward_components":
            components,
        "unclamped_reward":
            unclamped_reward,
        **diagnosis,
    }

    return reward, trade_score, metrics


def process_due_trade_reviews(
    now: datetime | None = None,
) -> dict:

    if now is None:
        now = utc_now()

    reviews = list(
        rl_trade_reviews.find(
            {
                "status":
                    "pending",
                "due_at": {
                    "$lte":
                        now
                },
            }
        ).sort(
            "due_at",
            1
        )
    )

    finalized = 0
    experiences_added = 0
    skipped = 0

    for review in reviews:
        symbol = review["symbol"]

        opened_at = aware_utc(
            review.get("opened_at")
        )

        sell_time = aware_utc(
            review.get("sell_timestamp")
        )

        due_at = aware_utc(
            review.get("due_at")
        )

        if (
            opened_at is None
            or sell_time is None
            or due_at is None
        ):
            skipped += 1
            continue

        held_path = _price_path(
            symbol,
            opened_at,
            sell_time,
        )

        future_path = _price_path(
            symbol,
            sell_time,
            due_at,
        )

        if not held_path or not future_path:
            rl_trade_reviews.update_one(
                {"_id": review["_id"]},
                {
                    "$set": {
                        "due_at":
                            now
                            + timedelta(
                                minutes=3
                            ),
                        "last_retry_at":
                            now,
                    }
                },
            )
            skipped += 1
            continue

        held_prices = [
            safe_float(row.get("price"))
            for row in held_path
            if safe_float(row.get("price")) > 0
        ]

        future_prices = [
            safe_float(row.get("price"))
            for row in future_path
            if safe_float(row.get("price")) > 0
        ]

        if not held_prices or not future_prices:
            skipped += 1
            continue

        entry_price = safe_float(
            review.get("entry_price")
        )

        sell_price = safe_float(
            review.get("sell_price")
        )

        if entry_price <= 0 or sell_price <= 0:
            skipped += 1
            continue

        entry_window_end = min(
            sell_time,
            opened_at
            + timedelta(
                minutes=BUY_REGRET_WINDOW_MINUTES
            ),
        )

        entry_path = _price_path(
            symbol,
            opened_at,
            entry_window_end,
        )

        entry_prices = [
            safe_float(row.get("price"))
            for row in entry_path
            if safe_float(row.get("price")) > 0
        ]

        if not entry_prices:
            entry_prices = [
                entry_price
            ]

        reward, trade_score, metrics = _trade_reward(
            entry_price=entry_price,
            sell_price=sell_price,
            entry_window_low=min(entry_prices),
            held_low=min(held_prices),
            held_high=max(held_prices),
            future_high=max(future_prices),
            future_low=min(future_prices),
        )

        sell_trade = agent_trades.find_one(
            {
                "trade_id":
                    review.get(
                        "sell_trade_id"
                    )
            }
        ) or {}

        sell_signal = (
            sell_trade.get(
                "state_snapshot"
            )
            or sell_trade.get(
                "signal"
            )
            or {}
        )

        if sell_trade.get("state_vector"):
            next_vector = (
                sell_trade[
                    "state_vector"
                ]
            )
        else:
            next_vector = (
                legacy_signal_to_state(
                    sell_signal,
                    owns_position=True,
                )[
                    "state_vector"
                ]
            )

        # Give each BUY that built this position a full hindsight outcome.
        for buy_trade_id in review.get(
            "buy_trade_ids",
            []
        ):
            buy_trade = agent_trades.find_one(
                {
                    "trade_id":
                        buy_trade_id
                }
            ) or {}

            if buy_trade.get("state_vector"):
                state_vector = (
                    buy_trade[
                        "state_vector"
                    ]
                )
            else:
                buy_signal = (
                    buy_trade.get(
                        "state_snapshot"
                    )
                    or buy_trade.get(
                        "signal"
                    )
                    or {}
                )

                state_vector = (
                    legacy_signal_to_state(
                        buy_signal,
                        owns_position=False,
                    )[
                        "state_vector"
                    ]
                )

            added = _insert_experience(
                experience_id=(
                    "trade_review:"
                    f"{buy_trade_id}:"
                    f"{review.get('sell_trade_id')}"
                ),
                symbol=symbol,
                timestamp=opened_at,
                state_vector=state_vector,
                action_index=ACTION_BUY,
                reward=reward,
                next_state_vector=next_vector,
                next_valid_actions=[
                    ACTION_WAIT,
                    ACTION_BUY,
                ],
                done=True,
                source="completed_trade_review",
                metadata={
                    "review_id":
                        review["review_id"],
                    "sell_trade_id":
                        review.get(
                            "sell_trade_id"
                        ),
                    "metrics":
                        metrics,
                    "trade_score":
                        trade_score,
                },
            )

            if added:
                experiences_added += 1

        rl_trade_reviews.update_one(
            {"_id": review["_id"]},
            {
                "$set": {
                    "status":
                        "finalized",
                    "reward":
                        reward,
                    "trade_score":
                        trade_score,
                    "metrics":
                        metrics,
                    "finalized_at":
                        now,
                }
            },
        )

        agent_trades.update_one(
            {
                "trade_id":
                    review.get(
                        "sell_trade_id"
                    )
            },
            {
                "$set": {
                    "rl_reward":
                        reward,
                    "trade_score":
                        trade_score,
                    "rl_review_metrics":
                        metrics,
                    "rl_review_id":
                        review[
                            "review_id"
                        ],
                }
            },
        )

        finalized += 1

    return {
        "finalized":
            finalized,
        "experiences_added":
            experiences_added,
        "skipped":
            skipped,
    }
