"""
trading_agent.py

Autonomous intraday PyTorch RL paper-trading agent.

Public API intentionally matches the earlier rule-based trading_agent.py:
    run_agent()
    get_agent_dashboard()
    set_agent_enabled()
    reset_agent()

That lets the existing Flask routes continue to work with minimal changes.

IMPORTANT:
- Paper trading only.
- Existing FinBERT news sentiment remains unchanged.
- The RL agent observes current data every collector cycle, but can choose WAIT.
- Hard portfolio/risk limits remain outside the neural network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from db import db, companies, headlines, market_data

from rl_brain import (
    acquire_model_lock,
    get_brain,
    get_model_status,
    release_model_lock,
)
from rl_config import (
    ACTION_BUY,
    ACTION_NAMES,
    ACTION_SELL,
    ACTION_WAIT,
    BUY_FRACTION,
    FORCE_CLOSE_HOUR,
    FORCE_CLOSE_MINUTE,
    HARD_STOP_LOSS_PERCENT,
    LAST_ENTRY_HOUR,
    LAST_ENTRY_MINUTE,
    MAX_POSITION_FRACTION,
    MAX_POSITIONS,
    MAX_TRADES_PER_OBSERVATION,
    MIN_VOLUME,
    STARTING_CASH,
    TRAIN_STEPS_PER_RUN,
)
from rl_reward import (
    create_trade_review,
    process_due_action_rewards,
    process_due_trade_reviews,
    record_pending_action,
)
from rl_prediction import process_due_predictions
from rl_state import (
    aware_utc,
    build_portfolio_context,
    build_state,
    can_open_new_positions_at,
    market_is_open_at,
    record_market_state,
    safe_float,
)


EASTERN = ZoneInfo("America/New_York")
AGENT_ID = "paper_agent"

agent_state = db["agent_state"]
agent_positions = db["agent_positions"]
agent_trades = db["agent_trades"]
agent_decisions = db["agent_decisions"]

rl_market_history = db["rl_market_history"]
rl_pending_actions = db["rl_pending_actions"]
rl_experiences = db["rl_experiences"]
rl_trade_reviews = db["rl_trade_reviews"]
rl_predictions = db["rl_predictions"]
rl_model_state = db["rl_model_state"]


ALLOWED_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "NYSE AMERICAN",
    "AMEX",
}


# =========================================================
# INDEXES
# =========================================================

def ensure_agent_indexes() -> None:

    agent_positions.create_index(
        [("symbol", 1)],
        unique=True,
    )

    # Match the existing project's trade_id index options exactly so MongoDB
    # does not raise IndexOptionsConflict when this file replaces the old agent.
    agent_trades.create_index(
        [("trade_id", 1)],
        unique=True,
    )

    agent_trades.create_index(
        [
            ("symbol", 1),
            ("timestamp", -1),
        ]
    )

    agent_trades.create_index(
        [("timestamp", -1)]
    )

    agent_decisions.create_index(
        [("symbol", 1)],
        unique=True,
    )

    rl_market_history.create_index(
        [
            ("symbol", 1),
            ("timestamp", 1),
        ],
        unique=True,
    )

    # Intentionally NO TTL: this is the AI's permanent memory.
    rl_market_history.create_index(
        [("timestamp", -1)]
    )

    rl_pending_actions.create_index(
        [("action_id", 1)],
        unique=True,
    )

    rl_pending_actions.create_index(
        [
            ("status", 1),
            ("due_at", 1),
        ]
    )

    # Pending action records are only temporary plumbing. The actual scored
    # experience remains permanently in rl_experiences.
    rl_pending_actions.create_index(
        [("processed_at", 1)],
        expireAfterSeconds=7 * 24 * 60 * 60,
        name="rl_pending_actions_7d_ttl",
    )

    rl_experiences.create_index(
        [("experience_id", 1)],
        unique=True,
    )

    rl_experiences.create_index(
        [("action_index", 1)]
    )

    rl_experiences.create_index(
        [("created_at", -1)]
    )

    rl_trade_reviews.create_index(
        [("review_id", 1)],
        unique=True,
    )

    rl_trade_reviews.create_index(
        [
            ("status", 1),
            ("due_at", 1),
        ]
    )

    rl_predictions.create_index(
        [("prediction_id", 1)],
        unique=True,
    )

    rl_predictions.create_index(
        [
            ("status", 1),
            ("next_due_at", 1),
        ]
    )

    rl_predictions.create_index(
        [
            ("symbol", 1),
            ("timestamp", -1),
        ]
    )

    # Prediction history is permanent training memory: intentionally NO TTL.

    rl_model_state.create_index(
        [("updated_at", -1)]
    )


# =========================================================
# TIME / MARKET HELPERS
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def eastern_now() -> datetime:
    return datetime.now(EASTERN)


def market_is_open() -> bool:
    return market_is_open_at(
        utc_now()
    )


def can_open_new_positions() -> bool:
    return can_open_new_positions_at(
        utc_now()
    )


def must_close_all_positions(
    now: datetime | None = None,
) -> bool:

    if now is None:
        now = utc_now()

    now_et = now.astimezone(EASTERN)

    if now_et.weekday() >= 5:
        return False

    force_close = now_et.replace(
        hour=FORCE_CLOSE_HOUR,
        minute=FORCE_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )

    market_close = now_et.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    return (
        force_close
        <= now_et
        < market_close
    )


def format_et(value) -> str:
    dt = aware_utc(value)

    if dt is None:
        return "N/A"

    return (
        dt.astimezone(EASTERN)
        .strftime(
            "%b %d, %Y %I:%M %p ET"
        )
    )


# =========================================================
# AGENT STATE
# =========================================================

def get_agent_state() -> dict:

    state = agent_state.find_one(
        {"_id": AGENT_ID}
    )

    if state:
        return state

    now = utc_now()

    state = {
        "_id":
            AGENT_ID,
        "enabled":
            False,
        "starting_cash":
            STARTING_CASH,
        "cash":
            STARTING_CASH,
        "session_date":
            eastern_now()
            .date()
            .isoformat(),
        "day_start_value":
            STARTING_CASH,
        "created_at":
            now,
        "updated_at":
            now,
        "last_run":
            None,
        "last_status":
            "PyTorch RL paper agent created",
    }

    agent_state.insert_one(
        state
    )

    return state


def set_agent_enabled(enabled: bool) -> None:

    get_agent_state()

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$set": {
                "enabled":
                    bool(enabled),
                "updated_at":
                    utc_now(),
                "last_status":
                    (
                        "RL agent enabled"
                        if enabled
                        else
                        "RL agent stopped"
                    ),
            }
        },
    )


# =========================================================
# SYMBOL UNIVERSE
# =========================================================

def get_active_universe() -> list[str]:

    raw_symbols = headlines.distinct(
        "securities.symbol"
    )

    symbols = []

    for raw in raw_symbols:
        if not isinstance(raw, str):
            continue

        symbol = raw.strip().upper()

        if not symbol:
            continue

        company = companies.find_one(
            {"symbol": symbol},
            {
                "_id": 0,
                "exchange": 1,
                "asset_type": 1,
            },
        ) or {}

        exchange = str(
            company.get("exchange")
            or ""
        ).strip().upper()

        if exchange not in ALLOWED_EXCHANGES:
            continue

        symbols.append(
            symbol
        )

    return sorted(
        set(symbols)
    )


# =========================================================
# PORTFOLIO
# =========================================================

def get_portfolio_snapshot() -> dict:

    state = get_agent_state()

    portfolio = (
        build_portfolio_context()
    )

    cash = safe_float(
        portfolio.get("cash")
    )

    positions_value = safe_float(
        portfolio.get(
            "positions_value"
        )
    )

    portfolio_value = safe_float(
        portfolio.get(
            "portfolio_value"
        )
    )

    rows = []

    unrealized_total = 0.0

    for symbol, position in (
        portfolio[
            "positions"
        ].items()
    ):

        shares = int(
            position.get(
                "shares"
            )
            or 0
        )

        avg_price = safe_float(
            position.get(
                "avg_price"
            )
        )

        current_price = safe_float(
            position.get(
                "current_price"
            )
        )

        market_value = safe_float(
            position.get(
                "market_value"
            )
        )

        cost_basis = (
            shares
            * avg_price
        )

        unrealized_pnl = (
            market_value
            - cost_basis
        )

        unrealized_total += (
            unrealized_pnl
        )

        if cost_basis:
            pnl_percent = (
                unrealized_pnl
                / cost_basis
                * 100.0
            )
        else:
            pnl_percent = 0.0

        rows.append(
            {
                "symbol":
                    symbol,
                "shares":
                    shares,
                "avg_price":
                    avg_price,
                "current_price":
                    current_price,
                "market_value":
                    market_value,
                "unrealized_pnl":
                    unrealized_pnl,
                "pnl_percent":
                    pnl_percent,
                "opened_at":
                    position.get(
                        "opened_at"
                    ),
                "opened_at_display":
                    format_et(
                        position.get(
                            "opened_at"
                        )
                    ),
            }
        )

    starting_cash = safe_float(
        state.get(
            "starting_cash"
        ),
        STARTING_CASH,
    )

    total_pnl = (
        portfolio_value
        - starting_cash
    )

    total_pnl_percent = (
        total_pnl
        / starting_cash
        * 100.0
        if starting_cash
        else 0.0
    )

    return {
        "cash":
            cash,
        "positions_value":
            positions_value,
        "portfolio_value":
            portfolio_value,
        "starting_cash":
            starting_cash,
        "total_pnl":
            total_pnl,
        "total_pnl_percent":
            total_pnl_percent,
        "unrealized_pnl":
            unrealized_total,
        "positions":
            sorted(
                rows,
                key=lambda row:
                    row["symbol"],
            ),
    }


def ensure_daily_session() -> bool:

    state = get_agent_state()

    today = (
        eastern_now()
        .date()
        .isoformat()
    )

    if (
        state.get(
            "session_date"
        )
        == today
    ):
        return False

    portfolio = (
        get_portfolio_snapshot()
    )

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$set": {
                "session_date":
                    today,
                "day_start_value":
                    portfolio[
                        "portfolio_value"
                    ],
                "updated_at":
                    utc_now(),
                "last_status":
                    "New trading day started",
            }
        },
    )

    return True


# =========================================================
# POSITION / ACTION MASKING
# =========================================================

def position_value(
    symbol: str,
    price: float,
) -> float:

    position = agent_positions.find_one(
        {"symbol": symbol}
    )

    if not position:
        return 0.0

    return (
        int(
            position.get(
                "shares"
            )
            or 0
        )
        * price
    )


def valid_actions_for_state(
    state: dict,
    portfolio: dict,
) -> list[int]:

    valid = [
        ACTION_WAIT
    ]

    symbol = state[
        "symbol"
    ]

    price = safe_float(
        state.get(
            "price"
        )
    )

    owns = bool(
        state.get(
            "owns_position"
        )
    )

    # BUY can be an initial entry or an add-on, but never beyond the 10%
    # portfolio cap.
    if state.get(
        "can_open_new_position"
    ):

        current_position_value = (
            position_value(
                symbol,
                price,
            )
        )

        max_position_value = (
            safe_float(
                portfolio.get(
                    "portfolio_value"
                )
            )
            * MAX_POSITION_FRACTION
        )

        existing_position_count = int(
            portfolio.get(
                "position_count"
            )
            or 0
        )

        can_add = (
            current_position_value
            < max_position_value
        )

        room_for_new_symbol = (
            owns
            or existing_position_count
            < MAX_POSITIONS
        )

        if (
            can_add
            and room_for_new_symbol
        ):
            valid.append(
                ACTION_BUY
            )

    if owns:
        valid.append(
            ACTION_SELL
        )

    return sorted(
        set(valid)
    )


# =========================================================
# PAPER EXECUTION
# =========================================================

def execute_buy(
    *,
    state: dict,
    decision: dict,
    now: datetime,
) -> dict | None:

    symbol = state[
        "symbol"
    ]

    price = safe_float(
        state.get(
            "price"
        )
    )

    if price <= 0:
        return None

    agent = get_agent_state()

    cash = safe_float(
        agent.get(
            "cash"
        )
    )

    portfolio = (
        build_portfolio_context()
    )

    portfolio_value = safe_float(
        portfolio.get(
            "portfolio_value"
        )
    )

    existing = agent_positions.find_one(
        {"symbol": symbol}
    )

    current_value = 0.0

    if existing:
        current_value = (
            int(
                existing.get(
                    "shares"
                )
                or 0
            )
            * price
        )

    max_position_value = (
        portfolio_value
        * MAX_POSITION_FRACTION
    )

    desired_purchase = (
        portfolio_value
        * BUY_FRACTION
    )

    available_room = (
        max_position_value
        - current_value
    )

    amount_to_spend = min(
        desired_purchase,
        available_room,
        cash,
    )

    shares = int(
        amount_to_spend
        // price
    )

    if shares < 1:
        return None

    trade_value = (
        shares
        * price
    )

    if existing:
        old_shares = int(
            existing.get(
                "shares"
            )
            or 0
        )

        old_avg = safe_float(
            existing.get(
                "avg_price"
            )
        )

        new_shares = (
            old_shares
            + shares
        )

        new_avg = (
            (
                old_shares
                * old_avg
            )
            + trade_value
        ) / new_shares

        agent_positions.update_one(
            {"symbol": symbol},
            {
                "$set": {
                    "shares":
                        new_shares,
                    "avg_price":
                        new_avg,
                    "last_price":
                        price,
                    "updated_at":
                        now,
                }
            },
        )

    else:
        agent_positions.insert_one(
            {
                "symbol":
                    symbol,
                "shares":
                    shares,
                "avg_price":
                    price,
                "last_price":
                    price,
                "opened_at":
                    now,
                "updated_at":
                    now,
            }
        )

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$inc": {
                "cash":
                    -trade_value
            },
            "$set": {
                "updated_at":
                    now
            },
        },
    )

    trade = {
        "trade_id":
            str(uuid4()),
        "timestamp":
            now,
        "symbol":
            symbol,
        "side":
            "BUY",
        "shares":
            shares,
        "price":
            price,
        "value":
            trade_value,
        "realized_pnl":
            None,
        "entry_price":
            price,
        "agent_type":
            "PYTORCH_DUELING_DDQN_V2",
        "state_vector":
            state.get(
                "state_vector",
                [],
            ),
        "state_snapshot":
            state,
        "rl_decision":
            decision,
        "reason":
            (
                "PyTorch RL chose BUY. "
                f"Q={decision.get('q_values')}; "
                f"epsilon={decision.get('epsilon', 0):.3f}; "
                f"explored={decision.get('explored', False)}"
            ),
    }

    agent_trades.insert_one(
        trade
    )

    record_pending_action(
        state=state,
        action_index=ACTION_BUY,
        decision=decision,
        trade_id=trade["trade_id"],
        trainable=True,
    )

    return trade


def execute_sell(
    *,
    state: dict,
    decision: dict,
    now: datetime,
    reason_override: str | None = None,
    trainable: bool = True,
) -> dict | None:

    symbol = state[
        "symbol"
    ]

    position = agent_positions.find_one(
        {"symbol": symbol}
    )

    if not position:
        return None

    shares = int(
        position.get(
            "shares"
        )
        or 0
    )

    if shares <= 0:
        return None

    price = safe_float(
        state.get(
            "price"
        )
    )

    avg_price = safe_float(
        position.get(
            "avg_price"
        )
    )

    if price <= 0 or avg_price <= 0:
        return None

    trade_value = (
        shares
        * price
    )

    realized_pnl = (
        price
        - avg_price
    ) * shares

    realized_return_percent = (
        (
            price
            - avg_price
        )
        / avg_price
        * 100.0
    )

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$inc": {
                "cash":
                    trade_value
            },
            "$set": {
                "updated_at":
                    now
            },
        },
    )

    agent_positions.delete_one(
        {"symbol": symbol}
    )

    trade = {
        "trade_id":
            str(uuid4()),
        "timestamp":
            now,
        "symbol":
            symbol,
        "side":
            "SELL",
        "shares":
            shares,
        "price":
            price,
        "value":
            trade_value,
        "entry_price":
            avg_price,
        "realized_pnl":
            realized_pnl,
        "realized_return_percent":
            realized_return_percent,
        "agent_type":
            "PYTORCH_DUELING_DDQN_V2",
        "state_vector":
            state.get(
                "state_vector",
                [],
            ),
        "state_snapshot":
            state,
        "rl_decision":
            decision,
        "reason":
            (
                reason_override
                or (
                    "PyTorch RL chose SELL. "
                    f"Q={decision.get('q_values')}; "
                    f"epsilon={decision.get('epsilon', 0):.3f}; "
                    f"explored={decision.get('explored', False)}"
                )
            ),
    }

    agent_trades.insert_one(
        trade
    )

    record_pending_action(
        state=state,
        action_index=ACTION_SELL,
        decision=decision,
        trade_id=trade["trade_id"],
        trainable=trainable,
        extra={
            "realized_return_percent":
                realized_return_percent,
        },
    )

    create_trade_review(
        trade
    )

    return trade


def force_close_all(
    state_map: dict[str, dict],
    now: datetime,
    reason: str,
    brain=None,
) -> list[dict]:

    trades = []

    for position in list(
        agent_positions.find({})
    ):
        symbol = position.get(
            "symbol"
        )

        state = state_map.get(
            symbol
        )

        if state is None:
            continue

        forced_decision = {
            "action_index":
                ACTION_SELL,
            "action":
                "SELL",
            "q_values": {
                "WAIT":
                    0.0,
                "BUY":
                    0.0,
                "SELL":
                    0.0,
            },
            "epsilon":
                0.0,
            "explored":
                False,
            "ready":
                True,
            "forced":
                True,
            "prediction": (
                brain.predict_future(
                    state.get("state_vector", [])
                )
                if brain is not None
                else None
            ),
        }

        trade = execute_sell(
            state=state,
            decision=forced_decision,
            now=now,
            reason_override=reason,
            trainable=False,
        )

        if trade:
            trades.append(
                trade
            )

    return trades


# =========================================================
# DECISION RECORD
# =========================================================

def save_decision(
    state: dict,
    decision: dict,
    executed_action: str | None = None,
    blocked_reason: str | None = None,
) -> None:

    q = decision.get(
        "q_values",
        {},
    )

    raw = state.get(
        "raw",
        {},
    )

    agent_decisions.update_one(
        {
            "symbol":
                state[
                    "symbol"
                ]
        },
        {
            "$set": {
                "symbol":
                    state[
                        "symbol"
                    ],
                "timestamp":
                    state.get(
                        "timestamp"
                    ),
                "updated_at":
                    utc_now(),
                "decision":
                    decision.get(
                        "action",
                        "WAIT",
                    ),
                "executed_action":
                    executed_action,
                "blocked_reason":
                    blocked_reason,
                "q_wait":
                    safe_float(
                        q.get("WAIT")
                    ),
                "q_buy":
                    safe_float(
                        q.get("BUY")
                    ),
                "q_sell":
                    safe_float(
                        q.get("SELL")
                    ),
                "epsilon":
                    safe_float(
                        decision.get(
                            "epsilon"
                        )
                    ),
                "explored":
                    bool(
                        decision.get(
                            "explored"
                        )
                    ),
                "model_ready":
                    bool(
                        decision.get(
                            "ready"
                        )
                    ),
                "prediction":
                    decision.get("prediction"),
                "pred_return_15m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("returns_percent", {})
                        .get("15m")
                    ),
                "pred_return_30m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("returns_percent", {})
                        .get("30m")
                    ),
                "pred_return_60m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("returns_percent", {})
                        .get("60m")
                    ),
                "pred_up_prob_30m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("direction_probabilities", {})
                        .get("30m", {})
                        .get("UP")
                    ),
                "pred_max_upside_30m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("max_upside_percent", {})
                        .get("30m")
                    ),
                "pred_max_downside_30m":
                    safe_float(
                        (decision.get("prediction") or {})
                        .get("max_downside_percent", {})
                        .get("30m")
                    ),
                "price":
                    state.get(
                        "price"
                    ),
                "has_position":
                    state.get(
                        "owns_position"
                    ),
                "state_vector":
                    state.get(
                        "state_vector",
                        [],
                    ),

                # Raw, readable features for the dashboard.
                "daily_change_percent":
                    raw.get(
                        "daily_change_raw"
                    ),
                "momentum_15m_percent":
                    raw.get(
                        "momentum_15m_raw"
                    ),
                "news_sentiment":
                    raw.get(
                        "news_sentiment_6h"
                    ),
                "news_sentiment_15m":
                    raw.get(
                        "news_sentiment_15m"
                    ),
                "news_count":
                    raw.get(
                        "news_count_6h_raw"
                    ),
                "news_count_15m":
                    raw.get(
                        "news_count_15m_raw"
                    ),
                "news_confidence_60m":
                    raw.get(
                        "news_confidence_60m"
                    ),
                "stocktwits_sentiment":
                    raw.get(
                        "stocktwits_sentiment_raw"
                    ),
                "stocktwits_density":
                    raw.get(
                        "stocktwits_density_raw"
                    ),
                "density_change_15m":
                    raw.get(
                        "stocktwits_density_change_15m_raw"
                    ),
                "volume":
                    raw.get(
                        "volume_raw"
                    ),
                "market_cap_millions":
                    raw.get(
                        "market_cap_raw"
                    ),
                "reason":
                    (
                        f"RL Q(WAIT)="
                        f"{safe_float(q.get('WAIT')):.3f}, "
                        f"Q(BUY)="
                        f"{safe_float(q.get('BUY')):.3f}, "
                        f"Q(SELL)="
                        f"{safe_float(q.get('SELL')):.3f}"
                    ),
            }
        },
        upsert=True,
    )


# =========================================================
# MAIN RL LOOP
# =========================================================

def _run_agent_impl() -> dict:

    ensure_agent_indexes()

    now = utc_now()
    agent = get_agent_state()

    new_session = (
        ensure_daily_session()
    )

    portfolio = (
        build_portfolio_context()
    )

    # Always observe held positions even if they disappear from the current
    # news-driven active universe. This keeps risk exits, predictions and
    # forced liquidation working for every open holding.
    symbols = set(
        get_active_universe()
    )

    symbols.update(
        str(symbol).strip().upper()
        for symbol in agent_positions.distinct("symbol")
        if str(symbol).strip()
    )

    symbols = sorted(symbols)

    states = []
    state_map = {}

    # -----------------------------------------------------
    # 1. Observe and permanently record the market.
    # -----------------------------------------------------

    for symbol in symbols:
        try:
            state = build_state(
                symbol,
                now=now,
                portfolio=portfolio,
            )

            if state is None:
                continue

            record_market_state(
                state
            )

            states.append(
                state
            )

            state_map[
                symbol
            ] = state

        except Exception as exc:
            print(
                f"RL state error {symbol}: {exc}"
            )

    # -----------------------------------------------------
    # 2. Score predictions/actions/trades whose future windows matured.
    # -----------------------------------------------------

    prediction_updates = (
        process_due_predictions(
            now
        )
    )

    action_rewards = (
        process_due_action_rewards(
            now
        )
    )

    trade_reviews = (
        process_due_trade_reviews(
            now
        )
    )

    # -----------------------------------------------------
    # 3. Multi-task train BEFORE making today's new choices.
    # -----------------------------------------------------

    brain = get_brain()

    training = (
        brain.train_from_replay(
            TRAIN_STEPS_PER_RUN
        )
    )

    # Even when disabled, finish previously pending scoring/training.
    if not agent.get(
        "enabled",
        False,
    ):
        brain.save()

        return {
            "enabled":
                False,
            "status":
                "RL agent disabled",
            "states_recorded":
                len(states),
            "trades":
                0,
            "action_rewards":
                action_rewards,
            "trade_reviews":
                trade_reviews,
            "predictions":
                prediction_updates,
            "training":
                training,
            "model":
                brain.status(),
        }

    # -----------------------------------------------------
    # 4. Do not open/close neural trades outside market hours.
    # -----------------------------------------------------

    if not market_is_open_at(
        now
    ):
        brain.save()

        agent_state.update_one(
            {"_id": AGENT_ID},
            {
                "$set": {
                    "last_run":
                        now,
                    "updated_at":
                        now,
                    "last_status":
                        (
                            "Market closed - "
                            "RL market memory updated only"
                        ),
                }
            },
        )

        return {
            "enabled":
                True,
            "status":
                "Market closed",
            "states_recorded":
                len(states),
            "trades":
                0,
            "action_rewards":
                action_rewards,
            "trade_reviews":
                trade_reviews,
            "predictions":
                prediction_updates,
            "training":
                training,
            "model":
                brain.status(),
        }

    # -----------------------------------------------------
    # 5. Recover positions accidentally left overnight.
    # -----------------------------------------------------

    if (
        new_session
        and agent_positions.count_documents({})
        > 0
    ):
        recovery = force_close_all(
            state_map,
            now,
            (
                "Overnight recovery exit. "
                "Intraday RL agent does not "
                "intentionally carry positions overnight."
            ),
            brain=brain,
        )

        brain.save()

        return {
            "enabled":
                True,
            "status":
                "Overnight recovery",
            "states_recorded":
                len(states),
            "trades":
                len(recovery),
            "action_rewards":
                action_rewards,
            "trade_reviews":
                trade_reviews,
            "predictions":
                prediction_updates,
            "training":
                training,
            "model":
                brain.status(),
        }

    # -----------------------------------------------------
    # 6. Mandatory 3:55 PM liquidation.
    # -----------------------------------------------------

    if must_close_all_positions(
        now
    ):
        forced = force_close_all(
            state_map,
            now,
            (
                "End-of-day forced exit at 3:55 PM ET. "
                "No overnight positions."
            ),
            brain=brain,
        )

        brain.save()

        agent_state.update_one(
            {"_id": AGENT_ID},
            {
                "$set": {
                    "last_run":
                        now,
                    "updated_at":
                        now,
                    "last_status":
                        (
                            f"End-of-day liquidation: "
                            f"{len(forced)} positions closed"
                        ),
                }
            },
        )

        return {
            "enabled":
                True,
            "status":
                "End-of-day liquidation",
            "states_recorded":
                len(states),
            "trades":
                len(forced),
            "action_rewards":
                action_rewards,
            "trade_reviews":
                trade_reviews,
            "predictions":
                prediction_updates,
            "training":
                training,
            "model":
                brain.status(),
        }

    # -----------------------------------------------------
    # 7. Neural decisions.
    # -----------------------------------------------------

    portfolio = (
        build_portfolio_context()
    )

    selected = []

    forced_stop_states = []

    for state in states:
        symbol = state[
            "symbol"
        ]

        raw = state.get(
            "raw",
            {},
        )

        # Do not let the neural network buy extremely illiquid names.
        if (
            safe_float(
                raw.get(
                    "volume_raw"
                )
            )
            < MIN_VOLUME
        ):
            valid = [
                ACTION_WAIT
            ]

            if state.get(
                "owns_position"
            ):
                valid.append(
                    ACTION_SELL
                )

        else:
            valid = valid_actions_for_state(
                state,
                portfolio,
            )

        # Hard stop-loss is a safety rule, not an RL choice.
        if (
            state.get(
                "owns_position"
            )
            and safe_float(
                raw.get(
                    "unrealized_return_raw"
                )
            )
            <= HARD_STOP_LOSS_PERCENT
        ):
            forced_stop_states.append(
                state
            )
            continue

        decision = brain.choose_action(
            state[
                "state_vector"
            ],
            valid_actions=valid,
            explore=True,
        )

        selected.append(
            (
                state,
                decision,
            )
        )

    # -----------------------------------------------------
    # 8. Execute hard stops first.
    # -----------------------------------------------------

    trades_made = []

    for state in forced_stop_states:
        decision = {
            "action_index":
                ACTION_SELL,
            "action":
                "SELL",
            "q_values": {
                "WAIT":
                    0.0,
                "BUY":
                    0.0,
                "SELL":
                    0.0,
            },
            "epsilon":
                0.0,
            "explored":
                False,
            "ready":
                brain.ready,
            "forced":
                True,
            "prediction":
                brain.predict_future(
                    state.get("state_vector", [])
                ),
        }

        trade = execute_sell(
            state=state,
            decision=decision,
            now=now,
            reason_override=(
                f"Hard risk stop triggered at "
                f"{safe_float(state['raw'].get('unrealized_return_raw')):+.2f}%."
            ),
            trainable=False,
        )

        save_decision(
            state,
            decision,
            executed_action=(
                "SELL"
                if trade
                else None
            ),
            blocked_reason=(
                None
                if trade
                else "Hard-stop sell could not execute"
            ),
        )

        if trade:
            trades_made.append(
                trade
            )

    # -----------------------------------------------------
    # 9. Execute model SELLs.
    # -----------------------------------------------------

    for state, decision in selected:
        if (
            len(trades_made)
            >= MAX_TRADES_PER_OBSERVATION
        ):
            break

        if (
            decision[
                "action_index"
            ]
            != ACTION_SELL
        ):
            continue

        trade = execute_sell(
            state=state,
            decision=decision,
            now=now,
            trainable=True,
        )

        save_decision(
            state,
            decision,
            executed_action=(
                "SELL"
                if trade
                else None
            ),
            blocked_reason=(
                None
                if trade
                else "No position available"
            ),
        )

        if trade:
            trades_made.append(
                trade
            )

    # Refresh portfolio after SELLs.
    portfolio = (
        build_portfolio_context()
    )

    # -----------------------------------------------------
    # 10. Rank BUY choices by neural advantage over WAIT.
    # -----------------------------------------------------

    buy_candidates = []

    for state, decision in selected:
        if (
            decision[
                "action_index"
            ]
            != ACTION_BUY
        ):
            continue

        q = decision.get(
            "q_values",
            {},
        )

        advantage = (
            safe_float(
                q.get(
                    "BUY"
                )
            )
            - safe_float(
                q.get(
                    "WAIT"
                )
            )
        )

        buy_candidates.append(
            (
                advantage,
                state,
                decision,
            )
        )

    buy_candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    executed_buy_symbols = set()

    for _, state, decision in buy_candidates:
        if (
            len(trades_made)
            >= MAX_TRADES_PER_OBSERVATION
        ):
            break

        valid_now = valid_actions_for_state(
            state,
            portfolio,
        )

        if ACTION_BUY not in valid_now:
            save_decision(
                state,
                decision,
                executed_action=None,
                blocked_reason=(
                    "Risk manager blocked BUY"
                ),
            )
            continue

        trade = execute_buy(
            state=state,
            decision=decision,
            now=now,
        )

        save_decision(
            state,
            decision,
            executed_action=(
                "BUY"
                if trade
                else None
            ),
            blocked_reason=(
                None
                if trade
                else "BUY could not be sized"
            ),
        )

        if trade:
            trades_made.append(
                trade
            )
            executed_buy_symbols.add(
                state[
                    "symbol"
                ]
            )

            portfolio = (
                build_portfolio_context()
            )

    # -----------------------------------------------------
    # 11. WAIT is an important RL action.
    #
    # Record only neural WAITs. BUYs blocked by portfolio limits are not
    # relabeled as WAIT, because the neural network did not choose WAIT.
    # -----------------------------------------------------

    for state, decision in selected:
        action_index = (
            decision[
                "action_index"
            ]
        )

        if action_index == ACTION_WAIT:
            record_pending_action(
                state=state,
                action_index=ACTION_WAIT,
                decision=decision,
                trainable=True,
            )

            save_decision(
                state,
                decision,
                executed_action="WAIT",
            )

        elif (
            action_index == ACTION_BUY
            and state["symbol"]
            not in executed_buy_symbols
        ):
            # Decision already may have been saved as blocked above. Upsert the
            # neural decision without creating a fake learning experience.
            save_decision(
                state,
                decision,
                executed_action=None,
                blocked_reason=(
                    "BUY not executed this cycle"
                ),
            )

        elif action_index == ACTION_SELL:
            # If no SELL happened because the global trade cap was reached,
            # preserve that information but do not train on a trade that was
            # never executed.
            if not any(
                trade.get("symbol")
                == state["symbol"]
                and trade.get("side")
                == "SELL"
                for trade in trades_made
            ):
                save_decision(
                    state,
                    decision,
                    executed_action=None,
                    blocked_reason=(
                        "SELL not executed this cycle"
                    ),
                )

    brain.save()

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$set": {
                "last_run":
                    now,
                "updated_at":
                    now,
                "last_status": (
                    f"RL observed {len(states)} stocks; "
                    f"executed {len(trades_made)} paper trades; "
                    f"experiences={brain.experience_count()}"
                ),
            }
        },
    )

    return {
        "enabled":
            True,
        "status":
            "completed",
        "states_recorded":
            len(states),
        "trades":
            len(trades_made),
        "action_rewards":
            action_rewards,
        "trade_reviews":
            trade_reviews,
        "predictions":
            prediction_updates,
        "training":
            training,
        "model":
            brain.status(),
    }



def run_agent() -> dict:
    """
    Public entry point with a Mongo-backed lease so the 3-minute scheduler and
    the Flask "Observe Now" button cannot train/write the same checkpoint at
    the same time.
    """

    owner = str(uuid4())

    if not acquire_model_lock(
        owner,
        lease_seconds=180,
    ):
        return {
            "enabled":
                bool(
                    get_agent_state().get(
                        "enabled",
                        False,
                    )
                ),
            "status":
                "RL agent busy in another process",
            "trades":
                0,
        }

    try:
        return _run_agent_impl()

    finally:
        release_model_lock(
            owner
        )


# =========================================================
# DAILY STATISTICS / DASHBOARD
# =========================================================

def get_today_trade_stats() -> dict:

    now_et = eastern_now()

    start_et = now_et.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    start_utc = start_et.astimezone(
        timezone.utc
    )

    trades = list(
        agent_trades.find(
            {
                "timestamp": {
                    "$gte":
                        start_utc
                }
            },
            {
                "_id": 0,
            },
        )
    )

    realized = 0.0
    closed = 0
    wins = 0

    for trade in trades:
        if trade.get("side") != "SELL":
            continue

        closed += 1

        pnl = safe_float(
            trade.get(
                "realized_pnl"
            )
        )

        realized += pnl

        if pnl > 0:
            wins += 1

    win_rate = (
        wins
        / closed
        * 100.0
        if closed
        else 0.0
    )

    return {
        "trade_count":
            len(trades),
        "closed_trades":
            closed,
        "winning_trades":
            wins,
        "win_rate":
            win_rate,
        "realized_pnl":
            realized,
    }


def get_lifetime_ai_trade_stats() -> dict:
    """Lifetime win/loss statistics for neural-agent SELL trades only.

    Legacy rule-based trades are deliberately excluded so this card reflects
    the performance of the PyTorch agent rather than mixing two strategies.
    Both the original DDQN tag and the v2 tag match the prefix below.
    """

    base_query = {
        "side": "SELL",
        "agent_type": {
            "$regex": r"^PYTORCH_DUELING_DDQN"
        },
    }

    closed = agent_trades.count_documents(
        base_query
    )

    wins = agent_trades.count_documents(
        {
            **base_query,
            "realized_pnl": {"$gt": 0},
        }
    )

    losses = agent_trades.count_documents(
        {
            **base_query,
            "realized_pnl": {"$lt": 0},
        }
    )

    breakeven = max(
        0,
        closed - wins - losses,
    )

    win_rate = (
        wins / closed * 100.0
        if closed
        else 0.0
    )

    loss_rate = (
        losses / closed * 100.0
        if closed
        else 0.0
    )

    return {
        "closed_trades": closed,
        "winning_trades": wins,
        "losing_trades": losses,
        "breakeven_trades": breakeven,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
    }


def get_prediction_accuracy_stats() -> dict:
    """Direction accuracy over every scored 15m/30m/60m forecast.

    Accuracy means the predicted UP/FLAT/DOWN class matched the realized
    direction class. Each matured horizon is one scored prediction example.
    """

    by_horizon = {}
    total_scored = 0
    total_correct = 0

    for minutes in (15, 30, 60):
        key = f"{minutes}m"
        status_field = (
            f"horizons.{key}.status"
        )
        correct_field = (
            f"horizons.{key}.direction_correct"
        )

        scored = rl_predictions.count_documents(
            {status_field: "scored"}
        )

        correct = rl_predictions.count_documents(
            {
                status_field: "scored",
                correct_field: True,
            }
        )

        accuracy = (
            correct / scored * 100.0
            if scored
            else 0.0
        )

        by_horizon[key] = {
            "scored": scored,
            "correct": correct,
            "accuracy": accuracy,
        }

        total_scored += scored
        total_correct += correct

    overall_accuracy = (
        total_correct / total_scored * 100.0
        if total_scored
        else 0.0
    )

    return {
        "scored": total_scored,
        "correct": total_correct,
        "accuracy": overall_accuracy,
        "by_horizon": by_horizon,
    }


def get_agent_dashboard() -> dict:

    ensure_agent_indexes()

    state = get_agent_state()

    portfolio = (
        get_portfolio_snapshot()
    )

    today = (
        get_today_trade_stats()
    )

    lifetime = (
        get_lifetime_ai_trade_stats()
    )

    prediction_accuracy = (
        get_prediction_accuracy_stats()
    )

    day_start_value = safe_float(
        state.get(
            "day_start_value"
        ),
        portfolio[
            "portfolio_value"
        ],
    )

    today_total = (
        portfolio[
            "portfolio_value"
        ]
        - day_start_value
    )

    today_pct = (
        today_total
        / day_start_value
        * 100.0
        if day_start_value
        else 0.0
    )

    today.update(
        {
            "day_start_value":
                day_start_value,
            "total_pnl":
                today_total,
            "total_pnl_percent":
                today_pct,
        }
    )

    trades = list(
        agent_trades.find(
            {},
            {
                "_id": 0,
                "state_vector": 0,
                "state_snapshot": 0,
            },
        ).sort(
            "timestamp",
            -1,
        )
    )

    for trade in trades:
        trade[
            "timestamp_display"
        ] = format_et(
            trade.get(
                "timestamp"
            )
        )

    decisions = list(
        agent_decisions.find(
            {},
            {
                "_id": 0,
                "state_vector": 0,
            },
        ).sort(
            "q_buy",
            -1,
        ).limit(
            200
        )
    )

    for decision in decisions:
        decision[
            "updated_at_display"
        ] = format_et(
            decision.get(
                "updated_at"
            )
        )

    recent_reviews = list(
        rl_trade_reviews.find(
            {
                "status":
                    "finalized"
            },
            {
                "_id": 0,
            },
        ).sort(
            "finalized_at",
            -1,
        ).limit(
            50
        )
    )

    recent_predictions = list(
        rl_predictions.find(
            {},
            {
                "_id": 0,
                "state_vector": 0,
            },
        ).sort(
            "timestamp",
            -1,
        ).limit(
            100
        )
    )

    for prediction in recent_predictions:
        prediction["timestamp_display"] = format_et(
            prediction.get("timestamp")
        )

    model = get_model_status()

    return {
        "state":
            state,
        "portfolio":
            portfolio,
        "today":
            today,
        "lifetime":
            lifetime,
        "prediction_accuracy":
            prediction_accuracy,
        "trades":
            trades,
        "decisions":
            decisions,
        "trade_reviews":
            recent_reviews,
        "predictions":
            recent_predictions,
        "model":
            model,
        "market_open":
            market_is_open(),
        "can_open_new_positions":
            can_open_new_positions(),
        "last_run_display":
            format_et(
                state.get(
                    "last_run"
                )
            ),
    }


# =========================================================
# RESET
# =========================================================

def reset_agent(
    reset_learning: bool = False,
    clear_trade_history: bool = False,
) -> None:

    now = utc_now()

    agent_positions.delete_many(
        {}
    )

    agent_decisions.delete_many(
        {}
    )

    rl_pending_actions.delete_many(
        {}
    )

    # Keep the complete BUY/SELL ledger and finalized reward reviews by
    # default. The dashboard should retain the agent's full history.
    if clear_trade_history:
        agent_trades.delete_many(
            {}
        )

        rl_trade_reviews.delete_many(
            {}
        )

    # By default keep learned experience/model. A portfolio reset should not
    # erase the AI's memory.
    if reset_learning:
        rl_experiences.delete_many(
            {}
        )

        rl_predictions.delete_many(
            {}
        )

        rl_model_state.delete_many(
            {}
        )

        model_dir = (
            Path(__file__).resolve().parent
            / "models"
        )

        checkpoint = (
            model_dir
            / "rl_trading_agent.pt"
        )

        if checkpoint.exists():
            checkpoint.unlink()

    agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$set": {
                "enabled":
                    False,
                "starting_cash":
                    STARTING_CASH,
                "cash":
                    STARTING_CASH,
                "session_date":
                    eastern_now()
                    .date()
                    .isoformat(),
                "day_start_value":
                    STARTING_CASH,
                "updated_at":
                    now,
                "last_run":
                    None,
                "last_status":
                    (
                        "Paper portfolio reset"
                        if not reset_learning
                        else
                        "Paper portfolio and RL learning reset"
                    ),
            }
        },
        upsert=True,
    )
