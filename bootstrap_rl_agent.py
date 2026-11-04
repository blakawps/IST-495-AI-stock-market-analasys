"""
bootstrap_rl_agent.py

One-time warm start for the custom PyTorch RL agent.

It studies the existing rule-based agent_trades collection rather than assuming
the old agent was correct. BUY/SELL actions are judged by their realized
outcomes and converted into initial RL experiences.

Because the old data does not preserve every 3-minute WAIT observation, this
bootstrap cannot reconstruct WAIT rewards perfectly. Live operation fixes that:
from installation onward WAIT/BUY/SELL actions are all evaluated with future
market data.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from db import db

from rl_brain import get_brain
from rl_config import (
    ACTION_BUY,
    ACTION_SELL,
    MAX_REWARD,
    MIN_REWARD,
)
from rl_state import legacy_signal_to_state, safe_float


agent_trades = db["agent_trades"]
rl_experiences = db["rl_experiences"]
rl_pending_actions = db["rl_pending_actions"]
rl_trade_reviews = db["rl_trade_reviews"]
rl_predictions = db["rl_predictions"]
rl_model_state = db["rl_model_state"]


def utc_now():
    return datetime.now(timezone.utc)


def clamp(value, low, high):
    return max(low, min(high, value))


def bootstrap_reward(return_percent: float) -> float:
    """
    Old history does not always have the future intraday price path needed for
    buy-regret and sell-regret. So bootstrap reward uses realized return only.
    Live rewards become much richer after installation.
    """
    reward = 5.0 * math.tanh(
        return_percent / 1.5
    )

    return clamp(
        reward,
        MIN_REWARD,
        MAX_REWARD,
    )


def insert_experience(
    *,
    experience_id,
    symbol,
    timestamp,
    state_vector,
    action_index,
    reward,
    next_state_vector,
    done,
    metadata,
):

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
                    action_index,
                "reward":
                    reward,
                "next_state_vector":
                    next_state_vector,
                "next_valid_actions": [
                    0,
                    1,
                    2,
                ],
                "done":
                    done,
                "source":
                    "legacy_rule_agent_bootstrap",
                "metadata":
                    metadata,
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


def build_bootstrap_experiences():

    trades = list(
        agent_trades.find(
            {}
        ).sort(
            "timestamp",
            1,
        )
    )

    open_buys = {}

    inserted = 0
    skipped = 0

    for trade in trades:

        symbol = str(
            trade.get(
                "symbol"
            )
            or ""
        ).strip().upper()

        side = str(
            trade.get(
                "side"
            )
            or ""
        ).strip().upper()

        if not symbol:
            skipped += 1
            continue

        if side == "BUY":
            open_buys.setdefault(
                symbol,
                [],
            ).append(
                trade
            )
            continue

        if side != "SELL":
            continue

        buys = open_buys.get(
            symbol,
            [],
        )

        if not buys:
            skipped += 1
            continue

        sell_price = safe_float(
            trade.get(
                "price"
            )
        )

        sell_signal = (
            trade.get(
                "state_snapshot"
            )
            or trade.get(
                "signal"
            )
            or {}
        )

        sell_state = (
            legacy_signal_to_state(
                sell_signal,
                owns_position=True,
            )
        )

        # Every BUY/add-on that built the position gets judged against the
        # closing SELL price.
        buy_returns = []

        for buy_trade in buys:

            buy_price = safe_float(
                buy_trade.get(
                    "price"
                )
            )

            if (
                buy_price <= 0
                or sell_price <= 0
            ):
                skipped += 1
                continue

            return_percent = (
                (
                    sell_price
                    - buy_price
                )
                / buy_price
                * 100.0
            )

            reward = bootstrap_reward(
                return_percent
            )

            buy_signal = (
                buy_trade.get(
                    "state_snapshot"
                )
                or buy_trade.get(
                    "signal"
                )
                or {}
            )

            buy_state = (
                legacy_signal_to_state(
                    buy_signal,
                    owns_position=False,
                )
            )

            buy_trade_id = (
                buy_trade.get(
                    "trade_id"
                )
                or str(
                    buy_trade.get(
                        "_id"
                    )
                )
            )

            sell_trade_id = (
                trade.get(
                    "trade_id"
                )
                or str(
                    trade.get(
                        "_id"
                    )
                )
            )

            added = insert_experience(
                experience_id=(
                    "bootstrap:BUY:"
                    f"{buy_trade_id}:"
                    f"{sell_trade_id}"
                ),
                symbol=symbol,
                timestamp=(
                    buy_trade.get(
                        "timestamp"
                    )
                    or utc_now()
                ),
                state_vector=(
                    buy_state[
                        "state_vector"
                    ]
                ),
                action_index=ACTION_BUY,
                reward=reward,
                next_state_vector=(
                    sell_state[
                        "state_vector"
                    ]
                ),
                done=True,
                metadata={
                    "buy_trade_id":
                        buy_trade_id,
                    "sell_trade_id":
                        sell_trade_id,
                    "return_percent":
                        return_percent,
                },
            )

            if added:
                inserted += 1

            buy_returns.append(
                return_percent
            )

        # Teach the SELL action from the total position outcome too.
        entry_price = safe_float(
            trade.get(
                "entry_price"
            )
        )

        if entry_price <= 0 and buys:
            total_shares = sum(
                int(
                    row.get(
                        "shares"
                    )
                    or 0
                )
                for row in buys
            )

            if total_shares:
                entry_price = (
                    sum(
                        safe_float(
                            row.get(
                                "price"
                            )
                        )
                        * int(
                            row.get(
                                "shares"
                            )
                            or 0
                        )
                        for row in buys
                    )
                    / total_shares
                )

        if (
            entry_price > 0
            and sell_price > 0
        ):
            position_return = (
                (
                    sell_price
                    - entry_price
                )
                / entry_price
                * 100.0
            )

            sell_reward = (
                bootstrap_reward(
                    position_return
                )
            )

            sell_trade_id = (
                trade.get(
                    "trade_id"
                )
                or str(
                    trade.get(
                        "_id"
                    )
                )
            )

            added = insert_experience(
                experience_id=(
                    "bootstrap:SELL:"
                    f"{sell_trade_id}"
                ),
                symbol=symbol,
                timestamp=(
                    trade.get(
                        "timestamp"
                    )
                    or utc_now()
                ),
                state_vector=(
                    sell_state[
                        "state_vector"
                    ]
                ),
                action_index=ACTION_SELL,
                reward=sell_reward,
                next_state_vector=(
                    sell_state[
                        "state_vector"
                    ]
                ),
                done=True,
                metadata={
                    "sell_trade_id":
                        sell_trade_id,
                    "return_percent":
                        position_return,
                    "buy_returns":
                        buy_returns,
                },
            )

            if added:
                inserted += 1

        open_buys[
            symbol
        ] = []

    return {
        "trade_rows":
            len(trades),
        "inserted":
            inserted,
        "skipped":
            skipped,
        "experience_total":
            rl_experiences.count_documents(
                {}
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete prior RL experiences/checkpoint "
            "before rebuilding from old rule-agent trades."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help=(
            "Gradient steps after building bootstrap experiences."
        ),
    )

    args = parser.parse_args()

    checkpoint = (
        Path(__file__).resolve().parent
        / "models"
        / "rl_trading_agent.pt"
    )

    if args.force:

        print(
            "Clearing RL-only training state..."
        )

        rl_experiences.delete_many(
            {}
        )

        rl_pending_actions.delete_many(
            {}
        )

        rl_trade_reviews.delete_many(
            {}
        )

        rl_predictions.delete_many(
            {}
        )

        rl_model_state.delete_many(
            {}
        )

        if checkpoint.exists():
            checkpoint.unlink()

    print(
        "Reading old paper trades..."
    )

    result = (
        build_bootstrap_experiences()
    )

    print()
    print(
        "Bootstrap experiences:"
    )
    print(
        result
    )

    brain = get_brain()

    print()
    print(
        f"Training custom PyTorch DDQN "
        f"for {args.steps} gradient steps..."
    )

    train_result = (
        brain.train_from_replay(
            args.steps
        )
    )

    brain.save()

    print()
    print(
        "Training result:"
    )
    print(
        train_result
    )

    print()
    print(
        "Model status:"
    )
    print(
        brain.status()
    )


if __name__ == "__main__":
    main()
