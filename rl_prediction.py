"""
rl_prediction.py

Permanent prediction-vs-reality memory for the PyTorch RL trading agent.

At decision time the neural network predicts future price behavior using only
information available at that moment. Later, after 15/30/60 minutes, this
module records what actually happened. Finalized prediction records become
supervised auxiliary targets for the shared neural representation.

No future information is exposed to the trading decision before it happens.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from db import db
from rl_config import (
    PREDICTION_FLAT_THRESHOLD_PERCENT,
    PREDICTION_HORIZONS_MINUTES,
)
from rl_state import aware_utc, safe_float


EASTERN = ZoneInfo("America/New_York")

rl_market_history = db["rl_market_history"]
rl_predictions = db["rl_predictions"]

DIRECTION_DOWN = 0
DIRECTION_FLAT = 1
DIRECTION_UP = 2

DIRECTION_NAMES = {
    DIRECTION_DOWN: "DOWN",
    DIRECTION_FLAT: "FLAT",
    DIRECTION_UP: "UP",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_change(new_value: float, old_value: float) -> float:
    if old_value is None or old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def direction_class(return_percent: float) -> int:
    threshold = float(PREDICTION_FLAT_THRESHOLD_PERCENT)

    if return_percent > threshold:
        return DIRECTION_UP

    if return_percent < -threshold:
        return DIRECTION_DOWN

    return DIRECTION_FLAT


def _market_close_for(timestamp: datetime) -> datetime:
    timestamp = aware_utc(timestamp) or utc_now()
    local = timestamp.astimezone(EASTERN)

    close_local = local.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    return close_local.astimezone(timezone.utc)


def _prediction_quality(
    predicted_return: float,
    actual_return: float,
    predicted_direction: int,
    actual_direction: int,
) -> float:
    """
    Human-readable 0-100 diagnostic score.

    This score is not the trading reward. The neural prediction head is trained
    directly from regression/classification losses. The score is only for
    inspection on the dashboard.
    """
    absolute_error = abs(
        float(predicted_return)
        - float(actual_return)
    )

    magnitude_score = 100.0 * (
        1.0 - min(absolute_error / 5.0, 1.0)
    )

    direction_score = (
        100.0
        if int(predicted_direction) == int(actual_direction)
        else 0.0
    )

    return clamp(
        0.70 * magnitude_score
        + 0.30 * direction_score,
        0.0,
        100.0,
    )


def record_prediction(
    *,
    state: dict,
    action_id: str,
    action_index: int,
    action: str,
    prediction: dict | None,
    trainable: bool = True,
    trade_id: str | None = None,
) -> str | None:
    """Persist the model's forecast made at action time."""

    if not prediction:
        return None

    timestamp = aware_utc(
        state.get("timestamp")
    ) or utc_now()

    price = safe_float(
        state.get("price")
    )

    if price <= 0:
        return None

    market_close = _market_close_for(timestamp)

    horizons = {}
    eligible_due_times = []

    for minutes in PREDICTION_HORIZONS_MINUTES:
        due_at = timestamp + timedelta(
            minutes=int(minutes)
        )

        key = f"{int(minutes)}m"

        # We only train fixed-horizon targets if the full horizon occurs before
        # the regular 4:00 PM close. Otherwise the horizon is explicitly N/A.
        if due_at <= market_close:
            status = "pending"
            eligible_due_times.append(due_at)
        else:
            status = "unavailable_market_close"

        horizons[key] = {
            "minutes": int(minutes),
            "due_at": due_at,
            "status": status,
        }

    prediction_id = str(uuid4())

    overall_status = (
        "pending"
        if eligible_due_times
        else "finalized"
    )

    rl_predictions.insert_one(
        {
            "prediction_id": prediction_id,
            "action_id": action_id,
            "trade_id": trade_id,
            "symbol": state.get("symbol"),
            "timestamp": timestamp,
            "action_index": int(action_index),
            "action": str(action),
            "price_at_prediction": price,
            "state_vector": state.get(
                "state_vector",
                [],
            ),
            "predicted": prediction,
            "horizons": horizons,
            "status": overall_status,
            "next_due_at": (
                min(eligible_due_times)
                if eligible_due_times
                else None
            ),
            "trainable": bool(trainable),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )

    return prediction_id


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
                "price": {"$ne": None},
            },
            {
                "_id": 0,
                "timestamp": 1,
                "price": 1,
            },
        ).sort("timestamp", 1)
    )


def _predicted_direction(
    prediction: dict,
    horizon_key: str,
) -> tuple[int, dict]:
    probabilities = (
        prediction
        .get("direction_probabilities", {})
        .get(horizon_key, {})
        or {}
    )

    ordered = [
        safe_float(probabilities.get("DOWN")),
        safe_float(probabilities.get("FLAT")),
        safe_float(probabilities.get("UP")),
    ]

    best = max(
        range(3),
        key=lambda idx: ordered[idx],
    )

    return best, {
        "DOWN": ordered[0],
        "FLAT": ordered[1],
        "UP": ordered[2],
    }


def process_due_predictions(
    now: datetime | None = None,
) -> dict:
    """Score prediction horizons that have reached their target time."""

    if now is None:
        now = utc_now()

    docs = list(
        rl_predictions.find(
            {
                "status": {
                    "$in": [
                        "pending",
                        "partial",
                    ]
                },
                "next_due_at": {
                    "$lte": now
                },
            }
        ).sort("next_due_at", 1)
    )

    horizons_scored = 0
    predictions_finalized = 0
    unavailable = 0

    for doc in docs:
        timestamp = aware_utc(
            doc.get("timestamp")
        )

        symbol = str(
            doc.get("symbol")
            or ""
        ).strip().upper()

        start_price = safe_float(
            doc.get("price_at_prediction")
        )

        if (
            timestamp is None
            or not symbol
            or start_price <= 0
        ):
            rl_predictions.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "status": "invalid",
                        "updated_at": now,
                    }
                },
            )
            continue

        horizons = dict(
            doc.get("horizons")
            or {}
        )

        prediction = dict(
            doc.get("predicted")
            or {}
        )

        changed = False

        for key, horizon in horizons.items():
            if horizon.get("status") != "pending":
                continue

            due_at = aware_utc(
                horizon.get("due_at")
            )

            if due_at is None or due_at > now:
                continue

            path = _price_path(
                symbol,
                timestamp,
                due_at,
            )

            prices = [
                safe_float(row.get("price"))
                for row in path
                if safe_float(row.get("price")) > 0
            ]

            if not prices:
                horizon["status"] = "unavailable_no_data"
                horizon["scored_at"] = now
                unavailable += 1
                changed = True
                continue

            end_price = prices[-1]
            actual_return = pct_change(
                end_price,
                start_price,
            )

            max_upside = max(
                0.0,
                pct_change(
                    max(prices),
                    start_price,
                ),
            )

            max_downside = max(
                0.0,
                -pct_change(
                    min(prices),
                    start_price,
                ),
            )

            actual_direction = direction_class(
                actual_return
            )

            predicted_return = safe_float(
                prediction
                .get("returns_percent", {})
                .get(key)
            )

            predicted_direction, probabilities = (
                _predicted_direction(
                    prediction,
                    key,
                )
            )

            quality = _prediction_quality(
                predicted_return,
                actual_return,
                predicted_direction,
                actual_direction,
            )

            horizon.update(
                {
                    "status": "scored",
                    "end_price": end_price,
                    "actual_return_percent": actual_return,
                    "actual_direction_class": actual_direction,
                    "actual_direction": DIRECTION_NAMES[
                        actual_direction
                    ],
                    "max_upside_percent": max_upside,
                    "max_downside_percent": max_downside,
                    "predicted_return_percent": predicted_return,
                    "predicted_direction_class": predicted_direction,
                    "predicted_direction": DIRECTION_NAMES[
                        predicted_direction
                    ],
                    "predicted_probabilities": probabilities,
                    "absolute_return_error_percent": abs(
                        predicted_return
                        - actual_return
                    ),
                    "direction_correct": (
                        predicted_direction
                        == actual_direction
                    ),
                    "quality_score": quality,
                    "scored_at": now,
                }
            )

            horizons_scored += 1
            changed = True

        if not changed:
            continue

        pending_due_times = []
        scored_scores = []

        for horizon in horizons.values():
            if horizon.get("status") == "pending":
                due = aware_utc(
                    horizon.get("due_at")
                )
                if due is not None:
                    pending_due_times.append(due)

            if horizon.get("status") == "scored":
                scored_scores.append(
                    safe_float(
                        horizon.get("quality_score")
                    )
                )

        if pending_due_times:
            status = "partial"
            next_due = min(pending_due_times)
        else:
            status = "finalized"
            next_due = None
            predictions_finalized += 1

        overall_score = (
            sum(scored_scores) / len(scored_scores)
            if scored_scores
            else None
        )

        update = {
            "horizons": horizons,
            "status": status,
            "next_due_at": next_due,
            "prediction_score": overall_score,
            "updated_at": now,
        }

        if status == "finalized":
            update["finalized_at"] = now

        rl_predictions.update_one(
            {"_id": doc["_id"]},
            {"$set": update},
        )

    return {
        "documents_checked": len(docs),
        "horizons_scored": horizons_scored,
        "predictions_finalized": predictions_finalized,
        "unavailable": unavailable,
    }
