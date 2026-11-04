"""
rl_brain.py

Custom PyTorch Dueling Double-DQN brain.

News receives a larger branch than market/social/portfolio inputs so the
existing FinBERT news signal has substantial representational capacity without
hard-coding a fixed percentage weight.
"""

from __future__ import annotations

import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from db import db

from rl_config import (
    ACTION_BUY,
    ACTION_NAMES,
    ACTION_SELL,
    ACTION_WAIT,
    BATCH_SIZE,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MIN_EXPERIENCES_BEFORE_AUTONOMOUS,
    REPLAY_MIN_SIZE,
    TARGET_SYNC_EVERY,
    PREDICTION_BATCH_SIZE,
    PREDICTION_HORIZONS_MINUTES,
    PREDICTION_LOSS_WEIGHT,
    PREDICTION_RETURN_SCALE_PERCENT,
)
from rl_state import (
    FEATURE_COUNT,
    MARKET_FEATURES,
    MARKET_SLICE,
    NEWS_FEATURES,
    NEWS_SLICE,
    POSITION_FEATURES,
    POSITION_SLICE,
    SOCIAL_FEATURES,
    SOCIAL_SLICE,
)


rl_experiences = db["rl_experiences"]
rl_predictions = db["rl_predictions"]
rl_model_state = db["rl_model_state"]
rl_locks = db["rl_locks"]

MODEL_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINT_PATH = MODEL_DIR / "rl_trading_agent.pt"

MODEL_STATE_ID = "ddqn_v2_multitask"
LOCK_ID = "ddqn_model_lock"
MODEL_VERSION = 2

# Prediction-head raw output layout. Return values are normalized during
# training; direction outputs are logits for DOWN / FLAT / UP.
PRED_RETURN_INDEX = {15: 0, 30: 1, 60: 2}
PRED_UPSIDE_INDEX = {30: 3, 60: 5}
PRED_DOWNSIDE_INDEX = {30: 4, 60: 6}
PRED_DIRECTION_SLICE = {
    15: slice(7, 10),
    30: slice(10, 13),
    60: slice(13, 16),
}
PREDICTION_OUTPUT_SIZE = 16


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# =========================================================
# NETWORK
# =========================================================

class DuelingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # News gets the largest dedicated branch on purpose.
        self.news_branch = nn.Sequential(
            nn.Linear(len(NEWS_FEATURES), 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        self.market_branch = nn.Sequential(
            nn.Linear(len(MARKET_FEATURES), 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 48),
            nn.ReLU(),
        )

        self.social_branch = nn.Sequential(
            nn.Linear(len(SOCIAL_FEATURES), 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.position_branch = nn.Sequential(
            nn.Linear(len(POSITION_FEATURES), 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
        )

        fused_size = 64 + 48 + 32 + 24

        self.fusion = nn.Sequential(
            nn.Linear(fused_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.ReLU(),
        )

        # Dueling head:
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        self.value_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        # Auxiliary prediction head. It shares the same learned representation
        # as the RL policy, then predicts future returns, risk/opportunity and
        # direction probabilities. This is multi-task learning: the model is
        # taught both "what should I do?" and "what happens next?".
        self.prediction_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, PREDICTION_OUTPUT_SIZE),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        news = self.news_branch(x[:, NEWS_SLICE])
        market = self.market_branch(x[:, MARKET_SLICE])
        social = self.social_branch(x[:, SOCIAL_SLICE])
        position = self.position_branch(x[:, POSITION_SLICE])

        fused = torch.cat(
            [news, market, social, position],
            dim=1,
        )

        return self.fusion(fused)

    def q_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)

        return (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        return self.q_from_hidden(hidden)

    def prediction_raw(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        return self.prediction_head(hidden)

    def forward_all(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(x)
        return (
            self.q_from_hidden(hidden),
            self.prediction_head(hidden),
        )


# =========================================================
# CROSS-PROCESS MODEL LOCK
# =========================================================

def acquire_model_lock(
    owner: str,
    lease_seconds: int = 120,
) -> bool:

    now = utc_now()
    expiry = now + timedelta(seconds=lease_seconds)

    try:
        result = rl_locks.find_one_and_update(
            {
                "_id": LOCK_ID,
                "$or": [
                    {"expires_at": {"$lte": now}},
                    {"owner": owner},
                    {"expires_at": {"$exists": False}},
                ],
            },
            {
                "$set": {
                    "owner": owner,
                    "expires_at": expiry,
                    "updated_at": now,
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    except DuplicateKeyError:
        # Another process currently owns the existing lock document.
        return False

    return bool(
        result
        and result.get("owner") == owner
    )


def release_model_lock(owner: str) -> None:
    rl_locks.update_one(
        {
            "_id": LOCK_ID,
            "owner": owner,
        },
        {
            "$set": {
                "expires_at": utc_now(),
                "updated_at": utc_now(),
            }
        },
    )


# =========================================================
# RL BRAIN
# =========================================================

class RLBrain:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.policy_net = DuelingQNetwork().to(self.device)
        self.target_net = DuelingQNetwork().to(self.device)

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5,
        )

        self.environment_steps = 0
        self.optimizer_steps = 0
        self.target_sync_count = 0
        self.last_loss = None
        self.last_dqn_loss = None
        self.last_prediction_loss = None
        self.created_at = utc_now()
        self.last_trained_at = None

        self._load_checkpoint()

        # Target network must start synchronized.
        if self.optimizer_steps == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        self.target_net.eval()

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    def _load_checkpoint(self) -> None:
        if not CHECKPOINT_PATH.exists():
            return

        try:
            payload = torch.load(
                CHECKPOINT_PATH,
                map_location=self.device,
                weights_only=False,
            )

            if payload.get("feature_count") != FEATURE_COUNT:
                print(
                    "RL checkpoint feature layout changed. "
                    "Ignoring old checkpoint; run bootstrap again."
                )
                return

            checkpoint_version = int(
                payload.get("model_version", 1)
            )

            # Version 1 did not have the auxiliary prediction head. Load all
            # compatible trading weights and initialize only the new head. This
            # preserves the policy's learned behavior instead of starting over.
            strict = checkpoint_version >= MODEL_VERSION

            self.policy_net.load_state_dict(
                payload["policy_state_dict"],
                strict=strict,
            )

            self.target_net.load_state_dict(
                payload["target_state_dict"],
                strict=strict,
            )

            if strict:
                try:
                    self.optimizer.load_state_dict(
                        payload["optimizer_state_dict"]
                    )
                except Exception as exc:
                    print(
                        "RL optimizer state could not be restored; "
                        f"using a fresh optimizer: {exc}"
                    )
            else:
                print(
                    "Migrated DDQN v1 checkpoint into multi-task v2. "
                    "Existing trading weights kept; prediction head starts fresh."
                )

            self.environment_steps = int(
                payload.get("environment_steps", 0)
            )

            self.optimizer_steps = int(
                payload.get("optimizer_steps", 0)
            )

            self.target_sync_count = int(
                payload.get("target_sync_count", 0)
            )

            self.last_loss = payload.get("last_loss")
            self.last_dqn_loss = payload.get("last_dqn_loss")
            self.last_prediction_loss = payload.get("last_prediction_loss")
            self.created_at = payload.get(
                "created_at",
                self.created_at,
            )
            self.last_trained_at = payload.get(
                "last_trained_at"
            )

        except Exception as exc:
            print(
                f"Could not load RL checkpoint: {exc}"
            )

    def save(self) -> None:
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_version": MODEL_VERSION,
            "feature_count": FEATURE_COUNT,
            "policy_state_dict":
                self.policy_net.state_dict(),
            "target_state_dict":
                self.target_net.state_dict(),
            "optimizer_state_dict":
                self.optimizer.state_dict(),
            "environment_steps":
                self.environment_steps,
            "optimizer_steps":
                self.optimizer_steps,
            "target_sync_count":
                self.target_sync_count,
            "last_loss":
                self.last_loss,
            "last_dqn_loss":
                self.last_dqn_loss,
            "last_prediction_loss":
                self.last_prediction_loss,
            "created_at":
                self.created_at,
            "last_trained_at":
                self.last_trained_at,
        }

        temp_path = CHECKPOINT_PATH.with_suffix(
            ".tmp"
        )

        torch.save(
            payload,
            temp_path,
        )

        os.replace(
            temp_path,
            CHECKPOINT_PATH,
        )

        self._write_model_state()

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    def experience_count(self) -> int:
        return rl_experiences.count_documents({})

    @property
    def ready(self) -> bool:
        return (
            self.experience_count()
            >= MIN_EXPERIENCES_BEFORE_AUTONOMOUS
        )

    def epsilon(self) -> float:
        fraction = min(
            1.0,
            self.environment_steps
            / max(EPSILON_DECAY_STEPS, 1),
        )

        return (
            EPSILON_START
            + fraction
            * (EPSILON_END - EPSILON_START)
        )

    def _write_model_state(self) -> None:
        rl_model_state.update_one(
            {"_id": MODEL_STATE_ID},
            {
                "$set": {
                    "ready": self.ready,
                    "experience_count":
                        self.experience_count(),
                    "environment_steps":
                        self.environment_steps,
                    "optimizer_steps":
                        self.optimizer_steps,
                    "target_sync_count":
                        self.target_sync_count,
                    "epsilon":
                        self.epsilon(),
                    "last_loss":
                        self.last_loss,
                    "last_dqn_loss":
                        self.last_dqn_loss,
                    "last_prediction_loss":
                        self.last_prediction_loss,
                    "prediction_count":
                        rl_predictions.count_documents({}),
                    "scored_prediction_count":
                        rl_predictions.count_documents({
                            "$or": [
                                {"horizons.15m.status": "scored"},
                                {"horizons.30m.status": "scored"},
                                {"horizons.60m.status": "scored"},
                            ]
                        }),
                    "model_version": MODEL_VERSION,
                    "last_trained_at":
                        self.last_trained_at,
                    "device":
                        str(self.device),
                    "checkpoint":
                        str(CHECKPOINT_PATH),
                    "updated_at":
                        utc_now(),
                }
            },
            upsert=True,
        )

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "experience_count":
                self.experience_count(),
            "environment_steps":
                self.environment_steps,
            "optimizer_steps":
                self.optimizer_steps,
            "target_sync_count":
                self.target_sync_count,
            "epsilon":
                self.epsilon(),
            "last_loss":
                self.last_loss,
            "last_dqn_loss":
                self.last_dqn_loss,
            "last_prediction_loss":
                self.last_prediction_loss,
            "prediction_count":
                rl_predictions.count_documents({}),
            "scored_prediction_count":
                rl_predictions.count_documents({
                    "$or": [
                        {"horizons.15m.status": "scored"},
                        {"horizons.30m.status": "scored"},
                        {"horizons.60m.status": "scored"},
                    ]
                }),
            "model_version": MODEL_VERSION,
            "last_trained_at":
                self.last_trained_at,
            "device":
                str(self.device),
            "checkpoint":
                str(CHECKPOINT_PATH),
        }

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    def _decode_prediction_row(
        self,
        raw_row: torch.Tensor,
    ) -> dict:
        scale = float(
            PREDICTION_RETURN_SCALE_PERCENT
        )

        returns = {}
        probabilities = {}

        for minutes in PREDICTION_HORIZONS_MINUTES:
            key = f"{int(minutes)}m"
            returns[key] = float(
                raw_row[
                    PRED_RETURN_INDEX[int(minutes)]
                ].detach().cpu().item()
                * scale
            )

            logits = raw_row[
                PRED_DIRECTION_SLICE[int(minutes)]
            ]

            probs = torch.softmax(
                logits,
                dim=0,
            ).detach().cpu().tolist()

            probabilities[key] = {
                "DOWN": float(probs[0]),
                "FLAT": float(probs[1]),
                "UP": float(probs[2]),
            }

        max_upside = {}
        max_downside = {}

        for minutes in (30, 60):
            key = f"{minutes}m"
            max_upside[key] = float(
                nn.functional.softplus(
                    raw_row[PRED_UPSIDE_INDEX[minutes]]
                ).detach().cpu().item()
                * scale
            )
            max_downside[key] = float(
                nn.functional.softplus(
                    raw_row[PRED_DOWNSIDE_INDEX[minutes]]
                ).detach().cpu().item()
                * scale
            )

        return {
            "returns_percent": returns,
            "direction_probabilities": probabilities,
            "max_upside_percent": max_upside,
            "max_downside_percent": max_downside,
            "model_version": MODEL_VERSION,
        }

    def evaluate_state(
        self,
        state_vector: list[float],
    ) -> tuple[list[float], dict]:

        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(
                [state_vector],
                dtype=torch.float32,
                device=self.device,
            )

            q, prediction_raw = (
                self.policy_net.forward_all(
                    state_tensor
                )
            )

        q_values = [
            float(value)
            for value in q[0].detach().cpu().tolist()
        ]

        prediction = self._decode_prediction_row(
            prediction_raw[0]
        )

        return q_values, prediction

    def q_values(
        self,
        state_vector: list[float],
    ) -> list[float]:
        q_values, _ = self.evaluate_state(
            state_vector
        )
        return q_values

    def predict_future(
        self,
        state_vector: list[float],
    ) -> dict:
        _, prediction = self.evaluate_state(
            state_vector
        )
        return prediction

    def choose_action(
        self,
        state_vector: list[float],
        valid_actions: Iterable[int],
        explore: bool = True,
    ) -> dict:

        valid_actions = list(
            dict.fromkeys(valid_actions)
        )

        if not valid_actions:
            valid_actions = [ACTION_WAIT]

        q_values, prediction = self.evaluate_state(
            state_vector
        )

        # Never let an untrained/random network place a paper trade.
        if not self.ready:
            action = (
                ACTION_WAIT
                if ACTION_WAIT in valid_actions
                else valid_actions[0]
            )

            self.environment_steps += 1

            return {
                "action_index": action,
                "action": ACTION_NAMES[action],
                "q_values": {
                    "WAIT": q_values[ACTION_WAIT],
                    "BUY": q_values[ACTION_BUY],
                    "SELL": q_values[ACTION_SELL],
                },
                "epsilon": 0.0,
                "explored": False,
                "ready": False,
                "prediction": prediction,
            }

        epsilon = (
            self.epsilon()
            if explore
            else 0.0
        )

        self.environment_steps += 1

        explored = False

        if (
            explore
            and self.ready
            and random.random() < epsilon
        ):
            action = random.choice(
                valid_actions
            )
            explored = True

        else:
            action = max(
                valid_actions,
                key=lambda idx: q_values[idx],
            )

        return {
            "action_index": action,
            "action": ACTION_NAMES[action],
            "q_values": {
                "WAIT": q_values[ACTION_WAIT],
                "BUY": q_values[ACTION_BUY],
                "SELL": q_values[ACTION_SELL],
            },
            "epsilon": epsilon,
            "explored": explored,
            "ready": self.ready,
            "prediction": prediction,
        }

    # -----------------------------------------------------
    # Balanced replay sampling
    # -----------------------------------------------------

    def _sample_action_group(
        self,
        action_index: int,
        count: int,
    ) -> list[dict]:

        if count <= 0:
            return []

        return list(
            rl_experiences.aggregate(
                [
                    {
                        "$match": {
                            "action_index":
                                action_index
                        }
                    },
                    {
                        "$sample": {
                            "size": count
                        }
                    },
                ]
            )
        )

    def sample_balanced_batch(
        self,
        batch_size: int,
    ) -> list[dict]:

        per_action = max(
            1,
            batch_size // 3,
        )

        rows = []

        for action_index in (
            ACTION_WAIT,
            ACTION_BUY,
            ACTION_SELL,
        ):
            rows.extend(
                self._sample_action_group(
                    action_index,
                    per_action,
                )
            )

        # Supplement any missing rows with a random sample from all actions.
        missing = batch_size - len(rows)

        if missing > 0:
            rows.extend(
                list(
                    rl_experiences.aggregate(
                        [
                            {
                                "$sample": {
                                    "size": missing
                                }
                            }
                        ]
                    )
                )
            )

        random.shuffle(rows)

        return rows[:batch_size]

    def sample_prediction_batch(
        self,
        batch_size: int,
    ) -> list[dict]:
        """Sample rows with at least one scored prediction horizon."""

        return list(
            rl_predictions.aggregate(
                [
                    {
                        "$match": {
                            "trainable": True,
                            "$or": [
                                {"horizons.15m.status": "scored"},
                                {"horizons.30m.status": "scored"},
                                {"horizons.60m.status": "scored"},
                            ],
                        }
                    },
                    {
                        "$sample": {
                            "size": int(batch_size)
                        }
                    },
                ]
            )
        )

    def _prediction_loss(
        self,
        batch: list[dict],
    ) -> torch.Tensor | None:
        if not batch:
            return None

        states = torch.tensor(
            [row["state_vector"] for row in batch],
            dtype=torch.float32,
            device=self.device,
        )

        raw = self.policy_net.prediction_raw(
            states
        )

        losses = []
        scale = float(PREDICTION_RETURN_SCALE_PERCENT)

        for minutes in PREDICTION_HORIZONS_MINUTES:
            key = f"{int(minutes)}m"
            indices = [
                idx
                for idx, row in enumerate(batch)
                if (
                    row.get("horizons", {})
                    .get(key, {})
                    .get("status")
                    == "scored"
                )
            ]

            if not indices:
                continue

            idx_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            )

            actual_returns = torch.tensor(
                [
                    float(
                        batch[idx]["horizons"][key][
                            "actual_return_percent"
                        ]
                    ) / scale
                    for idx in indices
                ],
                dtype=torch.float32,
                device=self.device,
            )

            return_pred = raw[
                idx_tensor,
                PRED_RETURN_INDEX[int(minutes)],
            ]

            losses.append(
                nn.functional.smooth_l1_loss(
                    return_pred,
                    actual_returns,
                )
            )

            direction_targets = torch.tensor(
                [
                    int(
                        batch[idx]["horizons"][key][
                            "actual_direction_class"
                        ]
                    )
                    for idx in indices
                ],
                dtype=torch.long,
                device=self.device,
            )

            logits = raw[
                idx_tensor,
                PRED_DIRECTION_SLICE[int(minutes)],
            ]

            losses.append(
                nn.functional.cross_entropy(
                    logits,
                    direction_targets,
                )
            )

            if int(minutes) in (30, 60):
                actual_upside = torch.tensor(
                    [
                        float(
                            batch[idx]["horizons"][key][
                                "max_upside_percent"
                            ]
                        ) / scale
                        for idx in indices
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                actual_downside = torch.tensor(
                    [
                        float(
                            batch[idx]["horizons"][key][
                                "max_downside_percent"
                            ]
                        ) / scale
                        for idx in indices
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                predicted_upside = nn.functional.softplus(
                    raw[
                        idx_tensor,
                        PRED_UPSIDE_INDEX[int(minutes)],
                    ]
                )

                predicted_downside = nn.functional.softplus(
                    raw[
                        idx_tensor,
                        PRED_DOWNSIDE_INDEX[int(minutes)],
                    ]
                )

                losses.append(
                    nn.functional.smooth_l1_loss(
                        predicted_upside,
                        actual_upside,
                    )
                )

                losses.append(
                    nn.functional.smooth_l1_loss(
                        predicted_downside,
                        actual_downside,
                    )
                )

        if not losses:
            return None

        return torch.stack(losses).mean()

    # -----------------------------------------------------
    # Double-DQN training
    # -----------------------------------------------------

    def train_from_replay(
        self,
        gradient_steps: int,
    ) -> dict:

        count = self.experience_count()

        if count < REPLAY_MIN_SIZE:
            return {
                "trained": False,
                "reason":
                    f"Need {REPLAY_MIN_SIZE} experiences; have {count}.",
                "steps": 0,
            }

        losses = []

        self.policy_net.train()

        for _ in range(gradient_steps):
            batch = self.sample_balanced_batch(
                BATCH_SIZE
            )

            if len(batch) < 2:
                break

            states = torch.tensor(
                [
                    row["state_vector"]
                    for row in batch
                ],
                dtype=torch.float32,
                device=self.device,
            )

            actions = torch.tensor(
                [
                    int(row["action_index"])
                    for row in batch
                ],
                dtype=torch.long,
                device=self.device,
            )

            rewards = torch.tensor(
                [
                    float(row["reward"])
                    for row in batch
                ],
                dtype=torch.float32,
                device=self.device,
            )

            next_states = torch.tensor(
                [
                    row["next_state_vector"]
                    for row in batch
                ],
                dtype=torch.float32,
                device=self.device,
            )

            dones = torch.tensor(
                [
                    1.0 if row.get("done") else 0.0
                    for row in batch
                ],
                dtype=torch.float32,
                device=self.device,
            )

            # Optional valid-action masks for the next state.
            next_masks = []

            for row in batch:
                valid = row.get(
                    "next_valid_actions",
                    [ACTION_WAIT, ACTION_BUY, ACTION_SELL],
                )

                mask = [
                    idx in valid
                    for idx in range(3)
                ]

                next_masks.append(mask)

            next_masks_tensor = torch.tensor(
                next_masks,
                dtype=torch.bool,
                device=self.device,
            )

            current_q = (
                self.policy_net(states)
                .gather(
                    1,
                    actions.unsqueeze(1),
                )
                .squeeze(1)
            )

            with torch.no_grad():
                # Double DQN:
                # policy net chooses next action,
                # target net evaluates it.
                next_policy_q = self.policy_net(
                    next_states
                )

                next_policy_q = next_policy_q.masked_fill(
                    ~next_masks_tensor,
                    float("-inf"),
                )

                next_actions = next_policy_q.argmax(
                    dim=1,
                    keepdim=True,
                )

                next_target_q = (
                    self.target_net(next_states)
                    .gather(
                        1,
                        next_actions,
                    )
                    .squeeze(1)
                )

                targets = (
                    rewards
                    + GAMMA
                    * next_target_q
                    * (1.0 - dones)
                )

            dqn_loss = nn.functional.smooth_l1_loss(
                current_q,
                targets,
            )

            prediction_batch = self.sample_prediction_batch(
                PREDICTION_BATCH_SIZE
            )

            prediction_loss = self._prediction_loss(
                prediction_batch
            )

            if prediction_loss is None:
                loss = dqn_loss
            else:
                loss = (
                    dqn_loss
                    + float(PREDICTION_LOSS_WEIGHT)
                    * prediction_loss
                )

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                GRADIENT_CLIP_NORM,
            )

            self.optimizer.step()

            self.optimizer_steps += 1
            self.last_loss = float(
                loss.detach().cpu().item()
            )
            self.last_dqn_loss = float(
                dqn_loss.detach().cpu().item()
            )
            self.last_prediction_loss = (
                float(
                    prediction_loss.detach().cpu().item()
                )
                if prediction_loss is not None
                else None
            )
            self.last_trained_at = utc_now()

            losses.append(
                self.last_loss
            )

            if (
                self.optimizer_steps
                % TARGET_SYNC_EVERY
                == 0
            ):
                self.target_net.load_state_dict(
                    self.policy_net.state_dict()
                )
                self.target_sync_count += 1

        if losses:
            self.save()

        return {
            "trained": bool(losses),
            "steps": len(losses),
            "average_loss": (
                sum(losses) / len(losses)
                if losses
                else None
            ),
            "last_dqn_loss": self.last_dqn_loss,
            "last_prediction_loss": self.last_prediction_loss,
            "prediction_examples": rl_predictions.count_documents({
                "$or": [
                    {"horizons.15m.status": "scored"},
                    {"horizons.30m.status": "scored"},
                    {"horizons.60m.status": "scored"},
                ]
            }),
            "experience_count": count,
        }


def get_brain() -> RLBrain:
    return RLBrain()


def get_model_status() -> dict:
    brain = RLBrain()
    return brain.status()
