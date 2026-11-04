"""
rl_config.py

Configuration for the custom PyTorch reinforcement-learning paper-trading agent.

The agent never submits brokerage orders. It only trades the existing simulated
portfolio in MongoDB.
"""

# -----------------------------
# Trading / portfolio rules
# -----------------------------

STARTING_CASH = 100_000.00

MAX_POSITION_FRACTION = 0.10
BUY_FRACTION = 0.05
MAX_POSITIONS = 8
MAX_TRADES_PER_OBSERVATION = 3

MIN_VOLUME = 100_000
MAX_MARKET_DATA_AGE_MINUTES = 6

LAST_ENTRY_HOUR = 15
LAST_ENTRY_MINUTE = 30

FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 55

# Keep this hard stop as a risk guardrail.
# There is intentionally NO hard take-profit. The neural agent learns when
# to sell profitable positions.
HARD_STOP_LOSS_PERCENT = -1.00

# -----------------------------
# RL action space
# -----------------------------

ACTION_WAIT = 0
ACTION_BUY = 1
ACTION_SELL = 2

ACTION_NAMES = {
    ACTION_WAIT: "WAIT",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
}

# -----------------------------
# Reward windows
# -----------------------------

BUY_REWARD_WINDOW_MINUTES = 30
WAIT_REWARD_WINDOW_MINUTES = 30
SELL_REWARD_WINDOW_MINUTES = 60

# Used when judging "could I have bought lower shortly after buying?"
BUY_REGRET_WINDOW_MINUTES = 30

# Used when judging "did I sell too early?"
POST_SELL_REGRET_WINDOW_MINUTES = 60

# -----------------------------
# Reward weights
# -----------------------------

# Full round-trip trade reward.
PROFIT_REWARD_WEIGHT = 5.0
BUY_REGRET_WEIGHT = 1.5
SELL_REGRET_WEIGHT = 2.0
DRAWDOWN_WEIGHT = 1.0
CAPTURE_EFFICIENCY_WEIGHT = 1.0
POST_SELL_PROTECTION_WEIGHT = 0.5

# Clamp rewards so one bizarre stock move cannot dominate training.
MIN_REWARD = -10.0
MAX_REWARD = 10.0

# -----------------------------
# DQN hyperparameters
# -----------------------------

GAMMA = 0.95
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

REPLAY_MIN_SIZE = 64
REPLAY_SAMPLE_LIMIT = 10_000

TARGET_SYNC_EVERY = 200
GRADIENT_CLIP_NORM = 5.0

# A few small updates each 3-minute cycle is safer than hammering the same
# recent data repeatedly.
TRAIN_STEPS_PER_RUN = 8

# Exploration is appropriate because this is paper trading.
EPSILON_START = 0.15
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 5_000

# The bootstrap from the old rule-based agent should create more than this.
MIN_EXPERIENCES_BEFORE_AUTONOMOUS = 64

# If there is no trained checkpoint yet, do not make random trades.
UNTRAINED_ACTION = ACTION_WAIT

# -----------------------------
# State windows
# -----------------------------

NEWS_WINDOW_HOURS = 6
STOCKTWITS_WINDOW_MINUTES = 60

RETURN_WINDOWS_MINUTES = (3, 6, 15, 30, 60)

# -----------------------------
# Auxiliary future-movement prediction
# -----------------------------

# The shared neural representation predicts these fixed horizons whenever the
# full horizon fits before the regular 4:00 PM market close.
PREDICTION_HORIZONS_MINUTES = (15, 30, 60)

# Returns inside +/- this threshold are classified as FLAT for the auxiliary
# direction-classification task.
PREDICTION_FLAT_THRESHOLD_PERCENT = 0.15

# Regression targets are divided by this value during neural training so they
# stay numerically well-scaled. Dashboard values are converted back to %.
PREDICTION_RETURN_SCALE_PERCENT = 5.0

# Weight of the auxiliary prediction objective relative to the DDQN loss.
# The trading reward remains the primary objective.
PREDICTION_LOSS_WEIGHT = 0.35

# Number of finalized prediction rows sampled for each multi-task update.
PREDICTION_BATCH_SIZE = 64
