import math

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from uuid import uuid4

from zoneinfo import ZoneInfo


from db import (
    headlines,
    companies,
    market_data,
    stocktwits_posts,
    analytics_snapshots,
    agent_state,
    agent_positions,
    agent_trades,
    agent_decisions,
)


# =========================================================
# BASIC AGENT SETTINGS
# =========================================================

AGENT_ID = "paper_agent"

EASTERN = ZoneInfo(
    "America/New_York"
)


STARTING_CASH = 100000.00


# Maximum 10% of the portfolio
# can be exposed to one ticker.
MAX_POSITION_FRACTION = 0.10


# Each individual purchase uses
# approximately 5% of the portfolio.
BUY_FRACTION = 0.05


# Maximum number of separate
# tickers held at the same time.
MAX_POSITIONS = 8


# Safety protection against suddenly
# opening/closing a huge number of
# positions from one observation.
MAX_TRADES_PER_OBSERVATION = 3


# =========================================================
# SIGNAL THRESHOLDS
# =========================================================

BUY_THRESHOLD = 0.35

SELL_THRESHOLD = -0.20


# =========================================================
# PROFIT / LOSS RULES
# =========================================================

TAKE_PROFIT_PERCENT = 1.50

STOP_LOSS_PERCENT = -1.00


# No fixed maximum holding duration.
#
# Agent may hold a position for as
# long as it wants during the day.


# =========================================================
# TRADE TIMING
# =========================================================

TRADE_COOLDOWN_MINUTES = 15


# No new positions after 3:30 PM ET.
LAST_ENTRY_HOUR = 15
LAST_ENTRY_MINUTE = 30


# Everything must be sold at 3:55 PM ET.
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 55


# =========================================================
# SIGNAL WINDOWS
# =========================================================

NEWS_WINDOW_HOURS = 6

STOCKTWITS_WINDOW_MINUTES = 60

MOMENTUM_MINUTES = 15


# =========================================================
# DATA QUALITY / LIQUIDITY
# =========================================================

MIN_VOLUME = 100000


# Agent refuses to paper trade
# using an old Finviz quote.
MAX_MARKET_DATA_AGE_MINUTES = 6


# =========================================================
# WHEN SHOULD THE AGENT RECONSIDER?
#
# New data arrives every 3 minutes,
# but the agent does NOT have to act.
# =========================================================

MIN_SCORE_CHANGE_TO_RECONSIDER = 0.10

MIN_PRICE_MOVE_TO_RECONSIDER = 0.25

MIN_DENSITY_CHANGE_TO_RECONSIDER = 10


# =========================================================
# EXCHANGES
# =========================================================

ALLOWED_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "NYSE AMERICAN",
    "AMEX",
}


# =========================================================
# GENERAL HELPERS
# =========================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def eastern_now():

    return datetime.now(
        EASTERN
    )


def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def as_aware_utc(
    value
):

    if value is None:
        return None

    if value.tzinfo is None:

        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def format_et(
    value
):

    value = as_aware_utc(
        value
    )

    if value is None:
        return "N/A"

    return (
        value
        .astimezone(
            EASTERN
        )
        .strftime(
            "%b %d, %Y %I:%M %p ET"
        )
    )


# =========================================================
# MARKET HOURS
# =========================================================

def market_is_open():

    now = eastern_now()

    # Monday-Friday only.
    #
    # Note:
    # This simple version does not
    # yet include exchange holidays.

    if now.weekday() >= 5:
        return False


    market_open = (
        now.replace(
            hour=9,
            minute=30,
            second=0,
            microsecond=0,
        )
    )


    market_close = (
        now.replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
        )
    )


    return (
        market_open
        <= now
        < market_close
    )


def can_open_new_positions():

    now = eastern_now()


    if not market_is_open():
        return False


    cutoff = (
        now.replace(
            hour=
                LAST_ENTRY_HOUR,

            minute=
                LAST_ENTRY_MINUTE,

            second=0,

            microsecond=0,
        )
    )


    return (
        now < cutoff
    )


def must_close_all_positions():

    now = eastern_now()


    if now.weekday() >= 5:
        return False


    close_time = (
        now.replace(
            hour=
                FORCE_CLOSE_HOUR,

            minute=
                FORCE_CLOSE_MINUTE,

            second=0,

            microsecond=0,
        )
    )


    market_close = (
        now.replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
        )
    )


    return (
        close_time
        <= now
        < market_close
    )


# =========================================================
# AGENT STATE
# =========================================================

def get_agent_state():

    state = (
        agent_state.find_one(
            {
                "_id":
                    AGENT_ID
            }
        )
    )


    if state:
        return state


    now = utc_now()

    today = (
        eastern_now()
        .date()
        .isoformat()
    )


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
            today,

        "day_start_value":
            STARTING_CASH,

        "created_at":
            now,

        "updated_at":
            now,

        "last_run":
            None,

        "last_status":
            "Paper agent created",
    }


    agent_state.insert_one(
        state
    )


    return state


def set_agent_enabled(
    enabled
):

    get_agent_state()


    agent_state.update_one(

        {
            "_id":
                AGENT_ID
        },

        {
            "$set": {

                "enabled":
                    bool(
                        enabled
                    ),

                "updated_at":
                    utc_now(),

                "last_status":
                    (
                        "Agent enabled"
                        if enabled
                        else
                        "Agent stopped"
                    ),
            }
        }
    )


# =========================================================
# FRESH FINVIZ MARKET DATA
# =========================================================

def get_fresh_market_data(
    symbol,
    now=None
):

    if now is None:

        now = utc_now()


    row = (
        market_data.find_one(

            {
                "symbol":
                    symbol
            },

            {
                "_id": 0,

                "symbol": 1,

                "price": 1,

                "change_percent": 1,

                "volume": 1,

                "market_cap_millions": 1,

                "updated_at": 1,
            }
        )
        or {}
    )


    price = row.get(
        "price"
    )


    updated_at = (
        as_aware_utc(
            row.get(
                "updated_at"
            )
        )
    )


    if (
        price is None
        or price <= 0
        or updated_at is None
    ):

        return None


    age = (
        now
        - updated_at
    )


    if age > timedelta(
        minutes=
            MAX_MARKET_DATA_AGE_MINUTES
    ):

        return None


    row[
        "updated_at"
    ] = updated_at


    return row


# =========================================================
# NEWS SIGNAL
#
# Uses the FinBERT sentiment that is
# already stored with each article.
# =========================================================

def get_news_signal(
    symbol,
    now
):

    cutoff = (
        now
        - timedelta(
            hours=
                NEWS_WINDOW_HOURS
        )
    )


    docs = list(

        headlines.find(

            {
                "securities.symbol":
                    symbol,

                "published_at": {
                    "$gte":
                        cutoff,

                    "$lte":
                        now,
                },

                "news_sentiment.score": {
                    "$in": [
                        -1,
                        0,
                        1,
                    ]
                },
            },

            {
                "_id": 0,

                "published_at": 1,

                "news_sentiment": 1,
            }
        )
    )


    weighted_sum = 0.0

    total_weight = 0.0


    for doc in docs:

        sentiment = (
            doc.get(
                "news_sentiment"
            )
            or {}
        )


        score = (
            sentiment.get(
                "score"
            )
        )


        if score not in (
            -1,
            0,
            1,
        ):

            continue


        confidence = (
            sentiment.get(
                "confidence",
                1.0
            )
        )


        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 1.0


        confidence = clamp(
            confidence,
            0.0,
            1.0
        )


        published_at = (
            as_aware_utc(
                doc.get(
                    "published_at"
                )
            )
        )


        if published_at is None:

            age_hours = (
                NEWS_WINDOW_HOURS
            )

        else:

            age_hours = max(

                0.0,

                (
                    now
                    - published_at
                )
                .total_seconds()
                / 3600.0
            )


        # Newer news receives
        # more influence.

        recency_weight = math.exp(

            -age_hours

            / max(
                NEWS_WINDOW_HOURS,
                1
            )
        )


        weight = max(

            0.05,

            confidence
            * recency_weight
        )


        weighted_sum += (
            float(
                score
            )
            * weight
        )


        total_weight += (
            weight
        )


    if total_weight > 0:

        sentiment_score = (
            weighted_sum
            / total_weight
        )

    else:

        sentiment_score = 0.0


    return {

        "sentiment":
            sentiment_score,

        "article_count":
            len(
                docs
            ),
    }


# =========================================================
# STOCKTWITS SIGNAL
# =========================================================

def get_stocktwits_signal(
    symbol,
    now
):

    cutoff = (
        now
        - timedelta(
            minutes=
                STOCKTWITS_WINDOW_MINUTES
        )
    )


    posts = list(

        stocktwits_posts.find(

            {
                "symbol":
                    symbol,

                "created_at": {
                    "$gte":
                        cutoff,

                    "$lte":
                        now,
                },
            },

            {
                "_id": 0,

                "sentiment": 1,
            }
        )
    )


    bullish = 0
    bearish = 0


    for post in posts:

        sentiment = (
            post.get(
                "sentiment"
            )
        )


        if sentiment == "Bullish":

            bullish += 1


        elif sentiment == "Bearish":

            bearish += 1


    labeled = (
        bullish
        + bearish
    )


    if labeled > 0:

        sentiment_score = (

            bullish
            - bearish

        ) / labeled

    else:

        sentiment_score = 0.0


    return {

        "sentiment":
            sentiment_score,

        "density":
            len(
                posts
            ),

        "messages":
            len(
                posts
            ),

        "bullish":
            bullish,

        "bearish":
            bearish,
    }


# =========================================================
# 15-MINUTE PRICE MOMENTUM
# =========================================================

def get_price_momentum(
    symbol,
    current_price,
    now
):

    target_time = (
        now
        - timedelta(
            minutes=
                MOMENTUM_MINUTES
        )
    )


    old_snapshot = (
        analytics_snapshots.find_one(

            {
                "symbol":
                    symbol,

                "timestamp": {
                    "$lte":
                        target_time
                },

                "price": {
                    "$ne":
                        None
                },
            },

            {
                "_id": 0,

                "price": 1,

                "timestamp": 1,
            },

            sort=[
                (
                    "timestamp",
                    -1
                )
            ]
        )
    )


    if not old_snapshot:

        return 0.0


    old_price = (
        old_snapshot.get(
            "price"
        )
    )


    if (
        old_price is None
        or old_price <= 0
    ):

        return 0.0


    return (

        (
            current_price
            - old_price
        )

        / old_price

    ) * 100.0


# =========================================================
# STOCKTWITS DENSITY CHANGE
# =========================================================

def get_density_change(
    symbol,
    current_density,
    now
):

    target_time = (
        now
        - timedelta(
            minutes=
                MOMENTUM_MINUTES
        )
    )


    old_snapshot = (
        analytics_snapshots.find_one(

            {
                "symbol":
                    symbol,

                "timestamp": {
                    "$lte":
                        target_time
                },

                "stocktwits_density": {
                    "$ne":
                        None
                },
            },

            {
                "_id": 0,

                "stocktwits_density":
                    1,
            },

            sort=[
                (
                    "timestamp",
                    -1
                )
            ]
        )
    )


    if not old_snapshot:

        return 0.0


    old_density = (
        old_snapshot.get(
            "stocktwits_density"
        )
    )


    if old_density is None:

        return 0.0


    return (

        float(
            current_density
        )

        -

        float(
            old_density
        )
    )


# =========================================================
# COMBINED AI / MARKET SIGNAL
# =========================================================

def calculate_signal(
    symbol,
    now=None
):

    if now is None:

        now = utc_now()


    market = (
        get_fresh_market_data(
            symbol,
            now
        )
    )


    if market is None:

        return None


    price = float(
        market[
            "price"
        ]
    )


    volume = int(
        market.get(
            "volume"
        )
        or 0
    )


    daily_change = float(
        market.get(
            "change_percent"
        )
        or 0.0
    )


    news = (
        get_news_signal(
            symbol,
            now
        )
    )


    stocktwits = (
        get_stocktwits_signal(
            symbol,
            now
        )
    )


    momentum = (
        get_price_momentum(
            symbol,
            price,
            now
        )
    )


    density_change = (
        get_density_change(
            symbol,
            stocktwits[
                "density"
            ],
            now
        )
    )


    # =====================================================
    # NORMALIZE SIGNALS
    # =====================================================

    news_component = clamp(

        news[
            "sentiment"
        ],

        -1.0,
        1.0
    )


    st_component = clamp(

        stocktwits[
            "sentiment"
        ],

        -1.0,
        1.0
    )


    # +/- 2% movement over 15m
    # becomes +/- 1.

    momentum_component = clamp(

        momentum
        / 2.0,

        -1.0,
        1.0
    )


    # +/- 5% daily movement
    # becomes +/- 1.

    daily_component = clamp(

        daily_change
        / 5.0,

        -1.0,
        1.0
    )


    # =====================================================
    # BASE SCORE
    # =====================================================

    base_score = (

        0.35
        * news_component

        +

        0.35
        * st_component

        +

        0.20
        * momentum_component

        +

        0.10
        * daily_component
    )


    # =====================================================
    # STOCKTWITS DENSITY
    #
    # Density itself has no bullish/
    # bearish direction, so it changes
    # confidence rather than direction.
    # =====================================================

    density_confidence = clamp(

        stocktwits[
            "density"
        ]
        / 100.0,

        0.0,
        1.0
    )


    confidence_multiplier = (

        0.80

        +

        0.20
        * density_confidence
    )


    score = (

        base_score
        * confidence_multiplier
    )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    if score >= BUY_THRESHOLD:

        decision = "BUY"


    elif score <= SELL_THRESHOLD:

        decision = "SELL"


    else:

        decision = "WATCH"


    # Avoid buying extremely
    # illiquid stocks.

    if (
        volume < MIN_VOLUME
        and decision == "BUY"
    ):

        decision = "WATCH"


    reason = (

        f"News="
        f"{news_component:+.2f}; "

        f"ST="
        f"{st_component:+.2f}; "

        f"15m momentum="
        f"{momentum:+.2f}%; "

        f"daily change="
        f"{daily_change:+.2f}%; "

        f"ST density="
        f"{stocktwits['density']}/hr; "

        f"15m density change="
        f"{density_change:+.0f}"
    )


    return {

        "symbol":
            symbol,

        "timestamp":
            now,

        "decision":
            decision,

        "score":
            round(
                score,
                4
            ),

        "price":
            price,

        "volume":
            volume,

        "market_cap_millions":
            market.get(
                "market_cap_millions"
            ),

        "market_updated_at":
            market.get(
                "updated_at"
            ),

        "daily_change_percent":
            daily_change,

        "momentum_15m_percent":
            momentum,

        "news_sentiment":
            news[
                "sentiment"
            ],

        "news_count":
            news[
                "article_count"
            ],

        "stocktwits_sentiment":
            stocktwits[
                "sentiment"
            ],

        "stocktwits_density":
            stocktwits[
                "density"
            ],

        "stocktwits_messages":
            stocktwits[
                "messages"
            ],

        "stocktwits_bullish":
            stocktwits[
                "bullish"
            ],

        "stocktwits_bearish":
            stocktwits[
                "bearish"
            ],

        "density_change_15m":
            density_change,

        "reason":
            reason,
    }


# =========================================================
# SHOULD AGENT RECONSIDER THIS STOCK?
#
# This is what prevents the agent from
# mechanically trading every 3 minutes.
# =========================================================

def meaningful_change(
    current_signal,
    previous_decision
):

    if not previous_decision:

        return True


    # If BUY/WATCH/SELL state itself
    # changed, definitely reconsider.

    if (
        current_signal.get(
            "decision"
        )
        !=
        previous_decision.get(
            "decision"
        )
    ):

        return True


    old_score = float(

        previous_decision.get(
            "score"
        )
        or 0.0
    )


    old_price = (
        previous_decision.get(
            "price"
        )
    )


    old_density = int(

        previous_decision.get(
            "stocktwits_density"
        )
        or 0
    )


    score_change = abs(

        current_signal[
            "score"
        ]

        -

        old_score
    )


    density_change = abs(

        current_signal[
            "stocktwits_density"
        ]

        -

        old_density
    )


    price_change_percent = 0.0


    if old_price:

        price_change_percent = abs(

            (
                current_signal[
                    "price"
                ]

                -

                float(
                    old_price
                )
            )

            /

            float(
                old_price
            )

            * 100.0
        )


    old_news_count = int(

        previous_decision.get(
            "news_count"
        )
        or 0
    )


    news_changed = (

        current_signal[
            "news_count"
        ]

        !=

        old_news_count
    )


    return (

        score_change
        >=
        MIN_SCORE_CHANGE_TO_RECONSIDER

        or

        price_change_percent
        >=
        MIN_PRICE_MOVE_TO_RECONSIDER

        or

        density_change
        >=
        MIN_DENSITY_CHANGE_TO_RECONSIDER

        or

        news_changed
    )


# =========================================================
# ACTIVE STOCK UNIVERSE
# =========================================================

def get_active_universe():

    raw_symbols = (
        headlines.distinct(
            "securities.symbol"
        )
    )


    valid_symbols = []


    for symbol in raw_symbols:

        if (
            not isinstance(
                symbol,
                str
            )
            or not symbol.strip()
        ):

            continue


        symbol = (
            symbol
            .strip()
            .upper()
        )


        company = (
            companies.find_one(

                {
                    "symbol":
                        symbol
                },

                {
                    "_id": 0,

                    "exchange": 1,
                }
            )
        )


        if not company:

            continue


        exchange = (

            company.get(
                "exchange"
            )
            or ""

        ).strip().upper()


        if exchange not in (
            ALLOWED_EXCHANGES
        ):

            continue


        valid_symbols.append(
            symbol
        )


    return sorted(
        set(
            valid_symbols
        )
    )


# =========================================================
# PORTFOLIO SNAPSHOT
# =========================================================

def get_portfolio_snapshot():

    state = get_agent_state()


    cash = float(

        state.get(
            "cash",
            STARTING_CASH
        )
    )


    positions = []


    positions_value = 0.0

    unrealized_total = 0.0


    for position in (
        agent_positions
        .find()
        .sort(
            "symbol",
            1
        )
    ):

        symbol = (
            position[
                "symbol"
            ]
        )


        shares = int(

            position.get(
                "shares"
            )
            or 0
        )


        avg_price = float(

            position.get(
                "avg_price"
            )
            or 0.0
        )


        market = (
            market_data.find_one(

                {
                    "symbol":
                        symbol
                },

                {
                    "_id": 0,

                    "price": 1,
                }
            )
            or {}
        )


        current_price = (
            market.get(
                "price"
            )
        )


        if current_price is None:

            current_price = (
                position.get(
                    "last_price",
                    avg_price
                )
            )


        current_price = float(

            current_price
            or 0.0
        )


        market_value = (

            shares
            * current_price
        )


        cost_basis = (

            shares
            * avg_price
        )


        unrealized_pnl = (

            market_value
            - cost_basis
        )


        if cost_basis:

            pnl_percent = (

                unrealized_pnl

                / cost_basis

                * 100.0
            )

        else:

            pnl_percent = 0.0


        positions_value += (
            market_value
        )


        unrealized_total += (
            unrealized_pnl
        )


        positions.append({

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
        })


    portfolio_value = (

        cash
        + positions_value
    )


    starting_cash = float(

        state.get(
            "starting_cash",
            STARTING_CASH
        )
    )


    total_pnl = (

        portfolio_value
        - starting_cash
    )


    if starting_cash:

        total_pnl_percent = (

            total_pnl

            / starting_cash

            * 100.0
        )

    else:

        total_pnl_percent = 0.0


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
            positions,
    }


# =========================================================
# DAILY SESSION
# =========================================================

def ensure_daily_session():

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

        {
            "_id":
                AGENT_ID
        },

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
        }
    )


    return True


# =========================================================
# TRADE COOLDOWN
# =========================================================

def trade_on_cooldown(
    symbol,
    now
):

    last_trade = (
        agent_trades.find_one(

            {
                "symbol":
                    symbol
            },

            sort=[
                (
                    "timestamp",
                    -1
                )
            ]
        )
    )


    if not last_trade:

        return False


    timestamp = (
        as_aware_utc(
            last_trade.get(
                "timestamp"
            )
        )
    )


    if timestamp is None:

        return False


    return (

        now
        - timestamp

        <

        timedelta(
            minutes=
                TRADE_COOLDOWN_MINUTES
        )
    )


# =========================================================
# EXECUTE PAPER BUY
# =========================================================

def execute_buy(
    signal,
    now
):

    symbol = (
        signal[
            "symbol"
        ]
    )


    price = float(

        signal[
            "price"
        ]
    )


    state = get_agent_state()


    cash = float(

        state.get(
            "cash"
        )
        or 0.0
    )


    portfolio = (
        get_portfolio_snapshot()
    )


    portfolio_value = (
        portfolio[
            "portfolio_value"
        ]
    )


    existing = (
        agent_positions.find_one(
            {
                "symbol":
                    symbol
            }
        )
    )


    current_position_value = 0.0


    if existing:

        current_position_value = (

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


    available_position_room = (

        max_position_value
        - current_position_value
    )


    amount_to_spend = min(

        desired_purchase,

        available_position_room,

        cash
    )


    if amount_to_spend <= 0:

        return None


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


        old_avg = float(

            existing.get(
                "avg_price"
            )
            or 0.0
        )


        new_shares = (

            old_shares
            + shares
        )


        new_avg_price = (

            (
                old_shares
                * old_avg
            )

            +

            trade_value

        ) / new_shares


        agent_positions.update_one(

            {
                "symbol":
                    symbol
            },

            {
                "$set": {

                    "shares":
                        new_shares,

                    "avg_price":
                        new_avg_price,

                    "last_price":
                        price,

                    "updated_at":
                        now,
                }
            }
        )


    else:

        agent_positions.insert_one({

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
        })


    agent_state.update_one(

        {
            "_id":
                AGENT_ID
        },

        {
            "$inc": {

                "cash":
                    -trade_value
            },

            "$set": {

                "updated_at":
                    now
            },
        }
    )


    trade = {

        "trade_id":
            str(
                uuid4()
            ),

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

        "score":
            signal[
                "score"
            ],

        "reason":
            signal[
                "reason"
            ],

        "signal":
            signal,
    }


    agent_trades.insert_one(
        trade
    )


    return trade


# =========================================================
# EXECUTE PAPER SELL
# =========================================================

def execute_sell(
    signal,
    now,
    reason_override=None
):

    symbol = (
        signal[
            "symbol"
        ]
    )


    position = (
        agent_positions.find_one(
            {
                "symbol":
                    symbol
            }
        )
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


    price = float(

        signal[
            "price"
        ]
    )


    trade_value = (

        shares
        * price
    )


    avg_price = float(

        position.get(
            "avg_price"
        )
        or 0.0
    )


    realized_pnl = (

        price
        - avg_price

    ) * shares


    agent_state.update_one(

        {
            "_id":
                AGENT_ID
        },

        {
            "$inc": {

                "cash":
                    trade_value
            },

            "$set": {

                "updated_at":
                    now
            },
        }
    )


    agent_positions.delete_one(
        {
            "symbol":
                symbol
        }
    )


    trade = {

        "trade_id":
            str(
                uuid4()
            ),

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

        "score":
            signal.get(
                "score",
                0.0
            ),

        "reason":
            (
                reason_override
                or
                signal.get(
                    "reason",
                    ""
                )
            ),

        "signal":
            signal,
    }


    agent_trades.insert_one(
        trade
    )


    return trade


# =========================================================
# FORCE CLOSE EVERY POSITION
# =========================================================

def force_close_all(
    signals,
    now,
    reason_text
):

    signal_map = {

        signal[
            "symbol"
        ]:
            signal

        for signal
        in signals
    }


    trades = []


    positions = list(
        agent_positions.find({})
    )


    for position in positions:

        symbol = (
            position[
                "symbol"
            ]
        )


        signal = (
            signal_map.get(
                symbol
            )
        )


        if signal is None:

            market = (
                get_fresh_market_data(
                    symbol,
                    now
                )
            )


            if market is None:

                continue


            signal = {

                "symbol":
                    symbol,

                "price":
                    float(
                        market[
                            "price"
                        ]
                    ),

                "score":
                    0.0,

                "reason":
                    reason_text,
            }


        trade = (
            execute_sell(

                signal,

                now,

                reason_override=
                    reason_text
            )
        )


        if trade:

            trades.append(
                trade
            )


    return trades


# =========================================================
# MAIN AGENT
# =========================================================

def run_agent():

    now = utc_now()


    state = get_agent_state()


    # -----------------------------------------------------
    # Agent can observe automatically only while enabled.
    # -----------------------------------------------------

    if not state.get(
        "enabled",
        False
    ):

        return {

            "enabled":
                False,

            "status":
                "Agent disabled",

            "signals":
                0,

            "trades":
                0,
        }


    new_session = (
        ensure_daily_session()
    )


    symbols = (
        get_active_universe()
    )


    signals = []


    # =====================================================
    # OBSERVE ALL CURRENT STOCKS
    #
    # Every 3-minute collection provides a
    # new observation opportunity.
    #
    # This does NOT mean a trade is required.
    # =====================================================

    for symbol in symbols:

        try:

            previous = (
                agent_decisions.find_one(
                    {
                        "symbol":
                            symbol
                    }
                )
            )


            signal = (
                calculate_signal(
                    symbol,
                    now
                )
            )


            if signal is None:

                continue


            changed = (
                meaningful_change(
                    signal,
                    previous
                )
            )


            existing_position = (
                agent_positions.find_one(
                    {
                        "symbol":
                            symbol
                    }
                )
            )


            signal[
                "meaningful_change"
            ] = changed


            signal[
                "has_position"
            ] = bool(
                existing_position
            )


            # Save latest state for dashboard.
            #
            # This is NOT a trade record.

            agent_decisions.update_one(

                {
                    "symbol":
                        symbol
                },

                {
                    "$set": {

                        **signal,

                        "updated_at":
                            now,
                    }
                },

                upsert=True
            )


            signals.append(
                signal
            )


        except Exception as exc:

            print(
                f"Agent signal error "
                f"{symbol}: {exc}"
            )


    # =====================================================
    # NO PAPER TRADING WHEN MARKET CLOSED
    # =====================================================

    if not market_is_open():

        agent_state.update_one(

            {
                "_id":
                    AGENT_ID
            },

            {
                "$set": {

                    "last_run":
                        now,

                    "updated_at":
                        now,

                    "last_status":
                        (
                            "Market closed - "
                            "signals observed only"
                        ),
                }
            }
        )


        return {

            "enabled":
                True,

            "status":
                "Market closed",

            "signals":
                len(
                    signals
                ),

            "trades":
                0,
        }


    # =====================================================
    # RECOVER FROM ACCIDENTAL OVERNIGHT POSITIONS
    #
    # Example:
    # computer was turned off at 3:55.
    # =====================================================

    if (
        new_session
        and
        agent_positions
        .count_documents({})
        > 0
    ):

        recovery_trades = (
            force_close_all(

                signals,

                now,

                (
                    "Overnight recovery exit. "
                    "This intraday agent does "
                    "not intentionally carry "
                    "positions overnight."
                )
            )
        )


        agent_state.update_one(

            {
                "_id":
                    AGENT_ID
            },

            {
                "$set": {

                    "last_run":
                        now,

                    "updated_at":
                        now,

                    "last_status":
                        (
                            f"Recovered "
                            f"{len(recovery_trades)} "
                            f"overnight positions"
                        ),
                }
            }
        )


        return {

            "enabled":
                True,

            "status":
                "Overnight recovery",

            "signals":
                len(
                    signals
                ),

            "trades":
                len(
                    recovery_trades
                ),
        }


    # =====================================================
    # 3:55 PM FORCED LIQUIDATION
    # =====================================================

    if must_close_all_positions():

        forced_trades = (
            force_close_all(

                signals,

                now,

                (
                    "End-of-day forced exit "
                    "at 3:55 PM ET. "
                    "Intraday agent does not "
                    "hold overnight."
                )
            )
        )


        agent_state.update_one(

            {
                "_id":
                    AGENT_ID
            },

            {
                "$set": {

                    "last_run":
                        now,

                    "updated_at":
                        now,

                    "last_status":
                        (
                            "End-of-day liquidation: "
                            f"{len(forced_trades)} "
                            "positions closed"
                        ),
                }
            }
        )


        return {

            "enabled":
                True,

            "status":
                "End-of-day liquidation",

            "signals":
                len(
                    signals
                ),

            "trades":
                len(
                    forced_trades
                ),
        }


    trades_made = []


    signal_map = {

        signal[
            "symbol"
        ]:
            signal

        for signal
        in signals
    }


    # =====================================================
    # EXIT EXISTING POSITIONS FIRST
    #
    # There is NO maximum holding time.
    # =====================================================

    positions = list(
        agent_positions.find({})
    )


    for position in positions:

        if (
            len(
                trades_made
            )
            >=
            MAX_TRADES_PER_OBSERVATION
        ):

            break


        symbol = (
            position[
                "symbol"
            ]
        )


        signal = (
            signal_map.get(
                symbol
            )
        )


        if signal is None:

            continue


        avg_price = float(

            position.get(
                "avg_price",
                signal[
                    "price"
                ]
            )
        )


        pnl_percent = (

            (
                signal[
                    "price"
                ]
                - avg_price
            )

            / avg_price

            * 100.0
        )


        take_profit = (

            pnl_percent
            >=
            TAKE_PROFIT_PERCENT
        )


        stop_loss = (

            pnl_percent
            <=
            STOP_LOSS_PERCENT
        )


        signal_exit = (

            signal[
                "score"
            ]

            <=

            SELL_THRESHOLD
        )


        # Agent keeps holding if
        # none of these exit rules fire.

        if not (

            take_profit

            or stop_loss

            or signal_exit

        ):

            continue


        if take_profit:

            reason = (

                "Take-profit triggered at "

                f"{pnl_percent:+.2f}%. "

                f"{signal['reason']}"
            )


        elif stop_loss:

            reason = (

                "Stop-loss triggered at "

                f"{pnl_percent:+.2f}%. "

                f"{signal['reason']}"
            )


        else:

            reason = (

                "Trading signal reversed. "

                f"{signal['reason']}"
            )


        trade = (
            execute_sell(

                signal,

                now,

                reason_override=
                    reason
            )
        )


        if trade:

            trades_made.append(
                trade
            )


    # =====================================================
    # NEW PURCHASES
    # =====================================================

    if can_open_new_positions():

        buy_candidates = sorted(

            [

                signal

                for signal
                in signals

                if (

                    signal[
                        "decision"
                    ]
                    == "BUY"

                    and

                    signal[
                        "meaningful_change"
                    ]
                )

            ],

            key=lambda item:
                item[
                    "score"
                ],

            reverse=True
        )


        for signal in buy_candidates:

            if (
                len(
                    trades_made
                )
                >=
                MAX_TRADES_PER_OBSERVATION
            ):

                break


            symbol = (
                signal[
                    "symbol"
                ]
            )


            # Don't repeatedly trade the
            # same ticker back and forth.

            if trade_on_cooldown(
                symbol,
                now
            ):

                continue


            existing = (
                agent_positions.find_one(
                    {
                        "symbol":
                            symbol
                    }
                )
            )


            current_positions = (
                agent_positions
                .count_documents({})
            )


            # Existing position can receive
            # another 5% purchase until its
            # 10% position cap is reached.
            #
            # New symbols are capped at
            # MAX_POSITIONS.

            if (
                not existing

                and

                current_positions
                >=
                MAX_POSITIONS
            ):

                continue


            trade = (
                execute_buy(
                    signal,
                    now
                )
            )


            if trade:

                trades_made.append(
                    trade
                )


    # =====================================================
    # UPDATE AGENT STATUS
    # =====================================================

    agent_state.update_one(

        {
            "_id":
                AGENT_ID
        },

        {
            "$set": {

                "last_run":
                    now,

                "updated_at":
                    now,

                "last_status":
                    (
                        f"Observed "
                        f"{len(signals)} stocks; "
                        f"{len(trades_made)} "
                        f"paper trades executed"
                    ),
            }
        }
    )


    return {

        "enabled":
            True,

        "status":
            "completed",

        "signals":
            len(
                signals
            ),

        "trades":
            len(
                trades_made
            ),
    }


# =========================================================
# TODAY'S TRADE STATS
# =========================================================

def get_today_trade_stats():

    now_et = eastern_now()


    start_et = (
        now_et.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    )


    start_utc = (
        start_et.astimezone(
            timezone.utc
        )
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
            }
        )
    )


    realized_pnl = 0.0

    closed_trades = 0

    wins = 0


    for trade in trades:

        if (
            trade.get(
                "side"
            )
            != "SELL"
        ):

            continue


        closed_trades += 1


        pnl = float(

            trade.get(
                "realized_pnl"
            )
            or 0.0
        )


        realized_pnl += (
            pnl
        )


        if pnl > 0:

            wins += 1


    if closed_trades:

        win_rate = (

            wins
            / closed_trades
            * 100.0
        )

    else:

        win_rate = 0.0


    return {

        "trade_count":
            len(
                trades
            ),

        "closed_trades":
            closed_trades,

        "winning_trades":
            wins,

        "win_rate":
            win_rate,

        "realized_pnl":
            realized_pnl,
    }


# =========================================================
# DASHBOARD
# =========================================================

def get_agent_dashboard():

    state = get_agent_state()


    portfolio = (
        get_portfolio_snapshot()
    )


    today = (
        get_today_trade_stats()
    )


    day_start_value = float(

        state.get(
            "day_start_value",
            portfolio[
                "portfolio_value"
            ]
        )
    )


    daily_total_pnl = (

        portfolio[
            "portfolio_value"
        ]

        -

        day_start_value
    )


    if day_start_value:

        daily_total_pnl_percent = (

            daily_total_pnl

            / day_start_value

            * 100.0
        )

    else:

        daily_total_pnl_percent = 0.0


    # =====================================================
    # ALL BUY / SELL ACTIVITY
    # =====================================================

    trades = list(

        agent_trades.find(

            {},

            {
                "_id": 0,

                # Don't send massive embedded
                # signal object to template.
                "signal": 0,
            }
        )
        .sort(
            "timestamp",
            -1
        )
    )


    for trade in trades:

        trade[
            "timestamp_display"
        ] = (
            format_et(
                trade.get(
                    "timestamp"
                )
            )
        )


    # =====================================================
    # LATEST OBSERVATION PER STOCK
    # =====================================================

    decisions = list(

        agent_decisions.find(

            {},

            {
                "_id": 0,
            }
        )
        .sort(
            "score",
            -1
        )
        .limit(
            100
        )
    )


    for decision in decisions:

        decision[
            "updated_at_display"
        ] = (
            format_et(
                decision.get(
                    "updated_at"
                )
            )
        )


    return {

        "state":
            state,

        "portfolio":
            portfolio,

        "today": {

            **today,

            "day_start_value":
                day_start_value,

            "total_pnl":
                daily_total_pnl,

            "total_pnl_percent":
                daily_total_pnl_percent,
        },

        "trades":
            trades,

        "decisions":
            decisions,

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
# RESET PAPER ACCOUNT
# =========================================================

def reset_agent():

    now = utc_now()


    today = (
        eastern_now()
        .date()
        .isoformat()
    )


    agent_positions.delete_many(
        {}
    )


    agent_trades.delete_many(
        {}
    )


    agent_decisions.delete_many(
        {}
    )


    agent_state.update_one(

        {
            "_id":
                AGENT_ID
        },

        {
            "$set": {

                "enabled":
                    False,

                "starting_cash":
                    STARTING_CASH,

                "cash":
                    STARTING_CASH,

                "session_date":
                    today,

                "day_start_value":
                    STARTING_CASH,

                "updated_at":
                    now,

                "last_run":
                    None,

                "last_status":
                    "Paper portfolio reset",
            }
        },

        upsert=True
    )