from datetime import (
    datetime,
    timedelta,
    timezone,
)

from db import (
    analytics_snapshots,
    market_data,
    stocktwits_posts,
)


def floor_to_three_minutes(value):

    minute = (
        value.minute
        - value.minute % 3
    )

    return value.replace(
        minute=minute,
        second=0,
        microsecond=0,
    )


def get_stocktwits_snapshot(
    symbol,
    now
):

    cutoff = (
        now
        - timedelta(minutes=60)
    )

    pipeline = [

        {
            "$match": {
                "symbol": symbol,

                "created_at": {
                    "$gte": cutoff,
                    "$lte": now,
                },
            }
        },

        {
            "$group": {

                "_id": "$symbol",

                "message_count": {
                    "$sum": 1
                },

                "bullish": {
                    "$sum": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$sentiment",
                                    "Bullish",
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },

                "bearish": {
                    "$sum": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$sentiment",
                                    "Bearish",
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
            }
        },
    ]

    result = list(
        stocktwits_posts.aggregate(
            pipeline
        )
    )

    if not result:

        return {
            "message_count": 0,
            "density": 0,
            "sentiment": None,
        }

    row = result[0]

    bullish = row.get(
        "bullish",
        0
    )

    bearish = row.get(
        "bearish",
        0
    )

    message_count = row.get(
        "message_count",
        0
    )

    labeled = (
        bullish
        + bearish
    )

    if labeled > 0:

        sentiment = (
            bullish - bearish
        ) / labeled

    else:

        sentiment = None

    return {
        "message_count":
            message_count,

        "density":
            message_count,

        "sentiment":
            sentiment,
    }


def save_analytics_snapshots(
    symbols
):

    clean_symbols = sorted({

        symbol.strip().upper()

        for symbol in symbols

        if (
            isinstance(
                symbol,
                str
            )
            and symbol.strip()
        )
    })

    now = datetime.now(
        timezone.utc
    )

    snapshot_time = (
        floor_to_three_minutes(
            now
        )
    )

    saved = 0

    for symbol in clean_symbols:

        market = market_data.find_one(
            {
                "symbol": symbol
            },
            {
                "_id": 0,
                "price": 1,
                "change_percent": 1,
            }
        ) or {}

        st = get_stocktwits_snapshot(
            symbol,
            now
        )

        analytics_snapshots.update_one(

            {
                "symbol": symbol,
                "timestamp":
                    snapshot_time,
            },

            {
                "$set": {

                    "symbol":
                        symbol,

                    "timestamp":
                        snapshot_time,

                    "price":
                        market.get(
                            "price"
                        ),

                    "change_percent":
                        market.get(
                            "change_percent"
                        ),

                    "stocktwits_sentiment":
                        st.get(
                            "sentiment"
                        ),

                    "stocktwits_density":
                        st.get(
                            "density",
                            0
                        ),

                    "stocktwits_messages":
                        st.get(
                            "message_count",
                            0
                        ),
                }
            },

            upsert=True,
        )

        saved += 1

    return saved