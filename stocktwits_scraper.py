import html
import time
from datetime import datetime, timedelta, timezone

import requests

from db import stocktwits_posts


BASE_URL = (
    "https://api.stocktwits.com/"
    "api/2/streams/symbol/"
    "{symbol}.json"
)


def parse_stocktwits_date(value):

    if not value:
        return None

    return datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00"
        )
    )


def parse_message(message, symbol):

    user = (
        message.get("user")
        or {}
    )

    entities = (
        message.get("entities")
        or {}
    )

    sentiment_data = (
        entities.get("sentiment")
        or {}
    )

    likes = (
        message.get("likes")
        or {}
    )

    conversation = (
        message.get("conversation")
        or {}
    )

    return {

        "message_id":
            message.get("id"),

        "symbol":
            symbol,

        "body":
            html.unescape(
                message.get(
                    "body",
                    ""
                )
            ),

        "created_at":
            parse_stocktwits_date(
                message.get(
                    "created_at"
                )
            ),

        "username":
            user.get(
                "username"
            ),

        "sentiment":
            sentiment_data.get(
                "basic"
            ),

        "likes":
            likes.get(
                "total",
                0
            ),

        "replies":
            conversation.get(
                "replies",
                0
            ),
    }


def save_posts(posts):

    inserted = 0

    now = datetime.now(
        timezone.utc
    )

    for post in posts:

        if not post.get(
            "message_id"
        ):
            continue

        result = (
            stocktwits_posts.update_one(

                {
                    "message_id":
                        post["message_id"],

                    "symbol":
                        post["symbol"],
                },

                {
                    "$set": {
                        **post,

                        "collected_at":
                            now,
                    }
                },

                upsert=True,
            )
        )

        if result.upserted_id:
            inserted += 1

    return inserted


def scrape_stocktwits(
    symbol,
    lookback_minutes=60,
    max_pages=50,
    session=None,
):

    symbol = (
        symbol.strip().upper()
    )

    cutoff = (
        datetime.now(
            timezone.utc
        )
        - timedelta(
            minutes=lookback_minutes
        )
    )

    if session is None:

        session = (
            requests.Session()
        )

        session.headers.update({
            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154 Safari/537.36"
        })

    url = BASE_URL.format(
        symbol=symbol
    )

    collected = []

    seen_ids = set()

    max_id = None

    for page_number in range(
        max_pages
    ):

        params = {
            "limit": 22
        }

        if max_id is not None:
            params["max"] = max_id

        try:

            response = session.get(
                url,
                params=params,
                timeout=20
            )

        except requests.RequestException as exc:

            print(
                f"Stocktwits {symbol}: "
                f"{exc}"
            )

            break

        if response.status_code == 429:

            print(
                f"Stocktwits rate limit "
                f"for {symbol}. Waiting..."
            )

            time.sleep(10)

            continue

        if response.status_code != 200:

            print(
                f"Stocktwits {symbol}: "
                f"HTTP "
                f"{response.status_code}"
            )

            break

        try:

            data = response.json()

        except ValueError:

            print(
                f"Stocktwits {symbol}: "
                f"invalid JSON"
            )

            break

        messages = data.get(
            "messages",
            []
        )

        if not messages:
            break

        reached_cutoff = False

        for message in messages:

            message_id = (
                message.get("id")
            )

            if message_id in seen_ids:
                continue

            seen_ids.add(
                message_id
            )

            created_at = (
                parse_stocktwits_date(
                    message.get(
                        "created_at"
                    )
                )
            )

            if created_at is None:
                continue

            # We reached messages older than
            # the rolling 60-minute window.
            if created_at < cutoff:

                reached_cutoff = True
                continue

            collected.append(
                parse_message(
                    message,
                    symbol
                )
            )

        if reached_cutoff:
            break

        cursor = (
            data.get("cursor")
            or {}
        )

        if not cursor.get(
            "more",
            False
        ):
            break

        next_max = cursor.get(
            "max"
        )

        if not next_max:
            break

        if next_max == max_id:
            break

        max_id = next_max

        time.sleep(0.25)

    return collected


def collect_stocktwits_for_symbols(
    symbols
):

    symbols = sorted({
        str(symbol)
        .strip()
        .upper()

        for symbol in symbols

        if symbol
    })

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/154 Safari/537.36"
    })

    stats = {

        "symbols_checked": 0,

        "messages_found": 0,

        "messages_inserted": 0,

        "errors": [],
    }

    for symbol in symbols:

        stats[
            "symbols_checked"
        ] += 1

        try:

            posts = scrape_stocktwits(
                symbol,
                lookback_minutes=60,
                session=session,
            )

            inserted = save_posts(
                posts
            )

            stats[
                "messages_found"
            ] += len(posts)

            stats[
                "messages_inserted"
            ] += inserted

            print(
                f"Stocktwits {symbol}: "
                f"{len(posts)} messages "
                f"in rolling 60m, "
                f"{inserted} new."
            )

        except Exception as exc:

            print(
                f"Stocktwits error "
                f"{symbol}: {exc}"
            )

            stats["errors"].append({
                "symbol": symbol,
                "error": str(exc),
            })

        time.sleep(0.35)

    return stats


if __name__ == "__main__":

    result = (
        collect_stocktwits_for_symbols(
            ["NVDA"]
        )
    )

    print(result)