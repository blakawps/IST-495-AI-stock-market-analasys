import csv
from datetime import datetime, timezone
from io import StringIO
import requests
from config import FINVIZ_API_KEY
from db import market_data
import time
from datetime import datetime, timezone, timedelta


FINVIZ_URL = "https://elite.finviz.com/export/screener"


def fetch_finviz_data(symbols):

    symbols = [
        symbol.strip().upper()
        for symbol in symbols
        if symbol
    ]

    if not symbols:
        return []

    params = {
        "v": "111",
        "t": ",".join(symbols),
        "auth": FINVIZ_API_KEY,
    }

    response = requests.get(
        FINVIZ_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    reader = csv.DictReader(
        StringIO(response.text)
    )

    results = []

    for row in reader:

        symbol = (
            row.get("Ticker")
            or ""
        ).strip().upper()

        if not symbol:
            continue

        try:
            market_cap_millions = float(
                row.get("Market Cap", 0)
            )
        except (ValueError, TypeError):
            market_cap_millions = None

        try:
            volume = int(
                float(
                    row.get("Volume", 0)
                )
            )
        except (ValueError, TypeError):
            volume = None

        try:
            price = float(
                row.get("Price", 0)
            )
        except (ValueError, TypeError):
            price = None

        results.append({
            "symbol": symbol,
            "company": row.get("Company"),
            "sector": row.get("Sector"),
            "industry": row.get("Industry"),

            "market_cap_millions":
                market_cap_millions,

            "volume": volume,
            "price": price,

            "updated_at":
                datetime.now(timezone.utc),

            "source": "finviz",
        })

    return results


def update_market_data(symbols):

    symbols = sorted(
        set(
            symbol.strip().upper()
            for symbol in symbols
            if symbol
        )
    )

    if not symbols:
        return {
            "requested": 0,
            "updated": 0,
            "skipped_cached": 0,
        }

    # Only refresh data older than 15 minutes
    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(minutes=15)
    )

    symbols_to_update = []

    for symbol in symbols:

        existing = market_data.find_one(
            {"symbol": symbol},
            {
                "_id": 0,
                "updated_at": 1,
            }
        )

        if not existing:
            symbols_to_update.append(symbol)
            continue

        updated_at = existing.get(
            "updated_at"
        )

        if updated_at is None:
            symbols_to_update.append(symbol)
            continue

        # MongoDB may return a naive UTC datetime
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(
                tzinfo=timezone.utc
            )

        if updated_at < cutoff:
            symbols_to_update.append(symbol)

    skipped_cached = (
        len(symbols)
        - len(symbols_to_update)
    )

    if not symbols_to_update:

        print(
            "Finviz data is still fresh. "
            "No update needed."
        )

        return {
            "requested": len(symbols),
            "updated": 0,
            "skipped_cached": skipped_cached,
        }

    updated = 0

    # Smaller batches are easier on Finviz
    batch_size = 25

    for start in range(
        0,
        len(symbols_to_update),
        batch_size
    ):

        batch = symbols_to_update[
            start:start + batch_size
        ]

        try:

            results = fetch_finviz_data(
                batch
            )

            for item in results:

                market_data.update_one(
                    {
                        "symbol":
                            item["symbol"]
                    },
                    {
                        "$set": item
                    },
                    upsert=True
                )

                updated += 1

            print(
                f"Finviz batch updated: "
                f"{len(results)} stocks"
            )

            # Slow down between API requests
            time.sleep(3)

        except requests.exceptions.HTTPError as exc:

            if (
                exc.response is not None
                and exc.response.status_code == 429
            ):

                print(
                    "Finviz rate limit reached. "
                    "Waiting 30 seconds..."
                )

                time.sleep(30)

                try:

                    results = fetch_finviz_data(
                        batch
                    )

                    for item in results:

                        market_data.update_one(
                            {
                                "symbol":
                                    item["symbol"]
                            },
                            {
                                "$set": item
                            },
                            upsert=True
                        )

                        updated += 1

                except Exception as retry_exc:

                    print(
                        "Finviz retry failed: "
                        f"{retry_exc}"
                    )

            else:

                print(
                    f"Finviz HTTP error: {exc}"
                )

        except Exception as exc:

            print(
                f"Finviz error: {exc}"
            )

    print(
        f"Finviz updated {updated} stocks. "
        f"Skipped {skipped_cached} cached stocks."
    )

    return {
        "requested": len(symbols),
        "needed_update":
            len(symbols_to_update),
        "updated": updated,
        "skipped_cached": skipped_cached,
    }