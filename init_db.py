import csv
from datetime import datetime, timezone
from pathlib import Path
from pymongo import ASCENDING, DESCENDING, UpdateOne
from db import (
    db,
    ensure_indexes,
    headlines,
    keywords,
    rss_feeds,
    companies,
)

DEFAULT_FEEDS = [
    {
        "name": "SEC Press Releases",
        "url": "https://www.sec.gov/news/pressreleases.rss",
        "enabled": True,
    },
    {
        "name": "GlobeNewswire Stock Market News",
        "url": "https://www.globenewswire.com/RssFeed/subjectcode/39-Stock%20Market%20News/feedTitle/GlobeNewswire%20-%20Stock%20Market%20News",
        "enabled": True,
    }
]

DEFAULT_KEYWORDS = [
    "stock",
    "stocks",
    "shares",
    "market",
    "markets",
    "earnings",
    "investor",
    "investors",
    "IPO",
    "acquisition",
    "merger",
    "dividend",
    "Nasdaq",
    "NYSE",
]


def seed_database():
    ensure_indexes()

    for feed in DEFAULT_FEEDS:
        rss_feeds.update_one(
            {"url": feed["url"]},
            {"$setOnInsert": feed},
            upsert=True,
        )

    for word in DEFAULT_KEYWORDS:
        keywords.update_one(
            {"word": word.lower()},
            {
                "$setOnInsert": {
                    "word": word.lower(),
                    "enabled": True,
                }
            },
            upsert=True,
        )

    print("MongoDB initialized successfully.")
    print(f"Feeds in database: {rss_feeds.count_documents({})}")
    print(f"Keywords in database: {keywords.count_documents({})}")

BASE_DIR = Path(__file__).resolve().parent
TICKERS_CSV = BASE_DIR / "tickers.csv"


AGENT_ID = "paper_agent"
STARTING_CASH = 100_000.0

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
rl_locks = db["rl_locks"]


def ensure_rl_indexes():
    """
    Create indexes required by the current paper-trading / RL system.

    Permanent learning collections intentionally have NO TTL.
    Only processed pending actions are temporary.
    """

    agent_positions.create_index(
        [("symbol", ASCENDING)],
        unique=True,
    )

    agent_trades.create_index(
        [("trade_id", ASCENDING)],
        unique=True,
    )
    agent_trades.create_index(
        [("timestamp", DESCENDING)]
    )
    agent_trades.create_index(
        [
            ("symbol", ASCENDING),
            ("timestamp", DESCENDING),
        ]
    )

    agent_decisions.create_index(
        [("symbol", ASCENDING)],
        unique=True,
    )
    agent_decisions.create_index(
        [("updated_at", DESCENDING)]
    )

    rl_market_history.create_index(
        [
            ("symbol", ASCENDING),
            ("timestamp", ASCENDING),
        ],
        unique=True,
    )
    rl_market_history.create_index(
        [("timestamp", DESCENDING)]
    )

    rl_pending_actions.create_index(
        [("action_id", ASCENDING)],
        unique=True,
    )
    rl_pending_actions.create_index(
        [
            ("status", ASCENDING),
            ("due_at", ASCENDING),
        ]
    )
    rl_pending_actions.create_index(
        [("processed_at", ASCENDING)],
        expireAfterSeconds=7 * 24 * 60 * 60,
        name="rl_pending_actions_7d_ttl",
    )

    rl_experiences.create_index(
        [("experience_id", ASCENDING)],
        unique=True,
    )
    rl_experiences.create_index(
        [("action_index", ASCENDING)]
    )
    rl_experiences.create_index(
        [("created_at", DESCENDING)]
    )
    rl_experiences.create_index(
        [
            ("source", ASCENDING),
            ("created_at", DESCENDING),
        ]
    )

    rl_trade_reviews.create_index(
        [("review_id", ASCENDING)],
        unique=True,
    )
    rl_trade_reviews.create_index(
        [
            ("status", ASCENDING),
            ("due_at", ASCENDING),
        ]
    )
    rl_trade_reviews.create_index(
        [("finalized_at", DESCENDING)]
    )

    rl_predictions.create_index(
        [("prediction_id", ASCENDING)],
        unique=True,
    )
    rl_predictions.create_index(
        [
            ("status", ASCENDING),
            ("next_due_at", ASCENDING),
        ]
    )
    rl_predictions.create_index(
        [
            ("symbol", ASCENDING),
            ("timestamp", DESCENDING),
        ]
    )
    rl_predictions.create_index(
        [
            ("trainable", ASCENDING),
            ("status", ASCENDING),
        ]
    )
    rl_predictions.create_index(
        [("finalized_at", DESCENDING)]
    )

    rl_model_state.create_index(
        [("updated_at", DESCENDING)]
    )

    rl_locks.create_index(
        [("expires_at", ASCENDING)]
    )

    print("RL / prediction indexes initialized.")


def ensure_initial_agent_state():
    """
    Create the paper account only if it does not already exist.

    This does NOT reset an existing account or erase learning.
    """

    now = datetime.now(timezone.utc)

    result = agent_state.update_one(
        {"_id": AGENT_ID},
        {
            "$setOnInsert": {
                "_id": AGENT_ID,
                "enabled": False,
                "starting_cash": STARTING_CASH,
                "cash": STARTING_CASH,
                "session_date": None,
                "day_start_value": STARTING_CASH,
                "created_at": now,
                "updated_at": now,
                "last_run": None,
                "last_status": "PyTorch RL paper agent initialized",
            }
        },
        upsert=True,
    )

    if result.upserted_id is not None:
        print(
            f"Created initial paper account with "
            f"${STARTING_CASH:,.2f}."
        )
    else:
        print("Existing paper-agent state preserved.")


def print_ai_database_summary():
    collections = {
        "agent_positions": agent_positions,
        "agent_trades": agent_trades,
        "agent_decisions": agent_decisions,
        "rl_market_history": rl_market_history,
        "rl_pending_actions": rl_pending_actions,
        "rl_experiences": rl_experiences,
        "rl_trade_reviews": rl_trade_reviews,
        "rl_predictions": rl_predictions,
        "rl_model_state": rl_model_state,
    }

    print("\nCurrent AI database state:")

    for name, collection in collections.items():
        print(
            f"- {name}: "
            f"{collection.count_documents({})}"
        )


def import_tickers_from_csv():

    if not TICKERS_CSV.exists():
        print(
            f"WARNING: tickers.csv was not found at:"
            f"\n{TICKERS_CSV}"
        )
        return

    print("\nImporting stock ticker database...")
    print(f"Reading: {TICKERS_CSV}")

    operations = []

    total_rows = 0
    imported_rows = 0
    skipped_rows = 0

    with open(
        TICKERS_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            total_rows += 1

            symbol = (
                row.get("ticker") or ""
            ).strip().upper()

            company_name = (
                row.get("name") or ""
            ).strip()

            exchange = (
                row.get("exchange") or ""
            ).strip().upper()

            asset_type = (
                row.get("asset_type") or ""
            ).strip()

            stock_sector = (
                row.get("stock_sector") or ""
            ).strip()

            country = (
                row.get("country") or ""
            ).strip()

            country_code = (
                row.get("country_code") or ""
            ).strip().upper()

            isin = (
                row.get("isin") or ""
            ).strip()

            aliases_raw = (
                row.get("aliases") or ""
            ).strip()

            # -----------------------------------------
            # We only want U.S. stocks for now.
            # Excludes ETFs and international companies.
            # -----------------------------------------

            if country_code != "US":
                skipped_rows += 1
                continue

            if asset_type.lower() != "stock":
                skipped_rows += 1
                continue

            if not symbol or not company_name:
                skipped_rows += 1
                continue

            # -----------------------------------------
            # CSV aliases are separated with |
            #
            # Example:
            # meta platforms|meta
            #
            # becomes:
            # ["meta platforms", "meta"]
            # -----------------------------------------

            aliases = []

            if aliases_raw:

                for alias in aliases_raw.split("|"):

                    alias = alias.strip().lower()

                    if (
                        alias
                        and alias not in aliases
                    ):
                        aliases.append(alias)

            company_document = {
                "symbol": symbol,
                "company_name": company_name,
                "exchange": exchange,
                "asset_type": asset_type,
                "stock_sector": stock_sector,
                "country": country,
                "country_code": country_code,
                "isin": isin,
                "aliases": aliases,
                "source": "tickers.csv",
            }

            operations.append(
                UpdateOne(
                    {
                        "symbol": symbol
                    },
                    {
                        "$set": company_document,

                        "$setOnInsert": {
                            "enabled": True,
                            "manual_aliases": [],
                        }
                    },
                    upsert=True
                )
            )

            imported_rows += 1

    if operations:

        result = companies.bulk_write(
            operations,
            ordered=False
        )

        print("\nTicker import complete.")

        print(
            f"CSV rows checked: {total_rows}"
        )

        print(
            f"Eligible U.S. stocks: "
            f"{imported_rows}"
        )

        print(
            f"Rows skipped: {skipped_rows}"
        )

        print(
            f"New companies inserted: "
            f"{result.upserted_count}"
        )

        print(
            f"Existing companies updated: "
            f"{result.modified_count}"
        )

        print(
            f"Companies currently in MongoDB: "
            f"{companies.count_documents({})}"
        )

    else:

        print(
            "No eligible ticker records "
            "were found."
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Initializing Stock News + PyTorch RL database")
    print("=" * 60)

    seed_database()
    ensure_rl_indexes()
    ensure_initial_agent_state()
    import_tickers_from_csv()
    print_ai_database_summary()

    print("\nInitialization complete.")
    print(
        "Existing trades, RL experiences, predictions, "
        "and learned model data were preserved."
    )
