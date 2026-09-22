import csv
from pathlib import Path
from pymongo import UpdateOne
from db import (
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
    seed_database()
    import_tickers_from_csv()
