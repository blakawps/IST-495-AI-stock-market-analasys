import os
import tempfile
import subprocess
import requests
import feedparser
import calendar
import hashlib
import re
from datetime import datetime, timezone, timedelta

import feedparser
from pymongo.errors import PyMongoError

from config import RSS_USER_AGENT
from db import ensure_indexes, headlines, keywords, rss_feeds


def download_globenewswire_feed(feed_url):
    fd, temp_path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)

    try:
        result = subprocess.run(
            [
                "curl.exe",
                "-4",
                "-L",
                "--max-time",
                "20",
                "--retry",
                "3",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                feed_url,
                "-o",
                temp_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise Exception(
                f"curl failed with code {result.returncode}: "
                f"{result.stderr.strip()}"
            )

        with open(temp_path, "rb") as file:
            feed_content = file.read()

        if not feed_content:
            raise Exception("GlobeNewswire returned an empty feed")

        print(
            f"Downloaded {len(feed_content)} bytes "
            f"from GlobeNewswire"
        )

        return feed_content

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def download_feed(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": "StockMarketNewsProject/1.0"
        },
        timeout=15
    )

    response.raise_for_status()

    return feedparser.parse(response.content)

def get_active_keywords():
    return [
        item["word"].strip().lower()
        for item in keywords.find({"enabled": True}, {"_id": 0, "word": 1})
        if item.get("word", "").strip()
    ]


def get_active_feeds():
    return list(
        rss_feeds.find(
            {"enabled": True},
            {"_id": 0, "name": 1, "url": 1},
        )
    )


def keyword_matches(headline, filter_words):
    """Return all configured words/phrases found in a headline."""
    matches = []
    for word in filter_words:
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        if re.search(pattern, headline, flags=re.IGNORECASE):
            matches.append(word)
    return matches


def parse_feed_datetime(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None

    timestamp = calendar.timegm(parsed)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

EXCHANGE_ALIASES = {
    "nasdaq": "NASDAQ",
    "nasdaq global market": "NASDAQ",
    "nasdaq capital market": "NASDAQ",
    "nasdaq global select market": "NASDAQ",

    "nyse": "NYSE",
    "nyse american": "NYSE AMERICAN",
    "amex": "NYSE AMERICAN",
    "nyse arca": "NYSE ARCA",

    "otc markets": "OTC",
    "other otc": "OTC",
    "otcqx": "OTCQX",
    "otcqb": "OTCQB",

    "tsx": "TSX",
    "tsx-v": "TSX-V",
    "tsx venture": "TSX-V",

    "lse": "LSE",
    "london": "LSE",

    "amsterdam": "EURONEXT AMSTERDAM",
}


def normalize_exchange(exchange):
    if not exchange:
        return None

    return EXCHANGE_ALIASES.get(
        exchange.strip().lower()
    )


def valid_ticker(symbol):
    if not symbol:
        return False

    symbol = symbol.strip().upper()

    return bool(
        re.fullmatch(
            r"[A-Z][A-Z0-9.\-]{0,9}",
            symbol
        )
    )


def extract_securities(entry, title):
    securities = []
    seen_symbols = set()

    def add_security(symbol, exchange=None, source=None):
        if not symbol:
            return

        symbol = (
            symbol.strip()
            .upper()
            .replace("$", "")
        )

        if not valid_ticker(symbol):
            return

        normalized_exchange = None

        if exchange:
            normalized_exchange = (
                normalize_exchange(exchange)
                or exchange.strip().upper()
            )

        # Avoid adding the same ticker twice
        if symbol in seen_symbols:
            return

        seen_symbols.add(symbol)

        securities.append({
            "symbol": symbol,
            "exchange": normalized_exchange,
            "source": source,
        })

    # -------------------------------------------------
    # 1. RSS metadata
    # Example GlobeNewswire:
    # Nasdaq:NVDA
    # NYSE:IBM
    # -------------------------------------------------

    for tag in entry.get("tags", []):

        term = str(
            tag.get("term", "")
        ).strip()

        if ":" not in term:
            continue

        exchange_text, symbol = term.split(
            ":",
            1
        )

        exchange = normalize_exchange(
            exchange_text
        )

        if not exchange:
            continue

        add_security(
            symbol=symbol,
            exchange=exchange,
            source="rss_metadata"
        )

    # -------------------------------------------------
    # 2. Exchange:ticker appearing directly in title
    # -------------------------------------------------

    exchange_pattern = re.compile(
        r"\b("
        r"NASDAQ|"
        r"NYSE|"
        r"NYSE\s+American|"
        r"NYSE\s+Arca|"
        r"AMEX|"
        r"OTCQX|"
        r"OTCQB|"
        r"OTC\s+Markets|"
        r"TSX|"
        r"TSX-V|"
        r"LSE"
        r")"
        r"\s*:\s*"
        r"\$?"
        r"([A-Z][A-Z0-9.\-]{0,9})\b",
        re.IGNORECASE
    )

    for match in exchange_pattern.finditer(title):

        add_security(
            symbol=match.group(2),
            exchange=match.group(1),
            source="headline"
        )

    # -------------------------------------------------
    # 3. $TICKER in headline
    # -------------------------------------------------

    dollar_pattern = re.compile(
        r"\$([A-Z][A-Z0-9.\-]{0,9})\b"
    )

    for match in dollar_pattern.finditer(title):

        add_security(
            symbol=match.group(1),
            exchange=None,
            source="headline"
        )

    return securities


def make_headline_hash(source_name, title, link):

    if link:
        raw = link.strip().encode("utf-8")
    else:
        raw = f"{source_name}|{title}".encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def collect_news():
    ensure_indexes()

    active_keywords = get_active_keywords()
    active_feeds = get_active_feeds()
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=3)

    stats = {
        "feeds_checked": 0,
        "entries_seen": 0,
        "matched": 0,
        "inserted": 0,
        "duplicates": 0,
        "with_tickers": 0,
        "feed_errors": [],
    }

    if not active_keywords:
        return {**stats, "message": "No enabled keywords found."}

    if not active_feeds:
        return {**stats, "message": "No enabled RSS feeds found."}

    for feed in active_feeds:
        source_name = feed["name"]
        feed_url = feed["url"]
        stats["feeds_checked"] += 1

        print(f"Checking feed: {source_name}")

        try:

            if "globenewswire.com" in feed_url.lower():

                print("Using curl for GlobeNewswire")

                feed_content = download_globenewswire_feed(feed_url)

                parsed_feed = feedparser.parse(feed_content)

            else:

                response = requests.get(
                    feed_url,
                    headers={"User-Agent": RSS_USER_AGENT},
                    timeout=15
                )

                response.raise_for_status()

                parsed_feed = feedparser.parse(response.content)

            print(
                f"Found {len(parsed_feed.entries)} entries "
                f"from {source_name}"
            )

        except Exception as exc:
            print(f"Error reading {source_name}: {exc}")

            stats["feed_errors"].append(
                {
                    "source": source_name,
                    "url": feed_url,
                    "error": str(exc),
                }
            )

            continue

        if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
            stats["feed_errors"].append(
                {
                    "source": source_name,
                    "url": feed_url,
                    "error": str(getattr(parsed_feed, "bozo_exception", "Feed parse error")),
                }
            )
            continue

        for entry in parsed_feed.entries:
            stats["entries_seen"] += 1

            title = entry.get("title", "").strip()

            if not title:
                continue

            # Keep description only for display
            summary = entry.get("summary", "").strip()

            # Check published date
            published_at = parse_feed_datetime(entry)

            if published_at is None:
                continue

            # Only accept news from the last 3 days
            if published_at < cutoff_time:
                continue

            # KEYWORDS ARE CHECKED AGAINST TITLE ONLY
            matched_keywords = keyword_matches(
                title,
                active_keywords
            )

            if not matched_keywords:
                continue

            # Extract ticker + exchange
            securities = extract_securities(
                entry,
                title
            )

            if securities:
                print(
                    f"TICKER FOUND: {securities} | {title}"
                )

            if securities:
                stats["with_tickers"] += 1

            stats["matched"] += 1

            link = entry.get("link", "").strip()

            headline_hash = make_headline_hash(
                source_name,
                title,
                link
            )

            document = {
                "headline_hash": headline_hash,
                "title": title,
                "link": link,
                "source": source_name,
                "feed_url": feed_url,
                "published_at": published_at,
                "collected_at": datetime.now(timezone.utc),
            }

            try:

                lookup = {
                    "headline_hash": headline_hash
                }

                if link:
                    lookup = {
                        "$or": [
                            {"headline_hash": headline_hash},
                            {"link": link},
                        ]
                    }

                result = headlines.update_one(
                    lookup,
                    {
                        "$setOnInsert": document,

                        "$set": {
                            "summary": summary,
                            "matched_keywords": matched_keywords,
                            "securities": securities,
                        }
                    },
                    upsert=True,
                )

                if result.upserted_id is not None:
                    stats["inserted"] += 1

                    print(
                        f"NEW ARTICLE INSERTED: {title}"
                    )

                else:
                    stats["duplicates"] += 1


            except PyMongoError as exc:

                stats["feed_errors"].append(
                    {
                        "source": source_name,
                        "headline": title,
                        "error": str(exc),
                    }
                )
    stats["database_total"] = headlines.count_documents({})
    stats["process_id"] = os.getpid()
    return stats


if __name__ == "__main__":
    print(collect_news())
