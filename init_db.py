from db import ensure_indexes, keywords, rss_feeds

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


if __name__ == "__main__":
    seed_database()
