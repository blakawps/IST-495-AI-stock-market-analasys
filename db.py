from pymongo import ASCENDING, DESCENDING, MongoClient

from config import MONGO_DB_NAME, MONGO_URI

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = _client[MONGO_DB_NAME]

headlines = db["headlines"]
rss_feeds = db["rss_feeds"]
keywords = db["keywords"]
companies = db["companies"]
market_data = db["market_data"]
stocktwits_posts = db["stocktwits_posts"]
analytics_snapshots = db["analytics_snapshots"]


def ensure_indexes():
    """Create indexes used by the news collector."""
    headlines.create_index([("headline_hash", ASCENDING)], unique=True)
    headlines.create_index([("collected_at", DESCENDING)])
    headlines.create_index([("published_at", DESCENDING)])
    headlines.create_index([("source", ASCENDING)])
    headlines.create_index([("matched_keywords", ASCENDING)])
    headlines.create_index([("securities.symbol", ASCENDING)])
    headlines.create_index([("securities.exchange", ASCENDING)])
    headlines.create_index([("securities.symbol", ASCENDING)])
    headlines.create_index([("published_at", ASCENDING)],expireAfterSeconds=48 * 60 * 60,name="headlines_48hr_ttl")
    rss_feeds.create_index([("url", ASCENDING)], unique=True)
    keywords.create_index([("word", ASCENDING)], unique=True)
    companies.create_index([("symbol", ASCENDING)],unique=True)
    companies.create_index([("company_name", ASCENDING)])
    companies.create_index([("aliases", ASCENDING)])
    companies.create_index([("country_code", ASCENDING)])
    companies.create_index([("asset_type", ASCENDING)])
    market_data.create_index([("symbol", ASCENDING)],unique=True)
    market_data.create_index([("updated_at", DESCENDING)])
    stocktwits_posts.create_index([("message_id", ASCENDING),("symbol", ASCENDING),],unique=True)
    stocktwits_posts.create_index([("symbol", ASCENDING),("created_at", DESCENDING),])
    stocktwits_posts.create_index(
        [("created_at", ASCENDING)],
        expireAfterSeconds=48 * 60 * 60,
        name="stocktwits_48hr_ttl"
    )
    analytics_snapshots.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING),],unique=True)
    analytics_snapshots.create_index([("timestamp", ASCENDING)], expireAfterSeconds= 48 * 60 * 60, name="analytics_48hr_ttl")


def ping_database():
    """Return True if MongoDB responds to a ping."""
    _client.admin.command("ping")
    return True
