from pymongo import ASCENDING, DESCENDING, MongoClient

from config import MONGO_DB_NAME, MONGO_URI

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = _client[MONGO_DB_NAME]

headlines = db["headlines"]
rss_feeds = db["rss_feeds"]
keywords = db["keywords"]


def ensure_indexes():
    """Create indexes used by the news collector."""
    headlines.create_index([("headline_hash", ASCENDING)], unique=True)
    headlines.create_index([("collected_at", DESCENDING)])
    headlines.create_index([("published_at", DESCENDING)])
    headlines.create_index([("source", ASCENDING)])
    headlines.create_index([("matched_keywords", ASCENDING)])

    rss_feeds.create_index([("url", ASCENDING)], unique=True)
    keywords.create_index([("word", ASCENDING)], unique=True)


def ping_database():
    """Return True if MongoDB responds to a ping."""
    _client.admin.command("ping")
    return True
