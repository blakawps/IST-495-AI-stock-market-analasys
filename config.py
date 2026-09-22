import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stock_news")
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
RSS_USER_AGENT = os.getenv("RSS_USER_AGENT", "StockNewsCollector/0.1")
FINVIZ_API_KEY = ("c445e0d1-4d7a-4d64-a2dc-d006f64d9933")