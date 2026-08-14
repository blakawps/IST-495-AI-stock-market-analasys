import argparse

from db import ensure_indexes, keywords, rss_feeds


def add_feed(name, url):
    ensure_indexes()
    rss_feeds.update_one(
        {"url": url},
        {"$set": {"name": name, "url": url, "enabled": True}},
        upsert=True,
    )
    print(f"Feed saved: {name} -> {url}")


def add_keyword(word):
    ensure_indexes()
    normalized = word.strip().lower()
    keywords.update_one(
        {"word": normalized},
        {"$set": {"word": normalized, "enabled": True}},
        upsert=True,
    )
    print(f"Keyword saved: {normalized}")


def list_config():
    print("\nRSS feeds:")
    for feed in rss_feeds.find().sort("name", 1):
        print(f"- {feed['name']}: {feed['url']} | enabled={feed.get('enabled', True)}")

    print("\nKeywords:")
    for item in keywords.find().sort("word", 1):
        print(f"- {item['word']} | enabled={item.get('enabled', True)}")


def main():
    parser = argparse.ArgumentParser(description="Manage stock-news RSS feeds and keywords.")
    sub = parser.add_subparsers(dest="command", required=True)

    feed_parser = sub.add_parser("add-feed")
    feed_parser.add_argument("name")
    feed_parser.add_argument("url")

    keyword_parser = sub.add_parser("add-keyword")
    keyword_parser.add_argument("word")

    sub.add_parser("list")

    args = parser.parse_args()

    if args.command == "add-feed":
        add_feed(args.name, args.url)
    elif args.command == "add-keyword":
        add_keyword(args.word)
    elif args.command == "list":
        list_config()


if __name__ == "__main__":
    main()
