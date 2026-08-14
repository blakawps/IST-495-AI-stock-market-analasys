from zoneinfo import ZoneInfo
from datetime import timezone
import os
import subprocess
import sys
from pathlib import Path
from flask import Flask, jsonify, redirect, render_template, request, url_for
from pymongo.errors import PyMongoError

from collector import collect_news
from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from db import ensure_indexes, headlines, keywords, ping_database, rss_feeds

app = Flask(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
SCHEDULER_FILE = PROJECT_DIR / "scheduler.py"
PID_FILE = PROJECT_DIR / "scheduler.pid"
LOG_FILE = PROJECT_DIR / "scheduler.log"

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(
    subprocess,
    "CREATE_NEW_PROCESS_GROUP",
    0
)

def get_scheduler_pid():
    if not PID_FILE.exists():
        return None

    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def process_is_running(pid):
    if pid is None:
        return False

    if sys.platform == "win32":
        result = subprocess.run(
            [
                "tasklist",
                "/FI",
                f"PID eq {pid}",
                "/NH"
            ],
            capture_output=True,
            text=True,
            creationflags=CREATE_NO_WINDOW
        )

        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def scheduler_is_running():
    pid = get_scheduler_pid()

    if pid is None:
        return False

    if process_is_running(pid):
        return True

    # Remove old PID file if process died
    try:
        PID_FILE.unlink()
    except OSError:
        pass

    return False

def serialize_headline(doc):
    published_at = doc.get("published_at")
    collected_at = doc.get("collected_at")

    if published_at:
        # MongoDB/PyMongo may return UTC as a naive datetime
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        published_et = published_at.astimezone(
            ZoneInfo("America/New_York")
        )

        published_display = published_et.strftime(
            "%b %d, %Y %I:%M %p ET"
        )
    else:
        published_display = None

    if collected_at:
        if collected_at.tzinfo is None:
            collected_at = collected_at.replace(tzinfo=timezone.utc)

        collected_display = collected_at.astimezone(
            ZoneInfo("America/New_York")
        ).strftime(
            "%b %d, %Y %I:%M %p ET"
        )
    else:
        collected_display = None

    return {
        "id": str(doc["_id"]),
        "title": doc.get("title"),
        "summary": doc.get("summary", ""),
        "link": doc.get("link"),
        "source": doc.get("source"),
        "matched_keywords": doc.get("matched_keywords", []),


        # Human-readable ET times for website
        "published_at": published_display,
        "collected_at": collected_display,
    }

@app.get("/api/headline-count")
def api_headline_count():
    try:
        count = headlines.count_documents({})

        return jsonify({
            "count": count
        })

    except PyMongoError as exc:
        return jsonify({
            "error": str(exc)
        }), 500

@app.get("/")
def home():
    try:
        docs = list(
            headlines.find()
            .sort([
                ("published_at", -1),
                ("collected_at", -1)
            ])
            .limit(500)
        )

        return render_template(
            "index.html",
            headlines=[
                serialize_headline(doc)
                for doc in docs
            ],
            error=None,
            scheduler_running=scheduler_is_running()
        )

    except PyMongoError as exc:
        return render_template(
            "index.html",
            headlines=[],
            error=str(exc),
            scheduler_running=scheduler_is_running()
        ), 500

@app.post("/collect")
def collect_now():
    collect_news()
    return redirect(url_for("home"))


@app.get("/health")
def health():
    try:
        ping_database()
        return jsonify({"status": "ok", "mongodb": "connected"})
    except PyMongoError as exc:
        return jsonify({"status": "error", "mongodb": "disconnected", "error": str(exc)}), 503


@app.get("/api/headlines")
def api_headlines():
    limit = min(max(request.args.get("limit", default=50, type=int), 1), 500)
    source = request.args.get("source", type=str)
    keyword = request.args.get("keyword", type=str)

    query = {}
    if source:
        query["source"] = source
    if keyword:
        query["matched_keywords"] = keyword.lower()

    docs = (
        headlines.find(query)
        .sort([
            ("published_at", -1),
            ("collected_at", -1)
        ])
        .limit(limit)
    )
    return jsonify([serialize_headline(doc) for doc in docs])


@app.post("/api/collect")
def api_collect():
    return jsonify(collect_news())


@app.get("/api/config")
def api_config():
    feeds = list(rss_feeds.find({}, {"_id": 0}).sort("name", 1))
    words = list(keywords.find({}, {"_id": 0}).sort("word", 1))
    return jsonify({"feeds": feeds, "keywords": words})

@app.route("/scheduler/start", methods=["POST"])
def start_scheduler():

    if scheduler_is_running():
        return redirect(url_for("home"))

    if not SCHEDULER_FILE.exists():
        return "scheduler.py was not found.", 500

    log_file = open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(SCHEDULER_FILE)
        ],
        cwd=PROJECT_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NEW_PROCESS_GROUP
    )

    PID_FILE.write_text(
        str(process.pid)
    )

    log_file.close()

    return redirect(url_for("home"))


@app.route("/scheduler/stop", methods=["POST"])
def stop_scheduler():

    pid = get_scheduler_pid()

    if pid and process_is_running(pid):

        if sys.platform == "win32":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(pid),
                    "/T",
                    "/F"
                ],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW
            )

        else:
            os.kill(pid, 15)

    if PID_FILE.exists():
        PID_FILE.unlink()

    return redirect(url_for("home"))


if __name__ == "__main__":
    ensure_indexes()

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False
    )
