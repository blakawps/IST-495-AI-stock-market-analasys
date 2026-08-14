import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

INTERVAL_MINUTES = 1

PROJECT_DIR = Path(__file__).resolve().parent
COLLECTOR_FILE = PROJECT_DIR / "collector.py"


def run_collector():
    print("\n" + "=" * 60)
    print(f"Starting collection: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(COLLECTOR_FILE)],
            cwd=PROJECT_DIR
        )

        if result.returncode == 0:
            print("Collection completed successfully.")
        else:
            print(f"Collector exited with code {result.returncode}")

    except Exception as exc:
        print(f"Scheduler error: {exc}")


def main():
    print("Stock News Collector Scheduler")
    print(f"Running every {INTERVAL_MINUTES} minutes.")
    print("Press Ctrl + C whenever you want to stop.\n")

    try:
        while True:
            run_collector()

            print(
                f"\nNext collection in {INTERVAL_MINUTES} minutes..."
            )

            time.sleep(INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user.")


if __name__ == "__main__":
    main()