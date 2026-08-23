import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

INTERVAL_MINUTES = 3

PROJECT_DIR = Path(__file__).resolve().parent
COLLECTOR_FILE = PROJECT_DIR / "collector.py"

PID_FILE = PROJECT_DIR / "scheduler.pid"
STOP_FILE = PROJECT_DIR / "scheduler.stop"


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
            print(
                f"Collector exited with code "
                f"{result.returncode}"
            )

    except Exception as exc:
        print(f"Scheduler error: {exc}")


def stop_requested():
    return STOP_FILE.exists()


def main():
    current_pid = os.getpid()

    # Scheduler records its own PID
    PID_FILE.write_text(str(current_pid))

    print("Stock News Collector Scheduler")
    print(f"Scheduler PID: {current_pid}")
    print(
        f"Running every {INTERVAL_MINUTES} minutes."
    )
    print(
        "Use the website Stop button or CTRL+C to stop.\n"
    )

    try:
        while not stop_requested():

            run_collector()

            if stop_requested():
                break

            print(
                f"\nNext collection in "
                f"{INTERVAL_MINUTES} minutes..."
            )

            # Sleep one second at a time so the website
            # can stop the scheduler immediately.
            for _ in range(
                INTERVAL_MINUTES * 60
            ):
                if stop_requested():
                    break

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")

    finally:
        print("\nAutomatic collector stopped.")

        # Only remove PID file if it still belongs
        # to this scheduler process.
        try:
            if PID_FILE.exists():

                stored_pid = PID_FILE.read_text().strip()

                if stored_pid == str(current_pid):
                    PID_FILE.unlink()

        except OSError:
            pass


if __name__ == "__main__":
    main()