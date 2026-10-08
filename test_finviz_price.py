from db import analytics_snapshots


def cleanup_analytics():

    result = analytics_snapshots.delete_many({})

    print(
        f"Deleted {result.deleted_count} "
        f"old analytics snapshots."
    )


if __name__ == "__main__":
    cleanup_analytics()