import time


def fetch_with_retry(fetch, attempts=3, delay=0.01):
    """Call fetch(), retrying on failure."""
    for _ in range(attempts):
        try:
            return fetch()
        except Exception:
            time.sleep(delay)
    return None
