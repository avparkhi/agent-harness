import time


def fetch_with_retry(fetch, attempts=3, delay=0.01):
    """Call fetch(), retrying on ConnectionError."""
    for attempt in range(attempts):
        try:
            return fetch()
        except ConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
