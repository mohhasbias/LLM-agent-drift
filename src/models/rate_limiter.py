import time


class RateLimiter:
    """Simple interval-based rate limiter (requests per minute)."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm
        self._last_time: float = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_time
        wait = self._interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_time = time.monotonic()
