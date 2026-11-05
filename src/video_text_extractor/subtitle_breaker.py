"""Process-local subtitle throttle state; never held across network or model work."""

import logging
import math
import threading
import time
from collections.abc import Callable

from .models import SubtitleBreakerStatus

logger = logging.getLogger(__name__)
PROBE_LEASE_SECONDS = 60


class SubtitleCircuitBreaker:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._generation = 0

    def acquire(self, seconds: int) -> tuple[bool, int | None]:
        """Return permission and an optional lease token for the sole probe."""
        if seconds == 0:
            return True, None
        with self._lock:
            now = self._clock()
            if self._blocked_until == 0:
                return True, None
            if now < self._blocked_until:
                return False, None
            self._blocked_until = now + PROBE_LEASE_SECONDS
            self._generation += 1
            return True, self._generation

    def trip(self, seconds: int, probe: int | None) -> None:
        if seconds == 0:
            return
        with self._lock:
            # Log the initial trip or a probe reopening the full window, not
            # each already-in-flight request that encounters the same throttle.
            opening = self._blocked_until == 0 or probe == self._generation
            self._blocked_until = self._clock() + seconds
            self._generation += 1
            if opening:
                logger.info("Subtitle breaker opened window_seconds=%s", seconds)

    def succeeded(self, probe: int | None) -> None:
        if probe is None:
            return
        with self._lock:
            # An old probe must not close a newer lease or a concurrent 429 trip.
            if probe == self._generation and self._blocked_until != 0:
                self._blocked_until = 0.0
                self._generation += 1
                logger.info("Subtitle breaker closed")

    def status(self, seconds: int) -> SubtitleBreakerStatus:
        if seconds == 0:
            return SubtitleBreakerStatus(breaker_open=False, retry_after_seconds=0)
        with self._lock:
            remaining = max(0, math.ceil(self._blocked_until - self._clock()))
            return SubtitleBreakerStatus(breaker_open=remaining > 0, retry_after_seconds=remaining)


# All extractor instances and authenticated info requests share one state per process.
subtitle_breaker = SubtitleCircuitBreaker()
