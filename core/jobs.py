"""Primitives for long-running work: progress, cancellation, throttling.

Deliberately free of Qt and of any threading policy, so the same job can be
driven from the desktop application, from a CLI, or from a test with no
threads at all. The application layer decides *where* a job runs; this module
only defines how it reports and how it is asked to stop.

Two things cross a thread boundary in this design and nothing else:

* **A flag, one way.** :class:`CancellationToken` wraps ``threading.Event``,
  which is built for exactly this producer/consumer pattern - set from the UI
  thread, read from the worker thread, no extra locking and no torn reads.
* **Copied snapshots, the other way.** Progress payloads are built fresh from
  plain scalars and from append-only objects that are never mutated after being
  created. No live mutable structure is ever handed across.

There is no lock here and no shared mutable state, because the design keeps the
amount of state crossing threads down to those two well-understood primitives.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROGRESS_INTERVAL_SECONDS = 0.1
"""Cap emission at roughly ten updates a second.

Progress is throttled on *time*, not on an event count, because event costs are
wildly uneven - a decision tick that fans out across every market costs orders
of magnitude more than a quote update. A flat every-N-events cadence would
therefore produce bursty, uneven updates; a time gate produces smooth ones
regardless of what the underlying loop is chewing through.
"""


class Cancelled(Exception):
    """Raised by :meth:`CancellationToken.raise_if_cancelled`.

    Jobs that can return a partial result should check
    :attr:`CancellationToken.cancelled` and return it instead of raising. This
    exists for jobs where a partial result would be meaningless.
    """


class CancellationToken:
    """A thread-safe "please stop" flag.

    Callable, so it can be passed directly wherever a ``should_stop`` predicate
    is expected::

        token = CancellationToken()
        run_backtest(..., should_stop=token)
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the job to stop. Safe to call from any thread, and idempotent."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    def __call__(self) -> bool:
        """Predicate form, so the token *is* a ``should_stop`` callable."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`Cancelled` if a stop has been requested."""
        if self._event.is_set():
            raise Cancelled("job cancelled")

    def reset(self) -> None:
        """Clear the flag so the token can be reused for another run."""
        self._event.clear()


@dataclass(frozen=True)
class JobProgress:
    """One progress report from a job.

    ``total`` of zero means the size is not known ahead of time (a corpus pull
    discovers its own size), in which case :attr:`fraction` is ``None`` and a
    caller should show an indeterminate indicator rather than inventing a
    percentage.
    """

    index: int = 0
    total: int = 0
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float | None:
        """Completion in ``[0, 1]``, or ``None`` when the total is unknown."""
        if self.total <= 0:
            return None
        return max(0.0, min(1.0, self.index / self.total))

    @property
    def percent(self) -> int:
        """Completion as a whole percentage; ``0`` when the total is unknown."""
        fraction = self.fraction
        return 0 if fraction is None else int(round(100 * fraction))


class ProgressThrottle:
    """Gate that lets a progress update through at most every N seconds.

    The final update of a run must always be delivered, or a UI is left showing
    99% forever, so :meth:`should_emit` takes a ``force`` flag for the last
    iteration rather than relying on the timer to happen to be open.
    """

    __slots__ = ("min_interval", "_last")

    def __init__(self, min_interval: float = DEFAULT_PROGRESS_INTERVAL_SECONDS) -> None:
        self.min_interval = float(min_interval)
        # None rather than 0.0: "never emitted" is not "emitted at time zero".
        # With a 0.0 sentinel the first update is suppressed whenever the clock
        # reads lower than the interval, which is a real case for any injected
        # clock and merely hidden by time.monotonic() returning a large number.
        self._last: float | None = None

    def should_emit(self, force: bool = False, now: float | None = None) -> bool:
        """Whether an update should be emitted now, updating the clock if so.

        The first call always passes: a caller that reports nothing until the
        interval has elapsed shows no progress at all for short runs.
        """
        moment = time.monotonic() if now is None else now
        if force or self._last is None or moment - self._last >= self.min_interval:
            self._last = moment
            return True
        return False

    def reset(self) -> None:
        """Forget the last emission, so the next check passes."""
        self._last = None
