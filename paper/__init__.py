"""Paper trading: the live agent loop that can never place an order.

Paper mode reads live market data and runs the *same* forecast, risk and
execution code the backtester uses, but every order it produces is
hypothetical: it is simulated against the observed book and written to a local
store. Nothing is ever sent to a venue.

Three independent mechanisms enforce that, so removing any one of them still
leaves the guarantee intact:

1. :class:`paper.guard.ReadOnlyGuard` wraps the HTTP transport and refuses any
   request that is not a GET to an allowlisted market-data path.
2. The venue client itself (``adapters.kalshi.KalshiClient``) has no
   order-placement method to call, and no code path that sends a request body.
3. :class:`paper.session.PaperSession` routes every approved order into
   ``backtest.fills.simulate_fill`` - a pure function over a quote - and then
   into a local store. It holds no venue write handle of any kind.
"""

from paper.guard import (
    ALLOWED_PATH_PREFIXES,
    FORBIDDEN_PATH_FRAGMENTS,
    OrderPlacementBlocked,
    ReadOnlyGuard,
)
from paper.session import (
    LiveSnapshot,
    PaperConfig,
    PaperSession,
    PollResult,
)
from paper.sources import KalshiQuoteSource, ReplayQuoteSource
from paper.store import PaperStore

__all__ = [
    "ALLOWED_PATH_PREFIXES",
    "FORBIDDEN_PATH_FRAGMENTS",
    "KalshiQuoteSource",
    "LiveSnapshot",
    "OrderPlacementBlocked",
    "PaperConfig",
    "PaperSession",
    "PaperStore",
    "PollResult",
    "ReadOnlyGuard",
    "ReplayQuoteSource",
]

PAPER_MODE_BANNER = (
    "PAPER MODE - every order below is hypothetical. Orders are simulated "
    "against observed market data and stored locally. Nothing is sent to any "
    "venue, and this application has no code path that could submit an order."
)
