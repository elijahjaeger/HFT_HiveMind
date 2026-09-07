"""Read-only historical-data adapters.

Every adapter in this package is strictly read-only: it issues HTTP GET
requests to public market-data endpoints and returns normalized
:class:`core.dataset.MarketDataset` objects. No adapter may place, modify or
cancel an order, and none of them handle credentials.
"""

from adapters.base import RawResponse, RawStore
from adapters.kalshi import KalshiAdapter, KalshiClient

__all__ = ["KalshiAdapter", "KalshiClient", "RawResponse", "RawStore"]
