"""Shared plumbing for read-only venue adapters.

Three things live here:

* :class:`Transport` - the only way an adapter can reach the network. It issues
  HTTP **GET** requests and nothing else. There is deliberately no code path
  for POST/PUT/DELETE anywhere in this package, so no adapter can place,
  amend or cancel an order even by accident.
* :class:`RawStore` - writes every response body to disk **byte for byte**
  before anything parses it, with provenance in a sidecar file.
* :class:`HistoricalDataSource` - the protocol adapters implement.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.config import resolve_path
from core.dataset import MarketDataset

USER_AGENT = "hft-hivemind-research/0.1 (read-only; simulation only)"


class AdapterError(RuntimeError):
    """An adapter could not fetch or interpret data from a venue."""


class RateLimitedError(AdapterError):
    """The venue returned HTTP 429 and the retry budget was exhausted."""


@dataclass(frozen=True)
class RawResponse:
    """One untouched response body plus the request that produced it."""

    url: str
    params: dict[str, Any]
    body: bytes
    fetched_at: datetime
    status: int = 200
    path: Path | None = None

    def json(self) -> Any:
        """Parse the body as JSON (the stored copy is never modified)."""
        return json.loads(self.body.decode("utf-8"))


class Transport(Protocol):
    """Callable that performs one HTTP GET and returns ``(status, body)``."""

    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        """Fetch ``url`` and return the status code and raw body."""
        ...


def urllib_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    """Perform a GET with the standard library. The only network call we make.

    ``urllib.request.Request`` is constructed without ``data``, which forces
    the method to GET; the method is also pinned explicitly.
    """
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:  # 4xx/5xx still carry a body
        return int(error.code), error.read()
    except urllib.error.URLError as error:
        raise AdapterError(f"network error requesting {url}: {error.reason}") from error


class RawStore:
    """Persists raw response bodies under ``data/raw/<venue>/`` unchanged.

    The body is written exactly as received. Provenance (URL, parameters,
    fetch time, status) goes into a separate ``.meta.json`` sidecar so the
    payload file stays a faithful copy of what the venue returned.
    """

    def __init__(self, venue: str, root: str | Path = "data/raw", enabled: bool = True) -> None:
        self.venue = venue
        self.root = resolve_path(root) / venue
        self.enabled = enabled
        self._counter = 0

    def save(self, response: RawResponse, endpoint: str) -> Path | None:
        """Write one response body verbatim; return the file path.

        Args:
            response: The response to persist.
            endpoint: Short endpoint slug used in the filename.

        Returns:
            The path written, or ``None`` when the store is disabled.
        """
        if not self.enabled:
            return None
        folder = self.root / endpoint.strip("/").replace("/", "_")
        folder.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        stamp = response.fetched_at.strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"{stamp}_{self._counter:04d}.json"
        path.write_bytes(response.body)
        meta = {
            "venue": self.venue,
            "url": response.url,
            "params": response.params,
            "status": response.status,
            "fetched_at": response.fetched_at.isoformat(),
            "bytes": len(response.body),
            "note": "Response body stored unchanged. This file is the parsed-from source.",
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path

    def load(self, path: str | Path) -> RawResponse:
        """Read back a stored raw response."""
        payload_path = resolve_path(path)
        meta_path = payload_path.with_suffix(".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = meta.get("fetched_at")
        return RawResponse(
            url=meta.get("url", ""),
            params=meta.get("params", {}),
            body=payload_path.read_bytes(),
            fetched_at=datetime.fromisoformat(fetched) if fetched else datetime.now(UTC),
            status=int(meta.get("status", 200)),
            path=payload_path,
        )


@dataclass
class ReadOnlyClient:
    """Base HTTP client: GET only, throttled, with retries on rate limiting."""

    base_url: str
    transport: Transport = urllib_get
    timeout: float = 20.0
    min_request_interval: float = 0.2
    max_retries: int = 3
    backoff_seconds: float = 1.0
    user_agent: str = USER_AGENT
    raw_store: RawStore | None = None
    sleep: Any = time.sleep
    _last_request_at: float = field(default=0.0, init=False, repr=False)
    responses: list[RawResponse] = field(default_factory=list, init=False, repr=False)

    def build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        """Join ``path`` onto the base URL and encode query parameters."""
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        encoded = urllib.parse.urlencode(
            {k: ("true" if v is True else "false" if v is False else v) for k, v in clean.items()}
        )
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        return f"{url}?{encoded}" if encoded else url

    def get(self, path: str, params: dict[str, Any] | None = None, endpoint: str = "") -> Any:
        """GET ``path`` and return decoded JSON, storing the raw body first.

        Args:
            path: Path below the base URL.
            params: Query parameters; ``None`` values are dropped.
            endpoint: Slug used when persisting the raw body.

        Returns:
            The parsed JSON payload.

        Raises:
            RateLimitedError: If HTTP 429 persists past ``max_retries``.
            AdapterError: On any other non-200 response or unparseable body.
        """
        url = self.build_url(path, params)
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}

        attempt = 0
        while True:
            self._throttle()
            status, body = self.transport(url, headers, self.timeout)
            response = RawResponse(
                url=url,
                params=dict(params or {}),
                body=body,
                fetched_at=datetime.now(UTC),
                status=status,
            )
            if status == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise RateLimitedError(
                        f"rate limited by the venue after {self.max_retries} retries: {url}"
                    )
                self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                continue
            if status != 200:
                raise AdapterError(f"HTTP {status} from {url}: {body[:200]!r}")
            break

        stored = self.raw_store.save(response, endpoint or path) if self.raw_store else None
        response = RawResponse(
            url=response.url,
            params=response.params,
            body=response.body,
            fetched_at=response.fetched_at,
            status=response.status,
            path=stored,
        )
        self.responses.append(response)
        try:
            return response.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterError(f"could not decode JSON from {url}: {error}") from error

    def _throttle(self) -> None:
        """Space requests out to stay well inside the venue rate limit."""
        if self.min_request_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_request_interval:
            self.sleep(self.min_request_interval - elapsed)
        self._last_request_at = time.monotonic()


@runtime_checkable
class HistoricalDataSource(Protocol):
    """What every read-only adapter provides."""

    venue: str

    def fetch_dataset(self, **kwargs: Any) -> MarketDataset:
        """Fetch and normalize historical data into a :class:`MarketDataset`."""
        ...
