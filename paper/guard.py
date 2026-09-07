"""A transport wrapper that makes order placement impossible at the wire level.

The venue client already has no order-placement method, but paper trading is
the one mode that touches a live venue, so it gets a second, independent
barrier that does not depend on the client behaving.

The guard is an allowlist, not a blocklist: a request must target a known
market-data path to pass. A blocklist would silently permit any endpoint nobody
thought to name. The blocklist is kept as well, purely so that an attempt to
reach a trading endpoint fails with an explicit, greppable error rather than a
generic "not allowed".
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from adapters.base import Transport

ALLOWED_PATH_PREFIXES: tuple[str, ...] = (
    "/trade-api/v2/markets",
    "/trade-api/v2/series",
    "/trade-api/v2/events",
    "/trade-api/v2/historical",
    "/trade-api/v2/exchange",
    # Bare forms, for base URLs that already include the API prefix.
    "/markets",
    "/series",
    "/events",
    "/historical",
    "/exchange",
)
"""Read-only market-data paths. Anything else is refused."""

FORBIDDEN_PATH_FRAGMENTS: tuple[str, ...] = (
    "/portfolio",
    "/orders",
    "/order",
    "/positions",
    "/fills",
    "/balance",
    "batch_create",
    "batch_cancel",
    "/cancel",
    "/decrease",
)
"""Fragments that name a trading or account endpoint on a prediction venue."""


class OrderPlacementBlocked(RuntimeError):
    """Raised when anything tries to reach a trading or account endpoint.

    This is a bug-catcher, not an error to handle: reaching it means code tried
    to do something paper mode must never do.
    """


@dataclass
class ReadOnlyGuard:
    """Wraps a :class:`adapters.base.Transport` and enforces read-only access.

    Every request is checked before it leaves the process. The guard also keeps
    an audit trail of what was requested, which the paper session surfaces so a
    human can confirm what the app actually talked to.
    """

    transport: Transport
    allowed_prefixes: tuple[str, ...] = ALLOWED_PATH_PREFIXES
    forbidden_fragments: tuple[str, ...] = FORBIDDEN_PATH_FRAGMENTS
    requests: list[str] = field(default_factory=list, init=False)
    blocked: list[str] = field(default_factory=list, init=False)

    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        """Check the request, then delegate to the wrapped transport.

        Args:
            url: Full request URL.
            headers: Request headers.
            timeout: Request timeout in seconds.

        Returns:
            ``(status, body)`` from the wrapped transport.

        Raises:
            OrderPlacementBlocked: If the URL names a trading or account
                endpoint, or is not an allowlisted market-data path.
        """
        self.check(url)
        self.requests.append(url)
        return self.transport(url, headers, timeout)

    def check(self, url: str) -> None:
        """Validate one URL against the blocklist and then the allowlist."""
        path = urllib.parse.urlparse(url).path or "/"
        lowered = path.lower()

        for fragment in self.forbidden_fragments:
            if fragment in lowered:
                self.blocked.append(url)
                raise OrderPlacementBlocked(
                    f"Refused to request {path!r}: it matches the trading/account "
                    f"endpoint fragment {fragment!r}. Paper mode is read-only and "
                    "must never reach an order endpoint."
                )

        if not any(lowered.startswith(prefix) for prefix in self.allowed_prefixes):
            self.blocked.append(url)
            raise OrderPlacementBlocked(
                f"Refused to request {path!r}: only read-only market-data paths "
                f"are allowed ({', '.join(self.allowed_prefixes[:5])}, ...)."
            )

    @property
    def request_count(self) -> int:
        """How many requests the guard has let through."""
        return len(self.requests)

    def audit(self) -> dict[str, Any]:
        """Summary of what this guard has seen, for display."""
        return {
            "requests_allowed": len(self.requests),
            "requests_blocked": len(self.blocked),
            "last_request": self.requests[-1] if self.requests else None,
        }
