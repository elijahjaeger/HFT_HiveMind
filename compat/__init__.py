"""Refusing stubs for external clients FinRL imports but this project forbids.

FinRL 0.3.7 unconditionally imports live-trading and credentialed data clients
from its package root (``finrl/__init__.py`` -> ``finrl.trade`` ->
``AlpacaPaperTrading`` -> ``alpaca_trade_api``). This project has no
order-placement code path and handles no credentials, so installing those real
clients is not an option - and the real ``alpaca-trade-api`` additionally pins
``websockets<11``, which breaks LangGraph (needs ``>=14``).

The stubs here satisfy the import so ``import finrl`` succeeds, while making any
actual *use* fail loudly with :class:`RefusedExternalClient`. That is strictly
safer than installing the genuine SDK: there is no working broker client in the
environment to reach for by accident.
"""

from __future__ import annotations

from typing import Any


class RefusedExternalClient(RuntimeError):
    """Raised when code tries to use a stubbed external trading/data client."""


def _refuse(name: str) -> Any:
    """Build a module-like object that imports fine and refuses on use."""

    class _Refusing:
        """Any attribute access or call on this raises."""

        __slots__ = ()

        def __getattr__(self, attr: str) -> Any:
            # Dunders must raise AttributeError, not our error: introspection
            # (pytest, inspect, copy) probes them and would otherwise blow up.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            raise RefusedExternalClient(
                f"{name}.{attr} is not available. This project is simulation-only: "
                f"it has no order-placement path and handles no credentials. "
                f"{name!r} is stubbed by compat/ purely so FinRL's package root "
                f"imports. See README 'Simulation-only statement'."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RefusedExternalClient(
                f"{name} is stubbed and cannot be called. Simulation-only."
            )

    return _Refusing()
