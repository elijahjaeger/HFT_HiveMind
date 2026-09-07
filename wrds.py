"""Refusing stub for `wrds` - see compat/__init__.py.

FinRL imports this module at package-import time. This project never uses it:
it is simulation-only, with no order-placement path and no credentials. Any
attempt to actually use it raises `compat.RefusedExternalClient`.
"""

from __future__ import annotations

from compat import RefusedExternalClient, _refuse

__all__ = ["RefusedExternalClient"]

_stub = _refuse("wrds")


def __getattr__(name: str) -> object:
    """Refuse every attribute this stub does not define."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(_stub, name)
