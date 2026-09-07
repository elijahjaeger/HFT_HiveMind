"""Paper trading: hypothetical orders only, proven at four levels.

The central claim this file has to establish is negative - that nothing can
submit an order - so it is attacked from four directions:

1. **Static.** The ``paper`` and ``adapters`` packages are parsed and searched
   for any call that looks like order placement.
2. **Interface.** A strict spy stands in for the venue client and raises on any
   attribute that is not an allowlisted read method.
3. **Wire.** The read-only guard rejects trading endpoints and non-market paths.
4. **Behavioural.** A full session runs end to end and every request it made is
   inspected.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import UTC, datetime, timedelta

import pytest

from adapters.kalshi import KalshiClient
from core.config import load_settings
from core.models import Market, Outcome, Quote, Side, Trade
from paper import PAPER_MODE_BANNER
from paper.guard import (
    ALLOWED_PATH_PREFIXES,
    FORBIDDEN_PATH_FRAGMENTS,
    OrderPlacementBlocked,
    ReadOnlyGuard,
)
from paper.session import PaperConfig, PaperSession
from paper.sources import KalshiQuoteSource, ReplayQuoteSource
from paper.store import PaperStore
from tests.fixtures import kalshi_payloads as fx

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)
BASE = "https://external-api.kalshi.com/trade-api/v2"

FORBIDDEN_CALL_NAMES = {
    "place_order",
    "submit_order",
    "create_order",
    "cancel_order",
    "send_order",
    "post",
    "put",
    "patch",
    "delete",
    "batch_create_orders",
    "batch_cancel_orders",
}


# ==========================================================================
# 1. Static: no order-placement call exists in the source
# ==========================================================================

PACKAGES = ["paper", "adapters", "research", "src"]


def source_files() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parent.parent
    return [f for package in PACKAGES for f in (root / package).rglob("*.py")]


def find_order_calls(tree: ast.AST) -> list[str]:
    """Return the names of any order-placement calls in a parsed module."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name in FORBIDDEN_CALL_NAMES:
            found.append(f"{name}() at line {node.lineno}")
    return found


def test_the_detector_actually_catches_an_order_call():
    """Confidence check: a vacuous scanner would make the next test worthless."""
    offending = ast.parse(
        "client.place_order(ticker='X', side='yes')\nsession.post(url)"
    )
    assert len(find_order_calls(offending)) == 2
    clean = ast.parse("client.market('X')\nclient.trades('X')")
    assert find_order_calls(clean) == []


def test_the_scan_covers_the_real_source_files():
    files = source_files()
    assert len(files) >= 6, f"expected to scan the packages, found {len(files)} files"
    assert any(f.name == "session.py" for f in files)
    assert any(f.name == "kalshi.py" for f in files)
    assert any(f.name == "pipeline.py" for f in files)


def test_the_packages_contain_no_order_placement_call():
    """Parse every module and reject any call that would submit an order."""
    offenders: list[str] = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(f"{path.name}: {hit}" for hit in find_order_calls(tree))
    assert not offenders, "order-placement calls found: " + "; ".join(offenders)


def test_no_http_method_other_than_get_is_named_anywhere():
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for verb in ('method="POST"', 'method="PUT"', 'method="DELETE"', 'method="PATCH"'):
            assert verb not in text, f"{path.name} names {verb}"


def test_the_venue_client_has_no_write_surface():
    surface = {name.lower() for name in dir(KalshiClient)}
    assert surface.isdisjoint(FORBIDDEN_CALL_NAMES)


# ==========================================================================
# 2. Interface: a strict spy rejects anything but reads
# ==========================================================================


class StrictReadOnlySpy:
    """Stands in for a venue client and refuses anything but read methods.

    Any attribute access outside the allowlist raises, so a session that tried
    to call an order method would fail loudly rather than silently no-op.
    """

    ALLOWED = {"market", "markets", "trades", "candlesticks", "cutoff", "responses"}

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.accessed: list[str] = []
        self.responses: list[object] = []

    def __getattr__(self, name: str) -> object:
        """Record the access and refuse anything outside the read allowlist."""
        if name.startswith("_"):
            raise AttributeError(name)
        self.accessed.append(name)
        if name not in self.ALLOWED:
            raise AssertionError(
                f"paper trading touched {name!r}, which is not a read-only method"
            )
        return lambda *args, **kwargs: self.payloads.get(name, [])


def test_a_session_only_touches_read_methods_on_the_venue_client():
    spy = StrictReadOnlySpy(
        {"market": fx.SETTLED_MARKET, "trades": fx.TRADES_PAGE_1["trades"]}
    )
    source = KalshiQuoteSource({"store_raw_responses": False})
    source.client = spy  # type: ignore[assignment]
    source.fetch(["KXHIGHNY-25AUG31-B70"])
    assert set(spy.accessed) <= StrictReadOnlySpy.ALLOWED
    assert "market" in spy.accessed


def test_the_spy_itself_would_catch_an_order_call():
    """Confidence check on the mechanism: the spy must actually fire."""
    spy = StrictReadOnlySpy({})
    with pytest.raises(AssertionError, match="not a read-only method"):
        spy.place_order()


# ==========================================================================
# 3. Wire: the read-only guard
# ==========================================================================


def allowing_transport(status: int = 200, body: bytes = b"{}"):
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], timeout: float):
        calls.append(url)
        return status, body

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


@pytest.mark.parametrize(
    "path",
    [
        "/trade-api/v2/portfolio/orders",
        "/trade-api/v2/portfolio/positions",
        "/trade-api/v2/portfolio/fills",
        "/trade-api/v2/portfolio/balance",
        "/trade-api/v2/portfolio/orders/batch_create",
        "/trade-api/v2/markets/ABC/orders",
        "/trade-api/v2/orders/cancel",
    ],
)
def test_the_guard_blocks_every_trading_endpoint(path):
    transport = allowing_transport()
    guard = ReadOnlyGuard(transport)
    with pytest.raises(OrderPlacementBlocked):
        guard(f"https://example.test{path}", {}, 10.0)
    assert transport.calls == [], "a blocked request must never reach the network"
    assert guard.blocked


def test_the_guard_blocks_unknown_paths_by_default():
    """An allowlist, not a blocklist: unknown endpoints are refused too."""
    guard = ReadOnlyGuard(allowing_transport())
    with pytest.raises(OrderPlacementBlocked, match="read-only market-data paths"):
        guard("https://example.test/trade-api/v2/something_new", {}, 10.0)


@pytest.mark.parametrize(
    "path",
    [
        "/trade-api/v2/markets",
        "/trade-api/v2/markets/KXHIGHNY-25AUG31-B70",
        "/trade-api/v2/markets/trades?ticker=ABC",
        "/trade-api/v2/historical/markets/ABC/candlesticks",
        "/trade-api/v2/series/KXHIGHNY/markets/ABC/candlesticks",
    ],
)
def test_the_guard_allows_market_data_paths(path):
    transport = allowing_transport()
    guard = ReadOnlyGuard(transport)
    status, _ = guard(f"https://example.test{path}", {}, 10.0)
    assert status == 200
    assert len(transport.calls) == 1


def test_the_guard_records_an_audit_trail():
    guard = ReadOnlyGuard(allowing_transport())
    guard("https://example.test/trade-api/v2/markets", {}, 10.0)
    with pytest.raises(OrderPlacementBlocked):
        guard("https://example.test/trade-api/v2/portfolio/orders", {}, 10.0)
    audit = guard.audit()
    assert audit["requests_allowed"] == 1
    assert audit["requests_blocked"] == 1


def test_the_allow_and_block_lists_do_not_overlap():
    for prefix in ALLOWED_PATH_PREFIXES:
        assert not any(fragment in prefix for fragment in FORBIDDEN_PATH_FRAGMENTS)


# ==========================================================================
# 4. Behavioural: a full session
# ==========================================================================


def paper_market(close_hours: int = 8) -> Market:
    return Market(
        market_id="M1",
        question="[PAPER] will the demo resolve YES?",
        created_at=T0,
        close_time=T0 + timedelta(hours=close_hours),
        resolution_time=T0 + timedelta(hours=close_hours + 1),
        outcome=Outcome.YES,
    )


def rising_quotes(market_id: str = "M1", n: int = 8) -> list[Quote]:
    return [
        Quote(
            market_id=market_id,
            ts=T0 + timedelta(minutes=i * 10),
            bid=round(0.20 + 0.05 * i, 4),
            ask=round(0.22 + 0.05 * i, 4),
            bid_size=1_000.0,
            ask_size=1_000.0,
        )
        for i in range(n)
    ]


def paper_tape(market_id: str = "M1", n: int = 12) -> list[Trade]:
    return [
        Trade(
            trade_id=f"T{i}",
            market_id=market_id,
            ts=T0 + timedelta(minutes=i),
            price=0.85,
            size=25.0,
            aggressor=Side.BUY,
        )
        for i in range(n)
    ]


def paper_settings() -> dict:
    settings = load_settings(use_env=False)
    settings["strategy"].update({"min_edge": 0.0, "require_positive_ev": False})
    settings["risk"].update({"max_data_age_seconds": 86_400, "max_spread": 1.0})
    settings["fills"].update({"max_quote_age_seconds": 86_400})
    return settings


class Clock:
    """A hand-cranked clock so tests never depend on wall time."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


def build_session(tmp_path, **config):
    source = ReplayQuoteSource([paper_market()], rising_quotes(), paper_tape())
    clock = Clock(T0)
    store = PaperStore(tmp_path / "paper.duckdb")
    session = PaperSession(
        PaperConfig(tickers=["M1"], execution_delay_seconds=600.0, **config),
        paper_settings(),
        source,
        store=store,
        session_id="test-session",
        clock=clock,
    )
    return session, source, clock, store


def run_polls(session, source, clock, polls: int = 6, minutes: float = 10.0):
    results = []
    for _ in range(polls):
        source.advance_to(clock.now)
        results.append(session.poll())
        clock.advance(minutes)
    return results


def test_a_session_produces_hypothetical_orders_and_fills(tmp_path):
    session, source, clock, store = build_session(tmp_path)
    results = run_polls(session, source, clock)
    assert any(r.proposals for r in results), "expected at least one proposal"
    assert any(r.executed for r in results), "expected at least one simulated fill"


def test_orders_and_fills_are_persisted_locally(tmp_path):
    session, source, clock, store = build_session(tmp_path)
    run_polls(session, source, clock)

    orders = store.orders("test-session")
    fills = store.fills("test-session")
    assert not orders.empty
    assert not fills.empty
    assert set(orders["mode"]) == {"PAPER"}
    assert set(fills["mode"]) == {"PAPER"}
    assert orders["is_hypothetical"].all()
    assert fills["is_hypothetical"].all()


def test_persisted_records_survive_reopening_the_store(tmp_path):
    session, source, clock, _ = build_session(tmp_path)
    run_polls(session, source, clock)
    reopened = PaperStore(tmp_path / "paper.duckdb")
    assert not reopened.orders("test-session").empty
    assert "test-session" in reopened.sessions()


def test_every_persisted_fill_traces_back_to_a_persisted_order(tmp_path):
    session, source, clock, store = build_session(tmp_path)
    run_polls(session, source, clock)
    orders = store.orders("test-session")
    fills = store.fills("test-session")
    assert set(fills["proposal_id"]) <= set(orders["proposal_id"])


def test_the_session_records_its_own_start_and_stop(tmp_path):
    session, source, clock, store = build_session(tmp_path)
    session.stop()
    kinds = set(store.events("test-session")["kind"])
    assert {"SESSION_START", "SESSION_STOP"} <= kinds


def test_equity_snapshots_are_recorded_each_poll(tmp_path):
    session, source, clock, store = build_session(tmp_path)
    run_polls(session, source, clock, polls=4)
    assert len(store.equity("test-session")) == 4


def test_execution_is_delayed_exactly_as_in_the_backtester(tmp_path):
    session, source, clock, _ = build_session(tmp_path)
    results = run_polls(session, source, clock)
    fills = [f for r in results for f in r.executed]
    assert fills
    for fill in fills:
        assert fill.decided_at is not None
        assert fill.ts > fill.decided_at
        assert fill.latency_seconds == pytest.approx(600.0)


def test_the_risk_agent_gates_every_order(tmp_path):
    """The kill switch must stop paper trading exactly as it stops a backtest."""
    source = ReplayQuoteSource([paper_market()], rising_quotes(), paper_tape())
    clock = Clock(T0)
    settings = paper_settings()
    settings["risk"]["kill_switch"] = True
    session = PaperSession(
        PaperConfig(tickers=["M1"], execution_delay_seconds=600.0),
        settings,
        source,
        store=PaperStore(tmp_path / "paper.duckdb"),
        session_id="killed",
        clock=clock,
    )
    results = run_polls(session, source, clock)
    assert any(r.decisions for r in results), "expected the risk agent to be consulted"
    assert all(d.action.value == "REJECT" for r in results for d in r.decisions)
    assert not [f for r in results for f in r.executed]


def test_a_fetch_failure_does_not_kill_the_session(tmp_path):
    class Broken:
        venue = "BROKEN"

        def fetch(self, tickers):
            raise ConnectionError("feed down")

    session = PaperSession(
        PaperConfig(tickers=["M1"]),
        paper_settings(),
        Broken(),
        store=PaperStore(tmp_path / "paper.duckdb"),
        session_id="broken",
        clock=Clock(T0),
    )
    result = session.poll()
    assert result.errors
    assert session.polls == 1


def test_a_session_refuses_to_start_if_live_trading_were_enabled(tmp_path):
    settings = paper_settings()
    settings["app"]["live_trading_enabled"] = True
    with pytest.raises(RuntimeError, match="live_trading_enabled must be False"):
        PaperSession(
            PaperConfig(tickers=["M1"]),
            settings,
            ReplayQuoteSource([paper_market()], rising_quotes(), paper_tape()),
            store=PaperStore(tmp_path / "paper.duckdb"),
            clock=Clock(T0),
        )


def test_the_session_status_always_declares_itself_hypothetical(tmp_path):
    session, _, _, _ = build_session(tmp_path)
    status = session.status()
    assert status["mode"] == "PAPER"
    assert status["is_hypothetical"] is True


def test_the_paper_mode_banner_is_unambiguous():
    assert "PAPER MODE" in PAPER_MODE_BANNER
    assert "hypothetical" in PAPER_MODE_BANNER
    assert "never" in PAPER_MODE_BANNER or "Nothing is sent" in PAPER_MODE_BANNER


def test_a_live_session_only_issues_guarded_get_requests(tmp_path):
    """End to end against a fake network: inspect every request that was made."""
    requests: list[tuple[str, dict[str, str]]] = []

    def transport(url: str, headers: dict[str, str], timeout: float):
        requests.append((url, headers))
        if "trades" in url:
            return 200, fx.encode(fx.TRADES_PAGE_2)
        return 200, fx.encode({"market": fx.SETTLED_MARKET})

    source = KalshiQuoteSource(
        {"store_raw_responses": False, "min_request_interval_seconds": 0.0},
        transport=transport,
    )
    session = PaperSession(
        PaperConfig(tickers=["KXHIGHNY-25AUG31-B70"], execution_delay_seconds=0.0),
        paper_settings(),
        source,
        store=PaperStore(tmp_path / "paper.duckdb"),
        session_id="live-ish",
        clock=Clock(T0),
    )
    session.poll()

    assert requests, "expected the session to read market data"
    for url, headers in requests:
        assert "/portfolio" not in url
        assert "/orders" not in url
        assert set(headers) <= {"User-Agent", "Accept"}
    assert source.guard.request_count == len(requests)
    assert source.guard.blocked == []


def test_the_live_source_reports_a_clean_audit(tmp_path):
    def transport(url: str, headers: dict[str, str], timeout: float):
        return 200, fx.encode({"market": fx.SETTLED_MARKET})

    source = KalshiQuoteSource(
        {"store_raw_responses": False, "min_request_interval_seconds": 0.0,
         "include_trades": False},
        transport=transport,
    )
    source.fetch(["KXHIGHNY-25AUG31-B70"])
    audit = source.audit()
    assert audit["requests_blocked"] == 0
    assert audit["requests_allowed"] >= 1


def test_a_market_that_cannot_be_read_is_skipped_not_fatal(tmp_path):
    def transport(url: str, headers: dict[str, str], timeout: float):
        return 200, fx.encode({"market": {"ticker": "BROKEN"}})  # no close_time

    source = KalshiQuoteSource(
        {"store_raw_responses": False, "min_request_interval_seconds": 0.0},
        transport=transport,
    )
    assert source.fetch(["BROKEN"]) == {}


# -- live quote normalization ---------------------------------------------


def test_the_live_book_comes_from_the_market_payload():
    from adapters.kalshi import normalize_quote_from_market

    quote = normalize_quote_from_market(fx.SETTLED_MARKET)
    assert quote is not None
    assert quote.bid == pytest.approx(0.42)
    assert quote.ask == pytest.approx(0.45)
    assert quote.bid_size == pytest.approx(250.0)
    assert quote.ask_size == pytest.approx(310.0)


def test_a_live_book_with_no_resting_orders_reports_no_quote():
    from adapters.kalshi import normalize_quote_from_market

    payload = dict(fx.SETTLED_MARKET, yes_bid_dollars="0.0000", yes_ask_dollars="1.0000")
    assert normalize_quote_from_market(payload) is None


def test_a_crossed_live_book_is_reported_as_no_quote():
    from adapters.kalshi import normalize_quote_from_market

    payload = dict(fx.SETTLED_MARKET, yes_bid_dollars="0.6000", yes_ask_dollars="0.4000")
    assert normalize_quote_from_market(payload) is None


def test_the_live_book_carries_real_depth_not_a_proxy():
    """Unlike candlesticks, the market payload has genuine displayed size."""
    from adapters.kalshi import normalize_quote_from_market

    quote = normalize_quote_from_market(fx.SETTLED_MARKET)
    assert quote is not None
    assert "PROXY" not in quote.source


def test_replaying_a_dataset_never_returns_a_future_quote():
    source = ReplayQuoteSource([paper_market()], rising_quotes(), paper_tape())
    moment = T0 + timedelta(minutes=25)
    source.advance_to(moment)
    snapshot = source.fetch(["M1"])["M1"]
    assert snapshot.quote is not None
    assert snapshot.quote.ts <= moment
    assert all(t.ts <= moment for t in snapshot.trades)


# -- display helpers -------------------------------------------------------


def test_status_rows_always_include_the_hypothetical_flag(tmp_path):
    from app import presenter

    session, _, _, _ = build_session(tmp_path)
    rows = {row["field"]: row["value"] for row in presenter.paper_status_rows(session)}
    assert rows["mode"] == "PAPER"
    assert rows["is hypothetical"] is True


def test_the_poll_summary_says_nothing_was_submitted(tmp_path):
    from app import presenter

    session, source, clock, _ = build_session(tmp_path)
    source.advance_to(clock.now)
    summary = presenter.paper_poll_summary(session.poll())
    assert "Nothing was submitted" in summary


def test_the_poll_summary_surfaces_errors():
    from app import presenter
    from paper.session import PollResult

    result = PollResult(ts=T0, errors=["ConnectionError: feed down"])
    assert "Poll failed" in presenter.paper_poll_summary(result)


def test_frame_rows_handle_an_empty_store(tmp_path):
    from app import presenter

    store = PaperStore(tmp_path / "empty.duckdb")
    assert presenter.paper_frame_rows(store.orders("nobody")) == []
    assert presenter.paper_frame_rows(None) == []
