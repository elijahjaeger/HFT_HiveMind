"""Close the research loop on real settled Kalshi markets.

Everything here is orchestration over modules that already exist. It adds no
data source, no model, no metric and no order path: it pulls settled markets
through the read-only adapter, accumulates them into a corpus, runs
:mod:`research.pipeline`'s four stages, and reports what the promotion gate
said.

The loop it closes
------------------
1. Discover settled markets in a window before Kalshi's historical cutoff.
2. Fetch and normalise them through ``adapters.kalshi``, caching via
   ``core.store``.
3. Run ``prepare_data -> train_and_evaluate -> run_model_backtest ->
   check_drift``.
4. Report ``effective_n``, fold count, Brier / log loss / ECE, and every gate
   check by name.
5. Widen the window and repeat, so ``effective_n`` grows across iterations.

Why it never "makes a run pass"
-------------------------------
The gate is evidence, not preference. This module never varies model
hyper-parameters, fold counts or thresholds in response to a failure, and never
calls ``approve``. A failing gate is reported with the registry's own wording
and the numbers behind it; the only thing the loop changes between iterations
is *how much real data it has*, which is the one lever that legitimately moves
the answer.

Widening the window is not p-hacking: the corpus only ever grows, the same
model and the same gate are applied each time, and every iteration is recorded.
What would be illegitimate - retrying with different parameters until something
clears - is exactly what this module refuses to do.

**Simulation-only.** No POST/PUT/DELETE, no credential, no ``live_trading``
toggle. The backtest stage runs the same event-driven simulator as everything
else and writes only to an in-memory portfolio and the local registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from adapters.base import AdapterError
from adapters.kalshi import KalshiAdapter, KalshiFetchSpec, parse_timestamp
from core.config import resolve_path
from core.dataset import MarketDataset
from core.store import read_dataset, write_dataset
from research.corpus import describe
from research.pipeline import (
    BacktestOutcome,
    PreparedData,
    TrainingOutcome,
    check_drift,
    prepare_data,
    run_model_backtest,
    train_and_evaluate,
)
from research.registry import PromotionGate, ResearchRegistry

RUN_LOOP_VERSION = "run-loop-v1"

KALSHI_LABEL = "KALSHI HISTORICAL"
KALSHI_SOURCE = "KALSHI_CANDLES"

# Statuses and results that mean a market genuinely settled. Mirrors
# research/labels.py rather than reinventing it: anything else is excluded so a
# market whose outcome can still change never enters the corpus.
SETTLED_RESULTS = frozenset({"yes", "no"})


class StopReason(StrEnum):
    """Why the loop stopped. Every run ends with exactly one of these."""

    GATE_PASSED = "GATE_PASSED"
    """The promotion gate cleared. A human decision is now required."""

    NO_NEW_MARKETS = "NO_NEW_MARKETS"
    """Discovery returned nothing new: the available history is exhausted."""

    NO_GROWTH = "NO_GROWTH"
    """effective_n stopped growing for the configured number of iterations."""

    MAX_ITERATIONS = "MAX_ITERATIONS"
    """The absolute iteration cap was reached."""

    NO_DATA = "NO_DATA"
    """Nothing could be fetched at all - usually no network."""

    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    """Discovery kept failing. A venue problem, not a finding about the data."""


@dataclass(frozen=True)
class LoopConfig:
    """How wide to cast the net, and when to give up."""

    max_iterations: int = 20
    """Absolute cap, so the loop cannot run forever even if it keeps growing."""

    max_stale_iterations: int = 5
    """Stop after this many consecutive iterations with no effective_n growth."""

    markets_per_iteration: int = 120
    window_days_initial: int = 30
    window_days_increment: int = 30
    """Each iteration reaches this many more days further back."""

    period_interval: int = 60
    include_trades: bool = False
    """Trades cost up to 20 extra requests per market and the pipeline's
    features are quote-driven, so the tape is off by default."""

    min_lifetime_hours: float = 6.0
    """Markets shorter than this are skipped.

    Not a quality filter and not a way to flatter the numbers: with
    ``research.dataset`` at ``horizon_seconds=3600`` and
    ``stride_seconds=1800``, the decision grid runs from a market's first quote
    to ``close_time - horizon``. A market that lives one hour therefore yields
    **zero** decision timestamps and cannot produce a single training row. The
    settled listing is dominated by such markets (1-6h crypto strikes), so
    without this filter the corpus grows while ``effective_n`` stays at zero.
    """

    historical_first: bool = True
    """Try ``/historical/...`` before the series path.

    Kalshi's historical candlestick endpoint returns 404 for every market
    tested here even when the market closed before the published cutoff, so the
    loop falls back to the series endpoint, which serves the same candles.
    """

    max_discovery_failures: int = 3
    """Consecutive discovery failures tolerated before giving up."""

    fetch_chunk_size: int = 25
    """Tickers fetched per request batch.

    ``KalshiAdapter.fetch_dataset`` guards the per-market call but *not* the
    candlestick call, so a single 404 there aborts the whole batch and loses
    every market in it. Chunking caps that blast radius: one bad ticker costs
    its chunk, not the iteration.
    """

    cache_path: str = "data/processed/kalshi_corpus.duckdb"
    max_discovery_pages: int = 12


@dataclass
class IterationReport:
    """What one pass of the loop pulled, measured and concluded."""

    iteration: int
    window_start: datetime
    window_end: datetime
    new_tickers: int = 0
    corpus_markets: int = 0
    corpus_effective_n: int = 0
    """Resolved markets in the *corpus*, before walk-forward splits it.

    Reported separately from :attr:`effective_n` because the two answer
    different questions and confusing them is misleading. A corpus can hold 94
    resolved markets while the out-of-sample count is 0, which means the folds
    came back empty - not that the data is missing. Only the second number
    gates promotion, but only the first tells you whether to pull more markets
    or to look at the fold configuration.
    """

    rows_labelled: int = 0
    effective_n: int = 0
    """Resolved markets scored **out-of-sample**. This is what the gate reads."""

    fold_count: int = 0
    brier: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    baseline_brier: float | None = None
    beats_baseline: bool | None = None
    gate_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)
    run_ids: dict[str, str] = field(default_factory=dict)
    drift_level: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        """One display row per iteration."""
        return {
            "iteration": self.iteration,
            "window_start": self.window_start.date().isoformat(),
            "new_tickers": self.new_tickers,
            "corpus_markets": self.corpus_markets,
            "corpus_effective_n": self.corpus_effective_n,
            "rows_labelled": self.rows_labelled,
            "effective_n": self.effective_n,
            "folds": self.fold_count,
            "brier": None if self.brier is None else round(self.brier, 4),
            "baseline_brier": (
                None if self.baseline_brier is None else round(self.baseline_brier, 4)
            ),
            "gate": "PASSED" if self.gate_passed else "failed",
        }


@dataclass
class LoopReport:
    """The whole run: every iteration, and why it stopped."""

    iterations: list[IterationReport] = field(default_factory=list)
    stopped_because: StopReason = StopReason.MAX_ITERATIONS
    recommendation: str = ""
    errors: list[str] = field(default_factory=list)
    """Transport failures seen while running. Never silently dropped."""

    @property
    def best_effective_n(self) -> int:
        """The largest independent-outcome count any iteration reached."""
        return max((i.effective_n for i in self.iterations), default=0)

    @property
    def gate_passed(self) -> bool:
        """Whether any iteration cleared the gate."""
        return any(i.gate_passed for i in self.iterations)

    def as_rows(self) -> list[dict[str, Any]]:
        """Every iteration as a display row."""
        return [i.as_row() for i in self.iterations]


# --------------------------------------------------------------------------
# Discovery and fetching
# --------------------------------------------------------------------------


def historical_cutoff(adapter: KalshiAdapter) -> datetime:
    """The boundary between Kalshi's live and historical tiers.

    Falls back to 90 days ago if the endpoint cannot be read, which is the
    documented approximate boundary.
    """
    try:
        payload = adapter.client.cutoff()
    except (AdapterError, OSError):
        return datetime.now(UTC) - timedelta(days=90)
    for key in ("market_settled_ts", "trades_created_ts"):
        raw = payload.get(key) if isinstance(payload, dict) else None
        if raw:
            try:
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
    return datetime.now(UTC) - timedelta(days=90)


def window_for(cutoff: datetime, iteration: int, config: LoopConfig) -> tuple[datetime, datetime]:
    """The date window for one iteration: fixed end, start reaching further back.

    The end is pinned at the historical cutoff so every market in the corpus is
    in the historical tier and has actually settled. Only the start moves.
    """
    days = config.window_days_initial + config.window_days_increment * max(0, iteration - 1)
    return cutoff - timedelta(days=days), cutoff


def market_lifetime_hours(payload: dict[str, Any]) -> float | None:
    """How long a listed market was open, or ``None`` if it cannot be told."""
    opened = parse_timestamp(payload.get("open_time"))
    closed = parse_timestamp(payload.get("close_time"))
    if opened is None or closed is None:
        return None
    return (closed - opened).total_seconds() / 3600.0


def is_settled_binary(payload: dict[str, Any], min_lifetime_hours: float = 0.0) -> bool:
    """Whether a listed market can contribute anything to the corpus.

    Mirrors the exclusions in ``research/labels.py`` - not binary, or no plain
    yes/no result, means no label - and adds one the labeller cannot know
    about: a market shorter than the feature horizon produces no decision
    timestamps at all, so including it grows the corpus without moving
    ``effective_n``.
    """
    if str(payload.get("market_type", "binary")).lower() not in ("binary", ""):
        return False
    if str(payload.get("result", "")).lower() not in SETTLED_RESULTS:
        return False
    if not payload.get("ticker"):
        return False
    if min_lifetime_hours > 0.0:
        lifetime = market_lifetime_hours(payload)
        if lifetime is None or lifetime < min_lifetime_hours:
            return False
    return True


@dataclass(frozen=True)
class DiscoveredMarket:
    """A settled market worth fetching, with the window it actually lived in."""

    ticker: str
    opened: datetime
    closed: datetime


def discover_settled_markets(
    adapter: KalshiAdapter,
    start: datetime,
    end: datetime,
    limit: int,
    seen: set[str] | None = None,
    config: LoopConfig | None = None,
    errors: list[str] | None = None,
) -> list[DiscoveredMarket]:
    """Discovery that keeps each market's own open/close window.

    The window matters: the candlestick endpoint rejects a request whose
    interval and span imply too many candles (HTTP 400). Fetching a market over
    the whole multi-week discovery window at 1-minute granularity asks for
    ~43,000 candles and is refused, while the market's own few days is fine.
    """
    cfg = config or LoopConfig()
    known = set(seen or ())
    try:
        payloads = adapter.client.markets(
            status="settled",
            min_close_ts=int(start.timestamp()),
            max_close_ts=int(end.timestamp()),
            limit=1000,
            max_pages=cfg.max_discovery_pages,
        )
    except (AdapterError, OSError) as error:
        if errors is not None:
            errors.append(f"{type(error).__name__}: {error}")
        return []

    found: list[DiscoveredMarket] = []
    for payload in payloads:
        ticker = str(payload.get("ticker", ""))
        if ticker in known or not is_settled_binary(payload, cfg.min_lifetime_hours):
            continue
        opened = parse_timestamp(payload.get("open_time"))
        closed = parse_timestamp(payload.get("close_time"))
        if opened is None or closed is None:
            continue
        known.add(ticker)
        found.append(DiscoveredMarket(ticker, opened, closed))
        if len(found) >= limit:
            break
    # Chronological, so a chunk of neighbours spans a narrow window.
    found.sort(key=lambda m: m.closed)
    return found


def discover_settled_tickers(
    adapter: KalshiAdapter,
    start: datetime,
    end: datetime,
    limit: int,
    seen: set[str] | None = None,
    config: LoopConfig | None = None,
    errors: list[str] | None = None,
) -> list[str]:
    """Settled binary tickers closing in ``[start, end)`` that we do not have.

    Args:
        adapter: The read-only Kalshi adapter.
        start: Earliest close time to consider.
        end: Latest close time to consider.
        limit: Stop after this many new tickers.
        seen: Tickers already in the corpus, which are skipped.
        config: Loop config, for the minimum-lifetime filter.
        errors: Appended to when the call fails. An empty result with no
            error means the history is genuinely exhausted; an empty result
            *with* one means the call failed and is worth retrying. Silently
            conflating the two reports a transient blip as "no data left".

    Returns:
        Up to ``limit`` ticker strings, in the order the venue listed them.
        The listing skews heavily toward very short markets, so many pages are
        read to find enough that can actually produce a training row.
    """
    cfg = config or LoopConfig()
    known = set(seen or ())
    try:
        payloads = adapter.client.markets(
            status="settled",
            min_close_ts=int(start.timestamp()),
            max_close_ts=int(end.timestamp()),
            limit=1000,
            max_pages=cfg.max_discovery_pages,
        )
    except (AdapterError, OSError) as error:
        if errors is not None:
            errors.append(f"{type(error).__name__}: {error}")
        return []

    found: list[str] = []
    for payload in payloads:
        ticker = str(payload.get("ticker", ""))
        if ticker in known or not is_settled_binary(payload, cfg.min_lifetime_hours):
            continue
        known.add(ticker)
        found.append(ticker)
        if len(found) >= limit:
            break
    return found


def fetch_corpus(
    adapter: KalshiAdapter,
    tickers: list[str],
    start: datetime,
    end: datetime,
    config: LoopConfig | None = None,
) -> MarketDataset | None:
    """Fetch and normalise ``tickers`` over a window, or ``None`` if none resolved.

    Returns ``None`` rather than raising when the whole batch fails, so one bad
    window cannot end the loop.
    """
    cfg = config or LoopConfig()
    if not tickers:
        return None

    def spec_for(historical: bool) -> KalshiFetchSpec:
        return KalshiFetchSpec(
            tickers=list(tickers),
            start_ts=int(start.timestamp()),
            end_ts=int(end.timestamp()),
            period_interval=cfg.period_interval,
            include_trades=cfg.include_trades,
            historical=historical,
        )

    # The historical endpoint 404s for every market observed here, even ones
    # that closed before the published cutoff, and a 404 on candlesticks aborts
    # the whole batch rather than skipping one ticker. The series endpoint
    # serves the same candles, so fall back to it rather than losing the pull.
    attempts = [True, False] if cfg.historical_first else [False]
    for historical in attempts:
        try:
            return adapter.fetch_dataset(spec_for(historical))
        except (AdapterError, OSError):
            continue
    return None


def fetch_corpus_chunked(
    adapter: KalshiAdapter,
    tickers: list[str],
    start: datetime,
    end: datetime,
    config: LoopConfig | None = None,
    errors: list[str] | None = None,
    markets: list[DiscoveredMarket] | None = None,
) -> MarketDataset | None:
    """Fetch ``tickers`` in chunks, keeping whatever succeeds.

    A candlestick 404 aborts a whole ``fetch_dataset`` batch, so fetching 300
    tickers at once means one bad market loses all 300 - which surfaces as an
    empty corpus and gets misread as "no data available". Chunking keeps the
    good markets and records which chunks failed.
    """
    cfg = config or LoopConfig()
    size = max(1, cfg.fetch_chunk_size)
    combined: MarketDataset | None = None
    spans = {m.ticker: (m.opened, m.closed) for m in (markets or ())}

    for index in range(0, len(tickers), size):
        chunk = tickers[index : index + size]
        # Bound the request to what this chunk actually spans. Using the whole
        # discovery window implies far more candles than the venue will return
        # at a fine interval, and it answers HTTP 400 rather than truncating.
        known = [spans[t] for t in chunk if t in spans]
        lo = min((o for o, _ in known), default=start)
        hi = max((c for _, c in known), default=end)
        fetched = fetch_corpus(adapter, chunk, max(lo, start - timedelta(days=1)), hi, cfg)
        if fetched is None:
            if errors is not None:
                errors.append(
                    f"chunk {index // size + 1} ({len(chunk)} ticker(s)) failed to fetch"
                )
            continue
        combined = merge_corpora(combined, fetched)
    return combined


def merge_corpora(
    existing: MarketDataset | None, new: MarketDataset | None
) -> MarketDataset | None:
    """Union two corpora, keyed by market id, newest provenance winning.

    The corpus only ever grows: this is what makes ``effective_n`` comparable
    across iterations rather than a fresh sample each time.
    """
    if existing is None:
        return new
    if new is None:
        return existing

    by_id = {m.market_id: m for m in existing.markets}
    by_id.update({m.market_id: m for m in new.markets})
    added = {m.market_id for m in new.markets} - {m.market_id for m in existing.markets}

    quotes = list(existing.quotes) + [q for q in new.quotes if q.market_id in added]
    trades = list(existing.trades) + [t for t in new.trades if t.market_id in added]
    quotes.sort(key=lambda q: (q.ts, q.market_id))
    trades.sort(key=lambda t: (t.ts, t.market_id))

    return MarketDataset(
        markets=sorted(by_id.values(), key=lambda m: m.market_id),
        quotes=quotes,
        trades=trades,
        seed=None,
        label=new.label or existing.label,
        source=new.source or existing.source,
        venue=new.venue or existing.venue,
        fetched_at=new.fetched_at or existing.fetched_at,
        notes=list(dict.fromkeys([*existing.notes, *new.notes])),
    )


def cache_corpus(dataset: MarketDataset, path: str) -> None:
    """Persist the accumulated corpus, provenance included."""
    write_dataset(dataset, path)


def load_cached_corpus(path: str) -> MarketDataset | None:
    """Load a previously cached corpus, or ``None`` when there is not one."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    try:
        return read_dataset(resolved)
    except Exception:  # noqa: BLE001 - a bad cache must not end the loop
        return None


# --------------------------------------------------------------------------
# Running the pipeline and reading the gate
# --------------------------------------------------------------------------


@dataclass
class StageResults:
    """What the four pipeline stages produced for one corpus."""

    prepared: PreparedData
    outcome: TrainingOutcome
    backtest: BacktestOutcome | None = None
    drift_level: str | None = None


def run_stages(
    corpus: MarketDataset,
    settings: dict[str, Any],
    registry: ResearchRegistry | None = None,
    model_name: str = "logistic_regression",
) -> StageResults:
    """Run all four pipeline stages against ``corpus``.

    The stages are called exactly as the dashboard calls them; nothing is
    reconfigured in response to a result.
    """
    prepared = prepare_data(corpus, settings, registry)
    outcome = train_and_evaluate(prepared, settings, model_name, registry)

    backtest: BacktestOutcome | None = None
    drift_level: str | None = None
    if outcome.report.usable_folds:
        backtest = run_model_backtest(prepared, outcome, corpus, settings, registry)
        drift = check_drift(prepared, outcome, settings, registry)
        drift_level = drift.worst_level.value
    return StageResults(prepared, outcome, backtest, drift_level)


def gate_failures(gate: PromotionGate) -> list[str]:
    """Every failing gate check, in the registry's own wording.

    The point is that a reader can tell *which* evidence is missing without
    re-deriving it, so the loop never has to guess whether more data would help.
    """
    return [f"{name}: {detail}" for name, ok, detail in gate.checks if not ok]


def summarise_iteration(
    iteration: int,
    window: tuple[datetime, datetime],
    corpus: MarketDataset,
    new_tickers: int,
    results: StageResults,
) -> IterationReport:
    """Turn one pass into a report row, gate verdict included."""
    outcome = results.outcome
    candidate = outcome.candidate
    gate = outcome.gate
    baseline = outcome.baseline

    report = IterationReport(
        iteration=iteration,
        window_start=window[0],
        window_end=window[1],
        new_tickers=new_tickers,
        corpus_markets=len(corpus.markets),
        corpus_effective_n=results.prepared.dataset.effective_n,
        rows_labelled=len(results.prepared.dataset.labelled),
        effective_n=candidate.effective_n,
        fold_count=len(outcome.report.usable_folds),
        brier=candidate.brier,
        log_loss=candidate.log_loss,
        ece=candidate.ece,
        baseline_brier=baseline.brier if baseline else None,
        beats_baseline=outcome.comparison.beats_baseline(),
        gate_passed=gate.passed,
        gate_failures=gate_failures(gate),
        drift_level=results.drift_level,
    )
    if results.prepared.run_id:
        report.run_ids["prepare"] = results.prepared.run_id
    if outcome.run_id:
        report.run_ids["train"] = outcome.run_id
    if results.backtest and results.backtest.run_id:
        report.run_ids["backtest"] = results.backtest.run_id
    report.notes.extend(candidate.warnings)
    return report


def recommendation_for(report: IterationReport) -> str:
    """The human-facing recommendation for a passing gate.

    Deliberately stops at "here is what to review". Approval needs a name and a
    rationale and stays a human action; nothing here calls ``approve``.
    """
    return (
        f"Gate PASSED on iteration {report.iteration}: {report.effective_n} resolved "
        f"out-of-sample markets across {report.fold_count} usable folds, Brier "
        f"{report.brier:.4f} vs market baseline {report.baseline_brier:.4f}.\n"
        f"Registry run ids: {report.run_ids}.\n"
        "This is a candidate for review, NOT an approval. Approving requires a named "
        "approver and a written rationale through research.registry.approve(). "
        "Nothing has been promoted, deployed or traded."
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_loop(
    settings: dict[str, Any],
    config: LoopConfig | None = None,
    adapter: KalshiAdapter | None = None,
    registry: ResearchRegistry | None = None,
    on_iteration: Any = None,
) -> LoopReport:
    """Grow a real corpus until the gate passes, data runs out, or growth stops.

    Args:
        settings: Full settings dictionary.
        config: Loop bounds. Defaults to :class:`LoopConfig`.
        adapter: Read-only Kalshi adapter. Built from settings when omitted.
        registry: Where runs are recorded.
        on_iteration: Optional callback invoked with each
            :class:`IterationReport` as it completes, for progress reporting.

    Returns:
        A :class:`LoopReport`. A passing gate yields a recommendation for a
        human; it never approves anything.
    """
    cfg = config or LoopConfig()
    kalshi = adapter or KalshiAdapter(settings.get("venues", {}).get("kalshi", {}))
    cutoff = historical_cutoff(kalshi)

    corpus = load_cached_corpus(cfg.cache_path)
    seen: set[str] = {m.market_id for m in corpus.markets} if corpus else set()
    result = LoopReport()
    best_effective_n = 0
    stale = 0
    failed_discoveries = 0
    notes: list[str] = []

    for iteration in range(1, cfg.max_iterations + 1):
        window = window_for(cutoff, iteration, cfg)
        discovery_errors: list[str] = []
        discovered = discover_settled_markets(
            kalshi, window[0], window[1], cfg.markets_per_iteration, seen, cfg,
            discovery_errors,
        )
        tickers = [m.ticker for m in discovered]
        if not tickers and discovery_errors:
            # A failed call is not an exhausted archive. Widen and try again
            # rather than reporting the venue being briefly unavailable as a
            # finding about the data.
            failed_discoveries += 1
            notes.extend(discovery_errors)
            if failed_discoveries >= cfg.max_discovery_failures:
                result.stopped_because = StopReason.DISCOVERY_FAILED
                result.errors = list(notes)
                break
            continue
        if not tickers and corpus is None:
            result.stopped_because = StopReason.NO_DATA
            break
        if not tickers:
            result.stopped_because = StopReason.NO_NEW_MARKETS
            break

        fetch_errors: list[str] = []
        fetched = fetch_corpus_chunked(
            kalshi, tickers, window[0], window[1], cfg, fetch_errors, discovered
        )
        notes.extend(fetch_errors)
        corpus = merge_corpora(corpus, fetched)
        if corpus is None:
            result.stopped_because = StopReason.NO_DATA
            result.errors = list(notes)
            break
        seen.update(m.market_id for m in corpus.markets)
        cache_corpus(corpus, cfg.cache_path)

        results = run_stages(corpus, settings, registry)
        report = summarise_iteration(iteration, window, corpus, len(tickers), results)
        # Provenance is what the gate's "Real market data" check reads; record
        # it so a misclassified corpus is visible rather than blamed on volume.
        if describe(corpus).is_synthetic:
            report.notes.append(
                "Corpus reports as synthetic - the gate cannot pass. Check that "
                "provenance survived caching (core/store.py)."
            )
        report.notes.extend(fetch_errors)
        result.iterations.append(report)
        if on_iteration is not None:
            on_iteration(report)

        if report.gate_passed:
            result.stopped_because = StopReason.GATE_PASSED
            result.recommendation = recommendation_for(report)
            break

        # Growth is measured on the corpus. Out-of-sample effective_n is 0
        # whenever the folds are empty, so using it would call a corpus that
        # is genuinely growing "stale" and stop early for the wrong reason.
        growth = max(report.corpus_effective_n, report.effective_n)
        if growth > best_effective_n:
            best_effective_n = growth
            stale = 0
        else:
            stale += 1
            if stale >= cfg.max_stale_iterations:
                result.stopped_because = StopReason.NO_GROWTH
                break
    else:
        result.stopped_because = StopReason.MAX_ITERATIONS

    if not result.recommendation:
        result.recommendation = _no_pass_summary(result)
    return result


def _no_pass_summary(result: LoopReport) -> str:
    """Say why the gate did not pass, without softening it."""
    if not result.iterations:
        detail = ""
        if result.errors:
            joined = "\n  ".join(result.errors[:5])
            detail = f"\nWhat went wrong:\n  {joined}"
        return (
            f"No iteration completed ({result.stopped_because.value}). "
            f"Nothing to review.{detail}"
        )
    last = result.iterations[-1]
    failures = "\n".join(f"  - {f}" for f in last.gate_failures) or "  - (none recorded)"

    # The two counts diverging is the single most informative thing here, and
    # the one most easily misread as "no data".
    diagnosis = ""
    if last.corpus_effective_n > 0 and last.effective_n == 0:
        diagnosis = (
            f"\nDIAGNOSIS: the corpus holds {last.corpus_effective_n} resolved market(s) "
            f"and {last.rows_labelled} labelled row(s), but {last.fold_count} usable "
            "walk-forward fold(s). The data is there; the folds are empty. Pulling more "
            "markets will not fix that on its own - look at research.walkforward "
            "(n_folds x test_markets_per_fold, purge and embargo) against the corpus size.\n"
        )
    elif last.corpus_effective_n == 0:
        diagnosis = (
            "\nDIAGNOSIS: no resolved market survived feature building. Check quote "
            "density (period_interval) and the two-sided-book requirement before "
            "pulling more markets.\n"
        )

    return (
        f"Gate did NOT pass. Stopped: {result.stopped_because.value} after "
        f"{len(result.iterations)} iteration(s).\n"
        f"Corpus: {last.corpus_markets} market(s), {last.corpus_effective_n} resolved, "
        f"{last.rows_labelled} labelled row(s).\n"
        f"Out-of-sample: {last.effective_n} resolved market(s) across "
        f"{last.fold_count} usable fold(s).\n"
        f"{diagnosis}"
        f"Failing checks on the last iteration:\n{failures}\n"
        "No parameters were varied to try to force a pass."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the loop from the command line and print a report.

    Usage::

        python -m research.run_loop --max-iterations 6 --markets 150
    """
    import argparse

    from core.config import get_settings

    parser = argparse.ArgumentParser(description="Close the research loop on real Kalshi data.")
    parser.add_argument("--max-iterations", type=int, default=LoopConfig.max_iterations)
    parser.add_argument("--stale-limit", type=int, default=LoopConfig.max_stale_iterations)
    parser.add_argument("--markets", type=int, default=LoopConfig.markets_per_iteration)
    parser.add_argument("--min-lifetime-hours", type=float, default=LoopConfig.min_lifetime_hours)
    parser.add_argument("--period-interval", type=int, default=LoopConfig.period_interval,
                        choices=(1, 60, 1440), help="candle length in minutes")
    parser.add_argument("--cache", default=LoopConfig.cache_path)
    args = parser.parse_args(argv)

    settings = get_settings()
    config = LoopConfig(
        max_iterations=args.max_iterations,
        max_stale_iterations=args.stale_limit,
        markets_per_iteration=args.markets,
        min_lifetime_hours=args.min_lifetime_hours,
        period_interval=args.period_interval,
        cache_path=args.cache,
    )

    def progress(report: IterationReport) -> None:
        print(
            f"[{report.iteration}] {report.window_start.date()} -> "
            f"{report.window_end.date()} | +{report.new_tickers} tickers | "
            f"corpus {report.corpus_markets} "
            f"(resolved {report.corpus_effective_n}, rows {report.rows_labelled}) | "
            f"OOS effective_n {report.effective_n} | "
            f"folds {report.fold_count} | brier "
            f"{'n/a' if report.brier is None else format(report.brier, '.4f')} | "
            f"gate {'PASSED' if report.gate_passed else 'failed'}",
            flush=True,
        )
        for failure in report.gate_failures:
            print(f"      - {failure}", flush=True)

    registry = ResearchRegistry(
        settings.get("research", {}).get("registry_path", "data/research/registry.duckdb")
    )
    result = run_loop(settings, config, registry=registry, on_iteration=progress)

    print()
    print(f"Stopped: {result.stopped_because.value}")
    print(result.recommendation)
    return 0 if result.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
