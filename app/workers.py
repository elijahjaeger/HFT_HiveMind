"""Background workers so the UI never blocks on a backtest or a network fetch."""

from __future__ import annotations

import atexit
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from adapters.kalshi import KalshiAdapter, KalshiFetchSpec
from backtest.engine import run_backtest
from core.dataset import MarketDataset
from core.jobs import CancellationToken
from core.store import load_or_create_dataset


class Worker(QObject):
    """Base worker: emits ``finished`` with a payload or ``failed`` with a message."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def run(self) -> None:  # pragma: no cover - overridden
        """Perform the work. Runs on the worker thread."""
        raise NotImplementedError


class BacktestWorker(Worker):
    """Runs one backtest off the UI thread, reporting progress and stoppable.

    ``progress_update`` is defined here rather than on :class:`Worker` because
    the demo-data and Kalshi-fetch workers have no analogous concept, and an
    unused signal on the shared base would loosen its contract.
    """

    progress_update = Signal(object)
    """Carries a :class:`backtest.engine.BacktestProgress`."""

    def __init__(self, dataset: MarketDataset, settings: dict[str, Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.settings = settings
        self._token = CancellationToken()

    def cancel(self) -> None:
        """Ask the run to stop. Safe to call from the UI thread.

        Setting the flag is the only thing that crosses into the worker thread.
        The run returns on its own within roughly one event's processing time
        and ``finished`` fires normally, so teardown is unchanged.
        """
        self._token.cancel()

    @property
    def cancelled(self) -> bool:
        """Whether a stop has been requested."""
        return self._token.cancelled

    def run(self) -> None:
        """Replay the dataset and emit the result, partial or complete."""
        try:
            self.progress.emit("Replaying events...")
            result = run_backtest(
                self.dataset.markets,
                self.dataset.quotes,
                self.dataset.trades,
                self.settings,
                on_progress=self.progress_update.emit,
                should_stop=self._token,
            )
            self.finished.emit(result)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"{type(error).__name__}: {error}")


class DemoDataWorker(Worker):
    """Generates (or loads from DuckDB) the synthetic demo dataset."""

    def __init__(self, settings: dict[str, Any], refresh: bool = False) -> None:
        super().__init__()
        self.settings = settings
        self.refresh = refresh

    def run(self) -> None:
        """Load or generate the demo dataset and emit it."""
        try:
            self.progress.emit("Generating DEMO DATA...")
            self.finished.emit(load_or_create_dataset(self.settings, refresh=self.refresh))
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"{type(error).__name__}: {error}")


class KalshiFetchWorker(Worker):
    """Fetches read-only Kalshi history off the UI thread."""

    def __init__(self, config: dict[str, Any], spec: KalshiFetchSpec) -> None:
        super().__init__()
        self.config = config
        self.spec = spec

    def run(self) -> None:
        """Fetch and normalize the requested markets."""
        try:
            self.progress.emit(
                f"Requesting {len(self.spec.tickers)} market(s) from Kalshi (read-only)..."
            )
            adapter = KalshiAdapter(self.config)
            dataset = adapter.fetch_dataset(self.spec)
            self.progress.emit(
                f"Fetched {len(dataset.quotes)} quotes and {len(dataset.trades)} trades."
            )
            self.finished.emit(dataset)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"{type(error).__name__}: {error}")


class CorpusLoopWorker(Worker):
    """Runs the corpus-building research loop off the UI thread.

    The loop is the longest job in the application - minutes of throttled venue
    requests followed by a full walk-forward pass per iteration - so it is the
    one that most needs to be stoppable. Stopping is safe: iterations already
    completed keep their results and their registry rows.
    """

    iteration_done = Signal(object)
    """Carries a :class:`research.run_loop.IterationReport` as each pass ends."""

    def __init__(self, settings: dict[str, Any], config: Any) -> None:
        super().__init__()
        self.settings = settings
        self.config = config
        self._token = CancellationToken()

    def cancel(self) -> None:
        """Ask the loop to stop after the current chunk or iteration."""
        self._token.cancel()

    @property
    def cancelled(self) -> bool:
        """Whether a stop has been requested."""
        return self._token.cancelled

    def run(self) -> None:
        """Grow the corpus until the gate passes, data runs out, or stopped."""
        try:
            from research.registry import ResearchRegistry
            from research.run_loop import run_loop

            self.progress.emit("Discovering settled markets (read-only)...")
            registry = ResearchRegistry(
                self.settings.get("research", {}).get(
                    "registry_path", "data/research/registry.duckdb"
                )
            )
            report = run_loop(
                self.settings,
                self.config,
                registry=registry,
                on_iteration=self.iteration_done.emit,
                should_stop=self._token,
            )
            self.finished.emit(report)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"{type(error).__name__}: {error}")


class CampaignWorker(Worker):
    """Runs many resampled backtests for a fixed time budget, off the UI thread.

    One backtest is one sample. This keeps running them over different random
    subsets of the same corpus so the user sees a distribution of outcomes
    rather than a single number that may be luck.
    """

    campaign_update = Signal(object)
    """Carries a :class:`backtest.campaign.CampaignProgress`."""

    def __init__(self, dataset: MarketDataset, settings: dict[str, Any],
                 config: Any) -> None:
        super().__init__()
        self.dataset = dataset
        self.settings = settings
        self.config = config
        self._token = CancellationToken()

    def cancel(self) -> None:
        """Stop after the current run. Completed runs are kept."""
        self._token.cancel()

    @property
    def cancelled(self) -> bool:
        """Whether a stop has been requested."""
        return self._token.cancelled

    def run(self) -> None:
        """Run the campaign and emit the accumulated result."""
        try:
            from backtest.campaign import run_campaign

            self.progress.emit("Running repeated backtests...")
            result = run_campaign(
                self.dataset,
                self.settings,
                self.config,
                on_progress=self.campaign_update.emit,
                should_stop=self._token,
            )
            self.finished.emit(result)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"{type(error).__name__}: {error}")


class WorkerRunner(QObject):
    """Owns a worker and the thread it runs on, and cleans both up afterwards.

    Two lifetime hazards are handled here, because getting either wrong
    produces a hard process crash rather than an exception:

    * **Premature collection.** A runner whose last Python reference goes out
      of scope while its ``QThread`` is still running is destroyed underneath
      that thread. Live runners are therefore held in :data:`_LIVE` until they
      shut down.
    * **Interpreter exit.** A thread still running when Python tears down dies
      with ``QThread: Destroyed while thread is still running`` and aborts the
      process (``0xC0000409``). :func:`shutdown_all` is registered with
      ``atexit`` to stop them first.
    """

    def __init__(self, worker: Worker, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.thread = QThread()
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.finished.connect(self._shutdown)
        worker.failed.connect(self._shutdown)

    def start(self) -> None:
        """Start the worker thread, keeping this runner alive until it ends."""
        _LIVE.add(self)
        self.thread.start()

    def _shutdown(self, *_: Any) -> None:
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5_000)
        _LIVE.discard(self)


# Runners with a live thread. A set, not a list, so double-shutdown is cheap.
_LIVE: set[WorkerRunner] = set()


def shutdown_all(timeout_ms: int = 5_000) -> None:
    """Stop every running worker thread. Safe to call more than once."""
    for runner in list(_LIVE):
        thread = runner.thread
        if thread.isRunning():
            thread.quit()
            thread.wait(timeout_ms)
        _LIVE.discard(runner)


atexit.register(shutdown_all)

