"""Background workers so the UI never blocks on a backtest or a network fetch."""

from __future__ import annotations

import atexit
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from adapters.kalshi import KalshiAdapter, KalshiFetchSpec
from backtest.engine import run_backtest
from core.dataset import MarketDataset
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
    """Runs one backtest off the UI thread."""

    def __init__(self, dataset: MarketDataset, settings: dict[str, Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.settings = settings

    def run(self) -> None:
        """Replay the dataset and emit the result."""
        try:
            self.progress.emit("Replaying events...")
            result = run_backtest(
                self.dataset.markets,
                self.dataset.quotes,
                self.dataset.trades,
                self.settings,
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

