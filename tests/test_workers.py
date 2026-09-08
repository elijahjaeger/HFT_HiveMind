"""Worker-layer tests: cancellation actually interrupts, progress reaches the UI.

Run against the offscreen Qt platform. The workers are driven on plain
``threading.Thread``s rather than real ``QThread``s, which keeps these tests
fast and headless while exercising the same code the UI runs.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# pandas before PySide6: see app/__init__.py for the import-order note.
pytest.importorskip("pandas")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.workers import BacktestWorker  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.jobs import CancellationToken, JobProgress, ProgressThrottle  # noqa: E402
from core.synthetic import generate_demo_dataset  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """A single QApplication for the module."""
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(scope="module")
def settings() -> dict:
    return get_settings()


@pytest.fixture(scope="module")
def dataset(settings):
    return generate_demo_dataset(settings["data"])


# --------------------------------------------------------------------------
# core/jobs primitives
# --------------------------------------------------------------------------


def test_a_fresh_token_is_not_cancelled():
    assert CancellationToken().cancelled is False


def test_a_token_is_callable_so_it_can_be_passed_as_should_stop():
    token = CancellationToken()
    assert token() is False
    token.cancel()
    assert token() is True


def test_cancelling_twice_is_harmless():
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.cancelled is True


def test_a_token_can_be_reset_for_another_run():
    token = CancellationToken()
    token.cancel()
    token.reset()
    assert token.cancelled is False


def test_a_token_set_on_another_thread_is_seen():
    """The one thing that crosses UI -> worker is this flag."""
    token = CancellationToken()
    threading.Thread(target=token.cancel).start()
    deadline = time.monotonic() + 2.0
    while not token.cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert token.cancelled is True


def test_progress_without_a_known_total_reports_no_fraction():
    """A corpus pull discovers its own size; do not invent a percentage."""
    update = JobProgress(index=5, total=0)
    assert update.fraction is None
    assert update.percent == 0


def test_progress_fraction_is_clamped():
    assert JobProgress(index=50, total=100).percent == 50
    assert JobProgress(index=500, total=100).fraction == 1.0


def test_the_throttle_admits_the_first_update_then_closes():
    throttle = ProgressThrottle(min_interval=10.0)
    assert throttle.should_emit(now=100.0) is True
    assert throttle.should_emit(now=100.1) is False


def test_the_throttle_reopens_after_its_interval():
    throttle = ProgressThrottle(min_interval=1.0)
    assert throttle.should_emit(now=100.0) is True
    assert throttle.should_emit(now=101.5) is True


def test_a_forced_update_always_passes_the_throttle():
    """Without this the UI is left showing 99% forever."""
    throttle = ProgressThrottle(min_interval=1000.0)
    assert throttle.should_emit(now=100.0) is True
    assert throttle.should_emit(now=100.001, force=True) is True


# --------------------------------------------------------------------------
# BacktestWorker
# --------------------------------------------------------------------------


def _drive(worker: BacktestWorker) -> dict:
    """Run a worker on a plain thread, collecting what it emitted."""
    seen: dict = {"progress": [], "result": None, "error": None}
    # Direct connections: the worker runs on a plain thread with no Qt event
    # loop, so an auto (queued) connection would post to a loop that never
    # spins and the assertions would see nothing.
    direct = Qt.ConnectionType.DirectConnection
    worker.progress_update.connect(seen["progress"].append, type=direct)
    worker.finished.connect(lambda r: seen.__setitem__("result", r), type=direct)
    worker.failed.connect(lambda m: seen.__setitem__("error", m), type=direct)
    thread = threading.Thread(target=worker.run)
    thread.start()
    thread.join(timeout=120)
    assert not thread.is_alive(), "worker did not finish"
    return seen


def test_a_worker_run_to_completion_reports_progress_and_a_full_result(
    qapp, dataset, settings
):
    seen = _drive(BacktestWorker(dataset, settings))
    assert seen["error"] is None
    assert seen["result"] is not None
    assert seen["result"].cancelled is False
    assert seen["progress"], "no progress was emitted"
    assert seen["progress"][-1].percent == 100


def test_cancelling_before_the_run_starts_stops_it_immediately(qapp, dataset, settings):
    worker = BacktestWorker(dataset, settings)
    worker.cancel()
    assert worker.cancelled is True

    started = time.monotonic()
    seen = _drive(worker)
    elapsed = time.monotonic() - started

    assert seen["result"] is not None
    assert seen["result"].cancelled is True
    assert elapsed < 30.0, "cancellation should not wait for the full replay"


def test_a_cancelled_worker_still_emits_a_usable_result(qapp, dataset, settings):
    """Stop yields a partial result, never a failure."""
    worker = BacktestWorker(dataset, settings)
    worker.cancel()
    seen = _drive(worker)

    assert seen["error"] is None, "cancellation is not an error"
    result = seen["result"]
    assert result.cancelled is True
    assert result.n_events > 0, "the stream size is known even when stopped early"
    assert isinstance(result.metrics, dict)


def test_cancelling_midway_truncates_the_run(qapp, dataset, settings):
    """The flag is checked once per event, so a stop lands promptly."""
    worker = BacktestWorker(dataset, settings)

    def stop_after_a_few(update) -> None:
        if update.index > 0:
            worker.cancel()

    worker.progress_update.connect(
        stop_after_a_few, type=Qt.ConnectionType.DirectConnection
    )
    seen = _drive(worker)

    full = _drive(BacktestWorker(dataset, settings))["result"]
    assert seen["result"].cancelled is True
    assert len(seen["result"].snapshots) < len(full.snapshots)


def test_the_worker_exposes_no_order_placement_surface(dataset, settings):
    worker = BacktestWorker(dataset, settings)
    forbidden = {"place_order", "submit_order", "create_order", "cancel_order", "post"}
    assert not (forbidden & set(dir(worker)))
    assert hasattr(worker, "cancel"), "cancel() stops a simulation, not an order"


# --------------------------------------------------------------------------
# CorpusLoopWorker
# --------------------------------------------------------------------------


def test_the_corpus_worker_can_be_cancelled_before_it_starts(qapp, settings):
    """The longest job in the app is the one that most needs a Stop."""
    from app.workers import CorpusLoopWorker
    from research.run_loop import LoopConfig

    worker = CorpusLoopWorker(settings, LoopConfig(max_iterations=1))
    worker.cancel()
    assert worker.cancelled is True

    seen: dict = {"result": None, "error": None}
    direct = Qt.ConnectionType.DirectConnection
    worker.finished.connect(lambda r: seen.__setitem__("result", r), type=direct)
    worker.failed.connect(lambda m: seen.__setitem__("error", m), type=direct)

    thread = threading.Thread(target=worker.run)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "a cancelled loop must not keep running"
    assert seen["error"] is None, "cancellation is not an error"
    assert seen["result"] is not None
    assert seen["result"].stopped_because.value == "CANCELLED"


def test_the_corpus_worker_exposes_no_order_placement_surface(settings):
    from app.workers import CorpusLoopWorker
    from research.run_loop import LoopConfig

    worker = CorpusLoopWorker(settings, LoopConfig())
    forbidden = {"place_order", "submit_order", "create_order", "cancel_order", "post"}
    assert not (forbidden & set(dir(worker)))
