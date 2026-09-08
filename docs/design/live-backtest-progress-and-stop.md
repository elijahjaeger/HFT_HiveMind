# Design Document: Live Progress & Cancellation for Backtests

## 1. Chosen approach and why

We are implementing the **callback-based design** (`on_progress` / `should_stop` optional callables threaded through the existing `BacktestEngine.run` loop), because across all three independent judge panels it won on the two criteria that matter most for this codebase: it is the smallest possible change to the exact code path that `tests/test_engine.py::test_the_backtest_is_reproducible` exercises, and it is the easiest to unit-test without any Qt or threading machinery. Both alternatives were good — the generator-based `run_streaming` design gives equally clean thread isolation but only by restructuring `run()` itself into a driver over a new generator (a mechanical rewrite of tested control flow), and the `LiveBacktestState` polled-shared-state design gives the smoothest, most flood-proof UI update story and best mirrors the Paper tab's actual `QTimer` polling idiom — but both add more new machinery inside `backtest/engine.py` than is justified for a workload the brief itself characterizes as sub-few-seconds and ~4,000 events. We therefore take the callback design as the backbone and graft on two ideas the judges explicitly called out as good from the runners-up: (1) a **time-based (not purely event-count-based) throttle gate** for progress emission, borrowed from the `LiveBacktestState` design's observation that CLOCK-driven decision ticks are far more expensive than QUOTE/TRADE events, so a flat event-count cadence risks bursty, uneven UI updates; and (2) an explicit **`positions_view`** (a plain list of per-market `(market_id, side, size, avg_price)` tuples), also borrowed from the `LiveBacktestState` design, since the existing `PortfolioSnapshot.open_positions` is only a count and a genuinely useful live view needs the per-market breakdown. Both additions are cheap, additive, and preserve the callback design's core property: `BacktestEngine.run` remains a plain, Qt-free, thread-free function whose default behavior (no callables passed) is byte-for-bit identical to today.

**This entire feature is local-only, simulation-only tooling.** It touches `backtest/engine.py`, `app/workers.py`, and `app/desktop.py`'s Backtest tab exclusively. It does not touch any live/paper trading session code (`PaperSession`), any order-placement or Kalshi-fetch code path, and it adds no network calls. "Stop" here means "stop replaying historical simulation events," never "cancel a live order" — there is no live order in this code path at all.

## 2. Exact API changes to `backtest/engine.py`

### 2.1 New dataclass in `core/models.py`

```python
@dataclass
class BacktestProgress:
    ts: datetime
    index: int                       # 0-based index of the event just processed
    n_events: int                    # total events in this run (== BacktestResult.n_events)
    last_snapshot: PortfolioSnapshot | None   # most recent snapshot, if one was recorded
    open_positions: int              # len(portfolio.positions) at this point
    positions_view: list[tuple[str, str, float, float]]  # (market_id, side, size, avg_price)
    last_audit: AuditEvent | None    # most recent AuditEvent appended this iteration, if any
```

This is a plain, immutable-in-practice snapshot dataclass — no references to the live `Portfolio`/`Position` objects, only copied scalars and, for `last_snapshot`/`last_audit`, references to objects that are never mutated after being appended (safe to share by reference, per the existing append-only pattern in `_maybe_decide` etc.).

### 2.2 New field on `BacktestResult` (`core/models.py`)

```python
cancelled: bool = False
```

Defaults to `False`, so existing code/tests reading `BacktestResult` fields see no change on the untouched path.

### 2.3 `BacktestEngine.run` signature change (`backtest/engine.py:411`)

```python
def run(
    self,
    markets, quotes, trades,          # exact existing positional signature, unchanged
    *,
    on_progress: Callable[[BacktestProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> BacktestResult:
```

Both new parameters are **keyword-only** (the `*`) specifically so no future positional-argument call site can accidentally collide with them, and both default to `None` so every existing call site (`run_backtest()`, `tests/test_engine.py`, and the untracked `research/run_loop.py` — verify by grep before landing, see §7) needs zero changes.

### 2.4 Loop body changes (inside `for event in events:`, `:448`)

Two additions inserted after the existing per-event work (after the snapshot-append block at `:496-497`), each independently guarded so they are true no-ops when the corresponding parameter is `None`:

1. **Cancellation check**, evaluated every single iteration (cheap boolean call, no lock):
   ```python
   if should_stop is not None and should_stop():
       result.cancelled = True
       break
   ```

2. **Throttled progress emission.** Use `enumerate(events)` to get the index `i`. Gate on a **time-based cadence**, not a flat event count, since a CLOCK tick's 8-market `_maybe_decide` fan-out costs far more than a QUOTE/TRADE event:
   ```python
   now = time.monotonic()
   if on_progress is not None and (
       now - last_emit_time >= MIN_PROGRESS_INTERVAL_SECONDS
       or i == len(events) - 1
   ):
       last_emit_time = now
       on_progress(BacktestProgress(
           ts=event.ts,
           index=i,
           n_events=result.n_events,
           last_snapshot=result.snapshots[-1] if result.snapshots else None,
           open_positions=len(portfolio.positions),
           positions_view=[
               (mid, p.side, p.size, p.avg_price)
               for mid, p in portfolio.positions.items()
           ],
           last_audit=result.audit_log[-1] if result.audit_log else None,
       ))
   ```
   `MIN_PROGRESS_INTERVAL_SECONDS = 0.1` (module-level constant) caps emission at ~10/sec regardless of event density, giving the same flood-proofing the polled-state design achieves via fixed-rate `QTimer` polling, but without adding a lock or a background poll — the check is inline in the same thread that's already running the loop. The final event (`i == len(events) - 1`) always emits, so the UI reliably receives a 100%-complete tick.

No changes anywhere to `_maybe_decide`'s `counter`-based `proposal_id`/`fill_id` generation (`:571`, `:738`) — both new hooks live outside that function and do not affect decision/fill ordering or ID sequencing.

### 2.5 What stays untouched

`run()`'s overall shape, the `for event in events:` structure, `_execute_pending`, `_maybe_decide`, and `result.n_events = len(events)` at `:446` are unchanged. `run_backtest()` (module-level convenience wrapper) is unchanged except that it now accepts and forwards `on_progress`/`should_stop` as optional keyword args to `BacktestEngine.run`.

## 3. Exact changes to `app/workers.py`

1. Add `import threading` and `import time` at the top of the file (if `time` is not already imported).

2. `Worker` base class (`:16-25`): add one new signal:
   ```python
   progress_update = Signal(object)   # carries a BacktestProgress instance
   ```
   Kept on the base class (rather than only on `BacktestWorker`) is a judgment call either way; recommend defining it on `BacktestWorker` specifically instead, since `DemoDataWorker`/`KalshiFetchWorker` have no analogous concept and an unused signal on the shared base slightly loosens `Worker`'s contract. Final placement: **`BacktestWorker` only**.

3. `BacktestWorker.__init__`: add
   ```python
   self._stop_event = threading.Event()
   ```

4. `BacktestWorker.run()` (`:36-48`): keep the existing `try/except Exception ... self.failed.emit(...)` envelope unchanged. Inside the `try`, build a small closure and pass both new hooks into `run_backtest(...)`:
   ```python
   def _emit_progress(p):
       self.progress_update.emit(p)

   result = run_backtest(
       self.dataset, self.settings,
       on_progress=_emit_progress,
       should_stop=self._stop_event.is_set,
   )
   self.finished.emit(result)
   ```

5. New public method:
   ```python
   def cancel(self) -> None:
       self._stop_event.set()
   ```
   Safe to call directly from the UI thread with no additional locking — `threading.Event.set()`/`.is_set()` are documented thread-safe for exactly this producer/consumer pattern.

6. `WorkerRunner`, `_shutdown`, and module-level `shutdown_all` (`:92-140`): **no changes**. Once `cancel()` is called, `run()` returns on its own within roughly one event's processing time (worst case, one CLOCK tick's full decision fan-out), `finished` fires normally, and `_shutdown`'s existing `thread.quit()` / `thread.wait(5_000)` proceeds exactly as it does for a normal completion — the 5-second timeout that today would silently no-op against a stuck synchronous call is no longer relied upon for interactive Stop.

## 4. Exact changes to the Backtest tab in `app/desktop.py`

### 4.1 New widgets in `_build_backtest_tab` (`:471-527`)

Layout, top to bottom, replacing the current single "Run backtest" button row:

- **Button row**: the existing "Run backtest" `QPushButton` (`:476-477`) stays where it is, unchanged in label and wiring. Immediately to its right, add a new **"Stop" `QPushButton`**, initially `setEnabled(False)`.
- **Progress row**, directly below the button row: a new tab-local `QProgressBar` with `setRange(0, 100)` — a real, determinate bar, distinct from the existing global marquee bar at `:212-216` (`range(0,0)`), which is left completely untouched and continues to serve the demo-data/Kalshi-fetch flows exactly as today.
- **Live status row**, directly below the progress bar: one `QLabel` showing the latest simulated timestamp and open-position count (e.g. "t=14:32:00 · 3 open positions"), and a second `QLabel` showing the most recent audit message (e.g. "FILL: BUY 12 @ 0.62 on MKT-7"). These are small, single-line labels — not a table — kept intentionally cheap to repaint on every throttled update.
- **Optional live positions strip** (nice-to-have, not required for v1): a small `QLabel` or three-column mini-row summarizing `positions_view` (market/side/size), updated on the same cadence as the other status labels. This is the one addition beyond the minimal callback design, justified by the runner-up's point that a per-market breakdown isn't otherwise visible mid-run.
- The existing static explanatory label, metric-tile row, equity/drawdown chart splitter, and the three result tables (metrics, calibration, risk decisions) (`:479-526`) are **all unchanged** — they continue to be populated exactly once, from the final `BacktestResult`, when `finished` fires.

### 4.2 `run_backtest` (`:1377-1384`)

- Keep the existing `self.dataset is None` guard and `self.run_button.setEnabled(False)` (`:1383`).
- Additionally: `self.backtest_stop_button.setEnabled(True)`, reset the new progress bar to 0, clear the status labels.
- Construct the worker exactly as today, but keep a reference: `self._backtest_worker = worker` so the Stop button's handler can reach it.
- Connect the new signal before starting: `worker.progress_update.connect(self._on_backtest_progress)`.

### 4.3 New slot `_on_backtest_progress(self, p: BacktestProgress)`

- `self.backtest_progress_bar.setValue(int(100 * p.index / max(p.n_events, 1)))`
- Update the timestamp/open-positions label from `p.ts` / `p.open_positions`.
- Update the audit-message label from `p.last_audit.message` if `p.last_audit is not None`.
- Optionally refresh the live positions strip from `p.positions_view`.
- Does **not** touch the metrics/calibration/risk-decision tables — those remain a single, final render from `finished`.

### 4.4 New slot `_stop_backtest(self)`, wired to the Stop button's `clicked`

- Guard on `self._backtest_worker is not None`.
- Call `self._backtest_worker.cancel()`.
- Immediately `self.backtest_stop_button.setEnabled(False)` to prevent double-clicks.
- Does nothing else — no thread teardown, no `wait()` call. The run finishes via the existing `finished`/`failed` → `_on_result_ready`/`_on_worker_failed` path, unchanged.

### 4.5 `_on_result_ready` (`:1565-1567`) and `_on_worker_failed` (`:1394-1398`)

Both already re-enable `self.run_button`. Extend both to also: disable the Stop button, clear `self._backtest_worker = None`, and reset the progress bar to 0. `_on_result_ready` additionally checks `result.cancelled` and, if `True`, shows a distinct status message (e.g. "Backtest stopped — partial results (N/M events)") instead of the normal completion message, while still rendering the (partial) metrics/calibration/risk tables from the returned `BacktestResult` exactly as it would for a complete run.

## 5. What "Stop" actually does to a running backtest

Clicking Stop calls `BacktestWorker.cancel()`, which sets a `threading.Event`. The engine's `for event in events:` loop checks this event once per iteration and `break`s out as soon as it sees it set — worst-case latency is one event's processing time, which is sub-millisecond for a QUOTE/TRADE event and bounded by one full CLOCK tick's 8-market decision fan-out in the worst case (the single most expensive per-event cost in this engine, per the brief). After the `break`, the function falls through to its existing tail: `result.final_equity = ...` and `return result` execute exactly as they do on natural completion, except `result.cancelled` is now `True` and the result reflects only a prefix of the full event stream (fewer snapshots, fewer proposals/fills/audit entries than a full run would have).

This is **cancel-and-discard-the-rest**, not pause/resume — there is no mechanism to continue a stopped run from where it left off. The returned `BacktestResult` is a fully-formed, valid object (same shape as a complete run), just built from fewer events; it is rendered into the existing metrics/calibration/risk-decision tables unchanged, with the UI adding a "stopped early / partial" label so the user doesn't mistake a truncated run for a complete one. No trade, order, or live-session state is touched by Stop — the only thing being interrupted is a `for` loop replaying historical simulated events in memory.

## 6. Concurrency/thread-safety argument in plain terms

- **Two threads exist in this feature**: the UI thread (where `app/desktop.py` runs) and the worker's `QThread` (where `BacktestEngine.run` actually executes, via the existing `moveToThread` pattern in `WorkerRunner`).
- **The only thing that crosses between them going one way (UI → worker) is a boolean flag**: `threading.Event`. Setting it from the UI thread (`cancel()`) and reading it from the worker thread (`should_stop()` / `is_set()`) is exactly the use case `threading.Event` is built for — internally lock-protected, no additional synchronization needed, and no risk of a torn read.
- **The only thing that crosses the other way (worker → UI) is a copied snapshot object**: each `BacktestProgress` is built fresh, out of plain scalars and references to append-only objects (`PortfolioSnapshot`, `AuditEvent`) that are never mutated after being created. Nothing about the live, mutable `Portfolio.positions` dict or `Position` objects is ever handed across the thread boundary — `positions_view` is a plain list of tuples copied out at emission time.
- **The cross-thread hop itself is handled by Qt, not by any code we write**: `Signal(object).emit()` called from a non-GUI thread is automatically queued by Qt onto the receiving (UI-thread) object's event loop — this is standard Qt signal/slot semantics, identical to how the existing `worker.progress` (`Signal(str)`) → `statusBar().showMessage` wiring already works today. No manual `invokeMethod`, no shared mutable state, no explicit lock is required anywhere in this design.
- **There is no lock inside `backtest/engine.py` at all.** The engine module gains a plain dataclass and two optional callables — it remains free of any threading/Qt import or concept. All thread-crossing is handled at the `BacktestWorker` boundary in `app/workers.py`, where Qt's own queued-connection machinery does the marshaling.
- **Net result**: the design that requires the least new reasoning about concurrency, because it deliberately keeps the amount of state that crosses threads to two small, well-understood primitives (a flag and copied snapshots) rather than introducing a shared, lockable object that both threads read and write.

## 7. Test plan

### 7.1 Existing tests that must still pass unmodified

- **`tests/test_engine.py::test_the_backtest_is_reproducible`** (`:293-296`) — calls `run_backtest(...)` twice with no `on_progress`/`should_stop` args and asserts `.metrics` equality. Since both new parameters default to `None` and every new branch is guarded by an `is not None` check, this path is a byte-for-bit no-op; the test requires no changes.
- All other tests in `tests/test_engine.py` that call `BacktestEngine.run`/`run_backtest` positionally with the existing arguments — unaffected, since the signature change is purely additive and keyword-only.
- Before landing: **grep the repo for every call site of `BacktestEngine.run` and `run_backtest`**, including the untracked `research/run_loop.py` and `tests/test_run_loop.py` shown in git status — confirm none constructs the call with positional arguments that could collide with the new keyword-only parameters (they can't, since they're keyword-only, but confirm none is monkeypatching `run`'s signature or introspecting it).
- Any test that asserts on the exact field set of `BacktestResult` (e.g. a `dataclasses.fields()` count check or strict schema comparison) needs updating for the new `cancelled: bool = False` field — grep for such usage before landing.

### 7.2 New tests needed

In `tests/test_engine.py`:
1. **`test_should_stop_halts_early`** — pass a `should_stop` callable that returns `True` after N calls; assert `result.cancelled is True` and that fewer events were processed than the full stream (e.g. `len(result.snapshots) < n_events` or a dedicated processed-count check).
2. **`test_on_progress_called_and_throttled`** — pass a list-appending `on_progress`; run the full event stream; assert at least one call occurred, the last call's `index == len(events) - 1`, and the call count is well below `len(events)` (proving the time-based throttle works, not just documenting it).
3. **`test_on_progress_payload_matches_final_state`** — assert the last progress call's `last_snapshot` has the same `equity`/`ts` as `result.snapshots[-1]`, and that `positions_view` matches `portfolio.positions` at that point (via a controlled fixture), so the payload isn't stale or decoupled from what was actually recorded.
4. **`test_default_run_unaffected`** — explicit regression test that calling `run()`/`run_backtest()` with no new kwargs on a fixture dataset produces identical `.metrics` to a reference call captured before this change (belt-and-suspenders alongside the existing reproducibility test).

In `app/workers.py`'s test suite (or a new `tests/test_workers.py` if none exists — confirm first):
5. **`test_backtest_worker_cancel`** — drive `BacktestWorker.run()` on a plain `threading.Thread` (not a real `QThread`, to stay headless/fast) with a fixture dataset large enough to not finish instantly; call `cancel()` shortly after starting; assert `finished` fires with a `BacktestResult` where `cancelled is True`, and that this happens within a bounded time (proving `Event.set()` actually interrupts promptly).

### 7.3 Out of scope for tests in this slice

UI-level tests of the Stop button's Qt wiring (`app/desktop.py`) are not required for this slice unless the project already has a precedent for driving real Qt widgets in tests — check for one before deciding whether to add it.

## 8. Known limitations / explicitly out of scope for this slice

- **No pause/resume** — Stop discards the remainder of the run; there is no way to continue a stopped backtest from where it left off. Building that would require lifting `books`/`tape`/`marks`/`pending`/`portfolio`/`counter` out of `run()`'s local scope into persisted session state (mirroring `PaperSession`), a much larger refactor not justified by "add a Stop button."
- **Fixed 0.1s progress throttle is a constant, not adaptive to dataset size.** For the shipped demo config (~4,000 events, sub-few-second runs) this is more than sufficient; a much larger user-supplied dataset (e.g. a multi-day real Kalshi fetch) has not been measured against this constant and may want it exposed as a `backtest_cfg` setting later. Not required for this slice.
- **No fine-grained interruption inside a single CLOCK tick's 8-market decision fan-out.** `should_stop()` is checked once per raw event, so the heaviest single per-event cost (the multi-agent forecast/consensus/risk pipeline across all markets on a CLOCK event) cannot itself be interrupted partway through. Given the brief's characterization of current per-tick cost as light, this is deferred; a finer-grained check inside the per-market loop could be added later if that cost grows.
- **No visual "partial" badge on the metrics/calibration/risk tables themselves** — only the status message/label distinguishes a cancelled run from a complete one. If partial-result semantics need to be unmistakable (e.g. for a report or export), a follow-up should add an explicit badge to those tables.
- **The live positions strip (§4.1) is a nice-to-have**, not required to ship the core Stop/progress feature; it can be deferred to a follow-up if implementation time is tight, since `positions_view` is already being computed and threaded through regardless (needed for the status labels' open-position count).
- **This feature does not add, modify, or touch any trading/order-placement code path.** No changes are made to `PaperSession`, any live-market connectivity, or any code that sends, cancels, or modifies a real order. Everything described here operates exclusively on the historical, simulated `Event` stream already loaded into `BacktestEngine.run`.