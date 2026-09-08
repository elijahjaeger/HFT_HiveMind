# Decisions log

Append-only. Newest first. Each entry records what changed, why, and what it
supersedes, so a later reader can tell a deliberate reversal from drift.

Before this file existed the recorded intent lived in `README.md` prose and
`research/README_DESIGN.md`. An audit on 2026-09-07 found no structured log and
no version control, meaning a change had nowhere to record its rationale and
nothing to diff against. This file closes that gap.

---

## 2026-09-08 - research/run_loop.py: the loop runs on real data; the protocol does not fit it

**Status:** adopted (the orchestrator). **Open** (what to do about empty folds).

### What was built

`research/run_loop.py`, a single orchestration script over the existing modules.
No new data source, no new model, no new metric, no order path. It discovers
settled Kalshi markets through the read-only adapter, accumulates them into a
growing corpus cached via `core/store.py`, runs all four `research/pipeline.py`
stages, and reports every promotion-gate check by name. Runnable as
`python -m research.run_loop`.

Stop conditions: gate passes, discovery exhausted, repeated discovery failure,
`max_stale_iterations` (default 5, chosen by the user) with no growth, or the
absolute iteration cap.

**It never loosens the gate.** It varies only how much real data it has. Tests
assert it passes no `min_markets`/`min_folds` override and calls no approval
function, with a confidence check proving the AST detector is not vacuous.

### First real-data result (5 iterations, 2026-06-09 back to 2026-02-09)

| Iter | Corpus | Resolved | Rows | OOS effective_n | Folds | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 400 | 81 | 5,147 | 0 | 0 | failed |
| 2 | 800 | 94 | 5,553 | 0 | 0 | failed |
| 3 | 1,200 | 125 | 6,765 | 0 | 0 | failed |
| 4 | 1,600 | 127 | 6,769 | 0 | 0 | failed |
| 5 | 2,000 | 127 | 6,769 | 0 | 0 | failed |

**The "Real market data" gate check passed on every iteration** - the first time
this project has run its pipeline on real settled data.

**Root cause of the failure, and it is not data volume.** All three folds build
a test set of 30 markets and a training set of **zero**:

```
fold 0: train=0 test=30 rows_out=0  skipped='Fold has an empty train, validation or test set after purging.'
fold 1: train=0 test=30 rows_out=0  ...
fold 2: train=0 test=30 rows_out=0  ...
```

Every training market is purged. This is the purge rule working exactly as
designed and exactly as `research/corpus.py` predicted: Kalshi's settled
population in this window is dominated by short-lived, heavily-overlapping
markets, so essentially every training market resolves after the first test
decision timestamp. Resolved markets plateaued at 127 while the corpus grew to
2,000 - iterations 4 and 5 added 800 markets and 2 resolved ones.

Also material: **33,728 of 40,497 rows (83%) were dropped for having no
two-sided book.** Candlestick history carries no book depth, and at these
market lifetimes many candles quote one side only.

### Findings about the venue, recorded because they are not in the README

- The **historical candlestick endpoint 404s for every market tested**, across
  three series and all lifetimes, even for markets that closed before the
  published cutoff (`2026-07-09`). The series endpoint serves the same candles.
  `run_loop` tries historical first and falls back.
- The candlestick endpoint answers **HTTP 400**, not a truncated response, when
  the interval and span imply too many candles. A 30-day window at 1-minute
  granularity (~43,000 candles) is refused. Requests are therefore bounded to
  each chunk's own market lifetimes.
- `KalshiAdapter.fetch_dataset` guards its per-market call but **not** its
  candlestick call, so one bad ticker aborts the whole batch. `run_loop` fetches
  in chunks of 25 so one 404 costs a chunk, not the iteration.
- The settled listing is dominated by 1-6h markets (7,992 of 12,000 in a 120-day
  sample). At `horizon_seconds=3600` those yield **zero** decision timestamps, so
  `run_loop` filters on a minimum market lifetime. This excludes markets that
  cannot produce a training row at all - the same kind of exclusion
  `research/labels.py` already makes - not markets whose results are inconvenient.

### Defects fixed in `core/store.py` (all with regression tests)

Three, all reachable only with real data, all of which would have made this
impossible or silently wrong:

1. **`read_dataset` crashed on any fetched corpus.** A fetched corpus has
   `seed=None`; `int(NULL)` raised `TypeError`. Real data could not be cached.
2. **Provenance was destroyed by the round trip.** Only `seed` and `label` were
   persisted, and `label` was never read back. A cached Kalshi corpus returned
   as `label="DEMO DATA", source="SYNTHETIC_DEMO"`, so the gate's "Real market
   data" check would have failed **permanently**, and the failure would have
   been blamed on data volume.
3. **An empty collection could not be cached.** `pd.DataFrame([])` has no
   columns and DuckDB refuses it, so a corpus fetched without the tape (which is
   how `run_loop` fetches, to stay inside the throttle) crashed on write.

### Reporting defects fixed in `run_loop` itself

- It reported the **walk-forward candidate's** `effective_n`, which is 0 whenever
  folds are empty. A corpus holding 127 resolved markets was displayed as 0,
  reading as "no data" when the real problem was fold construction. Corpus and
  out-of-sample counts are now reported separately, and the summary diagnoses
  which of the two is the blocker.
- A failed discovery call was indistinguishable from an exhausted archive. Both
  returned no tickers; only one is a finding about the data.

### Open question for a human

The gate cannot pass on this corpus, and the two ways forward are both decisions
this log should not make on its own:

1. **Change the market universe.** Target series whose markets live days or
   weeks and resolve sequentially - the ~7-concurrent shape `research/corpus.py`
   says walk-forward needs. This is a choice of research universe with real
   selection-bias consequences.
2. **Change the walk-forward parameters** - `test_markets_per_fold` (30),
   `n_folds` (3), or `embargo_seconds` - against a corpus this size. This edits a
   logged decision whose stated purpose is to be expensive and honest about it.

Doing neither is also legitimate: the finding "Kalshi's settled short-horizon
markets cannot support this protocol" is itself a result.

**Nothing was promoted, approved or deployed. No parameters were varied to force
a pass.**

---

## 2026-09-07 - Self-learning architecture: LangGraph + CrewAI + FinRL

**Status:** adopted. Supersedes three previously logged decisions.

### What changed

A control layer was added in `src/`, on top of the existing sandbox rather than
replacing it:

| Module | Role |
| --- | --- |
| `src/state_graph.py` | LangGraph supervisor state machine. One `supervisor` node owns all routing; workers `ingest -> research_desk -> rl_policy -> risk -> execute` each return to it. |
| `src/research_desk.py` | CrewAI research desk with two personas, `MacroAnalyst` and `MicrostructureQuant`, emitting an `AlphaVector`. |
| `src/rl_policy.py` | FinRL / Stable-Baselines3 PPO pipeline over a gymnasium env whose observations are alpha vectors and whose rewards run through `backtest.fills.simulate_fill`. |
| `src/contracts.py` | The typed payloads (`AlphaVector`, `PolicyMetrics`, `SupervisorState`) passed between nodes. |
| `compat/` | Refusing stubs for external clients FinRL imports but this project forbids. |

### Decisions superseded

1. **"No reinforcement learning."** Previously stated in `research/models.py`
   and enforced by `build_forecaster` raising on `rl` / `mappo`.
   **Now:** RL is part of the architecture, in `src/rl_policy.py`.
   `build_forecaster` still refuses `rl`, but as a *routing* error rather than a
   ban - a policy emits actions, not calibrated probabilities, so it cannot be
   scored by Brier or log loss and does not belong in the forecaster slot. The
   error message says where it does belong.

2. **"Drift never retrains or promotes anything"** and **"No model is ever
   deployed, replaced, or promoted automatically."**
   **Now:** the RL node updates policy metrics on each pass, which is a learning
   loop by design. The guard that survives is the *gate*, not the ban:
   `RLPolicyPipeline.evaluate_gate` still refuses on fabricated data, on fewer
   than 50 resolved markets, and on failing to beat a frozen baseline. Nothing
   auto-promotes today; the capability exists and the gate is what stands in
   front of it.

3. **"No scikit-learn dependency"** (kept so fitted parameters stay plain
   numbers the registry can diff).
   **Now:** torch, stable-baselines3, FinRL, CrewAI and LangGraph are
   dependencies. The registry consequence is real and accepted: a PPO policy's
   weights are not human-diffable the way logistic coefficients were. The
   `research/` forecasters are unchanged and still store plain numbers.

### Decisions explicitly NOT changed

**Simulation-only stands.** This remains the load-bearing property:

- No order-placement code path anywhere, including `src/`. The AST scan in
  `tests/test_paper_trading.py` was extended to cover `src/` and passes.
- The RL action space terminates in `simulate_fill` and an in-memory portfolio.
- No credentials are used or accepted. The CrewAI desk defaults to a
  deterministic offline path and only calls an LLM when a key is present *and*
  `use_llm=True` is passed explicitly.

### The dependency problem this created, and how it was resolved

`import finrl` (FinRL 0.3.7) transitively imports `finrl.trade` ->
`AlpacaPaperTrading` -> `alpaca_trade_api`, a **live brokerage order client**.
Two independent problems:

1. Installing a live broker SDK contradicts the simulation-only posture above.
2. `alpaca-trade-api` pins `websockets<11`; LangGraph needs `>=14`. Installing
   it broke `import langgraph`. The two requirements are mutually exclusive, so
   no amount of further installation resolves it.

**Resolution:** `compat/` provides refusing stubs for `alpaca_trade_api`, `wrds`
and `yfinance`. They satisfy FinRL's package-root import and raise
`compat.RefusedExternalClient` on any actual use. This is strictly safer than
installing the real SDKs - there is no working broker client in the environment
to reach for by accident - and it keeps `websockets` at a version LangGraph
supports. Dunder access falls through to `AttributeError` so `inspect`/pytest
introspection still works.

### Verification

- `python verify_architecture.py` exits 0, with assertions on graph topology and
  persona names added (it previously checked imports only, and crashed on
  Windows cp1252 before reaching any check).
- `python -m pytest`: **615 passed** (589 pre-existing, unchanged, + 26 new).
- `python -m ruff check .`: clean.
- The graph runs end-to-end on the demo corpus: 8 market states -> 8 alpha
  vectors (6 with a usable view) -> PPO trained over 6 resolved markets -> risk
  -> execute.

### Known weakness of this slice

The graph runs one decision instant per cycle, so the RL node trains on very few
episodes and its `mean_episode_reward` is noise. Both warnings fire on the demo
corpus, correctly: the data is fabricated, and 6 resolved markets is 6
independent outcomes regardless of timestep count. **Nothing here is evidence a
strategy works.**

---

## Pre-existing decisions (recorded retroactively, unchanged)

These were made before this log existed and remain in force. Their full
reasoning lives in `README.md` and `research/README_DESIGN.md`.

- **Information flows forward only.** Agents see a `MarketState` built solely
  from records at or before `as_of`, with `outcome` and `resolution_time`
  stripped.
- **Settlement labels never come from price.** `research/labels.py` is the only
  place an outcome is decided.
- **The statistical unit is the resolved market, not the row.** Every metric
  surface reports `effective_n`.
- **Walk-forward uses market grouping, purge and embargo**, even when that
  empties the folds.
- **Execution realism:** orders are queued and filled after
  `execution_delay_seconds` against a strictly earlier book; limit prices are
  enforced; the fill simulator has its own staleness guard.
- **No tree model.** `TREE_MODEL_BAR` records the bar one would have to clear.

---

# Track / roadmap

Current state, newest first. This section *is* edited in place; the dated
entries above are not.

| Track | State | Next |
| --- | --- | --- |
| Read-only Kalshi adapter | **Working against the live venue.** First exercised for real 2026-09-08; 50 tests remain fixture-only. | Guard the candlestick call inside `fetch_dataset` so one bad ticker skips itself rather than aborting the batch (`run_loop` chunks around this today). |
| Research pipeline on real data | **Runs end-to-end.** 2,000 markets pulled, 127 resolved, 6,769 labelled rows, "Real market data" gate check passing. | Blocked on the fold question below. |
| Walk-forward on Kalshi | **Blocked.** All folds purge to an empty training set. | Needs a human decision: change the market universe, or change the fold parameters. |
| Promotion gate | Never passed. Never bypassed. | Unchanged - it is doing its job. |
| Calibration | Only ever run on synthetic corpora. | Reachable once folds are usable. |
| Self-learning layer (`src/`) | Built, tested, runs on the demo corpus. | Has not been pointed at the real Kalshi corpus. |
| Paper trading | Unchanged, offline-capable, no order path. | Settlement detection (README roadmap 5). |

# Open questions

1. **Which market universe?** Kalshi's settled population in the sampled window
   is overwhelmingly short-lived and concurrent, which the purge rule correctly
   empties. Long-lived sequential series exist (`KXFEDDECISION` markets live
   months) but choosing them is a selection decision with bias consequences.
2. **Are the walk-forward parameters right for real data?** `n_folds=3 x
   test_markets_per_fold=30` needs 90 test markets *plus* a surviving training
   set. They were set against a synthetic corpus shaped to satisfy them.
3. **Is a candle-derived book good enough?** 83% of feature rows were dropped for
   having no two-sided book. If that holds on any real corpus, the feature set
   may need to tolerate one-sided quotes rather than discard them.
4. **Should `min_lifetime_hours` be derived rather than configured?** It is
   currently a constant that must be kept above `research.dataset.horizon_seconds`
   by hand; a test asserts the relationship but nothing enforces it at runtime.
