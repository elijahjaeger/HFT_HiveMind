# Mission

**Find a real, validated forecasting edge in prediction markets - and make it as
hard as possible to believe in one that is not there.**

Both halves are the mission. The second is why the first is worth anything: a
research pipeline that cannot tell you "no" cannot tell you "yes" either. Every
guard in this codebase exists because the failure mode being defended against
is not losing money, it is *being convinced you found something*.

## Scope boundaries

These are not aspirations. They are constraints that shape the code.

1. **This repository never executes an order.** It has no order-placement code
   path, no credentials, and no live-trading toggle, and it never will. If an
   edge validates, a **physically separate application** - own repository, own
   credentials, own review - performs execution, consuming approved artifacts
   from this one. The boundary is a repository boundary, not a config flag.
   Recorded 2026-09-08.
2. **Evidence is counted in resolved markets.** Never rows, never timesteps,
   never training steps. A market forecast 400 times resolves once.
3. **The promotion gate is the arbiter and is never loosened to make a run
   pass.** Parameters may change with a recorded justification; they may not
   change *in response to a failing run*.
4. **Real data or it does not count.** Synthetic corpora prove the machinery,
   never the strategy.

## Architecture

Six layers, each depending only on those above it.

| Layer | Responsibility | Modules |
| --- | --- | --- |
| **Venue adapters** | Read-only historical data, one class per venue behind a shared protocol | `adapters/` (Kalshi live-proven; Manifold next; Polymarket after) |
| **Core domain** | Typed models, the causal `MarketState`, provenance-preserving storage | `core/` |
| **Forecasters** | Anything that produces a probability behind `analyze()` | `agents/`, `src/research_desk.py` |
| **Research** | Labels, causal features, walk-forward, calibration, drift, registry, gate | `research/`, `calibration/` |
| **Execution simulation** | Frictions, risk veto, event loop, paper loop. Terminates in an in-memory portfolio | `backtest/`, `paper/` |
| **Product surface** | The Qt desktop application, driven by a shared job layer | `app/` |

**The application is the product.** Long-running work - corpus pulls,
walk-forward, backtests - runs through one job layer that both the UI and the
CLI drive, so a run behaves identically either way and progress and
cancellation work everywhere.

## What success looks like

A forecaster that clears the promotion gate on real data from at least one
venue: 50+ resolved out-of-sample markets, 3+ usable walk-forward folds, beating
the frozen market-price baseline, with calibration that holds. Then months of
paper trading whose realised calibration matches the backtest. Only then does
the question of a separate execution application arise.

---

# Decisions log

Append-only. Newest first. Each entry records what changed, why, and what it
supersedes, so a later reader can tell a deliberate reversal from drift.

Before this file existed the recorded intent lived in `README.md` prose and
`research/README_DESIGN.md`. An audit on 2026-09-07 found no structured log and
no version control, meaning a change had nowhere to record its rationale and
nothing to diff against. This file closes that gap.

---

## 2026-09-08 - Real data at last: Manifold, and the self-improving loop

**Status:** adopted. The central scientific blocker is cleared.

### The blocker, and why Manifold clears it

Every walk-forward training fold purged to zero on Kalshi because its settled
markets live one to six hours and overlap heavily. Manifold's resolved binary
markets have a **median lifetime of about 200 days** and resolve in a long
sequence. Measured on 40 of them: fold 0 came back **usable, with 453
out-of-sample rows** - the first usable fold this protocol has ever produced on
real data.

Manifold is play money, so an edge found there is not tradable. Accepted
deliberately: the blocker being cleared is scientific, not financial.

`adapters/manifold.py` is read-only by construction - GET-only transport, no
credentials, raw bodies stored before parsing, and a test asserting the client
has no write method.

### The order-book problem, handled honestly

Manifold is an AMM: one probability, no bid and no ask. The engine needs two
sides, so the adapter **derives** them - the mid is the AMM probability, the
half-spread is estimated from *observed price impact* (the median
`|probAfter - probBefore|` across real bets), and displayed size is a liquidity
proxy. Every such quote is stamped `MANIFOLD_AMM+IMPACT_SPREAD_PROXY` and the
dataset carries a note, exactly as Kalshi's `depth_from_volume` does.

Observed impact is usually below the 0.005 floor, so in practice the floor
binds rather than the estimate. The floor exists because a market with no
modelled cost looks free to trade, which would flatter every result.

Notably: **zero rows dropped for one-sided books**, against 83% on Kalshi.

### What "self-learning" means, and what it does not

`research/autopilot.py` improves by **accumulating evidence**, not by tuning
toward a number it likes. Each pass acquires more real resolved markets,
rebuilds features and labels, retrains every candidate under walk-forward, and
scores it out-of-sample against a frozen baseline.

Earnings are feedback - but only earnings on markets the candidate was never
fitted on. A loop that searched parameters and kept whatever maximised
in-sample profit would find something every time, and that something would be
noise. This is scope boundary 3 in the Mission, and the loop honours it: model
hyper-parameters are never varied in response to a result, and an AST test
asserts no approval function is ever called.

### The guard automation needs and manual work does not

Run enough candidates against the same test data and one looks good by luck:
twenty draws at a 5% bar produce a false winner about two-thirds of the time.
`TrialLedger` counts every evaluation and reports the bar a result would have to
clear given how many things were tried. It adjusts nothing silently - it makes
the cost of searching visible.

### First real run

Corpus grew 59 -> 119 -> 179 markets (175 resolved) across three batches.
Trial 3 scored **30 out-of-sample markets across 1 usable fold**.

**Brier 0.5452 against a market-price baseline of 0.1217.** The model is far
worse than simply using the market price - and the baseline is genuinely strong,
which is what a well-functioning prediction market looks like. Beating it is the
entire bar, and nothing here is close.

Gate failed on three of four checks. Nothing was promoted or approved.

### A defect this run exposed

The time budget was checked only at the top of each batch, so a 14-minute run
took **3.3 hours**: one candidate evaluation cannot be interrupted once started,
and the cost of rebuilding features grows with the corpus. The budget is now
rechecked before each candidate, and a significant overrun is recorded on the
report rather than passing silently.

The deeper cost remains: every pass rebuilds features for the entire corpus.
Incremental feature building is the fix and is not done.

Verification: 815 passed, 4 skipped, exit 0. ruff clean.

---

## 2026-09-08 - Simple mode, presets, and campaigns instead of single backtests

**Status:** adopted.

The owner is learning finance from the ground up. Thirty-five numeric controls
and nine tabs is not a neutral default for that: a wrong value in this codebase
produces a *plausible-looking result*, not an error, which is the worst kind of
confusing.

### Campaigns: a distribution, not a number

`backtest/campaign.py` runs the same strategy many times over different random
subsets of the same corpus for a fixed time budget, and reports the spread. A
single backtest is a single sample; on the demo corpus the median return moves
by more than its own size depending on which markets are drawn.

The chart is a median line inside 5-95% and 25-75% percentile bands, with a
dashed break-even line, because the question worth asking is not "is the line
up?" but **"does the band clear zero?"** `CampaignResult.verdict()` says so in
one sentence, and says "this has NOT been distinguished from doing nothing"
whenever the band straddles zero - which it does on the demo data.

**A campaign measures robustness to market selection, not out-of-sample
validity.** Every run draws from the same corpus. It cannot substitute for
walk-forward or the promotion gate, and both the module docstring and the UI
caveat say so, because this is exactly the number a beginner would over-read.

### Simple mode by default

Four tabs (Backtest, Research, Corpus, Market explorer), the settings dock
hidden, and four named presets - Cautious, Balanced, Adventurous, Do nothing -
each carrying a plain-language summary, a list of what it changes, and **what to
expect to see**. That last field matters most: "you will see very few trades,
that is the preset working" turns a confusing empty table into a confirmation.

One checkbox reveals all nine tabs, the full dock and the single-run detail.
Nothing is removed, only hidden. In simple mode the preset governs and the
hidden dock is ignored outright, because reading values the user cannot see
would be dishonest.

### Two bugs found while building it

- `ProgressThrottle` (added earlier the same day) suppressed the *first* update
  when the injected clock read below the interval - hidden in production only
  because `time.monotonic()` returns a large number.
- A `QAreaSeries` built from temporary `QLineSeries` is destroyed with them.
  The band chart holds explicit references.

Verification: 747 passed, 4 skipped, exit 0. ruff clean. A 6-second campaign
through the real UI produced 35 runs, a drawn band, and an honest verdict.

---

## 2026-09-08 - Job layer: live progress and cancellation

**Status:** adopted.

"The app is the product" was false while the longest job in the project - the
corpus loop, minutes of throttled venue requests plus a walk-forward pass per
iteration - was reachable only from a CLI, and while a backtest gave no sign of
life until it finished.

### What was built

`core/jobs.py` holds the Qt-free primitives: `CancellationToken` (callable, so
it *is* a `should_stop` predicate), `JobProgress` and `ProgressThrottle`. The
application layer decides where a job runs; this module only defines how it
reports and how it is asked to stop.

`BacktestEngine.run` and `run_backtest` gained keyword-only `on_progress` and
`should_stop`, following `docs/design/live-backtest-progress-and-stop.md`.
`BacktestResult` gained `cancelled`. `research.run_loop.run_loop` gained
`should_stop`, checked between iterations and between fetch chunks, with a new
`StopReason.CANCELLED`.

The Backtest tab has a determinate progress bar, live timestamp/position/audit
labels and a Stop button. A new **Corpus** tab drives the research loop from the
UI, appending each iteration as it lands.

### Two decisions worth recording

**The job layer does not supersede the LangGraph supervisor, and the supervisor
does not become the job layer.** They answer different questions - "what
analysis steps run for one instant" versus "how does a ten-minute task report
and stop". `src/state_graph.py` stays as a callable a job could run. Whether it
earns its place at all is still open.

**Progress is throttled on time, not on an event count.** Event costs are wildly
uneven: one CLOCK tick fans a decision across every market while a QUOTE is
nearly free, so an every-N-events cadence emits in bursts. Measured: 3 updates
over 1,567 events.

### What cancelling means

Cancel-and-discard, not pause/resume. A stopped backtest returns a fully-formed
`BacktestResult` built from fewer events, and the UI labels it partial so a
truncated run is never mistaken for a complete one. A stopped corpus loop keeps
every completed iteration and its registry rows. Nothing live is touched by
either - the only thing interrupted is a loop replaying stored events.

### A bug the tests caught

`ProgressThrottle` initialised its clock to `0.0`, so the *first* update was
suppressed whenever the injected clock read lower than the interval. Hidden in
production only because `time.monotonic()` returns a large number. Now `None`
means "never emitted" and the first call always passes.

Verification: 704 passed, 4 skipped, exit 0. ruff clean. verify_architecture
exit 0.

---

## 2026-09-08 - Mission and architecture fixed; direction set

**Status:** adopted.

Eight decisions taken with the user, establishing the Mission section above.

| Question | Decision |
| --- | --- |
| What is this for? | Find real edge, eventually trade it |
| The orphaned `src/` layer | Integrate the research desk; park the RL policy |
| Role of the Qt app | The app is the product |
| The empty-folds blocker | Add venues rather than retune walk-forward |
| Live-trading boundary | **Never in this repository** - a separate execution application |
| Next venues | Both, **Manifold first**, then Polymarket |
| RL dependencies | Remove from the tree, keep the code behind an optional import |
| First slice | Clean up the audit debt, then build the job layer |

### Why Manifold before Polymarket

Manifold is play money, so an edge found there is not tradable. That is
acceptable because the immediate blocker is *scientific, not financial*: every
walk-forward training fold purges to zero on Kalshi's short-horizon settled
markets, so the protocol has never once produced a usable fold on real data.
Manifold offers a large corpus of resolved markets with varied lifetimes and a
simple API - the cheapest way to prove the machinery works on a corpus that
actually fits. Polymarket then answers the real-money question with genuine
order-book depth, which also addresses the 83% row loss from one-sided books.

### Why the RL dependencies come out

`src/rl_policy.py` is parked, and a control policy cannot improve accuracy until
there is a validated forecast edge *and* enough independent outcomes - neither
of which holds. torch, FinRL, stable-baselines3 and gymnasium cost roughly 2GB
and force `compat/` to keep stub modules named after a live brokerage client at
the repository root, which is a poor thing to have in a tree whose central claim
is that it cannot trade. The code stays, behind an optional import, and the
decision reverses with one `pip install`.

This does not supersede the 2026-09-07 entry admitting RL. RL remains part of
the architecture; it is unfunded, not unwanted.

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
| Mission and architecture | **Fixed 2026-09-08.** See the Mission section at the top. | - |
| Audit debt | **Cleared.** Dead `evaluate_policy` engine removed (policy.py 417 -> 182 lines), `freshness_window` wired in, stale `corpus.py`/README concurrency claims corrected, Streamlit reference gone. | - |
| RL stack | **Parked and uninstalled.** `src/rl_policy.py` imports optionally; `compat/` and the three root broker stubs deleted. 4 tests skip. | Unpark only with a validated forecast edge and enough independent outcomes. |
| Job layer / app-as-product | **Done.** `core/jobs.py`; backtest, campaign and corpus loop all report progress and stop from the UI. | Extend to the Research tab's four stages, which still block without feedback. |
| Learner-facing UI | **Done.** Simple mode by default (4 tabs, presets with expectations), Advanced toggle reveals everything. | Glossary surface for `presets.TERMS`; plain-language read-out on the Research tab. |
| Evidence from backtests | **Campaigns.** Repeated resampled runs with a percentile band and a verdict that refuses to overclaim. | Run campaigns on a real corpus once a venue fills the folds. |
| Manifold adapter | **Done and live.** Read-only, derived book stamped as a proxy. 39 tests. | Deeper paging; more markets to fill folds 2 and 3. |
| Walk-forward on real data | **Unblocked.** First usable fold ever produced: 30 out-of-sample markets, 1 fold. | Reach 3 folds and 50 OOS markets - a volume problem now, not a protocol one. |
| Self-improving loop | **Done.** `research/autopilot.py` acquires, retrains, scores out-of-sample, reports the gate, never approves. Trial ledger charges for repeated search. | Incremental feature building: every pass currently rebuilds the whole corpus. |
| Polymarket adapter | Not started. | Real money, real book depth; addresses the 83% one-sided-book row loss. |
| Research desk integration | `ResearchDeskAgent` exists behind `analyze()` but is not graded by walk-forward or visible in the app. | Wire into the harness so the personas are measured rather than asserted. |
| Read-only Kalshi adapter | Working against the live venue. 50 tests remain fixture-only. | Guard the candlestick call inside `fetch_dataset` so one bad ticker skips itself. |
| Research pipeline on real data | Runs end-to-end; "Real market data" gate check passing. | Blocked on corpus shape, not on code. |
| Walk-forward on Kalshi | Blocked - all training folds purge to empty. | Superseded by "add venues" rather than retuning. |
| Promotion gate | Never passed. Never bypassed. | Unchanged. |
| Paper trading | Offline-capable, no order path. | Settlement detection (README roadmap 5). |
| Execution | **Out of scope for this repository, permanently.** | A separate application, if an edge ever validates. |

# Open questions

1. **Will Manifold's corpus actually fill the folds?** The hypothesis is that
   varied market lifetimes give the sequential structure Kalshi's short-horizon
   settled markets lack. If it does not, the walk-forward parameters become the
   next suspect rather than the data.
2. **Is a candle-derived book good enough?** 83% of Kalshi feature rows were
   dropped for having no two-sided book. Polymarket's CLOB should answer this;
   if the loss persists, the feature set may need to tolerate one-sided quotes.
3. **Should `min_lifetime_hours` be derived rather than configured?** It must
   stay above `research.dataset.horizon_seconds`; a test asserts the
   relationship but nothing enforces it at runtime.
4. **How much of `src/` survives the job layer?** The LangGraph supervisor and a
   job/orchestration layer are two answers to overlapping questions. One of them
   should probably go.
5. **What are the paper-trading criteria for the separate execution app?** The
   boundary is agreed; the bar that would justify crossing it is not yet
   written down.
