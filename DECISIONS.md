# Decisions log

Append-only. Newest first. Each entry records what changed, why, and what it
supersedes, so a later reader can tell a deliberate reversal from drift.

Before this file existed the recorded intent lived in `README.md` prose and
`research/README_DESIGN.md`. An audit on 2026-09-07 found no structured log and
no version control, meaning a change had nowhere to record its rationale and
nothing to diff against. This file closes that gap.

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
