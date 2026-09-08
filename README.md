# HFT Hivemind — prediction-market research sandbox

> **SIMULATION ONLY.** This project never places, amends or cancels an order. It has no
> order-placement code path and handles no credentials. It replays **fabricated demo data**
> or **read-only historical data** through a local agent stack. Nothing it produces is
> financial advice, and no claim of profitability is made or implied.

## What this is

A native desktop application and a small, modular Python foundation for researching
prediction-market trading ideas offline. It gives you a typed domain model, deterministic
agents, an event-driven backtester with realistic frictions, a read-only Kalshi historical
adapter, and a Qt UI — all of which run with no account and no API key.

It is deliberately a *skeleton with working plumbing* rather than a strategy. The included
toy strategy exists to exercise the proposal → risk → execution → settlement path end to
end, so the parts you actually care about (forecasting, calibration, more venues) have
somewhere to plug in.

On top of that foundation sits a **self-learning control layer** in `src/`: a LangGraph
supervisor state machine routing a CrewAI research desk and a FinRL/Stable-Baselines3
reinforcement-learning policy. See [The self-learning layer](#the-self-learning-layer) and
`DECISIONS.md`, which records what that pivot superseded. It changes what the project
*learns*; it does not change what it *can do* — there is still no order-placement code
path anywhere, including in `src/`.

## Architecture

```text
       config/settings.yaml            .env  (optional, no secrets — there are none to set)
               │                          │
               └────────────┬─────────────┘
                            ▼
                      core/config.py
                            │
     ┌──────────────────────┴───────────────────────┐
     ▼                                              ▼
core/synthetic.py                          adapters/kalshi.py
 seeded DEMO DATA                     read-only Kalshi Trade API v2
     │                                    (HTTP GET only, no auth)
     │                                              │
     │                                     adapters/base.py
     │                                  RawStore → data/raw/kalshi/
     │                                  (response bodies stored byte-for-byte)
     │                                              │
     └──────────────┬───────────────────────────────┘
                    ▼
          core/dataset.py  MarketDataset          core/models.py
      markets · quotes · trades · provenance    typed Pydantic models
                    │                            (shared vocabulary)
                    │                                   │
        core/store.py (DuckDB cache) ◀──────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        backtest/engine.py  (event loop)                     │
│                                                                             │
│   chronological stream:  QUOTE → TRADE → RESOLUTION → CLOCK (decision tick)  │
│                                                                             │
│   for each decision tick, per market:                                       │
│                                                                             │
│     MarketState  (only data with ts <= as_of; outcome masked out)           │
│          │                                                                  │
│          ├─▶ agents/market.py     MarketAgent      ─┐                       │
│          ├─▶ agents/tape.py       TapeAgent        ─┤                       │
│          ├─▶ agents/imbalance.py  BookImbalance    ─┼─▶ combine_forecasts() │
│          ├─▶ agents/baseline.py   BaseRateAgent    ─┘   (agents/base.py)    │
│          │                        (placeholder, excluded)     │             │
│          │                                                    ▼             │
│          ├─▶ agents/liquidity.py  LiquidityAgent          Consensus         │
│          │   spread · depth · staleness · warnings             │            │
│          │                                                     ▼            │
│          │                                        agents/strategy.py        │
│          │                                        build_proposal()          │
│          │                                                     │            │
│          │                                              TradeProposal       │
│          │                                                     ▼            │
│          └─────────────────────────────────────▶  agents/risk.py  RiskAgent │
│                                                   APPROVE / REDUCE / REJECT │
│                                                   + machine-readable reasons│
│                                                                │            │
│                                          (rejected ─▶ audit log, stop)      │
│                                                                ▼            │
│                                                     backtest/fills.py       │
│                                              fees · slippage · partial fills│
│                                                                ▼            │
│                                                Portfolio: cash, positions,  │
│                                                exposure, settlement at 0/1  │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
        BacktestResult: proposals, decisions, fills, settlements,
        equity snapshots, full audit log
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
backtest/metrics.py   calibration/       app/presenter.py  (pure, no Qt)
P&L · drawdown        splits.py                  │
turnover · win rate   methods.py         app/desktop.py  (PySide6 window)
Brier · log loss      evaluate.py        qt_models · qt_charts · workers
ECE · calibration     ─────────────
                      chronological train/validation/test
                      uncalibrated vs isotonic vs logistic
```

The rule the whole design serves: **information flows forward only.** Agents receive a
`MarketState` assembled exclusively from records at or before `as_of`, and the `Market` they
see has `outcome` and `resolution_time` stripped out. Realised outcomes are used in exactly
two places — settling positions and scoring forecasts after the fact.

## Setup

Python 3.11 or newer. From the project root:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(on macOS/Linux: `source .venv/bin/activate`)

```bash
python -m pip install -r requirements.txt
```

The self-learning layer pulls a large stack (torch, Stable-Baselines3, FinRL, CrewAI,
LangGraph). To check it wired up correctly:

```bash
python verify_architecture.py
```

Optionally copy `.env.example` to `.env` to override a few non-secret settings. **There are
no credentials to configure** — every endpoint this project touches is public and read-only.

## Run the app

```bash
python -m app.desktop
```

A native Qt window opens. No browser, no web server, no localhost port. It loads the demo
data on startup, so it works immediately and offline.

## Run the tests

```bash
python -m pytest
```

Lint:

```bash
python -m ruff check .
```

## Using the app

| Tab | What it does |
| --- | --- |
| **Market explorer** | Every market as of the selected instant, plus a bid/ask/mid chart with tape prints. Gaps are real gaps — stale periods and one-sided books are never interpolated. |
| **Agents & risk** | Runs the whole agent chain at one instant: each agent's forecast (with the placeholder clearly flagged), the blended probability and disagreement, liquidity warnings, the proposed trade with expected value after costs, and the risk decision with machine-readable reasons. |
| **Backtest** | Runs the event-driven simulation, then shows headline metrics, the equity curve, drawdown, the full metrics table, and calibration by probability bucket. |
| **Calibration** | Splits the backtest's forecasts chronologically, fits isotonic and logistic calibrators on the training fold only, and compares them against the uncalibrated baseline with Brier, log loss, ECE, a reliability diagram and calibration tables. |
| **Paper mode** | Runs the live agent loop against read-only market data (or a replayed dataset) and records **hypothetical** orders and fills to a local DuckDB file. Nothing is ever submitted. |
| **Research** | Builds causal feature datasets, trains and calibrates simple forecasters under a chronological walk-forward protocol, scores them against frozen baselines, backtests the out-of-sample forecasts, measures drift, and records every run locally. Promotion requires explicit human approval and grants a label, not a capability. |
| **Audit log** | Every proposal, risk decision, fill, rejection and settlement, filterable by event type. |
| **Data source** | Switch between the synthetic demo data and read-only Kalshi history, and inspect dataset provenance. |

The **as-of slider** at the top governs every view: agents only ever see data at or before
that instant. The left dock adjusts strategy, risk and execution settings live.

## The Kalshi adapter (read-only)

`adapters/kalshi.py` pulls historical data from the public Kalshi Trade API v2. Every
endpoint it uses is documented as requiring **no authentication**:

| Endpoint | Used for |
| --- | --- |
| `GET /markets/{ticker}` | Market definition, settlement result, resolution rules |
| `GET /markets` | Listing markets by status / event / close window |
| `GET /historical/markets/{ticker}/candlesticks` | Historical bid/ask series (archived markets; no series ticker needed) |
| `GET /series/{series}/markets/{ticker}/candlesticks` | The same for markets still in the live tier |
| `GET /markets/trades`, `GET /historical/trades` | The public trade tape |
| `GET /historical/cutoff` | The boundary between the live and historical tiers (~3 months) |

**Read-only by construction.** The transport issues HTTP GET and takes no request body;
there is no POST/PUT/DELETE code path anywhere in `adapters/`, no signing, and no credential
header is ever set. Tests assert all of this.

**Raw responses are stored unchanged.** Before anything is parsed, each response body is
written byte-for-byte to `data/raw/kalshi/<endpoint>/<timestamp>.json`, with the URL,
parameters, status and fetch time in a `.meta.json` sidecar so the payload file stays a
faithful copy of what the venue returned.

**Normalization.** Kalshi quotes fixed-point dollar strings for a contract that settles at
$1.00, so a price is already a probability in `[0, 1]`. Timestamps are preserved exactly and
converted to timezone-aware UTC. `result` maps to `YES`/`NO`; `settlement_ts` becomes
`resolution_time`; and `rules_primary`/`rules_secondary` are carried onto the `Market` so the
basis on which a market resolved travels with the data.

Two judgement calls, both configurable in `config/settings.yaml` and both worth knowing:

- **`treat_extremes_as_missing` (default true).** Kalshi order prices live in $0.01–$0.99, so
  a `yes_bid` of `0.00` or a `yes_ask` of `1.00` means *no resting order on that side*. These
  map to a missing side rather than to a real price.
- **`depth_from_volume` (default true).** Candlestick history carries **no book depth at all**.
  With this on, each candle's traded volume stands in for displayed size so fills can be
  simulated; every affected quote is stamped `KALSHI_CANDLES+VOLUME_DEPTH_PROXY` and the
  dataset carries a note saying so. It is an approximation, not real depth. Turn it off and
  the fill simulator will correctly refuse to fill anything, logging `NO_DISPLAYED_SIZE`.

Crossed candles (bid above ask) are dropped rather than coerced, so no fabricated book is
ever produced. Requests are throttled to 5/second by default — well inside Kalshi's
documented ~20 reads/second Basic tier — and HTTP 429 is retried with exponential backoff.

## Probability calibration

A forecast is *calibrated* when things it calls 70% likely happen about 70% of the
time. `calibration/` measures that and tries to fix it, comparing three methods:

| Method | What it does | When it helps |
| --- | --- | --- |
| `uncalibrated` | The raw probabilities, unchanged. The baseline everything else must beat. | - |
| `isotonic` | Exact monotone least-squares fit (pool-adjacent-violators). Non-parametric, so it can correct any monotone distortion. | Lots of resolved markets; it overfits thin folds badly. |
| `logistic` | Platt scaling: `sigmoid(a·logit(p) + b)`, fitted by Newton's method with Platt's target smoothing. | Little data; only corrects a smooth stretch. |

Both are implemented directly (about 120 lines) rather than by adding scikit-learn and
its dependency tree. They are checked against known-good answers: on already-calibrated
input the logistic fit recovers slope ≈ 1.0 and intercept ≈ 0.0, and on a forecaster made
exactly twice as confident as it should be, both methods map a raw 0.90 back to ≈ 0.70.

Metrics reported per fold: **Brier score**, **log loss**, **expected calibration error**,
and a **calibration table** (per-bucket count, mean predicted, observed frequency, gap)
that also drives the **reliability diagram**.

### How the split avoids leaking the future

Calibration is unusually easy to cheat at, because fitting needs outcomes and outcomes
only exist after a market resolves. `calibration/splits.py` blocks three distinct leaks:

1. **Random splitting** mixes late data into training. The split is always chronological.
2. **The same market on both sides.** A market is forecast many times, so its observations
   would otherwise land in several folds — the calibrator would learn that market's
   realised outcome from the training rows and then be scored on predicting it. Folds are
   therefore **grouped by market**: every observation for one market goes to exactly one fold.
3. **Using an outcome before it was known.** Even with grouped folds, a calibrator fitted
   on markets that resolve in June cannot be applied to a forecast made in March. Each fold
   records when its outcomes became knowable, and later folds **drop** observations made
   before that instant. Dropped counts are reported, never silently discarded.

Only the training fold is ever fitted on; the validation fold selects the method; the test
fold is scored once and never consulted to pick a winner. Train scores are shown but
labelled in-sample, so the gap to test is visible.

### The effective sample size is the market count

This is the most important thing on the tab. Calibration is learned **per market**, not per
observation: a market forecast 50 times still resolves once, so those 50 rows share a single
Bernoulli outcome. A fold with 4,000 observations across 8 markets carries about as much
information as 8 coin flips.

The app therefore reports `effective_n_markets` per fold and refuses to present results as
trustworthy when a fold has fewer than 20 resolved markets, when a fold's markets all
resolved the same way, or when the leakage guard dropped a large share of the data.

**The bundled demo dataset trips all of these**, and says so in a banner. With 8 rolling
markets, the training fold has 4 — the calibrators fit noise, and the numbers are
meaningless. This is not a bug to tune away; it is what calibration on 8 markets looks
like. Meaningful calibration needs hundreds of resolved markets, which is what the Kalshi
adapter is for.

## The book-imbalance agent

`BookImbalanceAgent` forecasts from the **microprice** — the price weighted by the size
resting on the *opposite* side of the book:

```
microprice = (bid × ask_size + ask × bid_size) / (bid_size + ask_size)
```

A heavy bid pushes the estimate toward the ask (buyers are queuing), a heavy ask pushes it
toward the bid. It reads **only the current quote**, which is what makes its provenance
trivially auditable: there is exactly one input timestamp to check.

It was added without touching any existing agent. `MarketAgent`, `TapeAgent` and
`BaseRateAgent` are unmodified, and the ensemble weights keep their original
`MarketAgent`/`TapeAgent` ratio — `combine_forecasts` normalises by the weight total, so
adding a fourth agent does not re-tune the first two.

### Its reporting contract

Three fields were added to `Forecast`, all optional and all defaulted, so agents that
predate them serialise and render exactly as before:

| Field | Meaning |
| --- | --- |
| `inputs` | Every piece of data used, with the timestamp it carried and its source. |
| `assumptions` | The modelling assumptions, stated on the forecast rather than buried in a docstring. They travel even when the forecast fails. |
| `status` | `OK`, `DEGRADED`, `NO_DATA`, or `REJECTED_FUTURE_INPUT`. |

Declared inputs make look-ahead checkable *after the fact* rather than only by reading
code: `forecast.uses_future_input` is true if any input is stamped after the forecast
itself. The agent also rejects a future-stamped quote itself, independently of the
engine's own guarantees, and returns `REJECTED_FUTURE_INPUT`.

### How it fails

Nothing here raises on missing data. Every degenerate book produces a structured
non-forecast — `probability=None`, `confidence=0.0`, a status and a plain-language reason:

| Situation | Status | Behaviour |
| --- | --- | --- |
| No quote observed | `NO_DATA` | No probability; no inputs declared. |
| One-sided book | `NO_DATA` | Imbalance is undefined with one side missing. |
| Both sides quoted, no size | `NO_DATA` | The weighting denominator would be zero. |
| Quote older than `max_data_age_seconds` | `DEGRADED` | Still forecasts, confidence scaled down linearly toward the cut-off. |
| Quote older than `hard_cutoff_seconds` | `NO_DATA` | Refuses outright. |
| Quote stamped after the decision instant | `REJECTED_FUTURE_INPUT` | Refuses; would be a look-ahead bug. |

Confidence is `base × depth_factor × freshness`, where `depth_factor` scales with the
*thinner* side of the book against `reference_depth`. A one-contract quote is not as
informative as a thousand-contract one, and the confidence says so.

On the shipped demo data all three paths occur naturally: of 384 forecasts, 274 are `OK`,
87 `DEGRADED` and 23 `NO_DATA`, with zero look-ahead violations.

### Does the imbalance agent help?

**On this demo data, no — and the weight is set low to reflect that.** Sweeping its
ensemble weight while holding everything else fixed:

| Weight | Proposals | Trades | Brier |
| --- | --- | --- | --- |
| 0.00 | 16 | 13 | 0.2247 |
| 0.05 | 12 | 11 | 0.2247 |
| **0.10** | **8** | **7** | **0.2246** |
| 0.15 | 7 | 6 | 0.2245 |
| 0.30 | 3 | 2 | 0.2244 |
| 0.60 | 1 | 0 | 0.2242 |

Brier score is flat to four decimal places while proposals fall monotonically. The reason
is structural: the microprice always lies **inside** the spread, so weighting it pulls the
blend back toward the midpoint and erases exactly the edges the strategy needs to cross
that spread. It suppresses trading without buying any forecast skill.

The default weight of `0.10` is a compromise that keeps the agent genuinely in the blend
and visible in the dashboard while leaving the demo active. It is **not** an
evidence-based weight, because on this data there is no evidence to base it on. A real
version of this signal would be fitted and validated on out-of-sample data — see the
calibration pipeline, which is the machinery for exactly that.

## The research pipeline

Four buttons, in order, on the **Research** tab. Each records what it did to a local
registry, and each can be inspected before moving on.

| Step | What it does |
| --- | --- |
| **1. Prepare data** | Builds causal features and settlement labels, then reports whether there is enough resolved data to train at all. |
| **2. Train & evaluate** | Runs the walk-forward protocol, calibrates on validation folds, and scores the candidate against frozen baselines. |
| **3. Backtest forecasts** | Replays the out-of-sample forecasts through the existing event-driven simulator, with the threshold policy in front and the risk agent behind. |
| **4. Check drift** | Compares the recent window against the training reference. Reports only; triggers nothing. |

### The research question

For a market and a decision time `t`, estimate `P(YES settles | information at t)`, then
ask whether that estimate beats the *executable* price by more than it costs to act. High
training accuracy is not evidence of anything, so the pipeline measures calibration,
out-of-sample performance and simulated trading separately.

### The number that governs everything: effective n

A market forecast 400 times still resolves **once**. The statistical unit is the resolved
market, not the feature row. Every metric surface reports `effective_n` (resolved markets)
next to the row count, and the promotion gate is expressed in markets. A result across
4,000 rows and 30 markets is 30 coin flips, and the app says so rather than letting the
row count flatter it.

### Settlement labels

Labels come from the venue's settlement fields and **never** from price. A market trading
at $0.98 did not "probably settle YES".

| Venue status | Result | Label |
| --- | --- | --- |
| `settled`, `finalized` | `yes`/`no` | YES / NO |
| `determined` | `yes`/`no` | YES / NO, once a settlement timestamp exists |
| `disputed`, `amended` | any | **excluded** - the outcome can still change |
| any | `scalar` | **excluded** - not a binary market |
| any | empty | unresolved: usable for features, never for labels |

A resolved market with no settlement timestamp is excluded too: without one we cannot prove
a forecast predates its outcome. Unresolved markets still produce feature rows for drift
monitoring, but carry `label=None` and are dropped from anything that trains or scores.

### Causal features

Every feature is built from data stamped at or before `t`, enforced in one place
(`_visible` in `research/features.py`) rather than trusted to each feature. Each declares
its sources, its window, and what it does when inputs are missing; the full dictionary is
shown in the app. Scaling is fitted on the training fold only.

Decision timestamps run on a fixed grid from a market's first quote until
`close_time - horizon` (default one hour). The horizon matters: a forecast made a minute
before settlement is nearly free and would flatter every metric.

### Walk-forward, purge and embargo

Prediction markets overlap in time, so a plain chronological cut leaks two ways:

- **Split markets.** One market resolves once. If its rows span train and test, the model
  has seen the answer. Markets are assigned whole to a single fold.
- **Overlapping horizons.** A training market that resolves *after* the test window opens
  was still unresolved when those forecasts were made. Such markets are **purged**.

An **embargo** then skips a configurable gap after the last training resolution. The
invariant the tests assert: *every training market resolves at least `embargo_seconds`
before the earliest test decision timestamp.*

This is expensive, and visibly so. Where markets overlap heavily the purge removes
nearly everything and folds come back empty. That is the guard working, and it is what
happened on real Kalshi data: 2,000 settled markets produced 127 resolved ones and
**zero** usable folds, because every training market resolved after the first test
decision timestamp (see `DECISIONS.md`, 2026-09-08). The synthetic research corpus
therefore uses short-lived markets spread over a long span (~7 concurrent), which is what
sequentially-resolving market data looks like.

### Models

Simple and interpretable only: a **market-price baseline** (the frozen bar), a
**persistence baseline**, and **L2-regularised logistic regression** fitted by Newton/IRLS.
No scikit-learn dependency - the fitted coefficients are plain numbers the registry stores
and displays.

**No tree model is included**, though the task permits one conditionally. The condition -
"enough data and a good justification" - is not met at these volumes. A tree would place
splits on noise from a handful of independent outcomes. `research/models.py` records the
bar a tree would have to clear first, and `build_forecaster` refuses tree, neural-network
and RL model names outright.

### Calibration

Uncalibrated, logistic and isotonic are compared on identical out-of-sample rows.
Calibrators are fitted on the **validation fold only**. Below 30 independent outcomes
isotonic is skipped with a visible warning and logistic is used instead - isotonic is
non-parametric and will fit a perfect-looking step function through a handful of points.

On the shipped demo this guard fires on every fold, and calibration measurably *worsens*
Brier score (0.2507 uncalibrated vs 0.2640 calibrated). That is the correct thing to
observe from a calibrator fitted on 15-27 markets, and the app shows it rather than hiding
it.

### The threshold policy

Trade only when the calibrated forecast beats the executable price by more than the whole
cost stack: half the spread, fees, modelled slippage, and an explicit safety margin for
what is not modelled (queue position, adverse selection, and the fact that the forecast is
an estimate). Otherwise stay flat.

**Every declined decision is recorded with a reason** - `INSUFFICIENT_EDGE`,
`NEGATIVE_EXPECTED_VALUE`, `MISSING_FORECAST`, `INVALID_DATA`, `STALE_DATA`, `RISK_CAP`,
`EXECUTION_CONSTRAINT`, `LOW_CONFIDENCE`, `MARKET_CLOSED`. A policy that trades twice out of
four thousand chances is not the same as one that found four thousand and rejected them on
cost, and the skip table is what tells them apart.

### Drift

Transparent measures only: missingness rates, mean and dispersion shifts, quantile-based
Population Stability Index, and rolling Brier/log loss once outcomes are known.

**Drift never retrains or promotes anything.** It raises a flag for a human. A drift measure
wired to an automatic trigger is a mechanism for fitting a model to recent noise.

> **Superseded in part (2026-09-07).** This still holds for `research/`: drift reports and
> triggers nothing. It no longer describes the whole project — `src/rl_policy.py` updates
> policy metrics on every pass, which is a learning loop by design. What survives is the
> *gate* rather than the ban: `RLPolicyPipeline.evaluate_gate` refuses fabricated data,
> fewer than 50 resolved markets, and any policy that fails to beat a frozen baseline.
> Nothing auto-promotes today. See `DECISIONS.md`.

### Promotion: what approval is and is not

Approval marks a stored run "approved for future paper-trading evaluation". It **deploys
nothing, starts nothing, and places no order**. There is no code path from an approved run
to an order anywhere in this project.

Four checks must all pass before the approve button is even enabled:

| Check | Requirement |
| --- | --- |
| Real market data | Fabricated demo data can never justify approval |
| Independent outcomes | At least 50 resolved markets out-of-sample |
| Walk-forward folds | At least 3 usable folds |
| Beats frozen baseline | Must beat the market price out-of-sample |

Approval then requires a named approver and a written rationale, both recorded. Nothing
promotes automatically, on drift or on any other trigger, and every promotion row is
written with `is_automatic = false` by the only code path that exists.

**On the shipped demo the gate fails**, on two counts: the corpus is synthetic, and the
model does not beat the market baseline. The approve button stays disabled. That is the
intended first-run experience.

### Real training is blocked until real settled data is loaded

The synthetic corpus exists to demonstrate that the machinery is correct. It cannot say
anything about whether a strategy works, and the promotion gate refuses it by design. To do
real research, load settled Kalshi markets through the read-only adapter on the **Data
source** tab, then point the pipeline at that corpus.

## The self-learning layer

`src/` adds a control layer over everything above. It changes how decisions are *routed*
and *learned*, not what the system is permitted to do.

### The supervisor

```text
          +---------------------------------------------------+
          v                                                   |
START -> supervisor -> ingest         (causal MarketStates) -->+
              |      -> research_desk (CrewAI personas)     -->+
              |      -> rl_policy     (FinRL / SB3 PPO)     -->+
              |      -> risk          (existing RiskAgent)  -->+
              |      -> execute       (existing simulate_fill)>+
              +-> END
```

Every worker returns to the supervisor, which owns all routing, so the order of operations
lives in one function instead of being smeared across edges. `risk` and `execute` call the
unchanged `RiskAgent` and `simulate_fill`, so the fee model, spread crossing, depth cap,
staleness guard and every risk limit apply exactly as they do in the backtester.

### The research desk

Two personas, given *different information* rather than different prompts over the same
information:

| Persona | Reads | Answers |
| --- | --- | --- |
| `MacroAnalyst` | Implied probability in the book, drift of recent prints | What does this market think, and which way has it moved? |
| `MicrostructureQuant` | Microprice, depth imbalance, spread, quote staleness | Is the top of book saying anything, and is it tradable right now? |

Both paths return the same `AlphaVector`. **Deterministic by default:** the personas are
backed by the existing `MarketAgent`/`TapeAgent` and `BookImbalanceAgent`/`LiquidityAgent`,
so the desk runs offline, reproducibly, with no key. The LLM path is opt-in and its output
is *merged over* the deterministic vector — an unparseable or out-of-range answer degrades
to the deterministic view instead of poisoning the policy.

### The RL policy

A gymnasium environment where one episode is one market walked forward over its alpha
vectors. Observation is the 9-feature alpha vector plus position, cash and progress;
action is `FLAT` / `LONG_YES` / `SHORT_YES`. The action space is deliberately tiny — a
continuous size head gives the policy more ways to overfit a handful of outcomes.

**The reward is not a mid-price delta.** Every position change runs through
`backtest.fills.simulate_fill`, so crossing the spread, fees, modelled slippage, the depth
cap and the staleness guard are all charged. A policy that looks profitable on mid moves
but cannot pay the spread scores badly here, which is the point.

### What governs the numbers

A market forecast 400 times still resolves **once**. Training for 200,000 timesteps across
200 markets is 200 independent outcomes, not 200,000. `PolicyMetrics` therefore leads with
`effective_n` (resolved markets) and `evaluate_gate` refuses in markets, never in steps:

| Check | Requirement |
| --- | --- |
| Real market data | Fabricated demo data can never justify promotion |
| Independent outcomes | At least 50 resolved markets |
| Beats frozen baseline | Must beat the baseline out-of-sample |

**On the shipped demo the gate fails**, on every count. That is the intended first-run
experience.

### It still cannot place an order

Adding a learned policy did not add a way to act on it:

1. No order-placement method exists in `src/` — the AST scan in
   `tests/test_paper_trading.py` covers `src/` and passes.
2. The RL action space terminates in `simulate_fill` and an in-memory portfolio.
3. No credentials are used or accepted; the desk defaults to the offline path.
4. FinRL's package root imports `alpaca_trade_api`, a live brokerage client. `compat/`
   replaces it with a stub that raises `RefusedExternalClient` on any use, so there is no
   working broker client in the environment at all. See `DECISIONS.md`.

## Paper trading

Paper mode runs the same loop the backtester runs — the same forecast agents, the same
`RiskAgent`, the same `simulate_fill`, the same `Portfolio` — but against a live read-only
feed instead of a stored dataset. Every order it produces is hypothetical: it is simulated
against the observed book and written to `data/paper/paper_trading.duckdb`.

```bash
python -m app.desktop
```

Then open **Paper mode**. Choose *Replay loaded dataset* to exercise the loop entirely
offline, or *Kalshi live* to poll real public market data. Start the session and it polls on
your chosen interval; every poll is recorded.

### It cannot submit an order

The guarantee rests on four independent mechanisms, so removing any one still leaves it
intact:

1. **No such method exists.** `KalshiClient` has no `place_order`, `post`, or any write
   method, and its transport is constructed without a request body, which forces GET.
2. **A read-only guard on the wire.** `paper.guard.ReadOnlyGuard` wraps the transport and
   refuses anything that is not a GET to an **allowlisted** market-data path. It is an
   allowlist, not a blocklist, so an endpoint nobody thought to name is refused by default;
   the blocklist is kept as well so an attempt to reach `/portfolio/orders` fails with an
   explicit, greppable `OrderPlacementBlocked` rather than a generic refusal.
3. **The session holds no write handle.** `PaperSession`'s only outbound capability is the
   quote source it is given. Approved orders go to `simulate_fill` — a pure function over a
   quote — and then to a local store.
4. **It refuses to start if live trading were ever enabled.** `live_trading_enabled` is
   force-set to `False` on load, and the session raises if it is not.

`tests/test_paper_trading.py` attacks the claim from four directions: it **parses** the
`paper` and `adapters` packages and fails on any call named like order placement (with a
confidence check proving the detector is not vacuous); it runs a session against a **strict
spy** that raises on any attribute outside a read allowlist; it drives the **guard** with
every trading endpoint Kalshi exposes; and it runs a **full session** against a fake network
and inspects every request and header that was actually sent.

### What is persisted

Four append-only tables, every row stamped `mode = 'PAPER'` and `is_hypothetical = true` so
a record cannot be mistaken for a real execution by whatever reads the file later:

| Table | Contents |
| --- | --- |
| `paper_orders` | Every hypothetical order with the risk decision that sized it, including rejected ones. |
| `paper_fills` | Every simulated fill, including zero-size non-fills and why they did not fill. |
| `paper_equity` | A mark-to-market snapshot of the hypothetical portfolio each poll. |
| `paper_events` | The session audit trail: start, stop, liquidity warnings, risk rejections, queued orders, cancellations. |

Execution matches the backtester exactly: an approved order is queued and filled on a
**later** poll against the book observed then, never against the quote that triggered the
decision.

## Execution realism

The engine was audited for look-ahead bias and unrealistic fills. Three defects were found
and fixed; each is now pinned by tests in `tests/test_execution_realism.py`.

**1. Orders executed with zero latency.** A fill was simulated against the very quote the
agents had just analysed, at the same timestamp — decide and fill in the same instant, at a
price that was on screen when the decision was made. Orders are now queued and executed
after `backtest.execution_delay_seconds` (default 60s) against the book in force **strictly
before** the execution instant. Draining the queue before each event is applied matters: an
order arriving at the same instant a new quote is published must not receive the new price.
An order still in flight when its market closes is cancelled, not filled.

The configured delay is a *minimum*: the simulator can only act when it next wakes, so on
5-minute demo quotes a 60s delay usually resolves to the same book. On 1-minute Kalshi
candles it does not.

**2. The limit price was computed but never enforced.** Every proposal carried a
`limit_price` that the fill simulator never received, so a marketable limit order could fill
at any price the book happened to offer. Combined with the delay above, that would have
filled orders into gaps that moved against them. `simulate_fill` now takes the limit and
returns `LIMIT_NOT_MARKETABLE` rather than filling worse than it.

**3. Execution had no staleness guard of its own.** Only the `RiskAgent` checked quote age,
so relaxing the risk configuration would have let the simulator trade against an
arbitrarily old book. The fill simulator now refuses any book older than
`fills.max_quote_age_seconds` independently of what risk allowed.

One unrealism is **not** fixed and is left configurable: `max_fraction_of_displayed`
defaults to `1.0`, meaning an order can take 100% of the displayed size at the touch. Real
orders compete for that liquidity. Lower it to model a more contested book.

Confirmed clean by the same audit: agents never receive a record timestamped after
`as_of`; the market handed to them has `outcome` and `resolution_time` stripped; decision
ticks sort after resolutions at the same instant so a resolved market is never traded; and
flipping every realised outcome leaves the sequence of proposals and fills byte-identical.

## Module guide

| Path | What it does |
| --- | --- |
| `core/models.py` | Every typed model: `Market`, `Quote`, `Trade`, `Forecast`, `TradeProposal`, `RiskDecision`, `SimulatedFill`, plus `Position`, `Settlement`, `PortfolioSnapshot`, `AuditEvent` and the `MarketState` handed to agents. Prices are probabilities in `[0, 1]`; a contract pays 1 on YES, 0 on NO. |
| `core/dataset.py` | `MarketDataset` — the container every data source produces, carrying provenance (`label`, `source`, `venue`, `notes`) so the UI can always say where the data came from. |
| `core/config.py` | Loads `config/settings.yaml`, applies optional `.env` overrides, and force-sets `live_trading_enabled = False` regardless of what the file says. |
| `core/synthetic.py` | Seeded generator for the DEMO DATA: price paths, quotes, tape prints and binary outcomes, including deliberate imperfections (variable spreads, one-sided books, stale freezes where the book stops updating but the tape keeps printing). Same seed ⇒ identical dataset. |
| `core/store.py` | DuckDB persistence under `data/processed/`, plus frame conversion and a read-only `query()` helper. |
| `adapters/base.py` | GET-only transport, the byte-faithful `RawStore`, throttling and 429 retry. |
| `adapters/kalshi.py` | The Kalshi client, field parsers and normalizers described above. |
| `agents/base.py` | The `Agent` interface (`analyze(market_state) -> structured result`) and `combine_forecasts()`, which blends agent views into a `Consensus` with a disagreement measure. |
| `agents/market.py` | `MarketAgent`: implied probability from the book — the midpoint, or a nudged one-sided price when half the book is missing. |
| `agents/tape.py` | `TapeAgent`: a size-weighted VWAP of recent prints blended with the midpoint. A **toy mean-reversion signal**, included because a mid-only view can never disagree with the price it would have to cross, so nothing would ever trade. |
| `agents/imbalance.py` | `BookImbalanceAgent`: the size-weighted microprice, with **declared input timestamps, stated assumptions and a machine-readable failure status** on every forecast. See below. |
| `agents/baseline.py` | `BaseRateAgent`: returns a constant from config, flagged `is_placeholder=True`, excluded from the blend by default. A labelled seat-holder for a real base-rate model, not a forecast. |
| `agents/liquidity.py` | `LiquidityAgent`: spread, depth, data age, staleness and warnings. Advisory only. |
| `agents/risk.py` | `RiskAgent`: the final veto. Deterministic, configurable limits returning APPROVE / REDUCE / REJECT plus `RiskReasonCode`s. No execution surface — it returns a decision object and nothing else. |
| `agents/strategy.py` | Turns a consensus probability into a `TradeProposal` when the edge over the crossing price survives modelled fees and slippage. |
| `backtest/fills.py` | The cost model: `compute_fee` (`kalshi` or `notional`), `estimate_slippage` (constant + linear participation impact) and `simulate_fill` (crosses the spread, caps at displayed depth, produces partial or zero fills with a reason). Shared by the strategy and the engine so quoted EV matches realised cost. |
| `backtest/engine.py` | The event loop: builds the chronological stream, maintains books/tape/marks, fires decision ticks on a fixed clock, runs the agent chain, applies fills, settles at resolution, and logs everything. Also holds the `Portfolio`. |
| `backtest/metrics.py` | P&L, return, max drawdown, trade count, win rate, turnover, Brier score, log loss, expected calibration error and a calibration table by probability bucket. |
| `calibration/splits.py` | `Observation`, `Fold` and `chronological_split` - the leakage rules described above, plus `effective_n` and the trust warnings. |
| `calibration/methods.py` | `IdentityCalibrator`, `IsotonicCalibrator` (PAVA) and `LogisticCalibrator` (Platt), with no third-party dependency. |
| `calibration/evaluate.py` | Fits on train, scores every fold, selects on validation, and builds the comparison tables and reliability series. |
| `research/labels.py` | Settlement labels. The only place an outcome is decided, and never from price. |
| `research/features.py` | Causal features with declared sources, windows and missing-data behaviour; causality enforced in one place. |
| `research/datasets.py` | Assembles training examples and answers whether there is enough resolved data to train at all. |
| `research/models.py` | The pluggable forecaster interface, two frozen baselines and regularised logistic regression. |
| `research/walkforward.py` | Chronological folds with market grouping, purge and embargo. |
| `research/evaluate.py` | Scoring against frozen baselines, always with `effective_n` attached. |
| `research/policy.py` | The threshold policy and the cost stack; serves stored forecasts to the backtester. |
| `research/drift.py` | Transparent drift measures that report and never act. |
| `research/registry.py` | Local run/artifact store and the promotion gate. |
| `research/pipeline.py` | The four stages behind the dashboard buttons. |
| `backtest/skips.py` | The reason codes for every decision that did not become a trade. |
| `paper/guard.py` | `ReadOnlyGuard`: the wire-level allowlist that makes an order endpoint unreachable. |
| `paper/session.py` | `PaperSession`: the live loop, reusing the backtester's agents, risk, fills and portfolio unchanged. |
| `paper/sources.py` | `KalshiQuoteSource` (guarded, live, read-only) and `ReplayQuoteSource` (offline replay of a loaded dataset). |
| `paper/store.py` | Append-only DuckDB persistence for hypothetical orders, fills, equity and events. |
| `app/presenter.py` | All UI logic as pure functions with no Qt import, so the interesting behaviour is testable without a GUI. |
| `app/desktop.py` | The PySide6 main window. `app/qt_models.py`, `app/qt_charts.py` and `app/workers.py` provide the table model, QtCharts helpers and background threads. |
| `config/settings.yaml` | Every tunable: data generation, venue adapters, agent weights, strategy thresholds, execution costs, risk limits, backtest parameters. |
| `src/state_graph.py` | The LangGraph supervisor state machine. One `supervisor` node owns every routing decision; workers `ingest → research_desk → rl_policy → risk → execute` each return to it. `risk` and `execute` call the unchanged `RiskAgent` and `simulate_fill`. |
| `src/research_desk.py` | The CrewAI desk: `MacroAnalyst` (implied price, print drift) and `MicrostructureQuant` (microprice, depth imbalance, spread, staleness), emitting an `AlphaVector`. Deterministic offline; LLM-backed only when a key exists and `use_llm=True`. |
| `src/rl_policy.py` | `AlphaTradingEnv` (gymnasium) and `RLPolicyPipeline` (Stable-Baselines3 PPO). Rewards run through `simulate_fill`, so the policy pays real modelled costs. Reports `effective_n` in resolved markets. |
| `src/contracts.py` | `AlphaVector`, `PolicyMetrics`, `SupervisorState` — the typed payloads passed between graph nodes. |
| `compat/` | Refusing stubs for `alpaca_trade_api`, `wrds` and `yfinance`, which FinRL imports at package-root but this project forbids. Any use raises `RefusedExternalClient`. |
| `DECISIONS.md` | Append-only decisions log: what changed, why, and what it superseded. |

### Risk limits

All of these live under `risk:` in `config/settings.yaml` and are adjustable live in the app:

| Limit | Effect |
| --- | --- |
| `kill_switch` | Rejects every proposal outright. |
| `max_position_size` | Cap on `\|contracts\|` in a single market. |
| `max_event_exposure` | Cap on worst-case loss per market. |
| `max_total_exposure` | Cap on worst-case loss across the whole book. |
| `max_spread` | Reject when the bid/ask width is too wide. |
| `max_slippage` | Reject when estimated slippage is too high. |
| `max_data_age_seconds` | Reject on stale quotes. |
| `max_drawdown` | Stop trading once portfolio drawdown exceeds this. |

Exposure is worst-case loss: `quantity × price` for a long position (it expires worthless on
NO) and `|quantity| × (1 − price)` for a short (it pays out on YES). Breaching a *sizing*
limit produces REDUCE when a smaller viable size exists and REJECT when it does not; hard
blocks always produce REJECT.

## Known limitations

- **The demo data is fabricated.** Every market, quote, print and outcome comes from a seeded
  RNG. Results on it tell you the machinery works, nothing more.
- **The strategy is a toy.** `TapeAgent` + `agents/strategy.py` implement a naive
  mean-reversion rule chosen to exercise the pipeline. Do not read anything into its P&L.
- **`BaseRateAgent` is a stub.** It returns a config constant and is excluded from the blend.
- **`BookImbalanceAgent` shows no skill on the demo data.** It is included as a
  well-instrumented example of the reporting contract, not as a signal that works. Its
  Brier score is indistinguishable from omitting it, and it monotonically suppresses
  trading. Do not read its inclusion as a claim that book imbalance predicts anything.
- **Top-of-book only.** The microprice sees one level. Real book imbalance work uses
  depth, queue position and order flow, none of which this data carries.
- **The research pipeline has never seen real data.** Everything shipped runs on a
  fabricated corpus. The machinery is tested; the strategy is not, and cannot be until
  settled Kalshi markets are loaded.
- **No tree model, by choice.** Justified in `research/models.py` rather than assumed.
- **Reinforcement learning is present but unproven.** `src/rl_policy.py` trains a PPO
  policy, and on the demo corpus it has seen a handful of independent outcomes. Both
  warnings fire. A policy trained for 200,000 timesteps across 200 markets has seen 200
  coin flips, and the metrics say so. Read `effective_n`, never the step count.
- **The CrewAI desk is deterministic by default.** With no LLM key it runs the existing
  agents behind the two personas. That path is reproducible and offline; the LLM path is
  opt-in, unvalidated, and merged over the deterministic vector so a bad answer degrades
  rather than propagates.
- **FinRL's package root imports a live brokerage client.** `compat/` stubs it out. See
  `DECISIONS.md` for why that is safer than installing the real SDK.
- **Purging is expensive and honest about it.** On heavily overlapping markets it can empty
  the folds entirely. The fix is more sequential data, not a weaker guard.
- **Backtest fills remain simulated.** The research backtest inherits the same fill model as
  everything else: displayed size assumed takeable, modelled slippage, no queue position.
- **Drift thresholds are conventions, not tests.** PSI cut-offs of 0.10/0.25 are rules of
  thumb, not statistical significance.
- **Kalshi candlesticks have no depth.** See `depth_from_volume` above. Any backtest run on
  Kalshi history is using a volume-derived stand-in for displayed size, which is the single
  biggest source of unrealism in those results.
- **Candles are not the book.** One quote per candle period means intra-period movement is
  invisible, and the 1-minute floor sets the finest resolution available.
- **The adapter is not backfill-complete.** It fetches the tickers you name; it has no
  discovery loop, no incremental sync and no local corpus management beyond the raw files.
- **Top-of-book only.** No depth beyond displayed size, no queue position, no resting limit
  orders, no cancels, no maker/taker distinction — every order crosses.
- **Slippage and fees are models, not measurements.** The `kalshi` fee formula approximates
  the shape of the published schedule; it is not the published schedule, and it ignores
  maker/taker distinctions and fee waivers.
- **No borrow, margin or capital cost.** Shorts are collateralised only by the risk agent's
  cash check, and cash earns nothing.
- **Forecast scoring is repeated-observation.** Each decision tick contributes one observation
  per market, so Brier/log-loss/calibration are dominated by markets evaluated most often.
  With a handful of markets these are wiring checks, not evaluation.
- **Latency is quantised.** The execution delay is a minimum; an order executes when the
  simulator next wakes, so on coarse data the realised latency is longer than configured.
- **The demo dataset cannot support calibration.** 8 rolling markets means 4 training
  markets. The Calibration tab runs and reports honestly, but every number on it is noise.
  Calibration needs hundreds of resolved markets.
- **Calibration is fitted once, not rolled forward.** A production setup would refit on an
  expanding window as markets resolve; here it is a single train/validation/test split.
- **ECE is bucket-count sensitive** and rewards a constant forecast at the base rate, which
  is perfectly calibrated and perfectly useless. Read it alongside Brier and log loss.
- **Binary markets only.** Scalar and multi-outcome markets are skipped with a note.
- **Paper fills are still simulated.** Paper mode uses real prices but the same fill model
  as the backtester: it assumes the displayed size is takeable, applies a modelled slippage,
  and has no queue position. A hypothetical fill is not evidence a real order would have
  filled there, or at all.
- **Paper mode does not detect settlement.** It marks positions to the live book and records
  them; it does not poll for resolution and settle them. Run a backtest for settled P&L.
- **Paper polling is coarse.** State advances only on poll boundaries, so anything that
  happens between polls is invisible to the session.
- **Import order matters.** pandas must be imported before PySide6 (Qt installs a `six`
  meta-path importer that breaks pandas' extension modules). `app/__init__.py` handles this.

## Roadmap

1. **More venues.** Polymarket (Gamma/CLOB) and Manifold behind the same `HistoricalDataSource`
   protocol, normalising into the same models so the engine is unchanged. Read-only, always.
2. **Corpus management.** Ticker discovery, incremental sync, and a DuckDB-backed local
   archive of fetched history rather than one-shot pulls.
3. **Deeper calibration.** Confidence intervals on reliability buckets, Brier
   decomposition into reliability/resolution/uncertainty, per-category calibration, rolling
   expanding-window refits, and feeding the fitted calibrator back into the live decision
   path rather than only scoring it offline.
4. **Real forecasting agents.** Base rates from reference classes and time-decay models.
   *Partly delivered:* `src/research_desk.ResearchDeskAgent` is an LLM-backed research desk
   behind the same `analyze()` interface, so the existing harness can grade it. It runs
   deterministically until an LLM key is supplied.
5. **Paper-trading depth.** Settlement detection so hypothetical positions resolve, an
   equity curve and calibration computed over live sessions rather than backtests, and
   reconstruction of an interrupted session's portfolio from its persisted fills.
6. **Better execution modelling.** Depth-aware fills from full order-book snapshots where a
   venue offers them, resting limit orders with queue position, latency, and maker/taker fees.

## Simulation-only statement

This system is for research and simulation. It has no code path that submits, routes, amends
or cancels an order; the venue adapter issues HTTP GET requests to public market-data
endpoints and nothing else; paper mode wraps even those in a read-only guard that refuses
trading endpoints; the `RiskAgent` can only return a decision object; the LangGraph nodes in
`src/` add routing rather than permissions, and the RL policy's action space terminates in
`simulate_fill`; the live brokerage client FinRL imports is replaced by a refusing stub in
`compat/`; and the execution layer writes only to an in-memory portfolio and a local file. Every paper record is stamped
hypothetical. No credentials are used or accepted.
Do not treat any output as a recommendation to trade, do not treat the demo data as real
market data, and do not treat a backtest on fetched history as evidence that anything works.
Nothing here is financial advice.

## API documentation referenced

- [Kalshi — Get Markets](https://docs.kalshi.com/api-reference/market/get-markets)
- [Kalshi — Get Trades](https://docs.kalshi.com/api-reference/market/get-trades)
- [Kalshi — Get Market Candlesticks](https://docs.kalshi.com/api-reference/market/get-market-candlesticks)
- [Kalshi — Get Historical Market Candlesticks](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
- [Kalshi — Get Historical Trades](https://docs.kalshi.com/api-reference/historical/get-historical-trades)
- [Kalshi — Historical Data](https://docs.kalshi.com/getting_started/historical_data)
- [Kalshi — Rate Limits](https://docs.kalshi.com/getting_started/rate_limits)
