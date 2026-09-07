# Research pipeline - design decisions

Recorded before implementation so the reasoning is auditable.

## Safety posture (unchanged from the rest of the project)

This package adds **no** trading capability. It has no order-placement code, no
broker/account calls, and no write path to any venue. It reads market data only
through the existing read-only `adapters/kalshi.py`, whose transport is GET-only
and whose paths are allowlisted by `paper/guard.py`. Every existing safety test
continues to run unchanged, and the static AST scan in
`tests/test_paper_trading.py` is extended to cover `research/` too.

No model is ever deployed, replaced, or promoted automatically. Promotion is a
manual, recorded, human decision (`research/registry.py`), and the only thing a
promotion grants is the label "approved for future paper-trading evaluation" -
it starts nothing.

> **Amended 2026-09-07.** This still describes `research/` exactly. It no longer
> describes the whole project: `src/rl_policy.py` updates policy metrics on each
> pass. The guard that survives is `RLPolicyPipeline.evaluate_gate` - fabricated
> data, fewer than 50 resolved markets, or failing to beat a frozen baseline all
> refuse. The static AST scan in `tests/test_paper_trading.py` was extended to
> cover `src/` as well, and the no-order-path guarantee is unchanged. See
> `DECISIONS.md`.

## Settlement labels

`research/labels.py` is the single source of truth. Labels come **only** from the
venue's settlement fields, never from price.

| Kalshi `status` | `result` | Label | Rationale |
| --- | --- | --- | --- |
| `settled`, `finalized` | `yes`/`no` | YES / NO | Terminal and final. |
| `determined` | `yes`/`no` | YES / NO, but held back until `settlement_ts` | Outcome determined; usable once timestamped. |
| `disputed`, `amended` | any | **excluded** | The outcome can still change. Training on it risks a label that later flips. |
| any | `scalar` | **excluded** | Not a binary market. |
| any | `""`/absent | **unresolved** | Usable for features, never for labels. |
| `closed`, `inactive`, `paused`, `active`, `initialized` | any | **unresolved** | Not settled. |

A market with a resolved result but no `settlement_ts` is **excluded**: without a
knowledge timestamp we cannot prove a forecast predates the outcome.

Unresolved markets may still produce feature rows (for drift monitoring and
inference), but they carry `label=None` and are dropped from any training or
scoring set. That separation is enforced in `datasets.py`, not left to callers.

## Decision timestamps and horizon

For each market, decision timestamps are sampled on a fixed grid from the
market's first observed quote up to `close_time - horizon`. Every example is
`(market, t)` with:

- features built **strictly** from data with `ts <= t`;
- the label from settlement, which becomes knowable at `resolution_ts > t`.

`horizon_seconds` is configurable and defaults to one hour: examples closer to
close than the horizon are dropped, because a forecast made seconds before
settlement is trivially easy and would flatter every metric.

## Purge and embargo

Prediction markets overlap in time, so an ordinary chronological cut leaks: a
market that resolves *during* the test window was open during training, and its
outcome was knowable before some test forecasts were made.

Policy, implemented in `walkforward.py`:

- **Group by market.** Every example for one market lands in exactly one fold.
  A market split across folds would leak its single outcome directly.
- **Purge.** Order markets by `resolution_ts`. A training market whose
  `resolution_ts` falls after the first test decision timestamp is dropped from
  training.
- **Embargo.** After the training fold's latest `resolution_ts`, skip
  `embargo_seconds` before the test fold may begin, so information that leaks
  through market co-movement has time to decay.

The rule the tests assert: **every training market resolves before the earliest
test decision timestamp, plus the embargo.**

## Model scope

Per the task: simple and interpretable only.

- `MarketPriceBaseline` - the frozen baseline. The market's own implied
  probability. Beating it is the entire bar.
- `PersistenceBaseline` - last observed probability, carried forward.
- `LogisticForecaster` - L2-regularised logistic regression, fitted by
  Newton/IRLS. Hand-written; no scikit-learn dependency is added.

**No tree model is included.** The task permits one "only if enough data exists
and its addition is well justified". With the resolved-market counts available
here, it is not justified: a tree would fit noise across a handful of
independent outcomes. `models.py` documents the sample-size bar a tree would
have to clear first.

> **Amended 2026-09-07.** The blanket "no reinforcement learning" that used to
> sit here has moved rather than been deleted. A policy emits actions, not
> calibrated probabilities, so it cannot be scored by Brier or log loss and does
> not belong behind `build_forecaster` - which still refuses `rl`, now as a
> routing error pointing at `src/rl_policy.py`. The forecasters in this package
> are unchanged and still store plain, diffable numbers. The sample-size
> argument survives as a reported warning rather than a prohibition.

## The binding constraint: independent outcomes

A market forecast 400 times still resolves **once**. The statistical unit is the
market, not the row. Every metric surface in this package reports
`effective_n` (resolved markets) alongside row counts, and the promotion gate
refuses on `effective_n`, not on row count. This is the single most important
guard against a demo that looks impressive and means nothing.
