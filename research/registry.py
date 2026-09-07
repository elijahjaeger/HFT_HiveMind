"""Local registry: every run, artifact, metric and promotion decision.

Two jobs:

1. **Reproducibility.** Every run records the data version, feature version,
   label policy, model configuration, fitted parameters and resulting metrics,
   so a number on screen can be traced back to what produced it.
2. **The promotion gate.** Promotion is the only state change in this whole
   pipeline that means anything, so it is deliberately hard: it requires an
   explicit human decision, it records who made it and why, and it refuses
   outright when the evidence does not support it.

What promotion is not
---------------------

Promotion marks a candidate "approved for future paper-trading evaluation". It
does **not** deploy anything, start anything, or place anything. There is no
code path from an approved model to an order, in this module or anywhere else
in the project. Nothing in this pipeline promotes automatically, on drift or on
any other trigger.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from core.config import resolve_path

DEFAULT_PATH = "data/research/registry.duckdb"
ARTIFACT_DIR = "data/research/artifacts"


class PromotionState(StrEnum):
    """Where a candidate stands."""

    CANDIDATE = "CANDIDATE"
    APPROVED_FOR_PAPER = "APPROVED_FOR_PAPER_EVALUATION"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


SCHEMA: dict[str, str] = {
    "research_runs": """
        CREATE TABLE IF NOT EXISTS research_runs (
            run_id             VARCHAR PRIMARY KEY,
            created_at         TIMESTAMPTZ,
            kind               VARCHAR,
            model_name         VARCHAR,
            model_version      VARCHAR,
            data_version       VARCHAR,
            feature_version    VARCHAR,
            label_policy       VARCHAR,
            corpus_label       VARCHAR,
            is_synthetic       BOOLEAN,
            effective_n        BIGINT,
            n_forecasts        BIGINT,
            config_json        VARCHAR,
            metrics_json       VARCHAR,
            notes              VARCHAR
        )
    """,
    "research_artifacts": """
        CREATE TABLE IF NOT EXISTS research_artifacts (
            artifact_id  VARCHAR PRIMARY KEY,
            run_id       VARCHAR,
            created_at   TIMESTAMPTZ,
            kind         VARCHAR,
            path         VARCHAR,
            summary      VARCHAR
        )
    """,
    "research_promotions": """
        CREATE TABLE IF NOT EXISTS research_promotions (
            promotion_id  VARCHAR PRIMARY KEY,
            run_id        VARCHAR,
            decided_at    TIMESTAMPTZ,
            state         VARCHAR,
            approved_by   VARCHAR,
            rationale     VARCHAR,
            gate_json     VARCHAR,
            is_automatic  BOOLEAN
        )
    """,
}


@dataclass(frozen=True)
class PromotionGate:
    """Whether the evidence permits promoting a candidate at all."""

    passed: bool
    checks: tuple[tuple[str, bool, str], ...] = ()

    @property
    def failures(self) -> list[str]:
        """Reasons promotion is not permitted."""
        return [detail for _, ok, detail in self.checks if not ok]

    def as_rows(self) -> list[dict[str, Any]]:
        """Display rows for the approval dialog."""
        return [
            {"check": name, "result": "PASS" if ok else "FAIL", "detail": detail}
            for name, ok, detail in self.checks
        ]


def evaluate_promotion_gate(
    *,
    effective_n: int,
    beats_baseline: bool | None,
    is_synthetic: bool,
    fold_count: int,
    min_markets: int = 50,
    min_folds: int = 3,
) -> PromotionGate:
    """Decide whether a candidate may be offered for approval.

    The gate is about evidence, not preference. A candidate that fails any
    check cannot be approved even by an insistent user, because the number that
    would justify approval does not exist yet.

    Args:
        effective_n: Independent out-of-sample outcomes.
        beats_baseline: Whether it beat the frozen market baseline.
        is_synthetic: Whether the corpus was fabricated.
        fold_count: Number of usable walk-forward folds.
        min_markets: Minimum independent outcomes required.
        min_folds: Minimum usable folds required.

    Returns:
        A :class:`PromotionGate`.
    """
    checks: list[tuple[str, bool, str]] = [
        (
            "Real market data",
            not is_synthetic,
            "Corpus is fabricated demo data; results cannot support approval."
            if is_synthetic
            else "Corpus is real venue data.",
        ),
        (
            "Independent outcomes",
            effective_n >= min_markets,
            f"{effective_n} resolved market(s); at least {min_markets} required."
            if effective_n < min_markets
            else f"{effective_n} resolved markets out-of-sample.",
        ),
        (
            "Walk-forward folds",
            fold_count >= min_folds,
            f"{fold_count} usable fold(s); at least {min_folds} required."
            if fold_count < min_folds
            else f"{fold_count} usable folds.",
        ),
        (
            "Beats frozen baseline",
            bool(beats_baseline),
            "Does not beat the market-price baseline out-of-sample."
            if not beats_baseline
            else "Beats the market-price baseline out-of-sample.",
        ),
    ]
    return PromotionGate(passed=all(ok for _, ok, _ in checks), checks=tuple(checks))


@dataclass
class RunRecord:
    """One recorded run."""

    run_id: str
    created_at: datetime
    kind: str
    model_name: str
    model_version: str
    data_version: str
    feature_version: str
    label_policy: str
    corpus_label: str
    is_synthetic: bool
    effective_n: int
    n_forecasts: int
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class ResearchRegistry:
    """Append-only local store for runs, artifacts and promotion decisions."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = resolve_path(ARTIFACT_DIR)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.path)) as con:
            for statement in SCHEMA.values():
                con.execute(statement)

    # -- runs --------------------------------------------------------------

    def record_run(
        self,
        kind: str,
        model_name: str,
        model_version: str,
        data_version: str,
        feature_version: str,
        label_policy: str,
        corpus_label: str,
        is_synthetic: bool,
        effective_n: int,
        n_forecasts: int,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
    ) -> str:
        """Record one run and return its id."""
        run_id = f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                "INSERT INTO research_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    run_id, datetime.now(UTC), kind, model_name, model_version,
                    data_version, feature_version, label_policy, corpus_label,
                    is_synthetic, int(effective_n), int(n_forecasts),
                    json.dumps(config or {}, default=str),
                    json.dumps(metrics or {}, default=str),
                    notes,
                ],
            )
        return run_id

    def save_artifact(
        self, run_id: str, kind: str, payload: Any, summary: str = ""
    ) -> str:
        """Write a JSON artifact to disk and index it against a run."""
        artifact_id = f"{kind}-{uuid.uuid4().hex[:10]}"
        path = self.artifact_dir / f"{run_id}__{artifact_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                "INSERT INTO research_artifacts VALUES (?,?,?,?,?,?)",
                [artifact_id, run_id, datetime.now(UTC), kind, str(path), summary],
            )
        return artifact_id

    def runs(self, limit: int = 200) -> pd.DataFrame:
        """Recorded runs, newest first."""
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(
                "SELECT * FROM research_runs ORDER BY created_at DESC LIMIT ?", [limit]
            ).fetch_df()

    def run(self, run_id: str) -> dict[str, Any] | None:
        """One run as a dict, or ``None``."""
        with duckdb.connect(str(self.path), read_only=True) as con:
            frame = con.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", [run_id]
            ).fetch_df()
        return None if frame.empty else frame.iloc[0].to_dict()

    def artifacts(self, run_id: str | None = None) -> pd.DataFrame:
        """Artifacts, optionally filtered to one run."""
        query = "SELECT * FROM research_artifacts"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(query + " ORDER BY created_at DESC", params).fetch_df()

    # -- promotion ---------------------------------------------------------

    def request_promotion(
        self,
        run_id: str,
        gate: PromotionGate,
        approved_by: str,
        rationale: str,
    ) -> str:
        """Record an explicit human approval of a candidate.

        Args:
            run_id: The run being approved.
            gate: The evaluated gate. Must have passed.
            approved_by: Who approved it. Required and recorded.
            rationale: Why. Required and recorded.

        Returns:
            The promotion id.

        Raises:
            ValueError: If the gate did not pass, if the approver or rationale
                is missing, or if the run does not exist. Approval is meant to
                be effortful; none of these are papered over.
        """
        if not gate.passed:
            raise ValueError(
                "promotion gate did not pass: " + "; ".join(gate.failures)
            )
        if not approved_by.strip():
            raise ValueError("promotion requires a named approver")
        if len(rationale.strip()) < 10:
            raise ValueError(
                "promotion requires a written rationale of at least 10 characters"
            )
        if self.run(run_id) is None:
            raise ValueError(f"unknown run: {run_id}")

        promotion_id = f"promo-{uuid.uuid4().hex[:10]}"
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                "INSERT INTO research_promotions VALUES (?,?,?,?,?,?,?,?)",
                [
                    promotion_id, run_id, datetime.now(UTC),
                    PromotionState.APPROVED_FOR_PAPER.value, approved_by.strip(),
                    rationale.strip(),
                    json.dumps([list(c) for c in gate.checks], default=str),
                    False,  # Never automatic. There is no code path that sets this True.
                ],
            )
        return promotion_id

    def record_rejection(self, run_id: str, decided_by: str, rationale: str) -> str:
        """Record an explicit decision not to promote a candidate."""
        promotion_id = f"promo-{uuid.uuid4().hex[:10]}"
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                "INSERT INTO research_promotions VALUES (?,?,?,?,?,?,?,?)",
                [
                    promotion_id, run_id, datetime.now(UTC),
                    PromotionState.REJECTED.value, decided_by.strip() or "unspecified",
                    rationale.strip(), "[]", False,
                ],
            )
        return promotion_id

    def promotions(self, run_id: str | None = None) -> pd.DataFrame:
        """Promotion decisions, newest first."""
        query = "SELECT * FROM research_promotions"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(query + " ORDER BY decided_at DESC", params).fetch_df()

    def state_of(self, run_id: str) -> PromotionState:
        """The current state of a run: its most recent recorded decision."""
        frame = self.promotions(run_id)
        if frame.empty:
            return PromotionState.CANDIDATE
        return PromotionState(str(frame.iloc[0]["state"]))

    def approved_runs(self) -> pd.DataFrame:
        """Runs whose latest decision was approval for paper evaluation.

        Returned for display only. Nothing consumes this to start a session.
        """
        frame = self.promotions()
        if frame.empty:
            return frame
        latest = frame.sort_values("decided_at").groupby("run_id").tail(1)
        return latest[latest["state"] == PromotionState.APPROVED_FOR_PAPER.value]
