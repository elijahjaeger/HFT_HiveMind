"""Local DuckDB persistence for hypothetical paper-trading orders and fills.

Every row written here carries ``mode = 'PAPER'`` and ``is_hypothetical = true``
so a record can never be mistaken for a real execution, including by whatever
reads the file later. The store is append-only: nothing is ever updated in
place, so the sequence of decisions stays auditable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from core.config import resolve_path
from core.models import PortfolioSnapshot, RiskDecision, SimulatedFill, TradeProposal

MODE = "PAPER"
DEFAULT_PATH = "data/paper/paper_trading.duckdb"

SCHEMA: dict[str, str] = {
    "paper_orders": """
        CREATE TABLE IF NOT EXISTS paper_orders (
            session_id       VARCHAR,
            recorded_at      TIMESTAMPTZ,
            decided_at       TIMESTAMPTZ,
            proposal_id      VARCHAR,
            market_id        VARCHAR,
            venue            VARCHAR,
            side             VARCHAR,
            requested_size   DOUBLE,
            approved_size    DOUBLE,
            limit_price      DOUBLE,
            reference_price  DOUBLE,
            fair_probability DOUBLE,
            edge             DOUBLE,
            expected_value   DOUBLE,
            risk_action      VARCHAR,
            risk_reasons     VARCHAR,
            mode             VARCHAR,
            is_hypothetical  BOOLEAN
        )
    """,
    "paper_fills": """
        CREATE TABLE IF NOT EXISTS paper_fills (
            session_id      VARCHAR,
            recorded_at     TIMESTAMPTZ,
            decided_at      TIMESTAMPTZ,
            filled_at       TIMESTAMPTZ,
            fill_id         VARCHAR,
            proposal_id     VARCHAR,
            market_id       VARCHAR,
            side            VARCHAR,
            requested_size  DOUBLE,
            filled_size     DOUBLE,
            reference_price DOUBLE,
            avg_price       DOUBLE,
            slippage        DOUBLE,
            fee             DOUBLE,
            unfilled_reason VARCHAR,
            latency_seconds DOUBLE,
            mode            VARCHAR,
            is_hypothetical BOOLEAN
        )
    """,
    "paper_equity": """
        CREATE TABLE IF NOT EXISTS paper_equity (
            session_id      VARCHAR,
            recorded_at     TIMESTAMPTZ,
            ts              TIMESTAMPTZ,
            cash            DOUBLE,
            position_value  DOUBLE,
            equity          DOUBLE,
            peak_equity     DOUBLE,
            drawdown        DOUBLE,
            total_exposure  DOUBLE,
            open_positions  BIGINT,
            mode            VARCHAR,
            is_hypothetical BOOLEAN
        )
    """,
    "paper_events": """
        CREATE TABLE IF NOT EXISTS paper_events (
            session_id      VARCHAR,
            recorded_at     TIMESTAMPTZ,
            ts              TIMESTAMPTZ,
            kind            VARCHAR,
            market_id       VARCHAR,
            message         VARCHAR,
            mode            VARCHAR,
            is_hypothetical BOOLEAN
        )
    """,
}


class PaperStore:
    """Append-only local record of a paper-trading session."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _initialise(self) -> None:
        with duckdb.connect(str(self.path)) as con:
            for statement in SCHEMA.values():
                con.execute(statement)

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                list(row.values()),
            )

    # -- writes ------------------------------------------------------------

    def record_order(
        self,
        session_id: str,
        proposal: TradeProposal,
        decision: RiskDecision,
        venue: str = "",
    ) -> None:
        """Persist one hypothetical order and the risk decision that sized it."""
        self._insert(
            "paper_orders",
            {
                "session_id": session_id,
                "recorded_at": datetime.now(UTC),
                "decided_at": proposal.ts,
                "proposal_id": proposal.proposal_id,
                "market_id": proposal.market_id,
                "venue": venue,
                "side": proposal.side.value,
                "requested_size": float(proposal.size),
                "approved_size": float(decision.approved_size),
                "limit_price": float(proposal.limit_price),
                "reference_price": float(proposal.reference_price),
                "fair_probability": float(proposal.fair_probability),
                "edge": float(proposal.edge),
                "expected_value": float(proposal.expected_value_total),
                "risk_action": decision.action.value,
                "risk_reasons": ",".join(c.value for c in decision.reason_codes),
                "mode": MODE,
                "is_hypothetical": True,
            },
        )

    def record_fill(self, session_id: str, fill: SimulatedFill) -> None:
        """Persist one hypothetical fill (including a zero-size non-fill)."""
        self._insert(
            "paper_fills",
            {
                "session_id": session_id,
                "recorded_at": datetime.now(UTC),
                "decided_at": fill.decided_at,
                "filled_at": fill.ts,
                "fill_id": fill.fill_id,
                "proposal_id": fill.proposal_id,
                "market_id": fill.market_id,
                "side": fill.side.value,
                "requested_size": float(fill.requested_size),
                "filled_size": float(fill.filled_size),
                "reference_price": fill.reference_price,
                "avg_price": fill.avg_price,
                "slippage": float(fill.slippage),
                "fee": float(fill.fee),
                "unfilled_reason": fill.unfilled_reason,
                "latency_seconds": fill.latency_seconds,
                "mode": MODE,
                "is_hypothetical": True,
            },
        )

    def record_equity(self, session_id: str, snapshot: PortfolioSnapshot) -> None:
        """Persist one mark-to-market snapshot of the hypothetical portfolio."""
        self._insert(
            "paper_equity",
            {
                "session_id": session_id,
                "recorded_at": datetime.now(UTC),
                "ts": snapshot.ts,
                "cash": snapshot.cash,
                "position_value": snapshot.position_value,
                "equity": snapshot.equity,
                "peak_equity": snapshot.peak_equity,
                "drawdown": snapshot.drawdown,
                "total_exposure": snapshot.total_exposure,
                "open_positions": snapshot.open_positions,
                "mode": MODE,
                "is_hypothetical": True,
            },
        )

    def record_event(
        self,
        session_id: str,
        ts: datetime,
        kind: str,
        message: str,
        market_id: str = "",
    ) -> None:
        """Persist one line of the paper-session audit trail."""
        self._insert(
            "paper_events",
            {
                "session_id": session_id,
                "recorded_at": datetime.now(UTC),
                "ts": ts,
                "kind": kind,
                "market_id": market_id,
                "message": message,
                "mode": MODE,
                "is_hypothetical": True,
            },
        )

    # -- reads -------------------------------------------------------------

    def read(self, table: str, session_id: str | None = None) -> pd.DataFrame:
        """Read one table back, optionally filtered to a session."""
        if table not in SCHEMA:
            raise ValueError(f"unknown paper-trading table: {table}")
        query = f"SELECT * FROM {table}"
        parameters: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            parameters.append(session_id)
        query += " ORDER BY recorded_at"
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(query, parameters).fetch_df()

    def orders(self, session_id: str | None = None) -> pd.DataFrame:
        """Hypothetical orders, newest last."""
        return self.read("paper_orders", session_id)

    def fills(self, session_id: str | None = None) -> pd.DataFrame:
        """Hypothetical fills, newest last."""
        return self.read("paper_fills", session_id)

    def equity(self, session_id: str | None = None) -> pd.DataFrame:
        """Portfolio snapshots, newest last."""
        return self.read("paper_equity", session_id)

    def events(self, session_id: str | None = None) -> pd.DataFrame:
        """Session audit trail, newest last."""
        return self.read("paper_events", session_id)

    def sessions(self) -> list[str]:
        """Every session id present in the store."""
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT DISTINCT session_id FROM paper_orders "
                "UNION SELECT DISTINCT session_id FROM paper_fills"
            ).fetchall()
        return sorted(str(row[0]) for row in rows if row[0])
