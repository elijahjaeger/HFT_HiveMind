"""Typed domain models for the prediction-market research sandbox.

All prices are expressed in *probability units* in the closed interval
``[0, 1]``: a binary contract pays exactly ``1.0`` if the market resolves
``YES`` and ``0.0`` if it resolves ``NO``.

Nothing in this module talks to a venue. These are plain data containers used
by the agents, the backtester and the dashboard.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

DEMO_SOURCE = "SYNTHETIC_DEMO"
"""Marker written onto every generated record: this is DEMO DATA, not a feed."""


class Outcome(StrEnum):
    """Terminal state of a binary prediction market."""

    YES = "YES"
    NO = "NO"

    @property
    def payout(self) -> float:
        """Payout per contract held long, in dollars."""
        return 1.0 if self is Outcome.YES else 0.0


class MarketStatus(StrEnum):
    """Lifecycle status of a market."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


class Side(StrEnum):
    """Direction of a proposal or fill, always in terms of the YES contract."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY (long YES), -1 for SELL (short YES)."""
        return 1 if self is Side.BUY else -1


class RiskAction(StrEnum):
    """The three decisions the risk agent is allowed to return."""

    APPROVE = "APPROVE"
    REDUCE = "REDUCE"
    REJECT = "REJECT"


class RiskReasonCode(StrEnum):
    """Machine-readable reason codes attached to every risk decision."""

    OK = "OK"
    KILL_SWITCH = "KILL_SWITCH"
    MARKET_NOT_OPEN = "MARKET_NOT_OPEN"
    NO_QUOTE = "NO_QUOTE"
    STALE_DATA = "STALE_DATA"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    SLIPPAGE_TOO_HIGH = "SLIPPAGE_TOO_HIGH"
    MAX_DRAWDOWN_BREACHED = "MAX_DRAWDOWN_BREACHED"
    POSITION_LIMIT = "POSITION_LIMIT"
    EVENT_EXPOSURE_LIMIT = "EVENT_EXPOSURE_LIMIT"
    TOTAL_EXPOSURE_LIMIT = "TOTAL_EXPOSURE_LIMIT"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    SIZE_BELOW_MINIMUM = "SIZE_BELOW_MINIMUM"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"


class EventKind(StrEnum):
    """Types of records written to the audit log."""

    QUOTE = "QUOTE"
    TRADE = "TRADE"
    FORECAST = "FORECAST"
    LIQUIDITY = "LIQUIDITY"
    PROPOSAL = "PROPOSAL"
    RISK_DECISION = "RISK_DECISION"
    FILL = "FILL"
    REJECTION = "REJECTION"
    SETTLEMENT = "SETTLEMENT"


class Market(BaseModel):
    """A binary prediction market that resolves YES or NO."""

    model_config = ConfigDict(frozen=True)

    market_id: str
    question: str
    category: str = "demo"
    created_at: datetime
    close_time: datetime
    resolution_time: datetime | None = None
    outcome: Outcome | None = None
    status: MarketStatus = MarketStatus.OPEN
    tick_size: float = Field(default=0.01, gt=0.0, le=0.5)
    fee_rate: float = Field(default=0.02, ge=0.0)
    source: str = DEMO_SOURCE
    is_demo: bool = True
    venue: str | None = None
    # Settlement rules exactly as published by the venue. Carried through so the
    # basis on which a market resolved travels with the data.
    rules_primary: str = ""
    rules_secondary: str = ""

    @model_validator(mode="after")
    def _check_resolution(self) -> Market:
        if self.status is MarketStatus.RESOLVED and self.outcome is None:
            raise ValueError("resolved market requires an outcome")
        return self


class Quote(BaseModel):
    """Top-of-book snapshot. Either side may be missing (a one-sided book)."""

    model_config = ConfigDict(frozen=True)

    market_id: str
    ts: datetime
    bid: float | None = Field(default=None, ge=0.0, le=1.0)
    ask: float | None = Field(default=None, ge=0.0, le=1.0)
    bid_size: float = Field(default=0.0, ge=0.0)
    ask_size: float = Field(default=0.0, ge=0.0)
    source: str = DEMO_SOURCE

    @model_validator(mode="after")
    def _check_crossed(self) -> Quote:
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"crossed quote: bid={self.bid} > ask={self.ask}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float | None:
        """Midpoint price, or ``None`` when either side is missing."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float | None:
        """Absolute bid/ask width in probability units, or ``None``."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def is_two_sided(self) -> bool:
        """True when both a bid and an ask are present."""
        return self.bid is not None and self.ask is not None

    def price_for(self, side: Side) -> float | None:
        """Price a marketable order of ``side`` would reference (ask for BUY)."""
        return self.ask if side is Side.BUY else self.bid

    def size_for(self, side: Side) -> float:
        """Displayed size available to a marketable order of ``side``."""
        return self.ask_size if side is Side.BUY else self.bid_size

    def age_seconds(self, as_of: datetime) -> float:
        """Seconds between this quote timestamp and ``as_of`` (never negative)."""
        return max(0.0, (as_of - self.ts).total_seconds())


class Trade(BaseModel):
    """A print on the public tape (context only, never a source of fills)."""

    model_config = ConfigDict(frozen=True)

    trade_id: str
    market_id: str
    ts: datetime
    price: Probability
    size: float = Field(gt=0.0)
    aggressor: Side
    source: str = DEMO_SOURCE


class AgentResult(BaseModel):
    """Common envelope returned by ``Agent.analyze``."""

    agent_name: str
    market_id: str
    ts: datetime
    kind: str
    details: dict[str, Any] = Field(default_factory=dict)


class ForecastInput(BaseModel):
    """One piece of data an agent used, with the timestamp it carried.

    Declaring inputs lets a forecast be audited for look-ahead after the fact:
    every input timestamp must be at or before the forecast's own ``ts``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    ts: datetime
    source: str = ""


class ForecastStatus(StrEnum):
    """Whether an agent produced a usable forecast, and if not, why."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    NO_DATA = "NO_DATA"
    REJECTED_FUTURE_INPUT = "REJECTED_FUTURE_INPUT"


class Forecast(AgentResult):
    """A probability estimate for the YES outcome produced by one agent.

    ``inputs``, ``assumptions`` and ``status`` are optional provenance fields.
    Agents that do not populate them behave exactly as before.
    """

    kind: str = "forecast"
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_placeholder: bool = False
    rationale: str = ""
    status: ForecastStatus = ForecastStatus.OK
    inputs: list[ForecastInput] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @property
    def latest_input_ts(self) -> datetime | None:
        """Most recent declared input timestamp, or ``None`` if undeclared."""
        return max((i.ts for i in self.inputs), default=None)

    @property
    def uses_future_input(self) -> bool:
        """True if any declared input is stamped after the forecast itself."""
        return any(i.ts > self.ts for i in self.inputs)


class LiquidityReport(AgentResult):
    """Tradability assessment of the current book."""

    kind: str = "liquidity"
    spread: float | None = None
    mid: float | None = None
    depth: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    data_age_seconds: float | None = None
    is_stale: bool = True
    is_two_sided: bool = False
    warnings: list[str] = Field(default_factory=list)


class TradeProposal(BaseModel):
    """A *proposed* trade. Proposals are never sent anywhere - by design."""

    proposal_id: str
    ts: datetime
    market_id: str
    agent_name: str
    side: Side
    size: float = Field(gt=0.0)
    limit_price: Probability
    reference_price: Probability
    fair_probability: Probability
    edge: float
    expected_fee_per_contract: float = 0.0
    expected_slippage: float = 0.0
    expected_value_per_contract: float = 0.0
    rationale: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expected_value_total(self) -> float:
        """Expected value of the whole proposal after modelled costs."""
        return self.expected_value_per_contract * self.size


class RiskReason(BaseModel):
    """One machine-readable justification for a risk decision."""

    model_config = ConfigDict(frozen=True)

    code: RiskReasonCode
    message: str
    observed: float | None = None
    limit: float | None = None


class RiskDecision(BaseModel):
    """Output of the risk agent: APPROVE, REDUCE or REJECT plus reasons."""

    proposal_id: str
    market_id: str
    ts: datetime
    action: RiskAction
    original_size: float
    approved_size: float
    reasons: list[RiskReason] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def reason_codes(self) -> list[RiskReasonCode]:
        """Reason codes in the order they were raised."""
        return [r.code for r in self.reasons]

    @property
    def is_blocked(self) -> bool:
        """True when nothing may be executed."""
        return self.action is RiskAction.REJECT or self.approved_size <= 0.0


class SimulatedFill(BaseModel):
    """Result of running an approved proposal through the fill simulator."""

    fill_id: str
    proposal_id: str
    market_id: str
    ts: datetime
    decided_at: datetime | None = None
    """When the decision was taken. Execution happens later, at ``ts``."""
    side: Side
    requested_size: float = Field(ge=0.0)
    filled_size: float = Field(ge=0.0)
    reference_price: float | None = None
    avg_price: float | None = None
    slippage: float = 0.0
    fee: float = 0.0
    unfilled_reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def notional(self) -> float:
        """Cash value of the fill before fees."""
        if self.avg_price is None:
            return 0.0
        return self.filled_size * self.avg_price

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_partial(self) -> bool:
        """True when only part of the requested size was filled."""
        return 0.0 < self.filled_size < self.requested_size

    @property
    def latency_seconds(self) -> float | None:
        """Delay between the decision and this execution, in seconds."""
        if self.decided_at is None:
            return None
        return max(0.0, (self.ts - self.decided_at).total_seconds())

    @property
    def cash_flow(self) -> float:
        """Signed cash impact: negative when buying, positive when selling."""
        if self.avg_price is None or self.filled_size <= 0.0:
            return -self.fee
        return -self.side.sign * self.filled_size * self.avg_price - self.fee


class Settlement(BaseModel):
    """Cash settlement of an open position when a market resolves."""

    market_id: str
    ts: datetime
    outcome: Outcome
    position: float
    payout_per_contract: float
    cash_flow: float


class Position(BaseModel):
    """Signed inventory in one market, in YES contracts."""

    market_id: str
    quantity: float = 0.0
    avg_price: float = 0.0
    fees_paid: float = 0.0
    realized_pnl: float = 0.0

    def exposure(self, price: float) -> float:
        """Worst-case loss on this position at ``price``.

        A long position loses its full cost if the market resolves NO; a short
        position loses ``1 - price`` per contract if it resolves YES.
        """
        if self.quantity >= 0.0:
            return self.quantity * price
        return abs(self.quantity) * (1.0 - price)


class PortfolioSnapshot(BaseModel):
    """Point-in-time view of the simulated account (used for the equity curve)."""

    ts: datetime
    cash: float
    position_value: float
    equity: float
    peak_equity: float
    drawdown: float
    total_exposure: float
    open_positions: int


class AuditEvent(BaseModel):
    """One immutable line in the backtest audit log."""

    ts: datetime
    kind: EventKind
    market_id: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MarketState(BaseModel):
    """Everything an agent is allowed to see at a single decision instant.

    The backtester builds this object exclusively from records whose timestamp
    is ``<= as_of``, which is what keeps agents free of look-ahead bias.
    """

    as_of: datetime
    market: Market
    quote: Quote | None = None
    recent_trades: list[Trade] = Field(default_factory=list)
    position: Position | None = None
    equity: float = 0.0
    peak_equity: float = 0.0
    cash: float = 0.0
    event_exposure: float = 0.0
    total_exposure: float = 0.0
    proposal: TradeProposal | None = None

    @property
    def drawdown(self) -> float:
        """Current drawdown as a positive fraction of the peak equity."""
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def position_quantity(self) -> float:
        """Signed contract inventory, ``0.0`` when flat."""
        return self.position.quantity if self.position else 0.0
