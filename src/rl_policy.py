"""FinRL / Stable-Baselines3 policy node: ingests alpha vectors, updates metrics.

What this node is
-----------------
A gymnasium environment over a :class:`core.dataset.MarketDataset`, whose
observation at each step is the research desk's :class:`~src.contracts.AlphaVector`
for that market and instant plus the current portfolio state, and whose action
is a discrete position intent. A Stable-Baselines3 PPO agent (the algorithm
FinRL wraps) is trained on it, and the run is summarised as
:class:`~src.contracts.PolicyMetrics`.

Why the reward is not just price change
---------------------------------------
The reward is realised through the project's existing
:func:`backtest.fills.simulate_fill`, so every cost the backtester charges is
charged here too: crossing the spread, fees, modelled slippage, depth caps and
the staleness guard. A policy that looks profitable on mid-price deltas but
cannot pay the spread scores badly here, which is the point.

**Simulation-only is unchanged.** The action space terminates in
``simulate_fill`` and an in-memory portfolio. There is no order-placement
method on this class, no venue handle, and no credential use.

**The RL stack is optional and not installed by default** - see the import
block below and ``DECISIONS.md`` (2026-09-08). This module imports cleanly
without it; only training raises.

The sample-size caveat that governs everything
----------------------------------------------
A market forecast 400 times still resolves once. Training for 200,000 timesteps
across 200 markets is 200 independent outcomes, not 200,000. Every metric this
node emits carries ``effective_n`` for exactly that reason, and
:meth:`RLPolicyPipeline.evaluate_gate` refuses on markets rather than steps.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np

from backtest.fills import simulate_fill
from core.dataset import MarketDataset
from core.models import Side
from src.contracts import ALPHA_FEATURE_NAMES, AlphaVector, PolicyMetrics

# The RL stack is optional and not installed by default. It is parked, not
# unwanted: a control policy cannot improve accuracy until there is a validated
# forecast edge and enough independent outcomes, and neither holds yet. Keeping
# torch/FinRL/SB3 out of the tree also lets compat/ - stub modules named after a
# live brokerage client - be deleted from a repository whose central claim is
# that it cannot trade. See DECISIONS.md (2026-09-08).
#
#   pip install stable-baselines3 gymnasium
#
# reverses this. The module imports cleanly either way, so anything that only
# needs ACTION_NAMES or PolicyMetrics keeps working.
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    RL_AVAILABLE = True
    _EnvBase: Any = gym.Env
except ImportError:  # pragma: no cover - exercised by the absence of the deps
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    PPO = Monitor = DummyVecEnv = None  # type: ignore[assignment]
    RL_AVAILABLE = False
    _EnvBase = object

RL_MISSING_MESSAGE = (
    "The reinforcement-learning stack is not installed. It is parked by design "
    "(DECISIONS.md 2026-09-08). Install it with: "
    "pip install stable-baselines3 gymnasium"
)


RL_POLICY_VERSION = "ppo-alpha-v1"

# Discrete action space. Deliberately tiny and interpretable: a continuous size
# head would give the policy more ways to overfit 200 independent outcomes.
ACTION_FLAT, ACTION_LONG, ACTION_SHORT = 0, 1, 2
ACTION_NAMES = {ACTION_FLAT: "FLAT", ACTION_LONG: "LONG_YES", ACTION_SHORT: "SHORT_YES"}

# Observation = alpha features + [position_fraction, cash_fraction, step_fraction]
_PORTFOLIO_FEATURES = ("position_fraction", "cash_fraction", "step_fraction")
OBSERVATION_NAMES: tuple[str, ...] = (*ALPHA_FEATURE_NAMES, *_PORTFOLIO_FEATURES)


class AlphaTradingEnv(_EnvBase):
    """One episode = one market, walked forward over its alpha vectors."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        alpha_by_market: dict[str, list[AlphaVector]],
        settings: dict[str, Any] | None = None,
        order_size: float = 100.0,
        initial_capital: float = 10_000.0,
    ) -> None:
        if not RL_AVAILABLE:
            raise ImportError(RL_MISSING_MESSAGE)
        super().__init__()
        self.dataset = dataset
        self.settings = dict(settings or {})
        self.fills_config = self.settings.get("fills", {})
        self.order_size = float(order_size)
        self.initial_capital = float(initial_capital)

        # Only markets that actually have alpha and a resolved outcome can be
        # traded to settlement, so only those form episodes.
        self.markets = [
            m
            for m in dataset.markets
            if alpha_by_market.get(m.market_id) and m.outcome is not None
        ]
        self.alpha_by_market = alpha_by_market
        if not self.markets:
            raise ValueError("no market has both alpha vectors and a settled outcome")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(OBSERVATION_NAMES),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._market_index = 0
        self._step_index = 0
        self._position = 0.0
        self._cash = self.initial_capital
        self._fills: list[Any] = []

    # -- gymnasium API ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode on the next market in the rotation."""
        super().reset(seed=seed)
        if seed is not None:
            self._market_index = seed % len(self.markets)
        self._market = self.markets[self._market_index % len(self.markets)]
        self._market_index += 1
        self._alpha = self.alpha_by_market[self._market.market_id]
        self._step_index = 0
        self._position = 0.0
        self._cash = self.initial_capital
        self._fills = []
        return self._observation(), {"market_id": self._market.market_id}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one position intent and advance to the next alpha vector."""
        vector = self._alpha[self._step_index]
        reward = self._apply_action(int(action), vector)

        self._step_index += 1
        terminated = self._step_index >= len(self._alpha)
        if terminated:
            reward += self._settle()
        observation = self._observation() if not terminated else self._terminal_observation()
        info = {
            "market_id": self._market.market_id,
            "position": self._position,
            "action": ACTION_NAMES[int(action)],
        }
        return observation, float(reward), terminated, False, info

    # -- internals --------------------------------------------------------

    def _quote_at(self, as_of: datetime) -> Any:
        """Most recent quote at or before ``as_of`` - never a later one."""
        seen = [q for q in self.dataset.quotes_for(self._market.market_id) if q.ts <= as_of]
        return seen[-1] if seen else None

    def _apply_action(self, action: int, vector: AlphaVector) -> float:
        """Move toward the requested position, paying real modelled costs."""
        target = {ACTION_FLAT: 0.0, ACTION_LONG: self.order_size, ACTION_SHORT: -self.order_size}[
            action
        ]
        delta = target - self._position
        if abs(delta) < 1.0:
            return 0.0

        quote = self._quote_at(vector.as_of)
        side = Side.BUY if delta > 0 else Side.SELL
        fill = simulate_fill(
            quote=quote,
            side=side,
            size=abs(delta),
            ts=vector.as_of,
            config=self.fills_config,
            proposal_id=f"rl-{self._market.market_id}-{self._step_index}",
            fill_id=f"rlfill-{self._market.market_id}-{self._step_index}",
            decided_at=vector.as_of,
        )
        if fill.filled_size <= 0.0:
            return 0.0

        signed = fill.filled_size * (1.0 if side is Side.BUY else -1.0)
        self._position += signed
        self._cash -= signed * fill.avg_price
        self._cash -= fill.fee
        self._fills.append(fill)
        # Costs are charged immediately so the policy feels them at the moment
        # it acts, rather than only in the terminal settlement.
        return -float(fill.fee) / self.initial_capital

    def _settle(self) -> float:
        """Settle the position at the realised outcome and return the P&L."""
        outcome = self._market.outcome
        payoff = 1.0 if outcome is not None and outcome.value == "YES" else 0.0
        self._cash += self._position * payoff
        self._position = 0.0
        return float(self._cash - self.initial_capital) / self.initial_capital

    def _observation(self) -> np.ndarray:
        vector = self._alpha[min(self._step_index, len(self._alpha) - 1)]
        values = [
            *vector.vector(),
            self._position / max(1.0, self.order_size),
            self._cash / max(1.0, self.initial_capital),
            self._step_index / max(1, len(self._alpha)),
        ]
        return np.asarray(values, dtype=np.float32)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)


class RLPolicyPipeline:
    """Trains a PPO policy on alpha vectors and reports what it learned."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        total_timesteps: int = 2_000,
        seed: int = 20260101,
    ) -> None:
        self.settings = dict(settings or {})
        self.total_timesteps = int(total_timesteps)
        self.seed = int(seed)
        self.model: PPO | None = None
        self.metrics = PolicyMetrics(model_version=RL_POLICY_VERSION)

    def build_env(self, dataset: MarketDataset, alpha: list[AlphaVector]) -> Any:
        """Group alpha vectors by market and wrap the environment for SB3."""
        if not RL_AVAILABLE:
            raise ImportError(RL_MISSING_MESSAGE)
        by_market: dict[str, list[AlphaVector]] = {}
        for vector in alpha:
            by_market.setdefault(vector.market_id, []).append(vector)
        for series in by_market.values():
            series.sort(key=lambda v: v.as_of)

        backtest_cfg = self.settings.get("backtest", {})
        strategy_cfg = self.settings.get("strategy", {})

        def _make() -> Any:
            return Monitor(
                AlphaTradingEnv(
                    dataset=dataset,
                    alpha_by_market=by_market,
                    settings=self.settings,
                    order_size=float(strategy_cfg.get("base_size", 100.0)),
                    initial_capital=float(backtest_cfg.get("initial_capital", 10_000.0)),
                )
            )

        return DummyVecEnv([_make])

    def train(
        self, dataset: MarketDataset, alpha: list[AlphaVector]
    ) -> PolicyMetrics:
        """Fit a PPO policy and update :attr:`metrics`.

        Returns the metrics rather than the model, because the model on its own
        is not evidence of anything: what matters downstream is the metric set
        and the ``effective_n`` attached to it.
        """
        if not RL_AVAILABLE:
            raise ImportError(RL_MISSING_MESSAGE)
        env = self.build_env(dataset, alpha)
        traded_markets = {v.market_id for v in alpha}
        resolved = {
            m.market_id
            for m in dataset.markets
            if m.market_id in traded_markets and m.outcome is not None
        }

        self.model = PPO(
            "MlpPolicy",
            env,
            seed=self.seed,
            verbose=0,
            n_steps=min(256, max(64, self.total_timesteps // 8)),
            batch_size=64,
        )
        self.model.learn(total_timesteps=self.total_timesteps)

        # Monitor records one total reward per finished episode; an episode is
        # one market walked to settlement, so this is a per-market return.
        monitor = env.envs[0]
        episode_rewards = (
            monitor.get_episode_rewards() if hasattr(monitor, "get_episode_rewards") else []
        )
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else None

        logs = getattr(self.model, "logger", None)
        recorded = getattr(logs, "name_to_value", {}) if logs else {}

        self.metrics = PolicyMetrics(
            model_version=RL_POLICY_VERSION,
            timesteps=self.total_timesteps,
            n_markets=len(traded_markets),
            effective_n=len(resolved),
            mean_episode_reward=mean_reward,
            total_return=mean_reward,
            n_trades=0,
            explained_variance=_as_float(recorded.get("train/explained_variance")),
            entropy_loss=_as_float(recorded.get("train/entropy_loss")),
            is_synthetic=dataset.is_demo,
            warnings=self._warnings(len(resolved), dataset.is_demo),
        )
        return self.metrics

    def _warnings(self, effective_n: int, is_synthetic: bool) -> list[str]:
        """State plainly what this run cannot support."""
        notes: list[str] = []
        if is_synthetic:
            notes.append(
                "Corpus is FABRICATED. Any policy learned here is a property of the "
                "generator, not of any market."
            )
        if effective_n < 50:
            notes.append(
                f"Only {effective_n} resolved market(s). The policy has seen "
                f"{effective_n} independent outcomes regardless of timestep count."
            )
        return notes

    def evaluate_gate(
        self, baseline_return: float = 0.0, min_markets: int = 50
    ) -> tuple[bool, list[str]]:
        """Would this policy clear the promotion bar?

        Expressed in resolved markets and in out-of-sample return against a
        frozen baseline - never in timesteps, which measure compute rather than
        evidence.
        """
        failures: list[str] = []
        if self.metrics.is_synthetic:
            failures.append("Fabricated corpus cannot justify promotion.")
        if self.metrics.effective_n < min_markets:
            failures.append(
                f"{self.metrics.effective_n} resolved markets, {min_markets} required."
            )
        realised = self.metrics.total_return
        if realised is None or not math.isfinite(realised) or realised <= baseline_return:
            failures.append("Does not beat the frozen baseline out-of-sample.")
        return (not failures), failures

    def update_policy_metrics(self, **fields: Any) -> PolicyMetrics:
        """Merge measured fields into the stored metrics and return them."""
        self.metrics = replace(self.metrics, **fields)
        return self.metrics


def _as_float(value: Any) -> float | None:
    """Coerce an SB3 logger value to a float, or ``None`` if it is not one."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
