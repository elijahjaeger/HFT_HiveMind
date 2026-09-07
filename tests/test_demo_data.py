"""The synthetic demo data: reproducibility, labelling and imperfections."""

from __future__ import annotations

import pytest

from core.models import DEMO_SOURCE, Outcome
from core.store import dataset_to_frames, read_dataset, write_dataset
from core.synthetic import DEMO_LABEL, generate_demo_dataset

SMALL = {"seed": 42, "n_markets": 3, "n_steps": 80}


@pytest.fixture(scope="module")
def dataset():
    return generate_demo_dataset(SMALL)


def test_the_same_seed_reproduces_the_same_dataset():
    first = generate_demo_dataset(SMALL)
    second = generate_demo_dataset(SMALL)
    assert [q.model_dump() for q in first.quotes] == [q.model_dump() for q in second.quotes]
    assert [m.outcome for m in first.markets] == [m.outcome for m in second.markets]


def test_a_different_seed_produces_a_different_dataset():
    other = generate_demo_dataset(dict(SMALL, seed=43))
    base = generate_demo_dataset(SMALL)
    assert [q.mid for q in other.quotes] != [q.mid for q in base.quotes]


def test_there_are_at_least_three_markets(dataset):
    assert len(dataset.markets) >= 3


def test_fewer_than_three_markets_is_rejected():
    with pytest.raises(ValueError):
        generate_demo_dataset(dict(SMALL, n_markets=2))


def test_every_record_is_labelled_as_demo_data(dataset):
    assert all(m.is_demo and m.source == DEMO_SOURCE for m in dataset.markets)
    assert all(DEMO_LABEL in m.question for m in dataset.markets)
    assert all(q.source == DEMO_SOURCE for q in dataset.quotes)
    assert all(t.source == DEMO_SOURCE for t in dataset.trades)


def test_every_market_has_a_binary_outcome(dataset):
    assert all(m.outcome in (Outcome.YES, Outcome.NO) for m in dataset.markets)


def test_quotes_and_trades_are_chronological(dataset):
    assert [q.ts for q in dataset.quotes] == sorted(q.ts for q in dataset.quotes)
    assert [t.ts for t in dataset.trades] == sorted(t.ts for t in dataset.trades)


def test_quotes_are_never_crossed(dataset):
    for quote in dataset.quotes:
        if quote.is_two_sided:
            assert quote.bid <= quote.ask


def test_the_dataset_contains_spread_variation(dataset):
    spreads = {q.spread for q in dataset.quotes if q.spread is not None}
    assert len(spreads) > 1


def test_the_dataset_contains_one_sided_books(dataset):
    assert any(not q.is_two_sided for q in dataset.quotes)


def test_the_dataset_contains_stale_gaps(dataset):
    """At least one market stops quoting for longer than the normal interval."""
    interval = 300
    found = False
    for market_id in dataset.market_ids:
        quotes = dataset.quotes_for(market_id)
        gaps = [
            (b.ts - a.ts).total_seconds()
            for a, b in zip(quotes, quotes[1:], strict=False)
        ]
        if any(gap > interval * 3 for gap in gaps):
            found = True
    assert found, "expected at least one stale gap in the demo data"


def test_the_tape_keeps_printing_during_a_stale_gap(dataset):
    """A frozen book must not freeze the tape - that is what makes it stale."""
    market_id = dataset.market_ids[0]
    quotes = dataset.quotes_for(market_id)
    trades = dataset.trades_for(market_id)
    gap = max(
        zip(quotes, quotes[1:], strict=False), key=lambda pair: pair[1].ts - pair[0].ts
    )
    during = [t for t in trades if gap[0].ts < t.ts < gap[1].ts]
    assert during


def test_frames_and_duckdb_round_trip(dataset, tmp_path):
    frames = dataset_to_frames(dataset)
    assert set(frames) == {"markets", "quotes", "trades"}
    assert len(frames["quotes"]) == len(dataset.quotes)

    path = write_dataset(dataset, tmp_path / "demo.duckdb")
    assert path.exists()

    restored = read_dataset(path)
    assert restored.market_ids == dataset.market_ids
    assert len(restored.quotes) == len(dataset.quotes)
    assert restored.quotes[0].bid == dataset.quotes[0].bid
    assert [m.outcome for m in restored.markets] == [m.outcome for m in dataset.markets]
