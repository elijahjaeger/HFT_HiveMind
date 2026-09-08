"""Presets: the numbers they set, and the honesty of the text around them.

These presets are the first thing a beginner touches, so the explanations are
as load-bearing as the values. A preset that quietly does nothing, or promises
an outcome it cannot deliver, is worse than no preset at all.
"""

from __future__ import annotations

import pytest

from app import presets
from core.config import get_settings


@pytest.fixture(scope="module")
def settings() -> dict:
    return get_settings()


def test_every_preset_is_reachable_by_key():
    for preset in presets.PRESETS:
        assert presets.get(preset.key) is preset


def test_an_unknown_key_falls_back_to_balanced():
    assert presets.get("nonsense") is presets.BALANCED
    assert presets.get("") is presets.BALANCED


def test_the_default_preset_exists():
    assert presets.DEFAULT_PRESET in presets.BY_KEY


def test_applying_a_preset_does_not_mutate_the_caller_settings(settings):
    before = settings["strategy"]["min_edge"]
    presets.apply(settings, "cautious")
    assert settings["strategy"]["min_edge"] == before


def test_cautious_demands_more_edge_than_adventurous(settings):
    cautious = presets.apply(settings, "cautious")["strategy"]["min_edge"]
    adventurous = presets.apply(settings, "adventurous")["strategy"]["min_edge"]
    assert cautious > adventurous


def test_balanced_changes_nothing(settings):
    """It is the project's configured default, not a fourth opinion."""
    assert presets.apply(settings, "balanced")["strategy"] == settings["strategy"]
    assert presets.apply(settings, "balanced")["risk"] == settings["risk"]


def test_do_nothing_actually_stops_trading(settings):
    assert presets.apply(settings, "flat")["risk"]["kill_switch"] is True


def test_a_preset_only_overrides_what_it_names(settings):
    """Merged one level in, so untouched keys keep the loaded configuration."""
    applied = presets.apply(settings, "cautious")
    assert applied["fills"] == settings["fills"]
    assert applied["risk"]["max_drawdown"] == settings["risk"]["max_drawdown"]
    assert applied["risk"]["max_spread"] != settings["risk"]["max_spread"]


def test_every_preset_says_what_to_expect():
    """Every preset must say what the user should expect to see.

    It turns a confusing empty table into a confirmation that the setting is
    doing its job.
    """
    for preset in presets.PRESETS:
        assert preset.expect, f"{preset.key} has no expectation text"
        assert len(preset.expect) > 40


def test_every_preset_has_a_summary_and_a_name():
    for preset in presets.PRESETS:
        assert preset.name and preset.summary


def test_presets_that_change_settings_explain_what_changed():
    for preset in presets.PRESETS:
        if preset.overrides:
            assert preset.changes, f"{preset.key} changes settings but explains nothing"


def test_the_campaign_explainer_leads_with_the_right_question():
    """Not "is the line up" - "does the band clear zero"."""
    text = presets.CAMPAIGN_EXPLAINER
    assert "zero" in text
    assert "median" in text.lower()


def test_the_campaign_caveat_refuses_to_claim_out_of_sample_validity():
    caveat = presets.CAMPAIGN_CAVEAT
    assert "NOT" in caveat
    assert "never seen" in caveat
    assert "walk-forward" in caveat


def test_the_glossary_covers_the_terms_the_ui_uses():
    terms = {name.lower() for name, _ in presets.TERMS}
    assert {"edge", "return", "drawdown", "confidence band"} <= terms


def test_effective_n_is_explained_as_markets_not_rows():
    """The single most misread number in the project."""
    text = dict(presets.TERMS)["Effective n"]
    assert "resolve" in text.lower()
    assert "400" in text
