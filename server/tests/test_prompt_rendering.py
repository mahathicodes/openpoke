"""Prompt pruning: the bound on what the interaction agent sees each turn.

This is the other half of the original problem - the roster used to be dumped in
full into every prompt, so cost grew without limit and the model had to pick a
reuse candidate by eyeballing a flat list of bare names.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from server.agents.interaction_agent import agent as interaction_agent
from server.services.execution import agent_matcher as matcher

UTC = timezone.utc


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")


@pytest.fixture
def counted_embeddings(monkeypatch):
    """Stub embeddings and count calls, so latency claims are testable."""
    state = {"calls": 0}

    async def fake_request_embeddings(*, model, texts, api_key=None, timeout=None, **kwargs):
        state["calls"] += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(matcher, "request_embeddings", fake_request_embeddings)

    class _Settings:
        openrouter_api_key = "test-key"
        embedding_model = "test/model"
        embedding_timeout_seconds = 10.0
        agent_prompt_top_k = 5
        agent_prompt_recent_count = 2
        agent_archive_after_days = 30
        agent_dedup_top_k = 5

    monkeypatch.setattr(matcher, "get_settings", lambda: _Settings())
    monkeypatch.setattr(interaction_agent, "get_settings", lambda: _Settings())
    return state


def _add(roster, name, description=""):
    return asyncio.run(roster.create_or_link(name, description=description))


def _render(query="a question about something"):
    return asyncio.run(interaction_agent._render_active_agents(query))


# ----------------------------------------------------------------------
# Base cases
# ----------------------------------------------------------------------


def test_empty_roster_renders_none(temp_roster, counted_embeddings, no_triggers):
    assert _render() == "None"


def test_small_roster_renders_every_agent(temp_roster, counted_embeddings, no_triggers):
    for i in range(4):  # under top_k of 5
        _add(temp_roster, f"Agent {i}")

    rendered = _render()

    for i in range(4):
        assert f'name="Agent {i}"' in rendered


def test_small_roster_makes_no_embedding_call(temp_roster, counted_embeddings, no_triggers):
    """The no-latency-regression guarantee: below the cap, retrieval is skipped."""
    for i in range(4):
        _add(temp_roster, f"Agent {i}")

    before = counted_embeddings["calls"]
    _render()

    assert counted_embeddings["calls"] == before


# ----------------------------------------------------------------------
# The bound
# ----------------------------------------------------------------------


def test_large_roster_is_capped_at_top_k(temp_roster, counted_embeddings, no_triggers):
    for i in range(40):
        _add(temp_roster, f"Agent {i}", description=f"thread number {i}")

    rendered = _render()

    assert rendered.count("<agent ") == 5  # agent_prompt_top_k


def test_prompt_size_stops_growing_with_roster_size(
    temp_roster, counted_embeddings, no_triggers
):
    """The actual fix: prompt cost decouples from roster size.

    Compares agent count rather than exact character length - which agents get
    selected changes with the roster, and their descriptions differ in length, so
    byte-identical output is the wrong thing to assert.
    """
    for i in range(10):
        _add(temp_roster, f"Agent {i}", description=f"thread number {i}")
    at_ten = _render()

    for i in range(10, 60):
        _add(temp_roster, f"Agent {i}", description=f"thread number {i}")
    at_sixty = _render()

    assert at_sixty.count("<agent ") == at_ten.count("<agent ") == 5
    # A 6x roster must not meaningfully move the prompt size.
    assert abs(len(at_sixty) - len(at_ten)) < 0.15 * len(at_ten)


def test_large_roster_does_embed_the_query(temp_roster, counted_embeddings, no_triggers):
    for i in range(40):
        _add(temp_roster, f"Agent {i}")

    before = counted_embeddings["calls"]
    _render()

    assert counted_embeddings["calls"] == before + 1


# ----------------------------------------------------------------------
# Selection behaviour
# ----------------------------------------------------------------------


def test_recent_agents_survive_even_when_semantically_distant(
    temp_roster, counted_embeddings, no_triggers
):
    """A thread being worked on right now must not vanish on cosine distance alone."""
    for i in range(40):
        _add(temp_roster, f"Agent {i}", description=f"thread number {i}")

    recent = _add(temp_roster, "Just Touched", description="completely unrelated wording")
    temp_roster.mark_active(recent.name)

    assert 'name="Just Touched"' in _render()


def test_descriptions_are_rendered_when_present(temp_roster, counted_embeddings, no_triggers):
    _add(temp_roster, "Keith lunch", description="planning lunch with keith")

    assert 'description="planning lunch with keith"' in _render()


def test_agents_without_descriptions_render_bare(temp_roster, counted_embeddings, no_triggers):
    _add(temp_roster, "No Description")

    rendered = _render()

    assert 'name="No Description"' in rendered
    assert "description=" not in rendered


def test_archived_agents_are_excluded(temp_roster, counted_embeddings, no_triggers):
    fresh = _add(temp_roster, "Fresh")
    stale = _add(temp_roster, "Stale")
    stale.last_active = _days_ago(90)
    temp_roster.save()

    rendered = _render()

    assert 'name="Fresh"' in rendered
    assert 'name="Stale"' not in rendered


def test_merged_agents_are_excluded(temp_roster, temp_logs, counted_embeddings, no_triggers):
    _add(temp_roster, "Canonical")
    _add(temp_roster, "Duplicate")
    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["test"])

    rendered = _render()

    assert 'name="Canonical"' in rendered
    assert 'name="Duplicate"' not in rendered


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_names_with_markup_characters_are_escaped(
    temp_roster, counted_embeddings, no_triggers
):
    """Agent names are LLM-authored and land inside XML-ish tags."""
    _add(temp_roster, 'Keith "the boss" & <co>')

    rendered = _render()

    assert "&quot;" in rendered or "&#x27;" in rendered
    assert "&amp;" in rendered
    assert "<co>" not in rendered


def test_embedding_failure_still_renders_a_bounded_roster(
    temp_roster, counted_embeddings, no_triggers, monkeypatch
):
    """A provider outage degrades ranking quality, never the ability to answer."""

    async def failing(**kwargs):
        raise matcher.OpenRouterError("provider down")

    monkeypatch.setattr(matcher, "request_embeddings", failing)

    for i in range(40):
        _add(temp_roster, f"Agent {i}", description=f"thread number {i}")

    rendered = _render("keith lunch")

    assert rendered != "None"
    assert rendered.count("<agent ") == 5
