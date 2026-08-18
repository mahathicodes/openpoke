"""Reachability of idle agents, merged-history recovery, and the embeddings client.

Groups three things that were previously untested and each underpin a claim made
elsewhere in the design:

  * `search_agents`   - the claim that going idle never makes an agent unreachable
  * merged transcripts - the claim that merging recovers context rather than just
                         relabelling records
  * embeddings client - the parsing the whole retrieval layer sits on
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.interaction_agent import tools as interaction_tools
from server.openrouter_client.client import OpenRouterError, _extract_embeddings
from server.services.execution import agent_matcher as matcher

UTC = timezone.utc


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="microseconds")


@pytest.fixture
def stub_search_deps(monkeypatch):
    async def fake_request_embeddings(*, model, texts, api_key=None, timeout=None, **kwargs):
        return [[1.0, 0.0] for _ in texts]

    class _Settings:
        openrouter_api_key = "test-key"
        embedding_model = "test/model"
        embedding_timeout_seconds = 10.0
        agent_dedup_top_k = 5
        agent_prompt_top_k = 5
        agent_prompt_recent_count = 2
        agent_archive_after_days = 30

    monkeypatch.setattr(matcher, "request_embeddings", fake_request_embeddings)
    monkeypatch.setattr(matcher, "get_settings", lambda: _Settings())
    monkeypatch.setattr(interaction_tools, "get_settings", lambda: _Settings())


# ----------------------------------------------------------------------
# search_agents - "idle is not unreachable"
# ----------------------------------------------------------------------


def test_search_finds_an_agent_too_idle_for_the_prompt(
    temp_roster, temp_logs, stub_search_deps, no_triggers
):
    """The central claim of the retirement design, previously unverified."""
    archived = asyncio.run(
        temp_roster.create_or_link("Old thread", description="lunch with keith last spring")
    )
    archived.last_active = _days_ago(200)
    temp_roster.save()

    # Confirm it really is out of the default view before asserting it is findable.
    assert [record.name for record in temp_roster.get_records()] == []

    result = asyncio.run(interaction_tools.search_agents("lunch with keith"))

    assert result.success
    assert any(m["agent_name"] == "Old thread" for m in result.payload["matches"])


def test_search_returns_descriptions_and_scores(
    temp_roster, temp_logs, stub_search_deps, no_triggers
):
    asyncio.run(temp_roster.create_or_link("Keith lunch", description="planning lunch"))

    match = asyncio.run(interaction_tools.search_agents("lunch")).payload["matches"][0]

    assert match["description"] == "planning lunch"
    assert "score" in match and "last_active" in match


def test_search_on_empty_roster_returns_no_matches(
    temp_roster, temp_logs, stub_search_deps, no_triggers
):
    result = asyncio.run(interaction_tools.search_agents("anything"))

    assert result.success
    assert result.payload["matches"] == []


def test_search_excludes_zero_score_matches(
    temp_roster, temp_logs, stub_search_deps, no_triggers, monkeypatch
):
    """Without embeddings, lexical overlap of zero means genuinely unrelated."""

    async def failing(**kwargs):
        raise matcher.OpenRouterError("down")

    monkeypatch.setattr(matcher, "request_embeddings", failing)
    asyncio.run(temp_roster.create_or_link("Zebra husbandry", description="zoo logistics"))

    matches = asyncio.run(
        interaction_tools.search_agents("quarterly tax filing")
    ).payload["matches"]

    assert matches == []


def test_search_survives_embedding_failure(
    temp_roster, temp_logs, stub_search_deps, no_triggers, monkeypatch
):
    async def failing(**kwargs):
        raise matcher.OpenRouterError("down")

    monkeypatch.setattr(matcher, "request_embeddings", failing)
    asyncio.run(temp_roster.create_or_link("Keith lunch", description="planning lunch keith"))

    result = asyncio.run(interaction_tools.search_agents("lunch with keith"))

    assert result.success
    assert result.payload["matches"]


# ----------------------------------------------------------------------
# Merged history - "merging recovers context"
# ----------------------------------------------------------------------


def test_survivor_inherits_the_absorbed_agents_history(temp_roster, temp_logs):
    """Without this, a merge is bookkeeping that recovers nothing."""
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_logs.record_agent_response("Duplicate", "asked about dietary restrictions")
    temp_logs.record_agent_response("Canonical", "proposed thursday")
    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["t"])

    prompt = ExecutionAgent("Canonical").build_system_prompt_with_history()

    assert "dietary restrictions" in prompt
    assert "proposed thursday" in prompt


def test_absorbed_history_precedes_the_survivors_own(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_logs.record_agent_response("Duplicate", "ABSORBED_MARKER")
    temp_logs.record_agent_response("Canonical", "OWN_MARKER")
    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["t"])

    prompt = ExecutionAgent("Canonical").build_system_prompt_with_history()

    assert prompt.index("ABSORBED_MARKER") < prompt.index("OWN_MARKER")


def test_absorbed_history_is_attributed_not_silently_spliced(temp_roster, temp_logs):
    """The survivor should be able to tell whose history it inherited."""
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_logs.record_agent_response("Duplicate", "something")
    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["t"])

    prompt = ExecutionAgent("Canonical").build_system_prompt_with_history()

    assert 'merged_agent name="Duplicate"' in prompt


def test_agent_without_merges_is_unaffected(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Solo"))
    temp_logs.record_agent_response("Solo", "only entry")

    prompt = ExecutionAgent("Solo").build_system_prompt_with_history()

    assert "only entry" in prompt
    assert "merged_agent" not in prompt


# ----------------------------------------------------------------------
# Triggers are not user engagement
# ----------------------------------------------------------------------


def test_trigger_execution_does_not_refresh_last_active(temp_roster, temp_logs, no_triggers):
    """A scheduled firing is not evidence the user is engaged with the thread.

    Guards the retirement rule: if triggers bumped recency, a recurring reminder
    would look permanently active with zero human contact.
    """
    from server.agents.execution_agent import batch_manager as batch_module
    from server.agents.execution_agent.runtime import ExecutionResult

    record = asyncio.run(temp_roster.create_or_link("Daily digest"))
    record.last_active = _days_ago(100)
    temp_roster.save()
    before = temp_roster.get_record("Daily digest").last_active

    class _StubRuntime:
        def __init__(self, agent_name):
            self.agent_name = agent_name

        async def execute(self, instructions):
            return ExecutionResult(agent_name=self.agent_name, success=True, response="done")

    original_runtime = batch_module.ExecutionAgentRuntime
    original_dispatch = batch_module.ExecutionBatchManager._dispatch_to_interaction_agent
    batch_module.ExecutionAgentRuntime = _StubRuntime

    async def _no_dispatch(self, payload):
        return None

    batch_module.ExecutionBatchManager._dispatch_to_interaction_agent = _no_dispatch
    try:
        # Exactly how trigger_scheduler reaches execution: a fresh manager per firing.
        asyncio.run(
            batch_module.ExecutionBatchManager().execute_agent("Daily digest", "trigger fired")
        )
    finally:
        batch_module.ExecutionAgentRuntime = original_runtime
        batch_module.ExecutionBatchManager._dispatch_to_interaction_agent = original_dispatch

    temp_roster.load()

    assert temp_roster.get_record("Daily digest").last_active == before


def test_live_chat_delegation_does_refresh_last_active(temp_roster, temp_logs, no_triggers):
    record = asyncio.run(temp_roster.create_or_link("Chatted about"))
    record.last_active = _days_ago(100)
    temp_roster.save()

    temp_roster.mark_active("Chatted about")

    assert temp_roster.get_record("Chatted about").last_active != _days_ago(100)
    assert temp_roster.is_archived(temp_roster.get_record("Chatted about")) is False


# ----------------------------------------------------------------------
# Embeddings client parsing
# ----------------------------------------------------------------------


def test_vectors_are_ordered_by_index_not_arrival():
    """The API returns an index per entry; list order must not be trusted."""
    payload = {
        "data": [
            {"embedding": [3.0], "index": 2},
            {"embedding": [1.0], "index": 0},
            {"embedding": [2.0], "index": 1},
        ]
    }

    assert _extract_embeddings(payload, expected=3) == [[1.0], [2.0], [3.0]]


def test_count_mismatch_is_an_error():
    """Silently returning fewer vectors would misalign every downstream pairing."""
    payload = {"data": [{"embedding": [1.0], "index": 0}]}

    with pytest.raises(OpenRouterError):
        _extract_embeddings(payload, expected=2)


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"embedding": [], "index": 0}]},
        {"data": [{"index": 0}]},
        {"data": "not-a-list"},
        {},
    ],
)
def test_malformed_payloads_raise(payload):
    with pytest.raises(OpenRouterError):
        _extract_embeddings(payload, expected=1)


def test_integer_values_are_coerced_to_float():
    result = _extract_embeddings({"data": [{"embedding": [1, 2], "index": 0}]}, expected=1)

    assert result == [[1.0, 2.0]]
    assert all(isinstance(value, float) for value in result[0])


def test_embed_texts_returns_none_rather_than_raising(monkeypatch):
    """Callers treat embeddings as optional, so failure must be a value not an exception."""

    async def failing(**kwargs):
        raise OpenRouterError("provider down")

    class _Settings:
        openrouter_api_key = "k"
        embedding_model = "m"
        embedding_timeout_seconds = 1.0

    monkeypatch.setattr(matcher, "request_embeddings", failing)
    monkeypatch.setattr(matcher, "get_settings", lambda: _Settings())

    assert asyncio.run(matcher.embed_texts(["anything"])) is None


def test_embed_texts_without_api_key_returns_none(monkeypatch):
    class _Settings:
        openrouter_api_key = None
        embedding_model = "m"
        embedding_timeout_seconds = 1.0

    monkeypatch.setattr(matcher, "get_settings", lambda: _Settings())

    assert asyncio.run(matcher.embed_texts(["anything"])) is None
