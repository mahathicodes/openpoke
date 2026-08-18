"""Roster records, name collisions, retirement, and merges."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from server.services.execution.log_store import _slugify
from server.services.execution.roster import AgentRoster, DuplicateLink


UTC = timezone.utc


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_legacy_string_list_roster_still_loads(tmp_path):
    """Old roster.json was a flat list of names; it must not crash the new code."""
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Email to Keith", "Vercel Job Offer"]))

    roster = AgentRoster(path)

    assert roster.get_agents() == ["Email to Keith", "Vercel Job Offer"]


def test_malformed_roster_file_degrades_to_empty(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text("{not json")

    assert AgentRoster(path).get_agents() == []


def test_record_round_trips_through_disk(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    asyncio.run(
        roster.create_or_link(
            "Keith Lunch",
            description="lunch planning with keith",
            embedding=[0.1, 0.2],
            embedding_model="openai/text-embedding-3-small",
        )
    )
    roster.set_duplicate_link(
        "Keith Lunch", DuplicateLink(name="Other", confidence=0.6, evidence=["text-similarity"])
    )

    reloaded = AgentRoster(path).get_record("Keith Lunch")

    assert reloaded.description == "lunch planning with keith"
    assert reloaded.embedding == [0.1, 0.2]
    assert reloaded.embedding_model == "openai/text-embedding-3-small"
    assert reloaded.possible_duplicate_of.name == "Other"
    assert reloaded.possible_duplicate_of.confidence == pytest.approx(0.6)


# ----------------------------------------------------------------------
# Name collisions
# ----------------------------------------------------------------------


def test_slug_colliding_names_get_distinct_records(temp_roster):
    """The log store slugifies names into filenames, and that slug is lossy.

    "Keith/Lunch" and "Keith Lunch" both slugify to "keith-lunch", so without
    disambiguation two deliberately distinct agents would share one history file.
    """
    first = asyncio.run(temp_roster.create_or_link("Keith Lunch"))
    second = asyncio.run(temp_roster.create_or_link("Keith/Lunch"))

    assert first.name != second.name
    assert _slugify(first.name) != _slugify(second.name)


def test_exact_same_name_reuses_the_record(temp_roster):
    first = asyncio.run(temp_roster.create_or_link("Keith Lunch"))
    second = asyncio.run(temp_roster.create_or_link("Keith Lunch"))

    assert first is second
    assert len(temp_roster.get_records(include_archived=True)) == 1


def test_repeated_collisions_keep_incrementing(temp_roster):
    names = {
        asyncio.run(temp_roster.create_or_link(name)).name
        for name in ("Keith Lunch", "Keith/Lunch", "Keith-Lunch")
    }
    assert len(names) == 3


# ----------------------------------------------------------------------
# Retirement
# ----------------------------------------------------------------------


def test_recent_agent_is_not_archived(temp_roster, no_triggers):
    record = asyncio.run(temp_roster.create_or_link("Fresh"))
    assert temp_roster.is_archived(record) is False


def test_idle_agent_is_archived(temp_roster, no_triggers):
    record = asyncio.run(temp_roster.create_or_link("Stale"))
    record.last_active = _days_ago(90)
    assert temp_roster.is_archived(record) is True


def test_live_trigger_exempts_an_idle_agent_from_archival(temp_roster, with_live_trigger):
    """A scheduled trigger means the user still wants this work happening."""
    record = asyncio.run(temp_roster.create_or_link("Daily Digest"))
    record.last_active = _days_ago(365)

    assert temp_roster.is_archived(record) is False


def test_get_records_hides_archived_by_default(temp_roster, no_triggers):
    asyncio.run(temp_roster.create_or_link("Fresh"))
    stale = asyncio.run(temp_roster.create_or_link("Stale"))
    stale.last_active = _days_ago(90)

    assert [r.name for r in temp_roster.get_records()] == ["Fresh"]
    assert len(temp_roster.get_records(include_archived=True)) == 2


def test_mark_active_revives_an_archived_agent(temp_roster, no_triggers):
    record = asyncio.run(temp_roster.create_or_link("Stale"))
    record.last_active = _days_ago(90)
    assert temp_roster.is_archived(record) is True

    temp_roster.mark_active("Stale")

    assert temp_roster.is_archived(temp_roster.get_record("Stale")) is False


# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------


def test_merge_redirects_name_resolution(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))

    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["test"])

    assert temp_roster.resolve_name("Duplicate") == "Canonical"
    assert temp_roster.resolve_name("Canonical") == "Canonical"


def test_merged_agent_is_no_longer_addressable(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))

    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["test"])

    assert temp_roster.get_agents() == ["Canonical"]
    assert temp_roster.merged_sources("Canonical") == ["Duplicate"]


def test_merge_preserves_the_absorbed_record(temp_roster, temp_logs):
    """Merging is a pointer, never a deletion - the history must survive."""
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate", description="absorbed thread"))

    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["test"])

    absorbed = temp_roster.get_record("Duplicate")
    assert absorbed is not None
    assert absorbed.description == "absorbed thread"
    assert absorbed.merged_into == "Canonical"


def test_merge_is_a_no_op_for_unknown_agents(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    temp_roster.merge_agent(source_name="Ghost", target_name="Canonical", evidence=[])
    assert temp_roster.get_agents() == ["Canonical"]


def test_chained_merges_preserve_every_ancestor(temp_roster, temp_logs):
    """Regression: A->B then B->C must not lose A's history.

    `resolve_name` already followed the chain, so work correctly reached C. But
    `merged_sources` only matched direct children, so C loaded B's transcript and
    silently dropped A's - breaking the guarantee that merging preserves context,
    in the one direction nobody would notice.
    """
    for name in ("A", "B", "C"):
        asyncio.run(temp_roster.create_or_link(name))

    temp_roster.merge_agent(source_name="A", target_name="B", evidence=["t"])
    temp_roster.merge_agent(source_name="B", target_name="C", evidence=["t"])

    assert temp_roster.resolve_name("A") == "C"
    assert set(temp_roster.merged_sources("C")) == {"A", "B"}


def test_chained_merge_history_reaches_the_final_survivor(temp_roster, temp_logs):
    """The same bug, observed where it actually bites: the execution prompt."""
    from server.agents.execution_agent.agent import ExecutionAgent

    for name in ("A", "B", "C"):
        asyncio.run(temp_roster.create_or_link(name))
    temp_logs.record_agent_response("A", "OLDEST_CONTEXT")
    temp_logs.record_agent_response("B", "MIDDLE_CONTEXT")
    temp_logs.record_agent_response("C", "SURVIVOR_CONTEXT")

    temp_roster.merge_agent(source_name="A", target_name="B", evidence=["t"])
    temp_roster.merge_agent(source_name="B", target_name="C", evidence=["t"])

    prompt = ExecutionAgent("C").build_system_prompt_with_history()

    assert "OLDEST_CONTEXT" in prompt
    assert "MIDDLE_CONTEXT" in prompt
    assert "SURVIVOR_CONTEXT" in prompt


def test_merged_sources_survives_a_cycle(temp_roster):
    """Defensive: a corrupt pointer cycle must not hang history loading."""
    a = asyncio.run(temp_roster.create_or_link("A"))
    b = asyncio.run(temp_roster.create_or_link("B"))
    a.merged_into = "B"
    b.merged_into = "A"

    assert isinstance(temp_roster.merged_sources("A"), list)


def test_resolve_name_terminates_on_a_pointer_cycle(temp_roster):
    """Defensive: a cycle must not hang the request."""
    a = asyncio.run(temp_roster.create_or_link("A"))
    b = asyncio.run(temp_roster.create_or_link("B"))
    a.merged_into = "B"
    b.merged_into = "A"

    assert temp_roster.resolve_name("A") in {"A", "B"}


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


def test_lock_for_is_stable_per_agent_and_distinct_across_agents(temp_roster):
    assert temp_roster.lock_for("Agent A") is temp_roster.lock_for("Agent A")
    assert temp_roster.lock_for("Agent A") is not temp_roster.lock_for("Agent B")


def test_same_agent_executions_serialize():
    """Two turns on one agent must not interleave their critical sections."""

    async def scenario():
        roster = AgentRoster.__new__(AgentRoster)  # bypass disk I/O
        roster._agent_locks = {}
        events = []

        async def worker(tag: str):
            async with roster.lock_for("Shared Agent"):
                events.append(f"{tag}-start")
                await asyncio.sleep(0.01)
                events.append(f"{tag}-end")

        await asyncio.gather(worker("first"), worker("second"))
        return events

    events = asyncio.run(scenario())

    # Each worker's start/end must be adjacent - no interleaving.
    assert events[0].endswith("-start")
    assert events[1] == events[0].replace("-start", "-end")


def test_concurrent_creates_do_not_duplicate_a_record(temp_roster):
    """The create critical section is held under a lock, so one record wins."""

    async def scenario():
        await asyncio.gather(
            *(temp_roster.create_or_link("Same Name") for _ in range(5))
        )

    asyncio.run(scenario())

    assert len(temp_roster.get_records(include_archived=True)) == 1
