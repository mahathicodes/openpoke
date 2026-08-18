"""The routing ladder in send_message_to_agent, with all network calls stubbed.

Covers every branch: exact-match fast path, similarity floor, staged linking,
embedding failure, and the promotion of a link into a merge.
"""

from __future__ import annotations

import asyncio

import pytest

from server.services.execution import agent_matcher as matcher
from server.services.execution.roster import DuplicateLink
from server.agents.interaction_agent import tools as interaction_tools
from conftest import make_embedding_response, make_tool_call_response


@pytest.fixture
def captured_dispatch(monkeypatch):
    """Capture dispatches instead of spawning real execution agents."""
    calls = []

    def fake_dispatch(agent_name, instructions, *, action, ambiguous_with=None):
        calls.append(
            {
                "agent_name": agent_name,
                "instructions": instructions,
                "action": action,
                "ambiguous_with": ambiguous_with,
            }
        )
        return interaction_tools.ToolResult(
            success=True,
            payload={"status": "submitted", "agent_name": agent_name},
        )

    monkeypatch.setattr(interaction_tools, "_dispatch_to_agent", fake_dispatch)
    return calls


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Return a fixed vector for every embedding request; count the calls."""
    state = {"calls": 0}

    async def fake_request_embeddings(*, model, texts, api_key, timeout, **kwargs):
        state["calls"] += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(matcher, "request_embeddings", fake_request_embeddings)
    monkeypatch.setattr(matcher, "get_settings", _settings_with_key(monkeypatch))
    return state


def _settings_with_key(monkeypatch):
    """Settings with an API key present, so matcher code paths are not skipped."""
    from server.config import get_settings

    real = get_settings()

    class _Settings:
        openrouter_api_key = "test-key"
        embedding_model = real.embedding_model
        embedding_timeout_seconds = real.embedding_timeout_seconds
        interaction_agent_model = real.interaction_agent_model
        agent_dedup_top_k = real.agent_dedup_top_k
        agent_min_candidate_similarity = real.agent_min_candidate_similarity
        agent_merge_commit_threshold = real.agent_merge_commit_threshold
        agent_archive_after_days = real.agent_archive_after_days
        agent_ambiguity_margin = real.agent_ambiguity_margin

    return lambda: _Settings()


@pytest.fixture
def stub_settings(monkeypatch):
    monkeypatch.setattr(interaction_tools, "get_settings", _settings_with_key(monkeypatch))


def _give_embedding(roster, name, vector):
    """Persist a vector for an agent.

    Must go through update_embedding rather than mutating the record: routing
    reloads the roster from disk first, which would discard an in-memory change.
    """
    roster.update_embedding(name, vector, matcher.get_settings().embedding_model)


# ----------------------------------------------------------------------
# (1) Exact-match fast path
# ----------------------------------------------------------------------


def test_exact_name_match_reuses_without_any_network_call(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings
):
    """The common case - the model reuses a name it just used - must stay free."""
    asyncio.run(temp_roster.create_or_link("Email to Keith"))
    embedding_calls_before = stub_embeddings["calls"]

    asyncio.run(interaction_tools.send_message_to_agent("Email to Keith", "follow up please"))

    assert stub_embeddings["calls"] == embedding_calls_before  # no embedding needed
    assert captured_dispatch[-1]["action"] == "reused"
    assert captured_dispatch[-1]["agent_name"] == "Email to Keith"


def test_exact_match_on_a_merged_agent_redirects_to_the_survivor(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings
):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_roster.merge_agent(source_name="Duplicate", target_name="Canonical", evidence=["test"])

    asyncio.run(interaction_tools.send_message_to_agent("Duplicate", "do the thing"))

    assert captured_dispatch[-1]["agent_name"] == "Canonical"


# ----------------------------------------------------------------------
# (2) Empty roster / similarity floor
# ----------------------------------------------------------------------


def test_first_agent_ever_is_created_without_a_judgment_call(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings, monkeypatch
):
    """Empty roster: there is nothing to compare against, so skip the LLM entirely."""
    called = {"judgment": False}

    async def fake_decide(**kwargs):
        called["judgment"] = True
        return matcher.RoutingDecision(description="")

    monkeypatch.setattr(interaction_tools, "decide_routing", fake_decide)

    asyncio.run(interaction_tools.send_message_to_agent("Brand New", "start something"))

    assert called["judgment"] is False
    assert captured_dispatch[-1]["action"] == "created"
    assert temp_roster.get_record("Brand New") is not None


def test_dissimilar_candidates_skip_the_judgment_call(
    temp_roster, temp_logs, captured_dispatch, stub_settings, monkeypatch
):
    """Below the similarity floor nothing is plausibly related - do not pay for an LLM call."""
    asyncio.run(temp_roster.create_or_link("Unrelated", description="something else"))

    # Orthogonal vectors => similarity 0.0, far below the floor.
    async def fake_request_embeddings(*, model, texts, api_key, timeout, **kwargs):
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(matcher, "request_embeddings", fake_request_embeddings)
    monkeypatch.setattr(matcher, "get_settings", _settings_with_key(monkeypatch))

    _give_embedding(temp_roster, "Unrelated", [1.0, 0.0])

    called = {"judgment": False}

    async def fake_decide(**kwargs):
        called["judgment"] = True
        return matcher.RoutingDecision(description="")

    monkeypatch.setattr(interaction_tools, "decide_routing", fake_decide)

    asyncio.run(interaction_tools.send_message_to_agent("Totally New", "unrelated work"))

    assert called["judgment"] is False
    assert captured_dispatch[-1]["action"] == "created"


# ----------------------------------------------------------------------
# (3) Staged linking - never an immediate merge
# ----------------------------------------------------------------------


def test_similar_task_stages_a_link_but_does_not_merge(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings, monkeypatch
):
    """The core safety property: text similarity alone must never merge histories."""
    asyncio.run(temp_roster.create_or_link("Email to Keith about lunch"))
    _give_embedding(temp_roster, "Email to Keith about lunch", [1.0, 0.0])

    async def fake_decide(**kwargs):
        return matcher.RoutingDecision(
            description="lunch thread with keith",
            duplicate_of="Email to Keith about lunch",
            confidence=0.85,
        )

    monkeypatch.setattr(interaction_tools, "decide_routing", fake_decide)

    asyncio.run(interaction_tools.send_message_to_agent("Lunch w Keith", "did keith reply?"))

    created = temp_roster.get_record("Lunch w Keith")
    assert created is not None, "a separate agent should still be created"
    assert created.possible_duplicate_of is not None
    assert created.possible_duplicate_of.name == "Email to Keith about lunch"
    # Both agents remain independently addressable until proof arrives.
    assert created.merged_into is None
    assert len(temp_roster.get_agents()) == 2


def test_hallucinated_duplicate_name_is_ignored(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings, monkeypatch
):
    """A link may only point at an agent that was actually on the shortlist."""
    asyncio.run(temp_roster.create_or_link("Real Agent"))
    _give_embedding(temp_roster, "Real Agent", [1.0, 0.0])

    async def fake_request_chat_completion(**kwargs):
        return make_tool_call_response(
            "record_agent_routing_decision",
            {
                "description": "some thread",
                "possible_duplicate_of": "Agent That Does Not Exist",
                "confidence": 0.95,
            },
        )

    monkeypatch.setattr(matcher, "request_chat_completion", fake_request_chat_completion)

    asyncio.run(interaction_tools.send_message_to_agent("New Thing", "do work"))

    created = temp_roster.get_record("New Thing")
    assert created.possible_duplicate_of is None


# ----------------------------------------------------------------------
# (3b) Ambiguity - two equally good matches
# ----------------------------------------------------------------------


@pytest.fixture
def two_identical_matches(temp_roster, monkeypatch):
    """Two different people, same first name, same topic - indistinguishable.

    The canonical ambiguous case: "follow up on lunch w/ Keith" when the roster
    holds a Keith Rivera lunch thread and a Keith Chen lunch thread.
    """

    async def fake_request_embeddings(*, model, texts, api_key=None, timeout=None, **kwargs):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(matcher, "request_embeddings", fake_request_embeddings)
    monkeypatch.setattr(matcher, "get_settings", _settings_with_key(monkeypatch))

    for name in ("Keith Rivera lunch", "Keith Chen lunch"):
        asyncio.run(temp_roster.create_or_link(name, description=f"lunch with {name}"))
        _give_embedding(temp_roster, name, [1.0, 0.0])  # identical -> tied scores
    return temp_roster


def test_tied_candidates_start_no_work(
    two_identical_matches, temp_logs, captured_dispatch, stub_settings, monkeypatch
):
    """Asking "which Keith?" while already emailing one of them would be theatre.

    Nothing is dispatched, so an irreversible action cannot race the clarification.
    """

    async def fake_decide(**kwargs):
        # Even a confident judge must not get the chance to break the tie.
        return matcher.RoutingDecision(
            description="lunch thread", duplicate_of="Keith Rivera lunch", confidence=0.95
        )

    monkeypatch.setattr(interaction_tools, "decide_routing", fake_decide)

    result = asyncio.run(
        interaction_tools.send_message_to_agent("Lunch w Keith", "follow up on lunch")
    )

    assert result.payload["status"] == "needs_clarification"
    assert captured_dispatch == []  # no execution agent started


def test_tied_candidates_create_no_agent_and_no_link(
    two_identical_matches, temp_logs, captured_dispatch, stub_settings, monkeypatch
):
    """Holding must not leave a half-built third agent behind."""
    asyncio.run(interaction_tools.send_message_to_agent("Lunch w Keith", "follow up"))

    assert two_identical_matches.get_record("Lunch w Keith") is None
    assert set(two_identical_matches.get_agents()) == {
        "Keith Rivera lunch",
        "Keith Chen lunch",
    }


def test_tied_candidates_are_reported_for_clarification(
    two_identical_matches, temp_logs, captured_dispatch, stub_settings
):
    """The model needs the names, and needs to know nothing was started."""
    result = asyncio.run(
        interaction_tools.send_message_to_agent("Lunch w Keith", "follow up")
    )

    assert set(result.payload["ambiguous_with"]) == {"Keith Rivera lunch", "Keith Chen lunch"}
    note = result.payload["note"].lower()
    assert "no work has been started" in note
    assert "ask the user" in note


def test_clarified_followup_routes_by_exact_match(
    two_identical_matches, temp_logs, captured_dispatch, stub_settings
):
    """The resume path: the user's answer is itself the mechanism.

    No pending state is parked anywhere - naming the agent hits the exact-match
    fast path, which is why holding the work needs no extra machinery.
    """
    asyncio.run(interaction_tools.send_message_to_agent("Lunch w Keith", "follow up"))
    assert captured_dispatch == []

    # User replies "Chen"; the model calls again with that agent's exact name.
    asyncio.run(interaction_tools.send_message_to_agent("Keith Chen lunch", "follow up"))

    assert captured_dispatch[-1]["agent_name"] == "Keith Chen lunch"
    assert captured_dispatch[-1]["action"] == "reused"


def test_clear_winner_still_links_normally(
    temp_roster, temp_logs, captured_dispatch, stub_embeddings, stub_settings, monkeypatch
):
    """Ambiguity handling must not suppress ordinary confident matches."""
    asyncio.run(temp_roster.create_or_link("Keith lunch", description="lunch with keith"))
    _give_embedding(temp_roster, "Keith lunch", [1.0, 0.0])
    asyncio.run(temp_roster.create_or_link("Unrelated", description="something else"))
    _give_embedding(temp_roster, "Unrelated", [0.0, 1.0])  # orthogonal -> no tie

    async def fake_decide(**kwargs):
        return matcher.RoutingDecision(
            description="lunch", duplicate_of="Keith lunch", confidence=0.9
        )

    monkeypatch.setattr(interaction_tools, "decide_routing", fake_decide)

    asyncio.run(interaction_tools.send_message_to_agent("Lunch f/u", "did keith reply"))

    assert temp_roster.get_record("Lunch f/u").possible_duplicate_of is not None


# ----------------------------------------------------------------------
# (4) Embedding failure fallback
# ----------------------------------------------------------------------


def test_embedding_failure_still_creates_the_agent(
    temp_roster, temp_logs, captured_dispatch, stub_settings, monkeypatch
):
    """A provider outage must not block delegation; it just skips linking."""

    async def failing_embeddings(**kwargs):
        raise matcher.OpenRouterError("provider down")

    monkeypatch.setattr(matcher, "request_embeddings", failing_embeddings)
    monkeypatch.setattr(matcher, "get_settings", _settings_with_key(monkeypatch))

    asyncio.run(interaction_tools.send_message_to_agent("Resilient", "do work anyway"))

    assert captured_dispatch[-1]["action"] == "created"
    assert temp_roster.get_record("Resilient") is not None


def test_embedding_failure_leaves_no_stale_model_tag(
    temp_roster, temp_logs, captured_dispatch, stub_settings, monkeypatch
):
    async def failing_embeddings(**kwargs):
        raise matcher.OpenRouterError("provider down")

    monkeypatch.setattr(matcher, "request_embeddings", failing_embeddings)
    monkeypatch.setattr(matcher, "get_settings", _settings_with_key(monkeypatch))

    asyncio.run(interaction_tools.send_message_to_agent("No Vector", "work"))

    record = temp_roster.get_record("No Vector")
    assert record.embedding is None
    assert record.embedding_model is None


# ----------------------------------------------------------------------
# (5) Evidence promotes a link to a merge
# ----------------------------------------------------------------------


def test_shared_thread_id_promotes_a_staged_link_to_a_merge(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_roster.set_duplicate_link(
        "Duplicate", DuplicateLink(name="Canonical", confidence=0.5, evidence=["text-similarity"])
    )

    for name in ("Canonical", "Duplicate"):
        temp_logs.record_action(name, 'tool: {"thread_id": "18c9f0aa77bb31"}')

    merged_into = matcher.reconcile_link("Duplicate")

    assert merged_into == "Canonical"
    assert temp_roster.resolve_name("Duplicate") == "Canonical"


@pytest.mark.parametrize("staged_confidence", [0.0, 0.5, 0.85, 0.9, 0.95, 0.99])
def test_llm_confidence_can_never_drive_a_merge(temp_roster, temp_logs, staged_confidence):
    """Regression: a confident LLM must not merge threads on weak evidence.

    `reconcile_link` previously took max(text_confidence, structural_confidence),
    so a 0.95-confident text guess plus a merely-shared correspondent cleared the
    0.9 commit bar and merged two provably different Gmail threads. Confidence now
    comes from structural evidence alone. Parameterised across the confidence range
    because the original test hardcoded 0.5 and so never reached the failing region.
    """
    asyncio.run(temp_roster.create_or_link("Keith lunch plans"))
    asyncio.run(temp_roster.create_or_link("Keith contract review"))
    temp_roster.set_duplicate_link(
        "Keith contract review",
        DuplicateLink(
            name="Keith lunch plans",
            confidence=staged_confidence,
            evidence=[f"text-similarity: {staged_confidence}"],
        ),
    )

    # Same person, but demonstrably DIFFERENT Gmail threads.
    temp_logs.record_action(
        "Keith lunch plans",
        'tool: {"recipient_email": "keith@example.com", "thread_id": "AAAA1111lunch"}',
    )
    temp_logs.record_action(
        "Keith contract review",
        'tool: {"recipient_email": "keith@example.com", "thread_id": "BBBB2222contract"}',
    )

    assert matcher.reconcile_link("Keith contract review") is None
    assert temp_roster.get_record("Keith contract review").merged_into is None
    assert temp_roster.resolve_name("Keith contract review") == "Keith contract review"


@pytest.mark.parametrize("staged_confidence", [0.0, 0.5, 0.95, 0.99])
def test_shared_thread_id_merges_regardless_of_staged_confidence(
    temp_roster, temp_logs, staged_confidence
):
    """The converse: structural proof merges even when the text guess was weak."""
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_roster.set_duplicate_link(
        "Duplicate", DuplicateLink(name="Canonical", confidence=staged_confidence, evidence=[])
    )

    for name in ("Canonical", "Duplicate"):
        temp_logs.record_action(name, 'tool: {"thread_id": "18c9f0aa77bb31"}')

    assert matcher.reconcile_link("Duplicate") == "Canonical"


def test_weak_evidence_revises_an_overconfident_guess_downward(temp_roster, temp_logs):
    """Weak structural evidence should correct a text-based overestimate, not ratify it."""
    asyncio.run(temp_roster.create_or_link("Keith lunch"))
    asyncio.run(temp_roster.create_or_link("Keith invoice"))
    temp_roster.set_duplicate_link(
        "Keith invoice", DuplicateLink(name="Keith lunch", confidence=0.95, evidence=[])
    )

    for name in ("Keith lunch", "Keith invoice"):
        temp_logs.record_action(name, 'tool: {"recipient_email": "keith@example.com"}')

    matcher.reconcile_link("Keith invoice")

    # 0.95 was the LLM's opinion; a shared correspondent is honestly worth 0.5.
    assert temp_roster.get_record("Keith invoice").possible_duplicate_of.confidence == 0.5


def test_weak_evidence_alone_never_merges(temp_roster, temp_logs):
    """Same correspondent, different threads: raises confidence but stays separate."""
    asyncio.run(temp_roster.create_or_link("Keith Lunch"))
    asyncio.run(temp_roster.create_or_link("Keith Contract"))
    temp_roster.set_duplicate_link(
        "Keith Contract",
        DuplicateLink(name="Keith Lunch", confidence=0.5, evidence=["text-similarity"]),
    )

    for name in ("Keith Lunch", "Keith Contract"):
        temp_logs.record_action(name, 'tool: {"recipient_email": "keith@example.com"}')

    merged_into = matcher.reconcile_link("Keith Contract")

    assert merged_into is None
    assert temp_roster.get_record("Keith Contract").merged_into is None
    # The link is retained for future reconciliation, scored by the evidence that
    # actually exists rather than by the text guess that proposed it.
    link = temp_roster.get_record("Keith Contract").possible_duplicate_of
    assert link is not None
    assert link.confidence < 0.9  # below the merge bar
    assert any("shared-correspondent" in item for item in link.evidence)


def test_evidence_arriving_on_the_target_side_still_merges(temp_roster, temp_logs):
    """Regression: reconciliation must not depend on which agent logged the evidence.

    A link lives on the source agent, so reconciling only the agent that just ran
    misses the common ordering - the newer agent carries the link and runs at once,
    while the older target sits idle until a trigger wakes it and logs the deciding
    thread id. Previously nothing was watching at that moment and the provable
    duplicate stayed unmerged forever.
    """
    asyncio.run(temp_roster.create_or_link("Keith lunch"))          # older target
    asyncio.run(temp_roster.create_or_link("Keith lunch f/u"))      # newer source
    temp_roster.set_duplicate_link(
        "Keith lunch f/u", DuplicateLink(name="Keith lunch", confidence=0.8, evidence=["text"])
    )

    # Source runs first; the target has logged nothing yet, so no evidence exists.
    temp_logs.record_action("Keith lunch f/u", 'tool: {"thread_id": "threadSHARED123"}')
    assert matcher.reconcile_around("Keith lunch f/u") == []

    # Days later the target runs and logs the same thread.
    temp_logs.record_action("Keith lunch", 'tool: {"thread_id": "threadSHARED123"}')
    merged = matcher.reconcile_around("Keith lunch")

    assert merged == ["Keith lunch"]
    assert temp_roster.resolve_name("Keith lunch f/u") == "Keith lunch"


def test_reconcile_around_still_handles_the_forward_direction(temp_roster, temp_logs):
    """The original ordering must keep working."""
    asyncio.run(temp_roster.create_or_link("Canonical"))
    asyncio.run(temp_roster.create_or_link("Duplicate"))
    temp_roster.set_duplicate_link(
        "Duplicate", DuplicateLink(name="Canonical", confidence=0.8, evidence=["text"])
    )

    temp_logs.record_action("Canonical", 'tool: {"thread_id": "threadAA11"}')
    temp_logs.record_action("Duplicate", 'tool: {"thread_id": "threadAA11"}')

    assert matcher.reconcile_around("Duplicate") == ["Canonical"]


def test_reconcile_around_is_quiet_when_nothing_is_staged(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Solo"))
    temp_logs.record_action("Solo", 'tool: {"thread_id": "threadBB22"}')

    assert matcher.reconcile_around("Solo") == []


def test_reconcile_is_a_no_op_without_a_staged_link(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Solo"))
    assert matcher.reconcile_link("Solo") is None


def test_link_to_a_vanished_agent_is_cleared(temp_roster, temp_logs):
    asyncio.run(temp_roster.create_or_link("Orphan"))
    temp_roster.set_duplicate_link(
        "Orphan", DuplicateLink(name="Gone", confidence=0.8, evidence=[])
    )

    assert matcher.reconcile_link("Orphan") is None
    assert temp_roster.get_record("Orphan").possible_duplicate_of is None
