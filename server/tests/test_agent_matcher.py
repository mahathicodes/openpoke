"""Retrieval primitives and structural-evidence scoring."""

from __future__ import annotations

import math

import pytest

from server.services.execution import agent_matcher as matcher
from server.services.execution.roster import AgentRecord


# ----------------------------------------------------------------------
# cosine_similarity
# ----------------------------------------------------------------------


def test_cosine_identical_vectors_is_one():
    assert matcher.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert matcher.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_negative_one():
    assert matcher.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_known_angle():
    # 45 degrees between (1,0) and (1,1).
    assert matcher.cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize(
    "a, b",
    [
        ([], []),
        ([1.0], []),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),  # dimension mismatch
        ([0.0, 0.0], [1.0, 1.0]),  # zero vector has no direction
    ],
)
def test_cosine_degenerate_inputs_return_zero(a, b):
    assert matcher.cosine_similarity(a, b) == 0.0


# ----------------------------------------------------------------------
# lexical_overlap (the zero-network fallback)
# ----------------------------------------------------------------------


def test_lexical_overlap_identical_text():
    assert matcher.lexical_overlap("email keith lunch", "email keith lunch") == pytest.approx(1.0)


def test_lexical_overlap_disjoint_text():
    assert matcher.lexical_overlap("alpha beta", "gamma delta") == 0.0


def test_lexical_overlap_is_case_and_punctuation_insensitive():
    assert matcher.lexical_overlap("Keith's Lunch!", "keith s lunch") == pytest.approx(1.0)


def test_lexical_overlap_ranks_related_above_unrelated():
    query = "lunch with keith"
    related = matcher.lexical_overlap(query, "Email to Keith about lunch")
    unrelated = matcher.lexical_overlap(query, "Vercel job offer negotiation")
    assert related > unrelated


# ----------------------------------------------------------------------
# find_candidates
# ----------------------------------------------------------------------


def _record(name: str, description: str = "", embedding=None, model="openai/text-embedding-3-small"):
    return AgentRecord(
        name=name,
        description=description,
        embedding=embedding,
        embedding_model=model if embedding else None,
    )


def test_find_candidates_ranks_by_embedding_similarity():
    records = [
        _record("Far", embedding=[0.0, 1.0]),
        _record("Near", embedding=[1.0, 0.0]),
    ]
    results = matcher.find_candidates(
        query_text="anything",
        query_embedding=[1.0, 0.0],
        records=records,
        top_k=2,
    )
    assert [r.record.name for r in results] == ["Near", "Far"]
    assert results[0].method == "embedding"


def test_find_candidates_respects_top_k():
    records = [_record(f"Agent {i}", embedding=[1.0, 0.0]) for i in range(5)]
    results = matcher.find_candidates(
        query_text="q", query_embedding=[1.0, 0.0], records=records, top_k=2
    )
    assert len(results) == 2


def test_find_candidates_falls_back_to_lexical_without_embeddings():
    records = [
        _record("Email to Keith about lunch", description="lunch planning"),
        _record("Vercel Job Offer", description="offer negotiation"),
    ]
    results = matcher.find_candidates(
        query_text="lunch with keith",
        query_embedding=None,
        records=records,
        top_k=2,
    )
    assert results[0].record.name == "Email to Keith about lunch"
    assert all(r.method == "lexical" for r in results)


def test_find_candidates_ignores_vectors_from_a_different_embedding_model():
    """A stale-model vector must not be scored by cosine.

    Two models can emit same-length vectors whose similarity looks plausible but
    means nothing, so such records fall back to lexical instead.
    """
    stale = _record("Stale", description="lunch with keith", embedding=[1.0, 0.0])
    stale.embedding_model = "some/older-model"

    results = matcher.find_candidates(
        query_text="lunch with keith",
        query_embedding=[1.0, 0.0],
        records=[stale],
        top_k=1,
    )
    assert results[0].method == "lexical"


def test_find_candidates_handles_empty_roster():
    assert matcher.find_candidates(
        query_text="q", query_embedding=[1.0], records=[], top_k=5
    ) == []


# ----------------------------------------------------------------------
# Structural evidence
# ----------------------------------------------------------------------


def test_extract_structural_evidence_separates_id_classes(temp_logs):
    """thread ids and message/draft ids are kept apart - they mean different things."""
    temp_logs.record_action(
        "Agent A",
        'Calling gmail_reply_to_thread with: {"thread_id": "18c9f0aa77bb31", '
        '"message_id": "msg99887766", "recipient_email": "keith@example.com"}',
    )
    found = matcher.extract_structural_evidence("Agent A")

    assert "18c9f0aa77bb31" in found.thread_ids
    assert "msg99887766" in found.object_ids
    assert "18c9f0aa77bb31" not in found.object_ids
    assert "keith@example.com" in found.emails


def test_extract_structural_evidence_empty_for_unknown_agent(temp_logs):
    found = matcher.extract_structural_evidence("Never Existed")
    assert found.thread_ids == set()
    assert found.object_ids == set()
    assert found.emails == set()


def test_shared_message_id_is_not_proof_of_same_thread(temp_logs):
    """Regression: two agents reading one email are not the same thread of work.

    An earlier version pooled thread/message/draft ids into a single set and scored
    any intersection 1.0, so two unrelated searches that happened to surface the
    same email could clear the merge threshold outright. Touching the same object
    is now suggestive, never conclusive.
    """
    for name in ("Search agent", "Summarise agent"):
        temp_logs.record_action(name, 'tool: {"message_id": "msg99887766"}')

    confidence, evidence = matcher.score_structural_evidence("Search agent", "Summarise agent")

    assert confidence < 0.9  # below the merge commit threshold
    assert any("shared-gmail-object" in item for item in evidence)


def test_shared_draft_id_is_not_proof_either(temp_logs):
    for name in ("A", "B"):
        temp_logs.record_action(name, 'tool: {"draft_id": "draft4455667788"}')

    confidence, _evidence = matcher.score_structural_evidence("A", "B")

    assert confidence < 0.9


def test_thread_id_outranks_object_id(temp_logs):
    """Only a shared conversation is conclusive."""
    temp_logs.record_action("A", 'tool: {"thread_id": "18c9f0aa77bb31"}')
    temp_logs.record_action("B", 'tool: {"thread_id": "18c9f0aa77bb31"}')
    thread_confidence, _ = matcher.score_structural_evidence("A", "B")

    temp_logs.record_action("C", 'tool: {"message_id": "msg99887766"}')
    temp_logs.record_action("D", 'tool: {"message_id": "msg99887766"}')
    object_confidence, _ = matcher.score_structural_evidence("C", "D")

    assert thread_confidence == 1.0
    assert object_confidence < thread_confidence


def test_shared_gmail_thread_is_conclusive(temp_logs):
    for name in ("A", "B"):
        temp_logs.record_action(name, 'tool: {"thread_id": "18c9f0aa77bb31"}')

    confidence, evidence = matcher.score_structural_evidence("A", "B")
    assert confidence == 1.0
    assert any("shared-gmail-thread" in item for item in evidence)


def test_shared_correspondent_is_suggestive_but_not_conclusive(temp_logs):
    """Same person, different threads: real signal, but not proof."""
    temp_logs.record_action("A", 'tool: {"recipient_email": "keith@example.com"}')
    temp_logs.record_action("B", 'tool: {"recipient_email": "keith@example.com"}')

    confidence, evidence = matcher.score_structural_evidence("A", "B")
    assert 0.0 < confidence < 1.0
    assert any("shared-correspondent" in item for item in evidence)


def test_no_shared_evidence_scores_zero(temp_logs):
    temp_logs.record_action("A", 'tool: {"thread_id": "aaaaaaaaaaaa"}')
    temp_logs.record_action("B", 'tool: {"thread_id": "bbbbbbbbbbbb"}')

    confidence, evidence = matcher.score_structural_evidence("A", "B")
    assert confidence == 0.0
    assert evidence == []
