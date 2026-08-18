"""Semantic candidate retrieval and evidence-based duplicate resolution for agents.

Two-stage design:

1. Retrieval (deterministic): embedding cosine similarity narrows the roster to a
   handful of candidates. Falls back to lexical overlap when embeddings are
   unavailable, so a provider outage degrades quality rather than breaking a turn.
2. Judgment (non-deterministic): one bounded LLM call over just those candidates.

Crucially, stage 2 can never merge anything on its own. Text similarity is weak
evidence, and acting on it irreversibly is how you corrupt an unrelated thread's
history. It may only stage a `possible_duplicate_of` link. A merge requires
structural proof - the same Gmail thread/message id turning up in both agents'
logs - which is a fact rather than an inference.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import OpenRouterError, request_chat_completion, request_embeddings
from .log_store import get_execution_agent_logs
from .roster import AgentRecord, DuplicateLink


# ----------------------------------------------------------------------
# Similarity primitives
# ----------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 for empty or mismatched input.

    Pure Python on purpose. At personal-assistant roster sizes a brute-force scan
    is microseconds, so numpy or a vector database would be dependency weight for
    no measurable gain.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    return set(_TOKEN_PATTERN.findall((text or "").lower()))


def lexical_overlap(query: str, text: str) -> float:
    """Jaccard overlap between two texts. The zero-network retrieval fallback."""
    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    intersection = len(query_tokens & text_tokens)
    if not intersection:
        return 0.0
    return intersection / len(query_tokens | text_tokens)


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------


async def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed texts, returning None on any failure.

    Never raises: callers treat embeddings as an optimisation and fall back to
    lexical ranking, so a provider hiccup must not block a user's turn.
    """
    if not texts:
        return []

    settings = get_settings()
    if not settings.openrouter_api_key:
        return None

    try:
        return await request_embeddings(
            model=settings.embedding_model,
            texts=texts,
            api_key=settings.openrouter_api_key,
            timeout=settings.embedding_timeout_seconds,
        )
    except OpenRouterError as exc:
        logger.warning(f"Embedding request failed: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Unexpected embedding failure: {exc}")
        return None


async def embed_text(text: str) -> Optional[List[float]]:
    """Embed a single text, or None on failure."""
    vectors = await embed_texts([text])
    if not vectors:
        return None
    return vectors[0]


def agent_embedding_text(name: str, description: str) -> str:
    """The text an agent is indexed under."""
    description = (description or "").strip()
    return f"{name}. {description}" if description else name


# ----------------------------------------------------------------------
# Candidate retrieval
# ----------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    record: AgentRecord
    score: float
    method: str  # "embedding" or "lexical"


def find_candidates(
    *,
    query_text: str,
    query_embedding: Optional[List[float]],
    records: Iterable[AgentRecord],
    top_k: int,
) -> List[ScoredCandidate]:
    """Rank roster records against a query, newest-relevant first.

    Uses embeddings when both sides have a usable vector from the *same* model,
    and lexical overlap otherwise. Mixed rosters are fine: each record is scored
    by whichever method it supports, since a record that has not been re-embedded
    yet should still be findable.
    """
    settings = get_settings()
    current_model = settings.embedding_model

    scored: List[ScoredCandidate] = []
    for record in records:
        usable_vector = (
            query_embedding
            and record.embedding
            # A stale-model vector is not comparable: different models can emit
            # same-length vectors whose cosine scores look plausible but mean
            # nothing. Fall back to lexical rather than trust a bogus number.
            and record.embedding_model == current_model
        )

        if usable_vector:
            score = cosine_similarity(query_embedding, record.embedding)
            method = "embedding"
        else:
            score = lexical_overlap(query_text, agent_embedding_text(record.name, record.description))
            method = "lexical"

        scored.append(ScoredCandidate(record=record, score=score, method=method))

    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    return scored[: max(top_k, 0)]


# ----------------------------------------------------------------------
# LLM judgment (may stage a link; may never merge)
# ----------------------------------------------------------------------


_DECISION_TOOL_NAME = "record_agent_routing_decision"

_DECISION_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _DECISION_TOOL_NAME,
        "description": (
            "Record how a new task should be routed: describe the new agent, and "
            "optionally flag an existing agent it may be a duplicate of."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "One-sentence description of this agent's ongoing thread of work "
                        "(who it involves and what it is about). Used for future matching."
                    ),
                },
                "possible_duplicate_of": {
                    "type": "string",
                    "description": (
                        "Exact name of an existing agent this task probably continues, or "
                        "omit entirely when it is new work. Only name an agent when it is "
                        "plausibly the SAME ongoing thread, not merely the same person or topic."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "0.0-1.0 confidence in possible_duplicate_of. Use below 0.5 when the "
                        "link is speculative. Omit when no duplicate is named."
                    ),
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    },
}

_DECISION_SYSTEM_PROMPT = (
    "You route tasks for a personal-assistant system that keeps one long-lived agent per "
    "ongoing thread of work (for example: an email thread with one person about one subject).\n\n"
    "Given a new task and a shortlist of existing agents, write a short description for the new "
    "agent, and decide whether it plausibly continues one of the existing threads.\n\n"
    "Be conservative when naming a duplicate. Two tasks involving the same person are NOT the "
    "same thread when they concern different subjects: 'lunch with Keith' and 'Keith's contract "
    "review' are separate threads. Wrongly linking distinct threads mixes unrelated private "
    "context together, which is far more damaging than briefly tracking two agents. When in "
    "doubt, name no duplicate at all.\n\n"
    "Nothing you return merges anything immediately: a named duplicate is only a hypothesis that "
    "will be confirmed later by hard evidence."
)


@dataclass
class RoutingDecision:
    description: str
    duplicate_of: Optional[str] = None
    confidence: float = 0.0


def _coerce_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _format_candidates(candidates: List[ScoredCandidate]) -> str:
    lines = []
    for candidate in candidates:
        description = candidate.record.description or "(no description)"
        lines.append(
            f'- name: "{candidate.record.name}" | description: {description} '
            f"| similarity: {candidate.score:.2f} | last_active: {candidate.record.last_active}"
        )
    return "\n".join(lines)


async def decide_routing(
    *,
    instructions: str,
    proposed_name: str,
    candidates: List[ScoredCandidate],
) -> RoutingDecision:
    """Ask the LLM to describe the new agent and optionally flag a duplicate.

    Returns a description-only decision if the call fails - degrading to "create a
    new agent with no link" is always safe, because a link is only a hypothesis.
    """
    settings = get_settings()
    fallback = RoutingDecision(description="")

    if not settings.openrouter_api_key:
        return fallback

    user_payload = (
        f"New task assigned to proposed agent name: {proposed_name}\n\n"
        f"Task instructions:\n{instructions}\n\n"
        f"Existing agents (shortlist):\n{_format_candidates(candidates) or '(none)'}"
    )

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_payload}],
            system=_DECISION_SYSTEM_PROMPT,
            api_key=settings.openrouter_api_key,
            tools=[_DECISION_TOOL_SCHEMA],
            # This call returns one short tool invocation. Without a cap OpenRouter
            # reserves the model's entire output window against the account.
            max_tokens=512,
        )
    except OpenRouterError as exc:
        logger.warning(f"Agent routing decision failed: {exc}")
        return fallback
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Unexpected agent routing failure: {exc}")
        return fallback

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    for tool_call in message.get("tool_calls") or []:
        function_block = tool_call.get("function") or {}
        if function_block.get("name") != _DECISION_TOOL_NAME:
            continue

        arguments = _coerce_arguments(function_block.get("arguments"))
        if arguments is None:
            logger.warning("Agent routing tool returned invalid arguments")
            return fallback

        duplicate_of = arguments.get("possible_duplicate_of")
        if not isinstance(duplicate_of, str) or not duplicate_of.strip():
            duplicate_of = None
        else:
            # Only trust a name that actually exists in the shortlist we supplied;
            # a hallucinated name must not create a link to nothing.
            known = {candidate.record.name for candidate in candidates}
            duplicate_of = duplicate_of.strip() if duplicate_of.strip() in known else None

        try:
            confidence = float(arguments.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return RoutingDecision(
            description=str(arguments.get("description") or "").strip(),
            duplicate_of=duplicate_of,
            confidence=max(0.0, min(1.0, confidence)),
        )

    return fallback


# ----------------------------------------------------------------------
# Structural evidence and link reconciliation
# ----------------------------------------------------------------------

# Gmail identifiers as they appear in logged tool arguments/responses, e.g.
#   Calling gmail_reply_to_thread with: {"thread_id": "18c9f0...", ...}
#
# thread_id is kept SEPARATE from message/draft ids, and the distinction is
# load-bearing. A thread is the conversation itself, so two agents working the same
# thread really are the same thread of work. A message or draft id is just an object
# both agents happened to touch: two unrelated searches can easily surface the same
# email, and one agent forwarding a message another agent read says nothing about
# them being the same ongoing task. Pooling them - as an earlier version did - meant
# a single incidentally shared message could trigger an irreversible merge.
_THREAD_ID_PATTERN = re.compile(r"\bthread_?id\b\W{0,4}([A-Za-z0-9_-]{8,})", re.IGNORECASE)
_OBJECT_ID_PATTERN = re.compile(
    r"\b(?:message_?id|draft_?id)\b\W{0,4}([A-Za-z0-9_-]{8,})", re.IGNORECASE
)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass
class StructuralEvidence:
    """Identifiers found in one agent's execution log, kept by strength."""

    thread_ids: Set[str] = field(default_factory=set)
    object_ids: Set[str] = field(default_factory=set)
    emails: Set[str] = field(default_factory=set)


def extract_structural_evidence(agent_name: str) -> StructuralEvidence:
    """Pull Gmail identifiers out of an agent's execution log.

    These come from the calling agent's own log entries, which the execution
    runtime writes for every tool call it makes. Note the runtime truncates logged
    arguments and responses, so this is best-effort: evidence can be missed. That
    is why a *missing* identifier never blocks anything, and only a positive
    thread-id match is allowed to authorise a merge.
    """
    found = StructuralEvidence()

    try:
        for _tag, _timestamp, payload in get_execution_agent_logs().iter_entries(agent_name):
            for match in _THREAD_ID_PATTERN.finditer(payload):
                found.thread_ids.add(match.group(1))
            for match in _OBJECT_ID_PATTERN.finditer(payload):
                found.object_ids.add(match.group(1))
            for match in _EMAIL_PATTERN.finditer(payload):
                found.emails.add(match.group(0).lower())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to read structural evidence for '{agent_name}': {exc}")

    return found


def score_structural_evidence(agent_a: str, agent_b: str) -> Tuple[float, List[str]]:
    """Confidence that two agents cover the same real-world thread.

    Only a shared *thread* id is treated as proof. Everything else is suggestive
    and deliberately scores below the merge threshold, so it can raise a link's
    confidence for later review without ever committing a merge on its own:

        shared thread id        1.0   proof - same conversation
        shared message/draft    0.6   both touched one email; not the same thread
        shared correspondent    0.5   same person, possibly unrelated threads
    """
    a = extract_structural_evidence(agent_a)
    b = extract_structural_evidence(agent_b)

    shared_threads = a.thread_ids & b.thread_ids
    shared_objects = a.object_ids & b.object_ids
    shared_emails = a.emails & b.emails

    evidence: List[str] = []
    confidence = 0.0

    if shared_threads:
        confidence = 1.0
        evidence.append(f"shared-gmail-thread: {', '.join(sorted(shared_threads)[:3])}")
    elif shared_objects:
        confidence = 0.6
        evidence.append(f"shared-gmail-object: {', '.join(sorted(shared_objects)[:3])}")
    elif shared_emails:
        confidence = 0.5
        evidence.append(f"shared-correspondent: {', '.join(sorted(shared_emails)[:3])}")

    return confidence, evidence


def reconcile_link(agent_name: str) -> Optional[str]:
    """Re-score an agent's staged duplicate link, merging when proof arrives.

    Returns the canonical agent name when a merge happened, else None. Merging is
    a metadata pointer, never a deletion: both logs stay on disk and the canonical
    agent reads them together.
    """
    from .roster import get_agent_roster

    roster = get_agent_roster()
    record = roster.get_record(agent_name)
    if record is None or record.possible_duplicate_of is None:
        return None

    link = record.possible_duplicate_of
    target = roster.get_record(link.name)
    if target is None:
        roster.set_duplicate_link(agent_name, None)
        return None

    structural_confidence, evidence = score_structural_evidence(agent_name, link.name)
    if structural_confidence <= 0.0:
        return None

    # Confidence is taken from structural evidence ALONE - never combined with, or
    # maxed against, the LLM's text-similarity score. Text similarity decides
    # whether to look for a duplicate; only structural proof decides whether to
    # merge one. Letting the text score contribute here would mean a sufficiently
    # confident LLM could merge two threads on resemblance, which is exactly the
    # silent-corruption failure this design exists to prevent.
    #
    # Note this can revise confidence *downward*: an overconfident text guess of
    # 0.95 backed by only a shared correspondent is honestly a 0.6.
    updated = DuplicateLink(
        name=link.name,
        confidence=structural_confidence,
        evidence=sorted(set(link.evidence) | set(evidence)),
    )

    threshold = get_settings().agent_merge_commit_threshold
    if updated.confidence < threshold:
        roster.set_duplicate_link(agent_name, updated)
        logger.info(
            f"Duplicate link '{agent_name}' -> '{link.name}' now at "
            f"{updated.confidence:.2f} (need {threshold:.2f})"
        )
        return None

    roster.merge_agent(source_name=agent_name, target_name=link.name, evidence=updated.evidence)
    logger.info(f"Merged agent '{agent_name}' into '{link.name}' on {updated.evidence}")
    return link.name


def reconcile_around(agent_name: str) -> List[str]:
    """Re-check every staged link this agent could settle, in either direction.

    Evidence is symmetric - a shared Gmail thread id proves two agents are the same
    conversation regardless of whose log it landed in - but a link is stored only on
    the source agent. Reconciling just the agent that finished therefore misses the
    common case entirely: the newer agent holds the link and runs immediately, while
    the older target may be idle for days before a trigger wakes it and logs the
    deciding id. At that point nothing was looking.

    Returns the canonical names of any merges performed.
    """
    from .roster import get_agent_roster

    merged: List[str] = []

    # Forward: this agent holds a link of its own.
    target = reconcile_link(agent_name)
    if target:
        merged.append(target)

    # Reverse: other agents are waiting on evidence that may have just arrived here.
    for source in get_agent_roster().links_pointing_at(agent_name):
        target = reconcile_link(source)
        if target:
            merged.append(target)

    return merged


__all__ = [
    "ScoredCandidate",
    "RoutingDecision",
    "reconcile_around",
    "agent_embedding_text",
    "cosine_similarity",
    "decide_routing",
    "embed_text",
    "embed_texts",
    "extract_structural_evidence",
    "find_candidates",
    "lexical_overlap",
    "reconcile_link",
    "score_structural_evidence",
]
