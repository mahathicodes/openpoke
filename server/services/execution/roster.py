"""Agent roster: records, semantic-dedup bookkeeping, and recency-based retirement."""

from __future__ import annotations

import asyncio
import json
import fcntl
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...logging_config import logger
# Reuse the log store's slug function rather than redefining it: the collision we
# guard against below is precisely a collision in *its* filenames, so the two must
# never drift apart.
from .log_store import _slugify


UTC = timezone.utc


def _utc_now_iso() -> str:
    # Microseconds, not seconds: `last_active` orders the recency slice of the
    # prompt roster, and at second granularity a burst of activity produces
    # identical timestamps, so the sort silently degrades to insertion order and
    # the most recently touched agent can be dropped from the prompt.
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class DuplicateLink:
    """A staged 'these might be the same thread' hypothesis.

    Text similarity alone never merges agents; it only records a link like this.
    Confidence rises when structural evidence (a shared Gmail thread/message id)
    turns up in both agents' logs, and only then can a real merge happen.
    """

    name: str
    confidence: float
    evidence: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> Optional["DuplicateLink"]:
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        if not isinstance(name, str) or not name:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = data.get("evidence")
        return cls(
            name=name,
            confidence=confidence,
            evidence=[str(item) for item in evidence] if isinstance(evidence, list) else [],
        )


@dataclass
class AgentRecord:
    """One execution agent's roster entry."""

    name: str
    description: str = ""
    embedding: Optional[List[float]] = None
    # Which model produced `embedding`. Two models can emit same-length vectors, so
    # without this tag a model swap yields plausible-looking but meaningless scores.
    embedding_model: Optional[str] = None
    possible_duplicate_of: Optional[DuplicateLink] = None
    # Set once this agent has been confirmed a duplicate of another. The record and
    # its log file are kept; it simply stops being addressable in its own right.
    merged_into: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    last_active: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["possible_duplicate_of"] = (
            asdict(self.possible_duplicate_of) if self.possible_duplicate_of else None
        )
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> Optional["AgentRecord"]:
        # Legacy format: roster.json used to be a plain list of name strings.
        if isinstance(data, str):
            return cls(name=data) if data else None

        if not isinstance(data, dict):
            return None

        name = data.get("name")
        if not isinstance(name, str) or not name:
            return None

        embedding = data.get("embedding")
        if isinstance(embedding, list):
            try:
                vector: Optional[List[float]] = [float(value) for value in embedding]
            except (TypeError, ValueError):
                vector = None
        else:
            vector = None

        created_at = data.get("created_at")
        last_active = data.get("last_active")

        return cls(
            name=name,
            description=str(data.get("description") or ""),
            embedding=vector,
            embedding_model=data.get("embedding_model") or None,
            possible_duplicate_of=DuplicateLink.from_dict(data.get("possible_duplicate_of")),
            merged_into=data.get("merged_into") or None,
            created_at=created_at if isinstance(created_at, str) else _utc_now_iso(),
            last_active=last_active if isinstance(last_active, str) else _utc_now_iso(),
        )


class AgentRoster:
    """Stores agent records in a JSON file, keyed by name."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._records: list[AgentRecord] = []
        # Guards the check-then-create-or-link sequence so two concurrent
        # delegations can't both miss each other and create duplicate records.
        self._write_lock = asyncio.Lock()
        # Per-agent locks so a single agent's executions serialize against each
        # other while unrelated agents still run in parallel. Mirrors the pattern
        # in ExecutionAgentLogStore._lock_for.
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load agent records from roster.json, tolerating the legacy format."""
        if not self._roster_path.exists():
            self._records = []
            self.save()
            return

        try:
            with open(self._roster_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load roster.json: {exc}")
            self._records = []
            return

        if not isinstance(data, list):
            self._records = []
            return

        records: list[AgentRecord] = []
        for entry in data:
            record = AgentRecord.from_dict(entry)
            if record is not None:
                records.append(record)
        self._records = records

    def save(self) -> None:
        """Save agent records to roster.json with file locking."""
        max_retries = 5
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                self._roster_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self._roster_path, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        json.dump([record.to_dict() for record in self._records], f, indent=2)
                        return
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except BlockingIOError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.warning("Failed to acquire lock on roster.json after retries")
            except Exception as exc:
                logger.warning(f"Failed to save roster.json: {exc}")
                break

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_record(self, agent_name: str) -> Optional[AgentRecord]:
        """Exact-name lookup."""
        for record in self._records:
            if record.name == agent_name:
                return record
        return None

    def get_records(self, *, include_archived: bool = False) -> list[AgentRecord]:
        """Return addressable records, optionally including idle ones.

        Merged-away records are always excluded: they are history, not agents you
        can still route work to.
        """
        live = [record for record in self._records if record.merged_into is None]
        if include_archived:
            return live
        return [record for record in live if not self.is_archived(record)]

    def get_agents(self) -> list[str]:
        """Get list of all agent names (kept for existing call sites)."""
        return [record.name for record in self._records if record.merged_into is None]

    def resolve_name(self, agent_name: str) -> str:
        """Follow merge pointers to the agent that now owns this thread.

        Anything still referring to a merged-away name - most importantly a trigger
        row, which stores the owner by name - lands on the canonical agent instead.
        """
        seen: set[str] = set()
        current = agent_name
        while current not in seen:
            seen.add(current)
            record = self.get_record(current)
            if record is None or record.merged_into is None:
                return current
            current = record.merged_into
        return current  # pragma: no cover - cycle guard

    def links_pointing_at(self, agent_name: str) -> list[str]:
        """Agents holding an open duplicate link that names this one as the target.

        Needed because structural evidence is symmetric but reconciliation is not.
        A link lives on the *source* agent, so `reconcile_link(target)` finds nothing
        and returns immediately - yet the deciding thread id may well be logged by
        the target. Without this reverse lookup, evidence arriving on the target side
        is never noticed and a provable duplicate stays unmerged indefinitely, which
        is the common ordering: the newer agent carries the link and runs at once,
        while the older one may sit idle for days before a trigger wakes it.
        """
        return [
            record.name
            for record in self._records
            if record.merged_into is None
            and record.possible_duplicate_of is not None
            and record.possible_duplicate_of.name == agent_name
        ]

    def merged_sources(self, agent_name: str) -> list[str]:
        """Every agent whose history now belongs to this one, oldest first.

        Walks the whole merge graph, not just direct children. Merges can chain:
        A folds into B, then B later folds into C because B carried its own staged
        link. `resolve_name` already follows that chain, so work correctly reaches
        C - but if this returned only direct children, C would load B's transcript
        and silently lose A's. That would quietly break the guarantee that merging
        preserves context, in the one direction nobody would notice.
        """
        by_target: Dict[str, list[AgentRecord]] = {}
        for record in self._records:
            if record.merged_into:
                by_target.setdefault(record.merged_into, []).append(record)

        collected: list[AgentRecord] = []
        seen: set[str] = {agent_name}
        frontier = [agent_name]
        while frontier:
            current = frontier.pop()
            for record in by_target.get(current, []):
                if record.name in seen:
                    continue  # cycle guard, same as resolve_name
                seen.add(record.name)
                collected.append(record)
                frontier.append(record.name)

        return [record.name for record in sorted(collected, key=lambda item: item.created_at)]

    def is_archived(self, record: AgentRecord) -> bool:
        """Whether an agent has gone quiet long enough to drop out of the default view.

        Computed from `last_active` rather than stored, so there is no status field
        to drift and no background job to recompute it. Agents that own a live
        trigger are exempt: a scheduled trigger is direct evidence the user wants
        this thread's work to continue, which is the opposite of an abandoned thread.
        """
        from ...config import get_settings

        cutoff_days = get_settings().agent_archive_after_days
        if cutoff_days <= 0:
            return False

        last_active = _parse_iso(record.last_active)
        if last_active is None:
            return False

        if datetime.now(UTC) - last_active <= timedelta(days=cutoff_days):
            return False

        return not self._has_live_trigger(record.name)

    def _has_live_trigger(self, agent_name: str) -> bool:
        """True when the agent owns at least one non-completed trigger."""
        # Imported lazily: the triggers package opens a SQLite database at import
        # time, and the roster should stay importable (and unit-testable) without it.
        try:
            from ..triggers import get_trigger_service

            triggers = get_trigger_service().list_triggers(agent_name=agent_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Trigger lookup failed for '{agent_name}': {exc}")
            return False

        return any((trigger.status or "").lower() != "completed" for trigger in triggers)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def resolve_available_name(self, candidate_name: str) -> str:
        """Return a name whose slug does not collide with a different existing agent.

        ExecutionAgentLogStore derives log filenames by slugifying the agent name,
        and that slug is lossy: "Keith/Lunch" and "Keith Lunch" both become
        "keith-lunch". Two agents we deliberately decided were distinct would then
        silently share one history file. Disambiguating here means the log store,
        triggers table, and runtime need no changes at all.
        """
        cleaned = (candidate_name or "agent").strip() or "agent"

        if self.get_record(cleaned) is not None:
            return cleaned  # Exact match: same agent, not a collision.

        taken = {_slugify(record.name) for record in self._records}
        if _slugify(cleaned) not in taken:
            return cleaned

        suffix = 2
        while _slugify(f"{cleaned} ({suffix})") in taken:
            suffix += 1
        return f"{cleaned} ({suffix})"

    def add_agent(
        self,
        agent_name: str,
        *,
        description: str = "",
        embedding: Optional[List[float]] = None,
        embedding_model: Optional[str] = None,
        possible_duplicate_of: Optional[DuplicateLink] = None,
    ) -> AgentRecord:
        """Add an agent if absent, returning the record either way."""
        existing = self.get_record(agent_name)
        if existing is not None:
            return existing

        record = AgentRecord(
            name=agent_name,
            description=description,
            embedding=embedding,
            embedding_model=embedding_model,
            possible_duplicate_of=possible_duplicate_of,
        )
        self._records.append(record)
        self.save()
        return record

    async def create_or_link(
        self,
        candidate_name: str,
        *,
        description: str = "",
        embedding: Optional[List[float]] = None,
        embedding_model: Optional[str] = None,
        possible_duplicate_of: Optional[DuplicateLink] = None,
    ) -> AgentRecord:
        """Create an agent under a collision-free name, or return the exact match.

        Held under `_write_lock` so concurrent delegations cannot both decide to
        create and then clobber one another.
        """
        async with self._write_lock:
            existing = self.get_record(candidate_name)
            if existing is not None:
                existing.last_active = _utc_now_iso()
                self.save()
                return existing

            resolved = self.resolve_available_name(candidate_name)
            return self.add_agent(
                resolved,
                description=description,
                embedding=embedding,
                embedding_model=embedding_model,
                possible_duplicate_of=possible_duplicate_of,
            )

    def mark_active(self, agent_name: str) -> None:
        """Refresh an agent's recency.

        Deliberately called only from live-chat delegation. A trigger firing on its
        own must never land here: a recurring reminder would otherwise look
        perpetually engaged even though the user has not touched it in months.
        """
        record = self.get_record(agent_name)
        if record is None:
            return
        record.last_active = _utc_now_iso()
        self.save()

    def set_duplicate_link(self, agent_name: str, link: Optional[DuplicateLink]) -> None:
        """Attach or clear a staged duplicate hypothesis."""
        record = self.get_record(agent_name)
        if record is None:
            return
        record.possible_duplicate_of = link
        self.save()

    def update_embedding(
        self, agent_name: str, embedding: List[float], embedding_model: str
    ) -> None:
        """Store a freshly computed vector (used for lazy re-embedding on model change)."""
        record = self.get_record(agent_name)
        if record is None:
            return
        record.embedding = embedding
        record.embedding_model = embedding_model
        self.save()

    def merge_agent(self, *, source_name: str, target_name: str, evidence: List[str]) -> None:
        """Fold `source_name` into `target_name` once evidence has proven them the same.

        Non-destructive by design: the source record and its log file both survive.
        Only addressability changes - the source stops being routable, and the
        target inherits its history (see ExecutionAgent.build_system_prompt_with_history).
        """
        source = self.get_record(source_name)
        target = self.get_record(target_name)
        if source is None or target is None or source is target:
            return

        source.merged_into = target.name
        source.possible_duplicate_of = None
        # The absorbed thread's activity counts as the survivor's activity.
        if source.last_active > target.last_active:
            target.last_active = source.last_active
        self.save()

        try:
            from .log_store import get_execution_agent_logs

            get_execution_agent_logs().record_action(
                target.name,
                f"Merged agent '{source.name}' into this agent | evidence={', '.join(evidence)}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to log merge marker: {exc}")

    def lock_for(self, agent_name: str) -> asyncio.Lock:
        """Get or create the per-agent execution lock."""
        slug = _slugify(agent_name)
        lock = self._agent_locks.get(slug)
        if lock is None:
            lock = asyncio.Lock()
            self._agent_locks[slug] = lock
        return lock

    def clear(self) -> None:
        """Clear the agent roster."""
        self._records = []
        self._agent_locks.clear()
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
