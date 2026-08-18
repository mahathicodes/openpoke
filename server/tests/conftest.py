"""Shared fixtures: isolate roster/log state and stub out network calls."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Allow `python -m pytest server/tests` from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.services.execution import log_store as log_store_module  # noqa: E402
from server.services.execution import roster as roster_module  # noqa: E402
from server.services.execution.log_store import ExecutionAgentLogStore  # noqa: E402
from server.services.execution.roster import AgentRoster  # noqa: E402


@pytest.fixture
def temp_roster(tmp_path, monkeypatch) -> AgentRoster:
    """A roster backed by a temp file, swapped in for the module singleton.

    Patching the module global is enough: get_agent_roster() reads it at call
    time, so every namespace that imported the *function* picks this up too.
    """
    roster = AgentRoster(tmp_path / "roster.json")
    monkeypatch.setattr(roster_module, "_agent_roster", roster)
    return roster


@pytest.fixture
def temp_logs(tmp_path, monkeypatch) -> ExecutionAgentLogStore:
    """An execution log store backed by a temp directory."""
    logs = ExecutionAgentLogStore(tmp_path / "execution_agents")
    monkeypatch.setattr(log_store_module, "_execution_agent_logs", logs)
    return logs


@pytest.fixture
def no_triggers(monkeypatch):
    """Report no triggers, so archival tests are not affected by a real DB."""
    monkeypatch.setattr(AgentRoster, "_has_live_trigger", lambda self, name: False)


@pytest.fixture
def with_live_trigger(monkeypatch):
    """Report every agent as owning a live trigger."""
    monkeypatch.setattr(AgentRoster, "_has_live_trigger", lambda self, name: True)


def make_embedding_response(vectors: List[List[float]]) -> Dict[str, Any]:
    """Shape an OpenRouter embeddings payload."""
    return {"data": [{"embedding": vector, "index": i} for i, vector in enumerate(vectors)]}


def make_tool_call_response(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Shape an OpenRouter chat completion that invokes one tool."""
    import json

    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }
