"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...config import get_settings
from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.execution import get_agent_roster, get_execution_agent_logs
from ...services.execution.agent_matcher import (
    agent_embedding_text,
    decide_routing,
    embed_text,
    find_candidates,
)
from ...services.execution.roster import DuplicateLink
from ..execution_agent.batch_manager import ExecutionBatchManager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Deliver instructions to a specific execution agent. Creates a new agent if the name doesn't exist in the roster, or reuses an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable agent name describing its purpose (e.g., 'Vercel Job Offer', 'Email to Sharanjeet'). This name will be used to identify and potentially reuse the agent."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["agent_name", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record an email draft so the user can review the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email for the draft.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agents",
            "description": (
                "Search all execution agents by meaning, including long-idle ones that are "
                "not listed in your active agent roster. Use this when the user refers to "
                "past work you cannot see in the roster above."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'lunch plans with Keith'.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()


# Create or reuse execution agent and dispatch instructions asynchronously
async def send_message_to_agent(agent_name: str, instructions: str) -> ToolResult:
    """Route instructions to an execution agent, reusing or linking where warranted.

    Routing ladder, cheapest first:
      1. Exact name match  -> reuse immediately, zero extra API calls.
      2. Semantic search   -> shortlist similar agents (lexical fallback on failure).
      3. LLM judgment      -> describe the new agent and optionally *stage* a
                              duplicate link. Never merges here; see agent_matcher.
    """
    roster = get_agent_roster()
    roster.load()

    # (1) Exact match: the common case when the model reuses a name it just used.
    # Keeping this ahead of retrieval means repeat delegations cost nothing extra.
    existing = roster.get_record(agent_name)
    if existing is not None:
        resolved_name = roster.resolve_name(agent_name)
        roster.mark_active(resolved_name)
        return _dispatch_to_agent(resolved_name, instructions, action="reused")

    settings = get_settings()
    candidates = []

    # (2) Retrieval. A failed embedding degrades to lexical overlap rather than
    # blocking the turn; see find_candidates.
    query_text = f"{agent_name}. {instructions}"
    query_embedding = await embed_text(query_text)
    searchable = roster.get_records(include_archived=True)
    if searchable:
        candidates = find_candidates(
            query_text=query_text,
            query_embedding=query_embedding,
            records=searchable,
            top_k=settings.agent_dedup_top_k,
        )
        # Nothing plausibly related: skip the judgment call entirely.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= settings.agent_min_candidate_similarity
        ]

    # (2b) Ambiguity check. When the top two candidates score within a hair of each
    # other, retrieval genuinely cannot separate them - two different people named
    # Keith, both about lunch, is the canonical case. Asking the judge to choose
    # produces a coin flip, and a 50/50 link is worse than no link: it is a wrong
    # answer wearing the costume of a decision. Surface the tie instead.
    if len(candidates) >= 2:
        margin = candidates[0].score - candidates[1].score
        if margin <= settings.agent_ambiguity_margin:
            tied = [
                candidate.record.name
                for candidate in candidates
                if candidates[0].score - candidate.score <= settings.agent_ambiguity_margin
            ]
            logger.info(
                f"Ambiguous match for '{agent_name}': {', '.join(tied)} "
                f"within {margin:.3f} - holding work for clarification"
            )
            # Start nothing. Dispatching while simultaneously asking "which one?"
            # would be theatre: the execution agent could email the wrong Keith
            # before the answer arrives, and asking after an irreversible action is
            # worthless. Returning without starting work also means the turn
            # produces nothing unless the model speaks to the user, so the pressure
            # to actually ask is structural rather than merely instructed.
            #
            # No pending-state machine is needed to resume: the user's reply names
            # the agent, and that call takes the exact-match fast path above.
            return ToolResult(
                success=True,
                payload={
                    "status": "needs_clarification",
                    "ambiguous_with": tied,
                    "note": (
                        "Several existing agents match this request equally well: "
                        f"{', '.join(tied)}. No work has been started and nothing was "
                        "linked. Ask the user which one they mean, then call this tool "
                        "again with that exact agent_name."
                    ),
                },
            )

    # (3) Judgment. Produces a description for future matching, plus at most a
    # staged hypothesis that this continues an existing thread.
    description = ""
    duplicate_link = None
    if candidates:
        decision = await decide_routing(
            instructions=instructions,
            proposed_name=agent_name,
            candidates=candidates,
        )
        description = decision.description
        if decision.duplicate_of:
            duplicate_link = DuplicateLink(
                name=roster.resolve_name(decision.duplicate_of),
                confidence=decision.confidence,
                evidence=[f"text-similarity: {decision.confidence:.2f}"],
            )
            logger.info(
                f"Staged duplicate link '{agent_name}' -> '{duplicate_link.name}' "
                f"at {decision.confidence:.2f} (awaiting structural evidence)"
            )

    embedding_model = settings.embedding_model if query_embedding else None
    description_embedding = query_embedding
    if description and query_embedding is not None:
        # Index the agent under its description rather than the raw first task, so
        # later matches compare like with like.
        description_embedding = (
            await embed_text(agent_embedding_text(agent_name, description)) or query_embedding
        )

    record = await roster.create_or_link(
        agent_name,
        description=description,
        embedding=description_embedding,
        embedding_model=embedding_model,
        possible_duplicate_of=duplicate_link,
    )

    roster.mark_active(record.name)
    return _dispatch_to_agent(record.name, instructions, action="created")


# Record the request and fire the execution agent without waiting for it
def _dispatch_to_agent(agent_name: str, instructions: str, *, action: str) -> ToolResult:
    get_execution_agent_logs().record_request(agent_name, instructions)
    logger.info(f"{action.capitalize()} agent: {agent_name}")

    async def _execute_async() -> None:
        try:
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(agent_name, instructions)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{agent_name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{agent_name}' failed: {str(exc)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    loop.create_task(_execute_async())

    return ToolResult(
        success=True,
        payload={
            "status": "submitted",
            # Report the canonical name back so the model's own context converges on
            # it: the name it proposed may have been disambiguated or redirected.
            "agent_name": agent_name,
            "new_agent_created": action == "created",
        },
    )


# Semantic search across every agent, including ones idle enough to leave the roster
async def search_agents(query: str) -> ToolResult:
    """Find agents by meaning, including long-idle ones absent from the prompt."""
    roster = get_agent_roster()
    roster.load()

    records = roster.get_records(include_archived=True)
    if not records:
        return ToolResult(success=True, payload={"matches": []})

    query_embedding = await embed_text(query)
    candidates = find_candidates(
        query_text=query,
        query_embedding=query_embedding,
        records=records,
        top_k=get_settings().agent_dedup_top_k,
    )

    matches = [
        {
            "agent_name": candidate.record.name,
            "description": candidate.record.description,
            "last_active": candidate.record.last_active,
            "score": round(candidate.score, 3),
        }
        for candidate in candidates
        if candidate.score > 0
    ]

    return ToolResult(success=True, payload={"matches": matches})


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    log = get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
async def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        # Only the routing tools need to await anything (embeddings / judgment);
        # the rest stay synchronous.
        if name == "send_message_to_agent":
            return await send_message_to_agent(**args)
        if name == "search_agents":
            return await search_agents(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
