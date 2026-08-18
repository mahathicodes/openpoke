"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List

from ...config import get_settings
from ...services.execution import get_agent_roster
from ...services.execution.agent_matcher import embed_text, find_candidates

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
async def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    rendered_agents = await _render_active_agents(latest_text)
    sections.append(f"<active_agents>\n{rendered_agents}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Format the most relevant execution agents into XML tags for LLM awareness
async def _render_active_agents(query_text: str = "") -> str:
    """Render a bounded, relevance-ranked slice of the roster.

    Previously this dumped every agent name into every turn, so prompt size grew
    without limit as the roster did. Now it stays bounded: small rosters render in
    full (no retrieval cost at all), and larger ones are capped at `top_k` total —
    the most recently active agents fill their slots first (never dropped for
    semantic distance alone), and semantically closest candidates fill whatever
    slots remain. This is a capped fill, not a union: recent agents can crowd out
    some of the top-k semantic matches, trading that for a fixed prompt-size cap.
    """
    roster = get_agent_roster()
    roster.load()
    records = roster.get_records()

    if not records:
        return "None"

    settings = get_settings()
    top_k = settings.agent_prompt_top_k

    if len(records) > top_k and query_text.strip():
        recent_count = min(settings.agent_prompt_recent_count, top_k)
        by_recency = sorted(records, key=lambda record: record.last_active, reverse=True)
        selected = list(by_recency[:recent_count])
        selected_names = {record.name for record in selected}

        # Only pay for an embedding when the roster is actually too big to show.
        query_embedding = await embed_text(query_text)
        for candidate in find_candidates(
            query_text=query_text,
            query_embedding=query_embedding,
            records=records,
            top_k=top_k,
        ):
            if len(selected) >= top_k:
                break
            if candidate.record.name not in selected_names:
                selected.append(candidate.record)
                selected_names.add(candidate.record.name)

        records = selected

    rendered: List[str] = []
    for record in records:
        name = escape(record.name or "agent", quote=True)
        if record.description:
            description = escape(record.description, quote=True)
            rendered.append(f'<agent name="{name}" description="{description}" />')
        else:
            rendered.append(f'<agent name="{name}" />')

    return "\n".join(rendered)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
