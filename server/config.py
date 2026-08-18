"""Simplified configuration management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


def _load_env_file() -> None:
    """Load .env from root directory if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


_load_env_file()


DEFAULT_APP_NAME = "OpenPoke Server"
DEFAULT_APP_VERSION = "0.3.0"


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


class Settings(BaseModel):
    """Application settings with lightweight env fallbacks."""

    # App metadata
    app_name: str = Field(default=DEFAULT_APP_NAME)
    app_version: str = Field(default=DEFAULT_APP_VERSION)

    # Server runtime
    server_host: str = Field(default=os.getenv("OPENPOKE_HOST", "0.0.0.0"))
    server_port: int = Field(default=_env_int("OPENPOKE_PORT", 8001))

    # LLM model selection
    interaction_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_search_model: str = Field(default="anthropic/claude-sonnet-4")
    summarizer_model: str = Field(default="anthropic/claude-sonnet-4")
    email_classifier_model: str = Field(default="anthropic/claude-sonnet-4")

    # Credentials / integrations
    openrouter_api_key: Optional[str] = Field(default=os.getenv("OPENROUTER_API_KEY"))
    composio_gmail_auth_config_id: Optional[str] = Field(default=os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID"))
    composio_api_key: Optional[str] = Field(default=os.getenv("COMPOSIO_API_KEY"))

    # HTTP behaviour
    cors_allow_origins_raw: str = Field(default=os.getenv("OPENPOKE_CORS_ALLOW_ORIGINS", "*"))
    enable_docs: bool = Field(default=os.getenv("OPENPOKE_ENABLE_DOCS", "1") != "0")
    docs_url: Optional[str] = Field(default=os.getenv("OPENPOKE_DOCS_URL", "/docs"))

    # Summarisation controls
    conversation_summary_threshold: int = Field(default=100)
    conversation_summary_tail_size: int = Field(default=10)

    # Agent matching / roster controls
    embedding_model: str = Field(default=os.getenv("OPENPOKE_EMBEDDING_MODEL", "openai/text-embedding-3-small"))
    embedding_timeout_seconds: float = Field(default=10.0)

    # How many roster candidates the dedup LLM call may consider.
    agent_dedup_top_k: int = Field(default=_env_int("OPENPOKE_AGENT_DEDUP_TOP_K", 5))
    # Roster entries rendered into the interaction agent prompt; retrieval only
    # kicks in once the roster is larger than this, so small rosters pay nothing.
    agent_prompt_top_k: int = Field(default=_env_int("OPENPOKE_AGENT_PROMPT_TOP_K", 15))
    # Recent agents always kept in the prompt regardless of semantic relevance.
    agent_prompt_recent_count: int = Field(default=_env_int("OPENPOKE_AGENT_PROMPT_RECENT", 5))
    # Below this cosine similarity a candidate is not worth showing the judge.
    #
    # Deliberately low, and NOT a safety threshold. An earlier value of 0.60 was
    # calibrated on one labeled set where true reuse scored 0.64-0.77 and adversarial
    # cases 0.50-0.57, which made a clean separating threshold look available. Across
    # both labeled sets measured against openai/text-embedding-3-small the classes
    # turn out to interleave completely:
    #
    #     true reuse    0.246 - 0.765   (0.246 is a pronoun reference: "ask *them*...")
    #     adversarial   0.505 - 0.665
    #
    # No threshold separates those. At 0.60 the floor was discarding genuine reuse
    # (recall fell to 33% on the ablation scenarios) while blocking nothing the judge
    # would not have refused anyway - in live runs it prevented zero false merges.
    #
    # So the floor is now only a noise filter and a cost saver: unrelated requests
    # score around 0.17, well under this. Everything above it is a mixed bag, and
    # discrimination is the judge's job - which live evaluation shows it does
    # reliably (correct abstention on every adversarial case tested).
    #
    # Tradeoff: more requests now reach the LLM, so this costs more per turn than
    # 0.60 did. Recall is worth more than the saving.
    #
    # 0.20 sits just above measured unrelated content (~0.17) and just below the
    # weakest true reuse (0.246). That is a narrow margin, and setting it by those
    # two points is the same overfitting this comment warns about - so treat it as
    # "low enough to be nearly inert" rather than as a tuned value. Pronoun-style
    # requests genuinely live near the noise floor and no threshold rescues them;
    # see the anaphora limitation in DESIGN.md.
    agent_min_candidate_similarity: float = Field(default=0.20)
    # When the top two candidates are within this margin, retrieval cannot tell them
    # apart (two people named Keith, both about lunch). Staging a link on a coin flip
    # is worse than staging none, so the tie is surfaced for clarification instead.
    agent_ambiguity_margin: float = Field(default=0.05)
    # Confidence a possible-duplicate link must reach before an actual merge.
    agent_merge_commit_threshold: float = Field(default=0.9)
    # Idle days before an agent drops out of the default prompt view. Agents that
    # own a live trigger are exempt (a scheduled trigger means the work is ongoing).
    agent_archive_after_days: int = Field(default=_env_int("OPENPOKE_AGENT_ARCHIVE_AFTER_DAYS", 30))

    @property
    def cors_allow_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_allow_origins_raw.strip() in {"", "*"}:
            return ["*"]
        return [origin.strip() for origin in self.cors_allow_origins_raw.split(",") if origin.strip()]

    @property
    def resolved_docs_url(self) -> Optional[str]:
        """Return documentation URL when docs are enabled."""
        return (self.docs_url or "/docs") if self.enable_docs else None

    @property
    def summarization_enabled(self) -> bool:
        """Flag indicating conversation summarisation is active."""
        return self.conversation_summary_threshold > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
