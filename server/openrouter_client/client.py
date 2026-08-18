from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

OpenRouterBaseURL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    base_url: str = OpenRouterBaseURL,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload.

    `max_tokens` is worth setting on calls with small bounded outputs. Left unset,
    OpenRouter reserves the model's full output window - 64k on some models - and
    bills/gates against that reservation, which fails outright on credit-limited
    accounts for a call that only ever returns a few dozen tokens.
    """

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    url = f"{base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=_headers(api_key=api_key),
                json=payload,
                timeout=60.0,  # Set reasonable timeout instead of None
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                _handle_response_error(exc)
            return response.json()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - handled above
            _handle_response_error(exc)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    raise OpenRouterError("OpenRouter request failed: unknown error")


async def request_embeddings(
    *,
    model: str,
    texts: List[str],
    api_key: Optional[str] = None,
    timeout: float = 10.0,
    base_url: str = OpenRouterBaseURL,
) -> List[List[float]]:
    """Embed one or more texts and return their vectors in input order.

    Uses a shorter timeout than chat completions: embeddings back a lightweight
    retrieval pre-filter, so a slow call should fall back rather than stall a turn.
    """

    payload: Dict[str, object] = {
        "model": model,
        "input": texts,
    }

    url = f"{base_url.rstrip('/')}/embeddings"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=_headers(api_key=api_key),
                json=payload,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                _handle_response_error(exc)
            return _extract_embeddings(response.json(), expected=len(texts))
        except httpx.HTTPStatusError as exc:  # pragma: no cover - handled above
            _handle_response_error(exc)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter embeddings request failed: {exc}") from exc

    raise OpenRouterError("OpenRouter embeddings request failed: unknown error")


def _extract_embeddings(payload: Dict[str, Any], *, expected: int) -> List[List[float]]:
    """Pull vectors out of an embeddings response, ordered by the API's index field."""

    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise OpenRouterError(
            f"OpenRouter embeddings response had {len(data) if isinstance(data, list) else 'no'} "
            f"entries, expected {expected}"
        )

    # The API returns an index per entry; don't assume the list is already ordered.
    ordered = sorted(data, key=lambda entry: entry.get("index", 0) if isinstance(entry, dict) else 0)

    vectors: List[List[float]] = []
    for entry in ordered:
        vector = entry.get("embedding") if isinstance(entry, dict) else None
        if not isinstance(vector, list) or not vector:
            raise OpenRouterError("OpenRouter embeddings response contained an empty vector")
        vectors.append([float(value) for value in vector])

    return vectors


__all__ = [
    "OpenRouterError",
    "request_chat_completion",
    "request_embeddings",
    "OpenRouterBaseURL",
]
