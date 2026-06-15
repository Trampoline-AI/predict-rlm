from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

_PERPLEXITY_CHAT_COMPLETIONS_URL = "https://api.perplexity.ai/chat/completions"
_DEFAULT_MODEL = "sonar"
_DEFAULT_TIMEOUT_SECONDS = 20
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_MAX_TOKENS = 512


def _build_perplexity_payload(
    query: str,
    max_results: int,
    domains: list[str] | None,
) -> dict[str, Any]:
    return {
        "model": _DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Search the web and answer concisely with citations.",
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "return_citations": True,
        "return_related_questions": False,
        "search_domain_filter": domains or [],
        "web_search_options": {"search_context_size": "medium"},
        "top_p": 0.9,
    }


def _truncate_results(search_results: Any, max_results: int) -> list[dict[str, Any]]:
    if not isinstance(search_results, list):
        return []
    truncated: list[dict[str, Any]] = []
    for item in search_results[:max_results]:
        if isinstance(item, dict):
            truncated.append(
                {
                    key: item[key]
                    for key in ("title", "url", "date", "snippet")
                    if key in item
                }
            )
    return truncated


def _normalize_domains(domains: list[str] | None) -> list[str] | None:
    if domains is None:
        return None
    normalized = [domain.strip() for domain in domains if domain.strip()]
    return normalized or None


def _call_perplexity(
    query: str,
    max_results: int,
    timeout: float,
    domains: list[str] | None,
) -> str:
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is required for web_search")

    payload = _build_perplexity_payload(query, max_results, domains)
    request = urllib.request.Request(
        _PERPLEXITY_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Perplexity web_search failed with HTTP {exc.code}: {body}") from exc

    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    result = {
        "query": query,
        "domains": domains or [],
        "answer": message.get("content", ""),
        "citations": data.get("citations") or [],
        "search_results": _truncate_results(data.get("search_results"), max_results),
    }
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def web_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    domains: list[str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Search the web and return a JSON string with answer, citations, and results.

    Args:
        query: The web search query or question.
        max_results: Maximum number of search result records to include.
        domains: Optional domains to limit search to, such as ["python.org"].
        timeout: HTTP timeout in seconds.

    Returns:
        JSON string with keys: query, domains, answer, citations, search_results.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("web_search query must be non-empty")
    if max_results < 1:
        raise ValueError("web_search max_results must be at least 1")
    return await asyncio.to_thread(
        _call_perplexity,
        normalized_query,
        max_results,
        timeout,
        _normalize_domains(domains),
    )
