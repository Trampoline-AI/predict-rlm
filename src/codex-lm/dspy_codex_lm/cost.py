from typing import Any

from dspy.clients.model_metadata import model_info

_KNOWN_MODEL_COSTS: dict[str, dict[str, Any]] = {
    "mercury-2": {
        "input_cost_per_token": 0.25 / 1_000_000,
        "cache_read_input_token_cost": 0.025 / 1_000_000,
        "output_cost_per_token": 0.75 / 1_000_000,
    },
    "mercury-edit-2": {
        "input_cost_per_token": 0.25 / 1_000_000,
        "cache_read_input_token_cost": 0.025 / 1_000_000,
        "output_cost_per_token": 0.75 / 1_000_000,
    },
}


# Codex-only slugs without a public API price, billed like their parent model.
_PRICE_ALIASES: dict[str, str] = {
    "gpt-5.3-codex-spark": "gpt-5.4",
}


def compute_cost(model: str, usage: dict[str, Any]) -> float:
    """Estimate equivalent API token cost; unpriced models return 0.0."""
    slug = model.split("/")[-1]
    slug = _PRICE_ALIASES.get(slug, slug)
    prices = _KNOWN_MODEL_COSTS.get(slug) or model_info("openai", slug)
    if not prices:
        return 0.0

    input_t = usage.get("input_tokens", 0) or 0
    output_t = usage.get("output_tokens", 0) or 0
    details = usage.get("input_tokens_details") or {}
    cached_t = (details.get("cached_tokens") if isinstance(details, dict) else 0) or 0

    in_rate = prices.get("input_cost_per_token", 0) or 0
    out_rate = prices.get("output_cost_per_token", 0) or 0
    cache_rate = prices.get("cache_read_input_token_cost", in_rate) or 0

    non_cached_in = max(input_t - cached_t, 0)
    return (non_cached_in * in_rate) + (cached_t * cache_rate) + (output_t * out_rate)
