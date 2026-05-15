from typing import Any

_KNOWN_MODEL_COSTS: dict[str, dict[str, Any]] = {
    "mercury-2": {
        "input_cost_per_token": 0.25 / 1_000_000,
        "cache_read_input_token_cost": 0.025 / 1_000_000,
        "output_cost_per_token": 0.75 / 1_000_000,
        "litellm_provider": "openai",
        "max_input_tokens": 128_000,
        "mode": "chat",
        "supported_endpoints": ["/v1/chat/completions"],
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_response_schema": True,
        "supports_reasoning": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
    },
    "mercury-edit-2": {
        "input_cost_per_token": 0.25 / 1_000_000,
        "cache_read_input_token_cost": 0.025 / 1_000_000,
        "output_cost_per_token": 0.75 / 1_000_000,
        "litellm_provider": "openai",
        "max_input_tokens": 32_000,
        "mode": "completion",
        "supported_endpoints": ["/v1/fim/completions", "/v1/edit/completions"],
    },
}


def register_known_model_costs() -> None:
    from litellm import model_cost

    for slug, prices in _KNOWN_MODEL_COSTS.items():
        model_cost.setdefault(slug, {**prices})
        model_cost.setdefault(f"openai/{slug}", {**prices})


def compute_cost(model: str, usage: dict[str, Any]) -> float:
    """Compute USD cost for a Codex call using LiteLLM's public price table.

    Looks up the model slug directly; all current Codex models
    (gpt-5.1-codex, gpt-5.3-codex, gpt-5.4, etc.) are in the LiteLLM price DB.
    Returns 0.0 if the model is not priced.
    """
    from litellm import model_cost

    register_known_model_costs()
    slug = model.split("/")[-1]
    prices = model_cost.get(slug)
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
