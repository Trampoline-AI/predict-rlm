from dspy_codex_lm.cost import compute_cost


def test_gpt_5_3_codex_normal_pricing():
    # Published rates: input 1.75e-6, output 1.4e-5
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
    }
    cost = compute_cost("gpt-5.3-codex", usage)
    expected = (100 * 1.75e-6) + (10 * 1.4e-5)
    assert abs(cost - expected) < 1e-12


def test_strips_openai_prefix_from_model_slug():
    # LM stores model as "openai/gpt-5.3-codex" — compute_cost should strip it
    usage = {"input_tokens": 100, "output_tokens": 10}
    direct = compute_cost("gpt-5.3-codex", usage)
    prefixed = compute_cost("openai/gpt-5.3-codex", usage)
    assert direct == prefixed > 0


def test_cached_tokens_priced_at_cache_rate():
    # 100 total input, 80 cached → 20 at normal rate, 80 at cache rate
    usage = {
        "input_tokens": 100,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 80},
    }
    cost = compute_cost("gpt-5.3-codex", usage)
    # gpt-5.3-codex: input 1.75e-6, cache 1.75e-7
    expected = (20 * 1.75e-6) + (80 * 1.75e-7)
    assert abs(cost - expected) < 1e-12


def test_unknown_model_returns_zero():
    usage = {"input_tokens": 1000, "output_tokens": 500}
    assert compute_cost("does-not-exist", usage) == 0.0


def test_empty_usage_returns_zero():
    assert compute_cost("gpt-5.3-codex", {}) == 0.0


def test_missing_details_key_does_not_crash():
    usage = {"input_tokens": 100, "output_tokens": 10}
    # No "input_tokens_details" key — should treat cached as 0
    cost = compute_cost("gpt-5.3-codex", usage)
    expected = (100 * 1.75e-6) + (10 * 1.4e-5)
    assert abs(cost - expected) < 1e-12


def test_gpt_5_4_has_its_own_rates():
    # Make sure we look up the actual slug (5.4), not collapse to 5.3
    usage = {"input_tokens": 100, "output_tokens": 10}
    c_5_3 = compute_cost("gpt-5.3-codex", usage)
    c_5_4 = compute_cost("gpt-5.4", usage)
    # gpt-5.4 is more expensive than gpt-5.3-codex
    assert c_5_4 > c_5_3


def test_mercury_2_pricing_from_inception_rate_card():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "input_tokens_details": {"cached_tokens": 100_000},
    }
    cost = compute_cost("openai/mercury-2", usage)
    expected = (900_000 * 0.25e-6) + (100_000 * 0.025e-6) + (1_000_000 * 0.75e-6)
    assert abs(cost - expected) < 1e-12


def test_mercury_edit_2_pricing_from_inception_rate_card():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    cost = compute_cost("mercury-edit-2", usage)
    expected = 0.25 + 0.75
    assert abs(cost - expected) < 1e-12
