from conftest import make_completed, make_text_delta
from litellm.types.llms.openai import ResponsesAPIResponse


def _run_events(lm, events):
    state = lm._fresh_state()
    for ev in events:
        lm._handle_event(ev, state)
    return lm._assemble(
        text_parts=state["text_parts"],
        usage_raw=state["usage_raw"],
        response_id=state["response_id"],
        model_name=state["model_name"],
    )


def test_assemble_returns_responses_api_response(lm):
    events = [make_text_delta("hi"), make_completed(input_tokens=5, output_tokens=2)]
    response = _run_events(lm, events)
    assert isinstance(response, ResponsesAPIResponse)


def test_assembled_text_is_concatenated(lm):
    events = [
        make_text_delta("he"),
        make_text_delta("llo"),
        make_text_delta(" world"),
        make_completed(),
    ]
    response = _run_events(lm, events)
    assert response.output[0].content[0].text == "hello world"


def test_assembled_cost_matches_rate_card(lm):
    events = [
        make_text_delta("x"),
        make_completed(input_tokens=100, output_tokens=10, model="gpt-5.3-codex"),
    ]
    response = _run_events(lm, events)
    expected = (100 * 1.75e-6) + (10 * 1.4e-5)
    assert abs(response.usage.cost - expected) < 1e-12
    assert abs(response._hidden_params["response_cost"] - expected) < 1e-12


def test_assemble_with_no_usage_defaults_to_zero_cost(lm):
    state = lm._fresh_state()
    # only text, never a completed event
    lm._handle_event(make_text_delta("x"), state)
    response = lm._assemble(
        text_parts=state["text_parts"],
        usage_raw=state["usage_raw"],
        response_id=state["response_id"],
        model_name=state["model_name"],
    )
    assert response.usage.cost == 0.0
