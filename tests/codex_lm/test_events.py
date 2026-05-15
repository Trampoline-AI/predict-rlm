from conftest import make_completed, make_text_delta


def test_text_delta_accumulates(lm):
    state = lm._fresh_state()
    lm._handle_event(make_text_delta("he"), state)
    lm._handle_event(make_text_delta("llo"), state)
    assert "".join(state["text_parts"]) == "hello"


def test_completed_captures_usage_and_ids(lm):
    state = lm._fresh_state()
    lm._handle_event(
        make_completed(
            input_tokens=20,
            output_tokens=8,
            response_id="resp-xyz",
            model="gpt-5.4",
        ),
        state,
    )
    assert state["response_id"] == "resp-xyz"
    assert state["model_name"] == "gpt-5.4"
    assert state["usage_raw"] is not None


def test_unknown_event_ignored(lm):
    from types import SimpleNamespace

    state = lm._fresh_state()
    original = dict(state)
    lm._handle_event(SimpleNamespace(type="some.unrelated.event"), state)
    assert state == original


def test_none_event_safe(lm):
    state = lm._fresh_state()
    original = dict(state)
    lm._handle_event(None, state)
    assert state == original


def test_empty_delta_not_appended(lm):
    state = lm._fresh_state()
    lm._handle_event(make_text_delta(""), state)
    assert state["text_parts"] == []
