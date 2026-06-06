import json
from types import SimpleNamespace
from unittest import mock

import pytest
from conftest import make_completed, make_text_delta
from dspy_codex_lm import CodexStreamError

from predict_rlm.debug import reset_debug_logger_for_tests


@pytest.fixture(autouse=True)
def reset_debug_logging(monkeypatch):
    for name in (
        "PREDICT_RLM_DEBUG",
        "RLM_DEBUG",
        "PREDICT_RLM_DEBUG_LOG",
        "PREDICT_RLM_DEBUG_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_debug_logger_for_tests()
    yield
    reset_debug_logger_for_tests()


def _patch_monotonic(monkeypatch, values):
    ticks = iter(values)
    monkeypatch.setattr("dspy_codex_lm.lm.monotonic", lambda: next(ticks))


def _debug_records(log_path):
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _records_by_event(log_path):
    return {record["event"]: record for record in _debug_records(log_path)}


def _enable_json_debug(monkeypatch, tmp_path):
    log_path = tmp_path / "codex-debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    return log_path


def _response_created_event():
    return SimpleNamespace(type="response.created")


def test_forward_emits_stream_timing_events(lm, monkeypatch, tmp_path):
    log_path = _enable_json_debug(monkeypatch, tmp_path)
    _patch_monotonic(monkeypatch, [10.0, 10.1, 10.25, 10.4, 10.5, 10.55])
    events = [
        _response_created_event(),
        make_text_delta("ok"),
        make_completed(input_tokens=5, output_tokens=1, cached_tokens=3),
    ]

    lm.kwargs["reasoning_effort"] = "xhigh"
    lm.kwargs["service_tier"] = "priority"
    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        result = lm.forward(prompt="hi", cache=False)

    assert result.output[0].content[0].text == "ok"
    records = _records_by_event(log_path)
    assert set(records) == {
        "codex_lm.stream.start",
        "codex_lm.stream.first_event",
        "codex_lm.stream.first_text_delta",
        "codex_lm.stream.end",
    }
    assert records["codex_lm.stream.start"]["attempt_number"] == 1
    assert records["codex_lm.stream.start"]["model"] == "gpt-5.3-codex"
    assert records["codex_lm.stream.start"]["transport"] == "http_sse"
    assert records["codex_lm.stream.start"]["reasoning_effort"] == "xhigh"
    assert records["codex_lm.stream.start"]["service_tier"] == "priority"
    assert records["codex_lm.stream.first_event"]["reasoning_effort"] == "xhigh"
    assert records["codex_lm.stream.first_text_delta"]["service_tier"] == "priority"
    assert records["codex_lm.stream.end"]["reasoning_effort"] == "xhigh"
    assert records["codex_lm.stream.end"]["service_tier"] == "priority"
    assert records["codex_lm.stream.first_event"]["first_event_type"] == "response.created"
    assert records["codex_lm.stream.first_event"]["tt_first_event_ms"] == 100.0
    assert records["codex_lm.stream.first_text_delta"]["ttft_ms"] == 250.0
    assert records["codex_lm.stream.end"]["stream_total_ms"] == 500.0
    assert records["codex_lm.stream.end"]["parse_overhead_ms"] == 50.0
    assert records["codex_lm.stream.end"]["output_text_chars"] == 2
    assert records["codex_lm.stream.end"]["completed"] is True
    assert records["codex_lm.stream.end"]["prompt_tokens"] == 5
    assert records["codex_lm.stream.end"]["cached_prompt_tokens"] == 3
    assert records["codex_lm.stream.end"]["prompt_cache_read_ratio"] == pytest.approx(0.6)


async def test_aforward_emits_stream_timing_events(lm, monkeypatch, tmp_path):
    log_path = _enable_json_debug(monkeypatch, tmp_path)
    _patch_monotonic(monkeypatch, [20.0, 20.2, 20.4, 20.6, 20.7, 20.72])
    events = [
        _response_created_event(),
        make_text_delta("async"),
        make_completed(input_tokens=5, output_tokens=1),
    ]

    async def fake_aresponses(**_):
        async def gen():
            for event in events:
                yield event

        return gen()

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
        result = await lm.aforward(prompt="hi", cache=False)

    assert result.output[0].content[0].text == "async"
    records = _records_by_event(log_path)
    assert records["codex_lm.stream.first_event"]["tt_first_event_ms"] == 200.0
    assert records["codex_lm.stream.first_text_delta"]["ttft_ms"] == 400.0
    assert records["codex_lm.stream.end"]["stream_total_ms"] == 700.0
    assert records["codex_lm.stream.end"]["output_text_chars"] == 5
    assert records["codex_lm.stream.end"]["completed"] is True


def test_codex_debug_log_gets_records_when_predict_rlm_debug_is_also_enabled(
    lm,
    monkeypatch,
    tmp_path,
):
    predict_log = tmp_path / "predict-debug.jsonl"
    codex_log = tmp_path / "codex-debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(predict_log))
    monkeypatch.setenv("CODEX_LM_DEBUG_LOG", str(codex_log))
    _patch_monotonic(monkeypatch, [10.0, 10.1, 10.2, 10.3, 10.35, 10.36])
    events = [make_text_delta("ok"), make_completed(input_tokens=5, output_tokens=1)]

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        lm.forward(prompt="hi", cache=False)

    records = _records_by_event(codex_log)
    assert records["codex_lm.stream.start"]["transport"] == "http_sse"
    assert records["codex_lm.stream.end"]["transport"] == "http_sse"
    assert not predict_log.exists()



def test_stream_end_reports_non_text_event_counts_and_inter_event_gap(lm, monkeypatch, tmp_path):
    log_path = _enable_json_debug(monkeypatch, tmp_path)
    _patch_monotonic(
        monkeypatch,
        [
            40.0,  # stream start
            40.1,  # response.created
            40.4,  # response.in_progress
            41.0,  # first text delta
            41.2,  # response.completed
            41.25,  # mark stream end
            41.27,  # emit end
        ],
    )
    events = [
        _response_created_event(),
        SimpleNamespace(type="response.in_progress"),
        make_text_delta("ok"),
        make_completed(input_tokens=5, output_tokens=1),
    ]

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        lm.forward(prompt="hi", cache=False)

    end = _records_by_event(log_path)["codex_lm.stream.end"]
    assert end["events_before_first_text"] == 2
    assert end["max_inter_event_gap_ms"] == 600.0
    assert end["non_text_event_counts"] == {
        "response.created": 1,
        "response.in_progress": 1,
        "response.completed": 1,
    }


def test_stream_events_include_selected_rotation_profile(monkeypatch, tmp_path):
    import json
    from pathlib import Path

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    def write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": access_token,
                        "account_id": account_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    log_path = _enable_json_debug(monkeypatch, tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import_auth_profile(
        "alpha",
        write_auth(tmp_path / "alpha.json", access_token="alpha-token", account_id="acct-alpha"),
    )
    import_auth_profile(
        "beta",
        write_auth(tmp_path / "beta.json", access_token="beta-token", account_id="acct-beta"),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    def choose_credentials(credentials):
        return next(
            credential
            for credential in credentials
            if credential.account_id == "acct-beta"
        )

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)
    _patch_monotonic(monkeypatch, [60.0, 60.1, 60.2, 60.3, 60.35, 60.36])
    events = [make_text_delta("ok"), make_completed(input_tokens=5, output_tokens=1)]

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        CodexLM(model="gpt-5.3-codex").forward(prompt="hi", cache=False)

    records = _records_by_event(log_path)
    assert records["codex_lm.stream.start"]["auth_profile"] == "beta"
    assert records["codex_lm.stream.start"]["auth_source"] == "rotation"
    assert records["codex_lm.stream.end"]["auth_profile"] == "beta"


def test_stream_error_reports_last_event_when_stream_stalls(monkeypatch, tmp_path):
    log_path = _enable_json_debug(monkeypatch, tmp_path)
    from dspy_codex_lm.lm import _StreamTiming

    timing = _StreamTiming(
        model="gpt-5.5",
        attempt_number=1,
        start_at=50.0,
        transport="websocket",
    )
    _patch_monotonic(monkeypatch, [50.2, 80.2])
    timing.observe_event(_response_created_event())
    timing.emit_error({}, CodexStreamError("stall"))

    error = _records_by_event(log_path)["codex_lm.stream.error"]
    assert error["transport"] == "websocket"
    assert error["last_event_type"] == "response.created"
    assert error["last_event_age_ms"] == 30000.0
    assert error["events_before_first_text"] == 1
    assert error["max_inter_event_gap_ms"] is None


def test_stream_error_event_is_sanitized(lm, monkeypatch, tmp_path):
    log_path = _enable_json_debug(monkeypatch, tmp_path)
    _patch_monotonic(monkeypatch, [30.0, 30.1, 30.2, 30.3])
    events = [
        make_text_delta("partial"),
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(
                    code="rate_limit_exceeded",
                    message="slow down",
                ),
            ),
        ),
    ]

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        with pytest.raises(CodexStreamError):
            lm.forward(prompt="hi", cache=False)

    records = _records_by_event(log_path)
    error = records["codex_lm.stream.error"]
    assert error["failure_kind"] == "failed"
    assert error["failure_code"] == "rate_limit_exceeded"
    assert error["exception_type"] == "CodexStreamError"
    assert error["completed"] is False
    assert error["output_text_chars"] == len("partial")
    log_text = log_path.read_text()
    assert "fake-access" not in log_text
    assert "fake-account" not in log_text
