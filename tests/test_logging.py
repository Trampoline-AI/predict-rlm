import logging

from predict_rlm._logging import _PredictRLMDebugFormatter


def _format_debug_message(message: str) -> str:
    record = logging.LogRecord(
        "predict_rlm.test",
        logging.DEBUG,
        __file__,
        1,
        message,
        (),
        None,
    )
    formatter = _PredictRLMDebugFormatter("%(levelname)s:%(message)s")
    return formatter.format(record)


def test_debug_formatter_colors_error_events_red():
    formatted = _format_debug_message("rlm.execute.error error_type=ValueError")

    assert formatted.startswith("\033[31m")
    assert formatted.endswith("\033[0m")


def test_debug_formatter_colors_status_error_records_red():
    formatted = _format_debug_message("sbx.runner.exited status=error")

    assert formatted.startswith("\033[31m")
    assert formatted.endswith("\033[0m")


def test_debug_formatter_leaves_non_error_records_plain():
    formatted = _format_debug_message("rlm.execute.ok duration_ms=12")

    assert formatted == "DEBUG:rlm.execute.ok duration_ms=12"
