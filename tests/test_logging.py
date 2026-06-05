import logging

import predict_rlm._logging as logging_module
from predict_rlm._logging import (
    DEBUG_HANDLER_MARKER,
    PACKAGE_LOGGER_NAME,
    TRACE_HANDLER_MARKER,
    TRACE_LOGGER_NAME,
    _PredictRLMDebugFormatter,
    configure_predict_rlm_logging,
)


def _snapshot_logger(logger: logging.Logger):
    return (
        logger.level,
        logger.propagate,
        logger.disabled,
        list(logger.handlers),
    )


def _restore_logger(logger: logging.Logger, state) -> None:
    level, propagate, disabled, handlers = state
    logger.setLevel(level)
    logger.propagate = propagate
    logger.disabled = disabled
    logger.handlers[:] = handlers


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


def test_debug_logging_can_restore_prior_logger_state():
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    original_logger = _snapshot_logger(logger)
    original_module_state = logging_module._debug_logger_state
    try:
        logging_module._debug_logger_state = None
        logger.handlers[:] = []
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.disabled = False

        configure_predict_rlm_logging(debug=True)

        assert logger.level == logging.DEBUG
        assert any(getattr(handler, DEBUG_HANDLER_MARKER, False) for handler in logger.handlers)

        configure_predict_rlm_logging(debug=False)

        assert logger.level == logging.WARNING
        assert logger.propagate is False
        assert logger.disabled is False
        assert not any(
            getattr(handler, DEBUG_HANDLER_MARKER, False) for handler in logger.handlers
        )
    finally:
        logging_module._debug_logger_state = original_module_state
        _restore_logger(logger, original_logger)


def test_verbose_trace_detail_does_not_hard_wrap_long_logical_lines():
    long_line = "x" * 160

    rendered = logging_module._render_trace_detail("output:", long_line)

    assert long_line in rendered


def test_verbose_trace_detail_preserves_code_syntax_color():
    rendered = logging_module._render_trace_detail(
        "code:",
        "value = {'x': 1}",
        syntax="python",
    )

    assert "\033[33m'" in rendered
    assert "\033[34m1" in rendered


def test_verbose_logging_can_restore_prior_trace_logger_state():
    logger = logging.getLogger(TRACE_LOGGER_NAME)
    original_logger = _snapshot_logger(logger)
    original_module_state = logging_module._trace_logger_state
    try:
        logging_module._trace_logger_state = None
        logger.handlers[:] = []
        logger.setLevel(logging.WARNING)
        logger.propagate = True
        logger.disabled = False

        configure_predict_rlm_logging(verbose=True)

        assert logger.level == logging.INFO
        assert logger.propagate is False
        assert any(getattr(handler, TRACE_HANDLER_MARKER, False) for handler in logger.handlers)

        configure_predict_rlm_logging(verbose=False)

        assert logger.level == logging.WARNING
        assert logger.propagate is True
        assert logger.disabled is False
        assert not any(
            getattr(handler, TRACE_HANDLER_MARKER, False) for handler in logger.handlers
        )
    finally:
        logging_module._trace_logger_state = original_module_state
        _restore_logger(logger, original_logger)
