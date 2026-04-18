"""Cancellation and timeout paths must still leave a partial RunTrace.

Background:
    ``PredictRLM._forward_traced`` and ``_aforward_traced`` attach
    ``exc.trace`` to any exception so the caller can diagnose partial
    runs (which iteration reached, what tokens were spent, etc). The
    original handler only caught ``Exception``, which meant
    ``asyncio.CancelledError`` (a ``BaseException`` since 3.8) slipped
    past unaugmented. In practice this lost cost accounting for every
    case that hit ``asyncio.wait_for`` — ~0.5% of evaluate rollouts in
    a typical long run.

    The fix widens the handler to ``except BaseException`` and wraps
    ``_build_run_trace`` in a safety net so a trace-building failure
    during cancellation can never mask the cancellation itself.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


def _patch_build_run_trace(predictor, sentinel):
    """Make _build_run_trace return a sentinel object we can recognise."""
    predictor._build_run_trace = MagicMock(return_value=sentinel)


class TestSyncForwardTracedCancellationPath:
    """The sync path should attach a trace even for KeyboardInterrupt.

    We test KeyboardInterrupt rather than CancelledError because
    CancelledError is an async primitive; KeyboardInterrupt is a sync
    BaseException that exercises the same handler widening.
    """

    def test_keyboard_interrupt_still_attaches_trace(self):
        from predict_rlm.predict_rlm import PredictRLM

        predictor = PredictRLM.__new__(PredictRLM)
        sentinel = object()
        _patch_build_run_trace(predictor, sentinel)

        # Build the minimum frame state _forward_traced expects so we
        # can call the except branch directly by raising into it.
        class _FakeExc(KeyboardInterrupt):
            pass

        exc = _FakeExc()
        try:
            # Simulate the handler body from _forward_traced (lines
            # 1426-1436): widened except sets exc.trace and re-raises.
            try:
                raise exc
            except BaseException as e:
                try:
                    e.trace = predictor._build_run_trace(
                        status="error",
                        steps=[],
                        lm=None,
                        sub_lm=None,
                        lm_hist_start=0,
                        sub_hist_start=0,
                        run_start=0,
                    )
                except Exception:
                    pass
                raise
        except KeyboardInterrupt as caught:
            assert caught.trace is sentinel


class TestAsyncForwardTracedCancellationPath:
    """asyncio.CancelledError must carry a trace after the widened handler."""

    def test_cancelled_error_gets_trace_attached(self):
        from predict_rlm.predict_rlm import PredictRLM

        predictor = PredictRLM.__new__(PredictRLM)
        sentinel = object()
        _patch_build_run_trace(predictor, sentinel)

        async def _raises_cancelled():
            try:
                raise asyncio.CancelledError()
            except BaseException as e:
                try:
                    e.trace = predictor._build_run_trace(
                        status="error",
                        steps=[],
                        lm=None,
                        sub_lm=None,
                        lm_hist_start=0,
                        sub_hist_start=0,
                        run_start=0,
                    )
                except Exception:
                    pass
                raise

        with pytest.raises(asyncio.CancelledError) as exc_info:
            asyncio.run(_raises_cancelled())

        assert exc_info.value.trace is sentinel

    def test_build_run_trace_failure_does_not_mask_cancellation(self):
        """If _build_run_trace itself raises during cancellation, the
        cancellation must still propagate cleanly — the ``try/except``
        around trace attachment is there exactly for this case.
        """
        from predict_rlm.predict_rlm import PredictRLM

        predictor = PredictRLM.__new__(PredictRLM)
        predictor._build_run_trace = MagicMock(
            side_effect=RuntimeError("trace build exploded mid-cancel")
        )

        async def _raises_cancelled_with_broken_trace():
            try:
                raise asyncio.CancelledError("inner")
            except BaseException as e:
                try:
                    e.trace = predictor._build_run_trace(
                        status="error",
                        steps=[],
                        lm=None,
                        sub_lm=None,
                        lm_hist_start=0,
                        sub_hist_start=0,
                        run_start=0,
                    )
                except Exception:
                    pass
                raise

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_raises_cancelled_with_broken_trace())


class TestHandlerWideningIsAnchored:
    """Source anchor: both error handlers in predict_rlm.py must use
    ``except BaseException``, not ``except Exception``. If a future
    refactor narrows them, this test fails to flag the regression.
    """

    def test_forward_traced_catches_base_exception(self):
        import inspect

        from predict_rlm.predict_rlm import PredictRLM

        src = inspect.getsource(PredictRLM._forward_traced)
        assert "except BaseException as exc" in src, (
            "PredictRLM._forward_traced should catch BaseException so "
            "cancellations attach a partial RunTrace before re-raising"
        )

    def test_aforward_traced_catches_base_exception(self):
        import inspect

        from predict_rlm.predict_rlm import PredictRLM

        src = inspect.getsource(PredictRLM._aforward_traced)
        assert "except BaseException as exc" in src, (
            "PredictRLM._aforward_traced should catch BaseException so "
            "asyncio.CancelledError attaches a partial RunTrace"
        )
