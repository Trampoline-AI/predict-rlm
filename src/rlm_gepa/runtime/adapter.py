from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gepa import EvaluationBatch

from predict_rlm import PredictRLM
from predict_rlm.trace import extract_trace_from_exc

from ..proposer.rlm import (
    RLMInstructionProposer,
    _acall_with_heartbeat,
    build_merge_signature,
    sum_traces,
)
from ..reporting.cost import CostRow, append_cost_rows, append_trace_cost_rows
from ..schema import (
    EvaluationContext,
    RLMGepaExampleResult,
    RLMGepaProject,
    validate_example_result,
)
from .progress import install_rlm_log_stream, progress_write, restore_rlm_log_stream
from .trace_rendering import render_inputs, render_trace, trace_to_json
from .utils import atomic_write_json, run_coro_sync

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when optional dep is missing
    tqdm = None


class RLMGepaAdapter:
    def __init__(
        self,
        *,
        project: RLMGepaProject,
        lm: Any,
        sub_lm: Any,
        max_iterations: int,
        concurrency: int,
        task_timeout: int,
        output_dir: str | Path,
        run_id: str,
        proposer_lm: Any | None = None,
        proposer_sub_lm: Any | None = None,
        proposer_max_iterations: int = 20,
        proposer_timeout: int = 600,
        heartbeat_interval_seconds: float = 30.0,
        verbose_rlm: bool = False,
        display_progress_bar: bool = False,
        valset_size: int | None = None,
    ):
        self.project = project
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.concurrency = concurrency
        self.task_timeout = task_timeout
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.verbose_rlm = verbose_rlm
        self.task_trace_dir = self.output_dir / "task_traces"
        self.proposer_trace_dir = self.output_dir / "proposer_traces"
        self.cost_log_path = self.output_dir / "cost_log.jsonl"
        self._eval_counts: dict[str, int] = {}
        self._merge_proposer_call_count = 0
        self.proposer_lm = proposer_lm
        self.proposer_sub_lm = proposer_sub_lm or sub_lm
        self.proposer_max_iterations = proposer_max_iterations
        self.proposer_timeout = max(1, int(proposer_timeout))
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.display_progress_bar = display_progress_bar
        self.valset_size = valset_size
        self._last_minibatch_signature: tuple[str, ...] | None = None
        self._progress_label_override: str | None = None
        self._reflective_progress: dict[str, int] | None = None
        self._queued_valset_progress_label: str | None = None

        if proposer_lm is None:
            self.propose_new_texts = None
        else:
            proposer = RLMInstructionProposer(
                spec=project.agent_spec,
                lm=proposer_lm,
                sub_lm=proposer_sub_lm or sub_lm,
                output_dir=self.output_dir,
                cost_log_path=self.cost_log_path,
                max_iterations=proposer_max_iterations,
                timeout=proposer_timeout,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                run_id=run_id,
                component_focus=project.component_focus,
            )
            self.propose_new_texts = proposer.propose_new_texts

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
        kind: str | None = None,
    ) -> EvaluationBatch:
        return run_coro_sync(self._evaluate_async(batch, candidate, capture_traces, kind))

    def set_reflective_progress_context(
        self,
        *,
        iteration: int,
        parent_idx: int,
        child_idx: int,
    ) -> None:
        self._reflective_progress = {
            "iteration": iteration,
            "parent_idx": parent_idx,
            "child_idx": child_idx,
        }

    def queue_valset_progress_label(self, label: str) -> None:
        self._queued_valset_progress_label = label

    @contextmanager
    def progress_label(self, label: str):
        previous = self._progress_label_override
        self._progress_label_override = label
        try:
            yield
        finally:
            self._progress_label_override = previous

    async def _evaluate_async(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool,
        kind: str | None = None,
    ) -> EvaluationBatch:
        eval_kind = kind or self._infer_eval_kind(batch, capture_traces)
        if eval_kind == "minibatch":
            self._last_minibatch_signature = _batch_signature(batch)
        eval_idx = self._eval_counts.get(eval_kind, 0)
        self._eval_counts[eval_kind] = eval_idx + 1
        event_id = f"{self.run_id}_eval_{eval_kind}_attempt_{eval_idx:04d}"
        operation_id = f"eval_{eval_kind}_{eval_idx:04d}"

        context = EvaluationContext(
            lm=self.lm,
            sub_lm=self.sub_lm,
            max_iterations=self.max_iterations,
            task_timeout=self.task_timeout,
            output_dir=self.output_dir,
            kind=eval_kind,
            verbose_rlm=self.verbose_rlm,
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(index: int, example: Any) -> tuple[int, RLMGepaExampleResult]:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self.project.evaluate_example(candidate, example, context),
                        timeout=self.task_timeout,
                    )
                except asyncio.TimeoutError:
                    return index, RLMGepaExampleResult(
                        score=0.0,
                        feedback=f"evaluation timeout at {self.task_timeout}s",
                        traces=[],
                        example_id=_example_id(example),
                        error=f"timeout at {self.task_timeout}s",
                    )
                except Exception as exc:
                    trace = extract_trace_from_exc(exc)
                    return index, RLMGepaExampleResult(
                        score=0.0,
                        feedback=f"evaluation {type(exc).__name__}: {exc}",
                        traces=[trace] if trace is not None else [],
                        example_id=_example_id(example),
                        error=str(exc),
                    )
                validate_example_result(result)
                return index, result

        progress_label = self._progress_label(eval_kind, eval_idx, capture_traces)
        progress_bar = self._open_progress_bar(progress_label, len(batch))
        log_stream = install_rlm_log_stream(progress_label) if self.verbose_rlm else None
        tasks = [asyncio.create_task(run_one(index, example)) for index, example in enumerate(batch)]
        results_by_index: list[RLMGepaExampleResult | None] = [None] * len(batch)
        try:
            for done in asyncio.as_completed(tasks):
                index, result = await done
                results_by_index[index] = result
                if progress_bar is not None:
                    progress_bar.set_postfix_str(_progress_postfix(result, batch[index], index))
                    progress_bar.update(1)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if log_stream is not None:
                restore_rlm_log_stream(log_stream)

        if any(result is None for result in results_by_index):
            raise RuntimeError("evaluation completed without a result for every example")
        results = [result for result in results_by_index if result is not None]
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []
        have_objective_scores = False

        self.task_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.task_trace_dir / f"{event_id}_{eval_kind}.jsonl"
        with trace_path.open("x", encoding="utf-8") as f:
            for index, result in enumerate(results):
                example_id = result.example_id or str(index)
                outputs.append({"example_id": example_id, "score": result.score})
                scores.append(float(result.score))
                if result.objective_scores is not None:
                    have_objective_scores = True
                    objective_scores.append(dict(result.objective_scores))
                else:
                    objective_scores.append({})
                record = reflective_record(result)
                if trajectories is not None:
                    trajectories.append({"example_id": example_id, "task_id": example_id, "record": record})
                row = {
                    "schema_version": 1,
                    "event_id": event_id,
                    "operation_id": operation_id,
                    "example_id": example_id,
                    "candidate_id": None,
                    "kind": eval_kind,
                    "status": "error" if result.error else "completed",
                    "score": result.score,
                    "feedback": result.feedback,
                    "rlm_inputs": dict(result.rlm_inputs),
                    "trace": trace_to_json(result.traces[0]) if result.traces else None,
                    "traces": [trace_to_json(trace) for trace in result.traces],
                    "error": result.error,
                }
                f.write(json.dumps(row, default=str) + "\n")

        self._write_eval_cost(
            event=eval_kind,
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=f"attempt_{eval_idx:04d}",
            traces=[trace for result in results for trace in result.traces],
        )
        if eval_kind == "valset" and progress_label == self._queued_valset_progress_label:
            self._queued_valset_progress_label = None

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores if have_objective_scores else None,
        )

    def _infer_eval_kind(self, batch: list[Any], capture_traces: bool) -> str:
        if capture_traces:
            return "minibatch"
        if self._last_minibatch_signature == _batch_signature(batch):
            return "minibatch"
        if self.valset_size is not None and len(batch) != self.valset_size:
            return "minibatch"
        return "valset"

    def _progress_label(self, eval_kind: str, eval_idx: int, capture_traces: bool) -> str:
        if self._progress_label_override is not None:
            return self._progress_label_override
        if eval_kind == "valset" and self._queued_valset_progress_label is not None:
            return self._queued_valset_progress_label
        if eval_kind == "minibatch" and self._reflective_progress is not None:
            iteration = self._reflective_progress["iteration"]
            parent_idx = self._reflective_progress["parent_idx"]
            child_idx = self._reflective_progress["child_idx"]
            if capture_traces:
                return f"Iteration {iteration} Parent #{parent_idx} Minibatch"
            self._queued_valset_progress_label = f"Candidate #{child_idx} Valset"
            return f"Iteration {iteration} Child #{child_idx} Minibatch"
        return _progress_label(eval_kind, eval_idx)

    def _open_progress_bar(self, label: str, total: int) -> Any | None:
        if not self.display_progress_bar:
            return None
        if tqdm is None:
            raise ImportError("tqdm must be installed when display_progress_bar is enabled")
        return tqdm(
            total=total,
            desc=f"  {label} ({total} tasks)",
            leave=False,
            unit="task",
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        records = [trajectory["record"] for trajectory in trajectories if "record" in trajectory]
        return {component: list(records) for component in components_to_update if component in candidate}

    def _reserve_merge_proposer_call_idx(self) -> int:
        self._merge_proposer_call_count += 1
        return self._merge_proposer_call_count

    def _rlm_propose_merge_texts(
        self,
        *,
        call_idx: int,
        attempt_idx: int,
        id1: int,
        id2: int,
        ancestor: int,
        current_instructions_a: str,
        current_instructions_b: str,
        common_ancestor_instructions: str,
        paired_traces_file: Any,
        trace_task_ids: list[str],
    ) -> tuple[str, list[str]]:
        if self.proposer_lm is None:
            raise ValueError("merge proposer requires proposer_lm")
        event_id = (
            f"{self.run_id}_merge_attempt_{attempt_idx:04d}_"
            f"cand_{id1}_cand_{id2}_ancestor_{ancestor}"
        )
        operation_id = f"merge_proposer_cand_{id1}_cand_{id2}_{call_idx:04d}"
        signature = build_merge_signature(self.project.agent_spec)
        predictor = PredictRLM(
            signature,
            lm=self.proposer_lm,
            sub_lm=self.proposer_sub_lm,
            skills=[],
            max_iterations=self.proposer_max_iterations,
            verbose=True,
            debug=False,
        )
        progress_write("\n" + "=" * 80)
        progress_write(f"RLM MERGE PROPOSER STARTING (call {call_idx})")
        progress_write("=" * 80)
        log_stream = install_rlm_log_stream(f"MERGE PROPOSER {call_idx:04d}")
        try:
            result = run_coro_sync(
                _acall_with_heartbeat(
                    predictor.acall(
                        current_instructions_a=current_instructions_a,
                        current_instructions_b=current_instructions_b,
                        common_ancestor_instructions=common_ancestor_instructions,
                        paired_traces_file=paired_traces_file,
                    ),
                    tag=f"MERGE-PROPOSER {event_id}",
                    timeout=self.proposer_timeout,
                    heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                )
            )
        finally:
            restore_rlm_log_stream(log_stream)
        progress_write("=" * 80)
        progress_write(
            f"RLM MERGE PROPOSER DONE (call {call_idx}, "
            f"{len(getattr(result, 'trajectory', []))} iterations)"
        )
        progress_write("=" * 80 + "\n")
        new_text = getattr(result, "new_instructions", None)
        if not isinstance(new_text, str) or not new_text.strip():
            exc = RuntimeError(f"RLM merge proposer returned empty new_instructions for {call_idx}")
            exc.trace = getattr(result, "trace", None)  # type: ignore[attr-defined]
            raise exc
        audit = list(getattr(result, "generalization_check", None) or [])
        trace = getattr(result, "trace", None)
        self._write_trace_cost(
            event="merge_proposer_call",
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=f"attempt_{attempt_idx:04d}",
            main_role="merge_proposer",
            sub_role="merge_proposer_sub_lm",
            trace=trace,
        )
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "operation_id": operation_id,
            "status": "completed",
            "kind": "merge_proposer",
            "call_idx": call_idx,
            "attempt_idx": attempt_idx,
            "cand_ids": [id1, id2],
            "ancestor_id": ancestor,
            "trace_task_ids": trace_task_ids,
            "paired_trace_path": str(getattr(paired_traces_file, "path", paired_traces_file)),
            "current_instructions_a": current_instructions_a,
            "current_instructions_b": current_instructions_b,
            "common_ancestor_instructions": common_ancestor_instructions,
            "new_instructions": new_text,
            "generalization_check": audit,
            "rlm_trajectory": getattr(result, "trajectory", []),
            "run_trace": trace_to_json(trace),
            "error": None,
        }
        atomic_write_json(
            self.proposer_trace_dir / f"{event_id}_merge_from_cand_{id1}_and_cand_{id2}.json",
            payload,
        )
        return new_text, audit

    def _persist_merge_proposer_error(
        self,
        *,
        call_idx: int | None,
        attempt_idx: int,
        id1: int,
        id2: int,
        ancestor: int,
        instructions_a: str,
        instructions_b: str,
        instructions_ancestor: str,
        trace_task_ids: list[str],
        exc: BaseException,
    ) -> None:
        trace = extract_trace_from_exc(exc)
        event_id = (
            f"{self.run_id}_merge_attempt_{attempt_idx:04d}_"
            f"cand_{id1}_cand_{id2}_ancestor_{ancestor}"
        )
        operation_id = f"merge_proposer_cand_{id1}_cand_{id2}_{attempt_idx:04d}"
        self._write_trace_cost(
            event="merge_proposer_error",
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=f"attempt_{attempt_idx:04d}",
            main_role="merge_proposer",
            sub_role="merge_proposer_sub_lm",
            trace=trace,
        )
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "operation_id": operation_id,
            "status": "error",
            "kind": "merge_proposer_error",
            "cand_ids": [id1, id2],
            "ancestor_id": ancestor,
            "trace_task_ids": trace_task_ids,
            "current_instructions_a": instructions_a,
            "current_instructions_b": instructions_b,
            "common_ancestor_instructions": instructions_ancestor,
            "new_instructions": None,
            "generalization_check": None,
            "run_trace": trace_to_json(trace),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        suffix = "ERROR"
        atomic_write_json(
            self.proposer_trace_dir
            / f"{event_id}_merge_from_cand_{id1}_and_cand_{id2}_{suffix}.json",
            payload,
        )

    def _write_eval_cost(
        self,
        *,
        event: str,
        event_id: str,
        operation_id: str,
        attempt_id: str,
        traces: Sequence[Any],
    ) -> None:
        main_usage, sub_usage, main_model, sub_model, main_calls, sub_calls = sum_traces(traces)
        if event == "merge_trace_capture":
            main_role = "merge_trace_executor"
            sub_role = "merge_trace_sub_lm"
        else:
            main_role = "executor"
            sub_role = "sub_lm"

        rows: list[CostRow] = []
        if main_usage is not None and (main_usage.input_tokens or main_usage.output_tokens or main_usage.cache_hits):
            rows.append(
                CostRow(
                    event_id=event_id,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    event=event,
                    role=main_role,
                    model=main_model,
                    calls=main_calls,
                    input_tokens=main_usage.input_tokens,
                    output_tokens=main_usage.output_tokens,
                    cost_usd=main_usage.cost,
                    cache_hits=main_usage.cache_hits,
                )
            )
        if sub_usage is not None and sub_model and (
            sub_usage.input_tokens or sub_usage.output_tokens or sub_usage.cache_hits
        ):
            rows.append(
                CostRow(
                    event_id=event_id,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    event=event,
                    role=sub_role,
                    model=sub_model,
                    calls=sub_calls,
                    input_tokens=sub_usage.input_tokens,
                    output_tokens=sub_usage.output_tokens,
                    cost_usd=sub_usage.cost,
                    cache_hits=sub_usage.cache_hits,
                )
            )
        if rows:
            append_cost_rows(self.cost_log_path, rows)

    def _write_trace_cost(
        self,
        *,
        event: str,
        event_id: str,
        operation_id: str,
        attempt_id: str,
        main_role: str,
        sub_role: str,
        trace: Any,
    ) -> None:
        append_trace_cost_rows(
            self.cost_log_path,
            event=event,
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            main_role=main_role,
            sub_role=sub_role,
            trace=trace,
            sum_traces=sum_traces,
        )


def reflective_record(result: RLMGepaExampleResult) -> dict[str, str]:
    traces = [render_trace(trace, f"TRACE {i + 1}") for i, trace in enumerate(result.traces)]
    return {
        "Inputs": render_inputs(result.rlm_inputs),
        "Generated Outputs": "\n\n\n".join(traces) if traces else "(no traces)",
        "Feedback": result.feedback,
    }


def _example_id(example: Any) -> str | None:
    value = getattr(example, "task_id", None) or getattr(example, "example_id", None)
    return str(value) if value is not None else None


def _batch_signature(batch: Sequence[Any]) -> tuple[str, ...]:
    return tuple(_example_id(example) or repr(example) for example in batch)


def _progress_label(eval_kind: str, eval_idx: int) -> str:
    labels = {
        "minibatch": "MB",
        "valset": "VALSET",
        "merge_trace_capture": "MERGE TRACE",
        "merge_subsample": "MERGE SUBSAMPLE",
    }
    label = labels.get(eval_kind, eval_kind.replace("_", " ").upper())
    return f"{label} {eval_idx:04d}"


def _progress_postfix(result: RLMGepaExampleResult, example: Any, index: int) -> str:
    example_id = result.example_id or _example_id(example) or str(index)
    return f"{example_id}={result.score:.0%}"
