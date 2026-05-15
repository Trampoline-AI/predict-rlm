from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from predict_rlm.telemetry import JsonlTelemetrySink, NoopTelemetrySink, TelemetryContext

from .reporting.cost import aggregate_costs_from_log, append_cost_rows
from .runtime.adapter import RLMGepaAdapter
from .runtime.lm_config import build_lm, validate_lm_env
from .runtime.utils import atomic_write_json
from .schema import (
    OptimizeConfig,
    OptimizeReport,
    ProjectValidation,
    RLMGepaProject,
    validate_project,
)


def run_optimization(
    project: RLMGepaProject,
    config: OptimizeConfig,
    *,
    command: str | None = None,
) -> OptimizeReport:
    validation = validate_project(project)
    run_dir, run_id = prepare_run_dir(project, config, command=command)
    telemetry_sink = (
        JsonlTelemetrySink(run_dir / "telemetry" / "events.jsonl")
        if config.telemetry_enabled
        else NoopTelemetrySink()
    )
    telemetry_context = TelemetryContext(
        sink=telemetry_sink,
        trace_id=run_id,
        run_id=run_id,
        telemetry_level=config.telemetry_level,
    )

    lm = build_lm(
        config.executor_lm,
        reasoning_effort=config.executor_reasoning_effort,
        cache=config.cache,
    )
    sub_lm = build_lm(
        config.executor_sub_lm,
        reasoning_effort=config.executor_sub_lm_reasoning_effort,
        cache=config.cache,
    )
    proposer_lm = build_lm(
        config.proposer_lm,
        reasoning_effort=config.proposer_reasoning_effort,
        cache=config.cache,
    )
    proposer_sub_lm = build_lm(
        config.proposer_sub_lm,
        reasoning_effort=config.proposer_sub_lm_reasoning_effort,
        cache=config.cache,
    )

    append_cost_rows(
        run_dir / "cost_log.jsonl",
        [
            {
                "schema_version": 1,
                "ts": datetime.now().isoformat(),
                "event_id": f"{run_id}_startup",
                "operation_id": "startup",
                "attempt_id": "startup",
                "event": "startup",
                "role": None,
                "model": None,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "cache_hits": 0,
            }
        ],
    )

    adapter = RLMGepaAdapter(
        project=project,
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=config.max_iterations,
        concurrency=config.concurrency,
        task_timeout=config.task_timeout,
        output_dir=run_dir,
        run_id=run_id,
        proposer_lm=proposer_lm,
        proposer_sub_lm=proposer_sub_lm,
        proposer_max_iterations=config.proposer_max_iterations,
        proposer_timeout=config.proposer_timeout,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        verbose_rlm=config.verbose_rlm,
        display_progress_bar=config.display_progress_bar,
        valset_size=len(validation.valset),
        telemetry_context=telemetry_context,
    )

    t0 = time.time()
    state = _run_gepa_engine(
        config=config,
        project=project,
        seed_candidate=validation.seed_candidate,
        trainset=list(validation.trainset),
        valset=list(validation.valset),
        adapter=adapter,
        run_dir=run_dir,
        seed=config.seed,
        reflection_lm=_reflection_lm_callable(proposer_lm),
    )
    elapsed = time.time() - t0

    candidates = list(getattr(state, "program_candidates", []))
    val_scores = list(getattr(state, "program_full_scores_val_set", []))
    if not val_scores:
        subscores = list(getattr(state, "prog_candidate_val_subscores", []))
        val_scores = [
            sum(scores.values()) / len(scores) if scores else 0.0
            for scores in subscores
        ]
    best_idx = max(range(len(candidates)), key=lambda idx: val_scores[idx]) if candidates else 0
    best_candidate = candidates[best_idx] if candidates else validation.seed_candidate
    costs = aggregate_costs_from_log(run_dir / "cost_log.jsonl")

    report = OptimizeReport(
        config=config,
        run_dir=str(run_dir),
        best_idx=best_idx,
        best_val_score=val_scores[best_idx] if val_scores else 0.0,
        total_candidates=len(candidates),
        total_metric_calls=int(getattr(state, "total_num_evals", 0)),
        duration_seconds=elapsed,
        best_candidate=best_candidate,
        val_aggregate_scores=val_scores,
        costs=costs,
    )
    write_summary_artifacts(run_dir, report, candidates)
    return report


def check_optimization(project: RLMGepaProject, config: OptimizeConfig) -> ProjectValidation:
    validation = validate_project(project)
    _validate_model_env(config.executor_lm)
    _validate_model_env(config.executor_sub_lm)
    _validate_model_env(config.proposer_lm)
    _validate_model_env(config.proposer_sub_lm)
    if config.run_dir is not None:
        parent = Path(config.run_dir).parent
        if not parent.exists():
            raise ValueError(f"run_dir parent does not exist: {parent}")
    return validation


def prepare_run_dir(
    project: RLMGepaProject,
    config: OptimizeConfig,
    *,
    command: str | None = None,
) -> tuple[Path, str]:
    run_dir = Path(config.run_dir) if config.run_dir is not None else _default_run_dir(project)
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists() and not config.resume:
        raise ValueError(f"run_dir already exists; pass --resume to reuse it: {run_dir}")
    if config.resume:
        state_path = run_dir / "gepa_state.bin"
        if not state_path.exists():
            raise ValueError(f"--resume requires existing checkpoint: {state_path}")
        (run_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        base_run_id = str(metadata.get("run_id") or f"run_{uuid.uuid4().hex[:8]}")
        run_id = f"{base_run_id}_resume_{uuid.uuid4().hex[:8]}"
        return run_dir, run_id

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "task_traces").mkdir()
    (run_dir / "proposer_traces").mkdir()
    (run_dir / "telemetry").mkdir()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "project_name": project.project_name,
        "created_at": datetime.now().isoformat(),
        "command": command,
        "resolved_config": config.to_dict(),
    }
    atomic_write_json(metadata_path, metadata)
    return run_dir, run_id


def write_summary_artifacts(
    run_dir: Path, report: OptimizeReport, candidates: list[dict[str, str]]) -> None:
    atomic_write_json(run_dir / "optimization_summary.json", report.to_dict())
    all_candidates = [
        {"idx": idx, "score": report.val_aggregate_scores[idx], **candidate}
        for idx, candidate in enumerate(candidates)
    ]
    atomic_write_json(run_dir / "all_candidates.json", all_candidates)


def _run_gepa_engine(
    *,
    config: OptimizeConfig,
    project: RLMGepaProject,
    seed_candidate: dict[str, str],
    trainset: list[Any],
    valset: list[Any],
    adapter: RLMGepaAdapter,
    run_dir: Path,
    seed: int,
    reflection_lm: Any,
) -> Any:
    from gepa.core.data_loader import ensure_loader
    from gepa.logging.experiment_tracker import create_experiment_tracker
    from gepa.logging.logger import StdOutLogger
    from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
    from gepa.strategies.candidate_selector import (
        CurrentBestCandidateSelector,
        EpsilonGreedyCandidateSelector,
        ParetoCandidateSelector,
    )
    from gepa.strategies.component_selector import (
        AllReflectionComponentSelector,
        RoundRobinReflectionComponentSelector,
    )
    from gepa.strategies.eval_policy import FullEvaluationPolicy
    from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper

    from .runtime.engine import RLMGepaEngine

    rng = random.Random(seed)
    logger = StdOutLogger()
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset)

    candidate_selector = {
        "pareto": lambda: ParetoCandidateSelector(rng=rng),
        "current_best": CurrentBestCandidateSelector,
        "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
    }[config.candidate_selection_strategy]()
    candidate_selector = _ProgressCandidateSelector(candidate_selector, adapter)
    component_selector = {
        "round_robin": RoundRobinReflectionComponentSelector,
        "all": AllReflectionComponentSelector,
    }[config.component_selection_strategy]()
    batch_sampler = _build_minibatch_sampler(
        project,
        train_loader,
        minibatch_size=config.minibatch_size,
        rng=rng,
    )
    experiment_tracker = create_experiment_tracker(
        use_wandb=False,
        wandb_api_key=None,
        wandb_init_kwargs=None,
        use_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name=None,
    )
    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_loader,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=component_selector,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        reflection_prompt_template=None,
    )
    stoppers = [
        FileStopper(str(run_dir / "gepa.stop")),
        MaxMetricCallsStopper(config.max_metric_calls),
    ]
    merge_proposer = None
    if config.merge_proposer:
        from .proposer.merge import RlmMergeProposer

        if len(seed_candidate) != 1:
            raise ValueError("merge_proposer currently supports single-component projects only")

        def evaluator_fn(inputs: list[Any], candidate: dict[str, str]) -> tuple[list[Any], list[float], Any]:
            result = adapter.evaluate(inputs, candidate, capture_traces=False, kind="merge_subsample")
            return result.outputs, result.scores, result.objective_scores

        merge_proposer = RlmMergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator_fn,
            adapter=adapter,
            trainset=train_loader,
            use_merge=True,
            max_merge_invocations=config.max_merge_invocations,
            max_rlm_merge_attempts=config.max_merge_attempts,
            min_each=config.merge_min_each,
            merge_minibatch_size=config.merge_minibatch_size,
            component_name=next(iter(seed_candidate)),
            rlm_merge_state_path=run_dir / "rlm_merge_state.json",
            rng=rng,
        )

    engine = RLMGepaEngine(
        adapter=adapter,
        run_dir=str(run_dir),
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type="instance",
        logger=logger,
        experiment_tracker=experiment_tracker,
        track_best_outputs=False,
        display_progress_bar=config.display_progress_bar,
        raise_on_exception=True,
        stop_callback=CompositeStopper(*stoppers),
        val_evaluation_policy=FullEvaluationPolicy(),
        use_cloudpickle=True,
        evaluation_cache=None,
    )
    with experiment_tracker:
        return engine.run()


class GroupAwareBatchSampler:
    def __init__(
        self,
        minibatch_size: int,
        group_id_fn: Callable[[Any], str | None],
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.group_id_fn = group_id_fn
        self.rng = rng or random.Random(0)
        self.batches: list[list[Any]] = []
        self.epoch = -1
        self.last_trainset_size = 0

    def next_minibatch_ids(self, loader: Any, state: Any) -> list[Any]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        if not self.batches or trainset_size != self.last_trainset_size:
            self._refresh(loader, state.i)
        else:
            curr_epoch = state.i // len(self.batches)
            if curr_epoch > self.epoch:
                self._refresh(loader, state.i)

        return self.batches[state.i % len(self.batches)]

    def _refresh(self, loader: Any, iteration: int) -> None:
        grouped_ids = self._group_ids(loader)
        self.rng.shuffle(grouped_ids)
        self.batches = self._pack_batches(grouped_ids)
        self.last_trainset_size = len(loader)
        self.epoch = iteration // len(self.batches)

    def _group_ids(self, loader: Any) -> list[list[Any]]:
        all_ids = list(loader.all_ids())
        examples = loader.fetch(all_ids)
        groups: dict[tuple[str, str | int], list[Any]] = {}
        for index, (data_id, example) in enumerate(zip(all_ids, examples, strict=True)):
            group_id = self.group_id_fn(example)
            group_key = ("group", group_id) if group_id is not None else ("ungrouped", index)
            groups.setdefault(group_key, []).append(data_id)
        return list(groups.values())

    def _pack_batches(self, grouped_ids: list[list[Any]]) -> list[list[Any]]:
        batches: list[list[Any]] = []
        current: list[Any] = []
        for group_ids in grouped_ids:
            if len(group_ids) > self.minibatch_size:
                if current:
                    batches.append(current)
                    current = []
                for start in range(0, len(group_ids), self.minibatch_size):
                    batches.append(group_ids[start : start + self.minibatch_size])
                continue

            current.extend(group_ids)
            if len(current) >= self.minibatch_size:
                batches.append(current)
                current = []

        if current:
            batches.append(current)
        return self._pad_batches(batches, grouped_ids)

    def _pad_batches(self, batches: list[list[Any]], grouped_ids: list[list[Any]]) -> list[list[Any]]:
        pad_groups = [group_ids for group_ids in grouped_ids if group_ids]
        if not pad_groups:
            return batches
        for batch in batches:
            pad_index = 0
            while len(batch) < self.minibatch_size:
                batch.extend(pad_groups[pad_index % len(pad_groups)])
                pad_index += 1
        return batches


def _build_minibatch_sampler(
    project: RLMGepaProject,
    train_loader: Any,
    *,
    minibatch_size: int,
    rng: random.Random,
) -> Any:
    from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

    if _loader_has_minibatch_groups(project, train_loader):
        return GroupAwareBatchSampler(
            minibatch_size=minibatch_size,
            group_id_fn=project.minibatch_group_id,
            rng=rng,
        )
    return EpochShuffledBatchSampler(minibatch_size=minibatch_size, rng=rng)


def _loader_has_minibatch_groups(project: RLMGepaProject, loader: Any) -> bool:
    all_ids = list(loader.all_ids())
    if not all_ids:
        return False
    return any(project.minibatch_group_id(example) is not None for example in loader.fetch(all_ids))


def _reflection_lm_callable(lm: Any) -> Any:
    def call(prompt: str) -> str:
        return _coerce_reflection_lm_text(lm(prompt))

    return call


class _ProgressCandidateSelector:
    def __init__(self, selector: Any, adapter: RLMGepaAdapter):
        self.selector = selector
        self.adapter = adapter

    def select_candidate_idx(self, state: Any) -> int:
        candidate_idx = self.selector.select_candidate_idx(state)
        self.adapter.set_reflective_progress_context(
            iteration=state.i,
            parent_idx=candidate_idx,
            child_idx=len(state.program_candidates),
        )
        return candidate_idx


def _coerce_reflection_lm_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        for key in ("output_text", "text", "completion", "content", "output", "choices", "message"):
            if key in value:
                return _coerce_reflection_lm_text(value[key])
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        texts = [_coerce_reflection_lm_text(item) for item in value]
        return "\n".join(text for text in texts if text)
    for attr in ("output_text", "text", "content", "message"):
        if hasattr(value, attr):
            return _coerce_reflection_lm_text(getattr(value, attr))
    raise TypeError(f"Reflection LM returned non-text response ({type(value).__name__})")


def _default_run_dir(project: RLMGepaProject) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{project.project_name}_{timestamp}"


def _validate_model_env(model_or_lm: Any) -> None:
    if isinstance(model_or_lm, str):
        validate_lm_env(model_or_lm)
