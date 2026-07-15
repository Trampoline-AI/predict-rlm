# Runtime observability

PredictRLM can send ordered lifecycle evidence to custom loggers, monitoring
systems, GEPA pipelines, and durable evidence stores. This guide explains the
event sink contract and its delivery guarantees.

Use an `EventSink` when the consumer needs run lifecycle events. Use the returned
`RunTrace` when it only needs the completed RLM trace. An event sink observes the
runtime; it is not an input adapter or execution backend.

## Event model

Each `RunEvent` contains:

- `run_id`: invocation identity;
- `sequence`: order within that invocation;
- `kind`: a `RunEventKind` value;
- `timestamp_ns`: emission time; and
- `data`: event-specific structured fields.

Kinds cover run start/termination, input preparation, session lifecycle,
artifacts, generated code, iterations, `predict()` calls, tools, and output
materialization. Consumers should branch on `kind` and treat unrelated `data`
fields as event-specific.

## Implement a monitoring sink

An event sink is shared construction-time configuration, so it must support
concurrent runs. This JSONL example serializes writes with a lock. It is
best-effort monitoring (`strict = False`), not a durable evidence ledger.

```python
import asyncio
import json
import threading
from pathlib import Path

from predict_rlm import RunEvent


class JsonlEventSink:
    strict = False

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    async def emit(self, event: RunEvent) -> None:
        await self._append(event)

    async def flush(self, run_id: str) -> None:
        pass  # Each append opens, writes, and closes the file.

    async def close(
        self,
        run_id: str,
        terminal_event: RunEvent | None = None,
    ) -> None:
        if terminal_event is not None:
            await self._append(terminal_event)

    async def _append(self, event: RunEvent) -> None:
        payload = {
            "run_id": event.run_id,
            "sequence": event.sequence,
            "kind": event.kind.value,
            "timestamp_ns": event.timestamp_ns,
            "data": dict(event.data),
        }
        line = json.dumps(payload, default=str) + "\n"
        await asyncio.to_thread(self._write, line)

    def _write(self, line: str) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
```

Register sinks directly:

```python
rlm = PredictRLM(MySignature, events=[JsonlEventSink("runs/events.jsonl")])
```

Or package them with other host-side runtime contributions:

```python
from predict_rlm import RuntimeContribution


def monitoring_module() -> RuntimeContribution:
    return RuntimeContribution(events=(JsonlEventSink("runs/events.jsonl"),))
```

## Terminal events and sink lifecycle

The recorder calls:

1. `emit(event)` for non-terminal lifecycle events;
2. `flush(run_id)` before committing the terminal result; and
3. `close(run_id, terminal_event)` exactly once for the terminal event.

A sink must persist `terminal_event` in `close`; terminal events are not also sent
through `emit`. Sink instances may receive interleaved calls for different run
IDs, while `sequence` remains ordered within each run.

## Best-effort monitoring versus strict evidence

Set `strict = False` for monitoring that must not determine run correctness. Sink
failures are isolated from the primary workload where possible.

Set `strict = True` only when complete evidence is part of the run's correctness
contract—for example, a GEPA data ledger that must not publish a successful run
without durable trace evidence. A strict sink must provide real durability in
`flush` and `close`; an in-memory buffer or ordinary logger is not sufficient.
Strict sink failure can prevent success from being published.

`InMemoryEvidenceSink` is useful for tests and in-process evidence inspection. It
is not a replacement for a durable cross-process store.

## What to test

For a custom sink, verify:

- event order by `(run_id, sequence)`;
- concurrent runs do not corrupt or mix records;
- the terminal event is written once;
- `flush` makes prior events durable before success;
- `close` runs on success, failure, and cancellation; and
- strict and best-effort failure behavior matches the intended contract.

For the runtime lifecycle that produces these events, see
[Custom adapters and the runtime kernel](custom-adapters.md). For process and
backend boundaries, see [predict-rlm Architecture](../ARCHITECTURE.md).
