## Pipeline Overview

### Coordinator

`Coordinator` is the global request router. It registers stage endpoints, sends
new requests to the entry stage, receives `CompleteMessage` and `StreamMessage`
events, and resolves client futures or streams.

Key responsibilities:

- route new requests to `entry_stage`
- track request state: pending, running, completed, failed, aborted
- collect terminal stage completions
- merge results when a pipeline has multiple terminal stages, such as `decode`
  and `code2wav`
- broadcast abort messages to all stages

The coordinator is stage-implementation agnostic. In a tensor
parallel stage group, it only talks to rank 0. Peer ranks stay internal to the
stage group.

#### Admission policy

Native admission is the coordinator's in-flight cap (`max_running_requests +
max_queued_requests` of the generation stage, HTTP 503 when full). A deployment
can add its own decision behind that cap by naming a factory in
`PipelineConfig.admission_policy` (YAML `admission_policy:` or CLI
`--admission_policy module.make_policy`). The factory returns a policy object,
or `None` to keep native admission:

```python
class ConcurrencyCap:
    """Admit while fewer than `limit` admitted requests are still in the pipeline."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight: set[str] = set()

    def admit(self, request_id, request) -> bool:   # False -> HTTP 429
        if len(self.in_flight) >= self.limit:
            return False
        self.in_flight.add(request_id)
        return True

    def completed(self, request_id) -> None:        # terminal stage(s) succeeded
        self.in_flight.discard(request_id)

    aborted = completed                              # failed, aborted, or shutdown


def make_policy(*, config):
    return ConcurrencyCap(limit=8)
```

This static cap is the smallest policy that fits the interface; anything that
needs more than a count, such as the request itself or the time budget left,
goes in `admit` the same way. A deadline-aware policy can read a
deployment-defined field such as `request.metadata["deadline_monotonic_s"]`
and compare it with `time.monotonic()`; the deployment must fill that field
before submit, on the coordinator's clock. The hook adds no HTTP deadline field.

The contract:

- `admit(request_id, request)` runs synchronously on the coordinator loop, after
  the in-flight cap, request validation and routing resolution, for every submit
  that did not set `should_bypass_admission` (session control operations). It
  sees the `OmniRequest`. Returning `False`, or raising
  `AdmissionRejectedError(reason)`, rejects the request before anything reaches
  a stage; the speech endpoints answer HTTP 429 `rate_limit_error` /
  `admission_rejected`, distinct from the 503 of a full queue. Any other
  exception propagates and fails the request.
- Every admitted request gets exactly one release call: `completed` when its
  terminal stage(s) succeed, `aborted` on failure, abort, submit error,
  `fail_pending_requests` or `stop`. A release that raises is logged and not
  retried. `stop` also calls the optional `close` once.
- Callbacks must not block. They count logical requests in the coordinator, not
  GPU occupancy, and `aborted` does not mean every stage has stopped executing.
- The hook decides admission only; it does not schedule, reorder or preempt, and
  the in-flight cap stays in force underneath it.

##### Built-in policy: capacity table

`sglang_omni.admission_policies.capacity_table.make_policy` rejects, at
admission, the requests that are unlikely to produce their first output within
their deadline given how many requests are already executing. Above a
deployment's knee this trades a few early 429s for a goodput plateau instead of
a collapse (numbers in the PR that introduced it). It needs a *profile* fitted
from the deployment's own traffic; the three modes make that a loop that stays
inside the repository:

```yaml
admission_policy: sglang_omni.admission_policies.capacity_table.make_policy
admission_policy_options:
  mode: record                  # 1. admit everything, write events.jsonl
  record_path: /var/log/sgl-omni/admission-events.jsonl
```

```bash
# 2. fit: per occupancy, first-output latencies, departure rate, q and prices
python -m sglang_omni.admission_policies.fit_capacity_table admission-events.jsonl \
  --kind asr --deadline-s 0.5 --arrival-rate-rps 48 --output profile.json
```

```yaml
admission_policy_options:
  mode: shadow                  # 3. decide and record, never reject; then
  profile: profile.json         # 4. mode: apply
  arrival_rate_rps: 48          # re-solves the prices for the operating rate
```

The decision for a request arriving at occupancy `n` with `remaining` seconds
of budget is: admit iff `n < capacity` and `q(n, remaining) > prices[n]`, where
`q` is the recorded share of requests admitted at `n` whose first output came
within `remaining - guard_s`, and `prices[n]` is the average-reward value of a
slot solved from the fitted birth/death chain at `arrival_rate_rps`. `remaining`
is `request.metadata["deadline_monotonic_s"] - time.monotonic()` when the
deployment sets it, else the profile's `deadline_s`. The profile is specific to
the model, hardware class, SLO and operating rate; refit when any of them
changes (prices alone are re-solved from `arrival_rate_rps`). Single-route ASR
and TTS deployments are supported; `kind: tts` scores time to first audio.

### Stage

`Stage` is an IO shell. It handles all inter-stage communication. It receives control messages, reads
and writes relay payloads, performs fan-in when needed, and pushes all executable
work into `scheduler.inbox`.

```python
class Stage:
    def __init__(
        self,
        name,
        control_plane,
        relay,
        get_next,
        input_handler,
        scheduler,
        stream_targets,
        same_gpu_targets,
    ):
        self.scheduler = scheduler
```

Stage responsibilities:

- receive `SubmitMessage`, `DataReadyMessage`, `ShutdownMessage`, and profiler
  control messages over ZMQ
- receive `AbortMessage` over the coordinator broadcast channel
- read and write full `StagePayload` objects through relay
- aggregate inputs with `AggregatedInput` for fan-in stages
- route normal results to downstream stages or the coordinator
- route streaming chunks, including same-GPU CUDA IPC and cross-GPU relay
- drain `scheduler.outbox` and convert scheduler output into control-plane
  messages

The important invariant is that `Stage` does not branch on scheduler type.
`SimpleScheduler`, `OmniScheduler`, and streaming schedulers all present the
same surface.

### Scheduler

All schedulers implement the same interface:

```python
class Scheduler:
    inbox: Queue[IncomingMessage]
    outbox: Queue[OutgoingMessage]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

Scheduler messages are used to communicate with stage layer:

```python
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any

class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any
    target: str | None
    metadata: dict[str, Any] | None
```

#### OmniScheduler

`OmniScheduler` is used by autoregressive stages. It composes with SGLang's
upstream scheduler. The goal is to reuse SGLang's
batch selection, KV cache management, prefill/decode scheduling, and tree cache
while keeping SGLang-Omni's transport, request objects, and streaming behavior
outside the upstream scheduler. (Overlap scheduling is explicitly unsupported:
`OmniScheduler._event_loop_overlap` refuses to run because the
`Req.inflight_middle_chunks` decrement would lag one iteration on that loop.)

#### SimpleScheduler

`SimpleScheduler` is for non-AR stages such as preprocessing, encoders,
aggregation, and decode. It has no KV cache and no SGLang batching. The loop is:

```text
inbox.get() -> compute function -> outbox.put(result or error)
```

It supports a batch compute function for stages where local batching is
useful.

#### Code2WavScheduler

`Code2WavScheduler` is a streaming vocoder scheduler. It handles:

- `new_request`: initialize per-request state
- `stream_chunk`: accumulate and decode code chunks
- `stream_done`: flush remaining audio and emit a final result

### Model Runner

The model runner layer owns the AR forward path. The design target is:

```text
ForwardBatch -> before/custom forward hooks -> model forward -> post hook -> output processing
```

The shared base runner owns common mechanics: `ForwardBatch` construction,
sampling, logit processing, repetition penalty handling, output processing, and
conversion into scheduler output.

#### ThinkerModelRunner

`ThinkerModelRunner` is for Qwen-omni thinker-style AR models. Its model-specific job is
to prepare the forward batch by injecting multimodal embeddings such as image,
video, audio, and deepstack inputs before the model forward.

#### FeedbackARModelRunner

The refactor design identifies a shared `FeedbackARModelRunner` role for AR
models whose next decode step depends on feedback produced by the previous step
inside the same model runner. Qwen3-Omni talker and Fish Audio S2-Pro both fit
this shape; Qwen3 currently implements the pattern in its talker runner.

The abstraction covers self-contained feedback loops only:

- write previous-step feedback into model buffers before forward
- run the AR backbone and secondary head inside model `forward()`
- extract codebook outputs and feedback tensors after forward
- push stream or result output to the scheduler outbox

Cross-stage feedback, where the producer and consumer live in different
schedulers and communicate through relay, is out of scope for this runner.

The design groups model-specific feedback behavior into a small strategy:

```python
class FeedbackStrategy:
    def write_buffers(self, model, schedule_batch, requests) -> None: ...
    def extract_output(self, model, schedule_batch, requests, outbox) -> None: ...
    def prefill_forward(self, tp_worker, forward_batch, ...) -> object | None: ...
```
