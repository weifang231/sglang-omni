# Runtime Queueing Control

Working requirements are maintained in
[Project Requirements and Audit](../../../experiments/workload/REQUIREMENTS.md).
This document describes runtime APIs; experiment settings belong to the run configuration.

## Control contract

The primary mechanism is coordinator-level, end-to-end class credit.
A request is accepted into the coordinator queue,
acquires one credit immediately before dispatch to the entry stage, and releases
that credit exactly once when the full request completes, fails, or is aborted.
Credit decreases are non-preemptive. The optional per-stage gate is an extension
for mechanism studies and is disabled unless a stage explicitly configures it.
It is rejected on stream-receiver stages because payload-only admission cannot
bound work triggered by side-channel chunks.

Both gates support stable FIFO or non-preemptive EDF ordering. EDF reads an
absolute Unix timestamp from request metadata; requests without a deadline sort
after requests with finite deadlines, and arrival sequence breaks ties. Per-class
limits are checked without head-of-line blocking: an ineligible class head does
not prevent another class with free credit from dispatching.

The default configuration contains no `queue_control` block and follows the
existing direct-dispatch path. A typical primary configuration is:

```yaml
queue_control:
  discipline: edf
  max_active_requests: 8
  max_waiting_requests: 64
  trust_request_metadata: true
  class_limits:
    text: 1
    speech: 7
```

Requests may carry the following metadata through `/v1/chat/completions`,
`/v1/audio/speech`, or `/generate`:

- `sglang_omni.request_class`: non-empty class name; defaults to `default`.
- `sglang_omni.first_output_deadline_unix_s`: finite absolute Unix timestamp.

These client-supplied scheduling hints are ignored by default. Deployments that
derive or authenticate them at a trusted ingress may set
`queue_control.trust_request_metadata: true`; doing so delegates class and EDF
priority assignment to that ingress and is not an authorization boundary.

The authenticated `POST /update_queue_control` endpoint replaces the live
limits and discipline. `scope: pipeline` updates the primary coordinator gate;
`scope: stage` forwards the update to the selected stages. Lowering a limit does
not cancel active work. The multi-worker router does not expose this endpoint:
cross-worker ordering, versioning, and configuration replay for newly registered
workers need a separate consistency protocol before that surface is safe.

`max_waiting_requests` bounds requests that have entered the runtime queue but
have not acquired a credit; active requests do not consume this waiting budget.
Lowering the bound does not evict accepted waiters; it rejects subsequent
requests that cannot dispatch immediately until the waiting count is below the
new bound.
At pipeline scope, omitting it uses the native generation `max_in_flight` value
as a safe default when that value is known. The native value remains an upper
bound on active dispatch even if a larger runtime limit is requested. With no
native bound, an omitted waiting limit is unbounded and should be set explicitly
for production use. Default-off configurations retain the original behavior in
which `max_in_flight` bounds all coordinator-owned requests.

## Observability and boundaries

Queue snapshots expose the configured discipline and limits plus active/waiting
request counts globally and by class. Event traces record queue entry, credit
acquisition (including queue wait), release, and cancellation. The scheduler,
code2wav, and TTS vocoder JSON channels remain opt-in via
`OMNI_RUNTIME_METRICS_DIR` and `OMNI_RUNTIME_CONTROL_FILE`.

EDF controls admission order only. It does not preempt active requests or replace
the engine's token-level batching policy. The stage extension counts requests,
not outstanding chunks or bytes, and therefore is not an edge-flow-control
implementation. In a multi-worker router deployment the coordinator credit pool
is per worker rather than a globally shared cross-worker pool.

An active abort releases the coordinator's logical credit after the abort
broadcast is accepted by the transport. The abort path has no per-stage
termination acknowledgement, so residual device work can briefly overlap a
successor after cancellation. Strict physical-WIP accounting across aborts
requires an acknowledgement protocol and is not provided by this revision.

On supported non-stream-receiver stages, the optional stage gate controls
scheduler admission of the complete request payload. Stream-receiver stages
must instead rely on the primary coordinator credit until a bounded,
incarnation-aware chunk-credit protocol exists.
