# Runtime Queueing Control

## Development baseline

- Feature branch: `feature/runtime-queueing-control`
- User fork remote: `origin https://github.com/weifang231/sglang-omni.git`
- User fork base: `eb72964c62dc38d6e98eb0a1cba5da5772dd7a10`
- Read-only official remote: `upstream https://github.com/sgl-project/sglang-omni.git`
- Implementation base: `1e6646be4ea0abfee5ddfb5044fcf9525dc98768`
- Upstream push URL: `DISABLED`

The feature branch was fast-forwarded from the user-fork base to the official
upstream commit above before implementation. No commits from this branch have
been pushed.

## Control contract

The primary mechanism is the coordinator-level, end-to-end class credit used by
the paper's `c_i` decision. A request is accepted into the coordinator queue,
acquires one credit immediately before dispatch to the entry stage, and releases
that credit exactly once when the full request completes, fails, or is aborted.
Credit decreases are non-preemptive. The optional per-stage gate is an extension
for mechanism studies and is disabled unless a stage explicitly configures it.
It is rejected on stream-receiver stages because payload-only admission cannot
bound work triggered by side-channel chunks, and it must not be described as the
paper's end-to-end `c_i` controller.

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

## Observability and boundaries

Queue snapshots expose the configured discipline and limits plus active/waiting
request counts globally and by class. Event traces record queue entry, credit
acquisition (including queue wait), release, and cancellation. The previously
validated scheduler, code2wav, and TTS vocoder JSON channels remain opt-in via
`OMNI_RUNTIME_METRICS_DIR` and `OMNI_RUNTIME_CONTROL_FILE`.

EDF controls admission order only. It does not preempt active requests or replace
the engine's token-level batching policy. The stage extension counts requests,
not outstanding chunks or bytes, and therefore is not an edge-flow-control
implementation. In a multi-worker router deployment the coordinator credit pool
is per worker rather than a globally shared cross-worker pool.

On supported non-stream-receiver stages, the optional stage gate controls
scheduler admission of the complete request payload. Stream-receiver stages
must instead rely on the primary coordinator credit until a bounded,
incarnation-aware chunk-credit protocol exists.
