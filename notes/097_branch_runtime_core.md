# Branch-preserving writer runtime core

The `writer_shaped` runtime facade is now the boundary adapters must consume.
The semantic center is the branch-preserving transition core underneath
`writer_runtime.py`.

## Contract

A checked writer step is:

```text
one retained writer state
  -> one selected raw transition support
  -> one exact residual update already carried by that support
  -> one exact successor writer state
```

The emitted token text is payload. It is not branch identity. Same-text supports
may be grouped by the current public text/determinized projection, but that
projection remains above the branch-preserving support layer.

## Current shape

`writer_runtime.py` exposes the checked text projection through
`writer_runtime_choices(...)` and `writer_runtime_choice_transitions(...)`, but
both now route through the branch-preserving runtime surface. Internally, the
runtime validates the retained snapshot, runs one checked frontier schedule,
records branch-preserving supports from that schedule's raw `next_token_supports`,
and projects the public text choices from the same schedule outcome. Runtime no
longer derives branch supports from public text-choice entries or from a
separately replayed text snapshot.

```text
raw writer transitions
  -> checked frontier schedule outcome
  -> branch-preserving runtime supports
  -> branch-provenance successor states
  -> branch-local lifecycle events
  -> text/determinized runtime choices from the same schedule
  -> adapters: writer_online_decoder, writer_support, snapshots
```

Branch successor states are selected by branch provenance, not by emitted text.
They still use the same token-boundary snapshot packaging as checked text
successors so decoder-boundary accounting observes one selected writer step.

Ring label allocation/release is carried on raw writer transition event streams.
Opening a closure endpoint carries label allocation evidence before the endpoint
event; pairing a closure endpoint carries label release evidence after the pair
event. The state update remains authoritative: allocated/reusable label
accounting is read from the exact branch successor.

The lifecycle installer is now restricted to closure-transition factory
installation. The installed closure factories construct the lifecycle events
before calling the raw `_transition(...)` constructor, so `_transition(...)`,
stereo advancement, runtime preservation, and lifecycle validation all see the
same event stream. The remaining scaffold is the import-time factory
installation point itself; the next cleanup is to move these closure-factory
bodies directly into `writer_transitions.py` and delete the installer.

`count_writer_runtime_completions(...)` and
`count_writer_runtime_branch_completions(...)` both consume the branch-preserving
layer. They count completions by memoized recursion over canonical writer state
keys and do not materialize support strings or route through the support-image
adapter. Distinct-string support count and support streaming remain text adapters.

## Guardrails

- `writer_support.py` remains an adapter and must not import writer frontier or
  transition internals.
- `writer_runtime.py` must not import support adapters, RDKit adapters, artifact
  code, or writer online decoder adapters.
- Branch counting stays separate from support-string counting.
- Duplicate emitted text is legal below the text projection.
