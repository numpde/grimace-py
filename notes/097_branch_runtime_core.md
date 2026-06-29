# Branch-preserving writer runtime core

The `writer_shaped` runtime facade is now the boundary adapters must consume.
The next semantic center is therefore not a sibling runtime module, but a
branch-preserving transition core underneath `writer_runtime.py`.

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
projection must remain above the branch-preserving support layer.

## Current slice

`writer_runtime.py` exposes the existing checked text projection through
`writer_runtime_choice_transitions(...)`. Internally, it validates the retained
snapshot, runs one checked frontier schedule, records branch-preserving supports
from that schedule's raw `next_token_supports`, and projects the public text
choices from the same schedule outcome. It no longer derives runtime branch
supports from public text-choice entries or from a separately replayed text
snapshot.

```text
raw writer transitions
  -> checked frontier schedule outcome
  -> branch-preserving runtime supports
  -> branch-provenance successor states
  -> text/determinized runtime choices from the same schedule
  -> adapters: writer_online_decoder, writer_support, snapshots
```

This is still an intermediate inversion point. The checked frontier schedule
lives in `writer_frontier.py`; the public text snapshot remains the adapter
surface. The branch-preserving runtime layer now consumes the schedule's raw
supports beneath that text projection and can package exact successor runtime
states by branch provenance.

`count_writer_runtime_branch_completions(...)` is the first consumer of the
branch-preserving layer. It counts completions by memoized recursion over
canonical writer state keys and does not materialize support strings or route
through the support-image adapter. It uses the branch-provenance successor
packaging rather than advancing by emitted text.

## Guardrails

- `writer_support.py` remains an adapter and must not import writer frontier or
  transition internals.
- `writer_runtime.py` must not import support adapters, RDKit adapters, artifact
  code, or writer online decoder adapters.
- Branch counting must stay separate from support-string counting.
- Duplicate emitted text is legal below the text projection.
