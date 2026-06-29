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
`writer_runtime_choice_transitions(...)`. Internally, it records the
branch-preserving supports from the checked frontier schedule outcome's raw
`next_token_supports`, not from public text-choice entries. This keeps adapters
on the existing runtime facade while preparing the dependency direction we want:

```text
raw writer transitions
  -> branch-preserving runtime supports
  -> text/determinized runtime choices
  -> adapters: writer_online_decoder, writer_support, snapshots
```

This is an intermediate inversion point. The checked frontier schedule still
lives in `writer_frontier.py`; the public text snapshot remains the adapter
surface. The branch-preserving runtime layer now consumes the schedule's raw
supports beneath that text projection.

`count_writer_runtime_branch_completions(...)` is the first consumer of the
branch-preserving layer. It counts completions by memoized recursion over
canonical writer state keys and does not materialize support strings or route
through the support-image adapter.

## Guardrails

- `writer_support.py` remains an adapter and must not import writer frontier or
  transition internals.
- `writer_runtime.py` must not import support adapters, RDKit adapters, artifact
  code, or writer online decoder adapters.
- Branch counting must stay separate from support-string counting.
- Duplicate emitted text is legal below the text projection.
