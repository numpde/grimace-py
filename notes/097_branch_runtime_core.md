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

`writer_frontier.py` owns checked branch supports, branch completion counts, and
diagnostics. `writer_snapshot.py` owns checked token-boundary snapshot
advancement. `writer_runtime.py` validates retained snapshots and wraps
frontier/snapshot products in public runtime dataclasses.

```text
raw writer transitions
  -> checked frontier schedule outcome
  -> checked frontier branch supports
  -> snapshot-owned token-boundary successor snapshots
  -> branch-local lifecycle events
  -> text/determinized runtime choices from the same schedule
  -> adapters: writer_online_decoder, writer_support, snapshots
```

Branch successor states are selected by branch provenance, not by emitted text.
Branch supports carry their exact successor cursor, and snapshot-owned
token-boundary packaging advances the decoder boundary for both checked branch
supports and checked text choices.

Closure lifecycle events are now constructed directly by the raw closure
transition factories in `writer_transitions.py`. Opening a closure endpoint
constructs allocation evidence before the endpoint event; pairing constructs
release evidence after the pair event. There is no import-time lifecycle
installer or post-construction event rewriting.

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
