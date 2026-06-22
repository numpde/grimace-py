# South Star branch

This branch is for South Star, a writer-state model for exact SMILES support.

A SMILES writer is usually exposed as a procedure that returns strings. South
Star treats it as a state machine. The state records graph traversal, branch
returns, ring labels, closure endpoints, atom and bond text choices, stereo
obligations, and terminal closure. A token is allowed only if applying it gives
a valid successor state.

## Questions

The branch is organized around these queries:

- Which complete strings can this prepared molecule emit?
- Which tokens are legal after this prefix?
- Which successor state does a token produce?
- Can this stored cursor resume without replaying its prefix?
- Which operation makes a molecule or transition unsupported?

It is not a parser-equivalence project. South Star keeps two checks separate:

- whether a string is meaningful for the prepared molecular facts;
- whether the writer model can emit that string.

## Model

The implementation is moving toward one live transition engine.

Graph state records the writer traversal, pending entries, branch frames, ring
labels, open closure endpoints, and closed closures. Stereo state is carried by
residual factors over live variables. Snapshot validation reconstructs residual
state from recorded writer history instead of trusting stored domains and
assignments.

The intended shape is:

```text
prepared facts
+ writer policy
+ current graph/ring/stereo state
=> legal next transitions
=> exact successor states
```

Enumeration, online decoding, snapshot resumption, capability auditing, and
diagnostics should all read the same state.

## Current Work

Recent commits have been replacing topology-derived positive admission with
live execution certificates. A retained transition records the writer
capabilities it used. A public admission is positive only when the reachable
transition set stays inside the supported capability envelope.

The current phase is about shared operations. The immediate case is a
ring-closure bond that also acts as a directional stereo carrier. That forces
the ring syntax relation and the stereo residual relation to be updated by the
same live transition.

## Repository Notes

The package is still distributed as `grimace-py` and imported as `grimace`.
This branch is not written as mainline package documentation; it is a working
branch for South Star internals.

Useful starting points:

- [South Star developer contract](python/grimace/_south_star1/DEV-CONTRACT.md)
- [Python API](docs/api/python.md)
- [Correctness contracts](docs/correctness-contracts.md)
- [Testing fixtures](docs/testing-fixtures.md)

## License

`grimace` is source-available under [PolyForm Noncommercial 1.0.0](LICENSE).
Commercial use requires a separate commercial license from the author.
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
