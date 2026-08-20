# Fixed-order disconnected writer composition

## Product fact

`MoleculeFacts.components` owns fragment order. A nonnegative writer root fixes
the root of its containing component without moving that component; every other
component retains its ordinary root domain. The writer may emit `.` only after
the current component is graph-complete and before the declared root of the
next component becomes active.

`component_boundary` is therefore a graph/state relation, not text evidence.
The v11 support artifact and v3 branch-transition artifact classify the
existing `WriterComponentBoundaryEmitted` event under that relation without a
new transition term. The producer-free verifier resolves the exact source and
successor states, proves the completed component prefix, verifies the `i + 1`
cursor and root-vector rule, checks the new inactive root frame, and then gives
relation credit only after linked stereo obligations replay successfully.

Terminalization and component-boundary replay share the completed-prefix graph
partition checker. The terminal lane additionally requires the final component;
the boundary lane additionally proves that later components are untouched.

## Independent product gate

The seven pinned cases were decomposed into standalone RDKit fragments in the
same order. Each rooted component used its declared singleton root; every other
fragment used its full root domain. The global support and witness counts equal:

```text
support = {".".join(parts) for parts in product(*component_supports)}
completion_count = product(component_completion_counts)
```

This distinction matters for `CC.CC`: it has one support string and two writer
witnesses. The continuation/Rust recurrence retains that multiplicity.

## Authority table

| Field | Authority |
| --- | --- |
| Component order | `MoleculeFacts.components` |
| Root vector | runtime root domains |
| Current component | source writer state |
| Next component | exact `i + 1` rule |
| Separator | DOT transition kind |
| Next root | unchanged component root vector |
| Completed graph prefix | facts plus written and closed bonds |
| Future isolation | exact source/successor writer states |
| Local-order closure | existing residual/lifecycle semantic replay |
| Global support | fixed-order component product |
| Global completions | product of component witness counts |
| Online counts | unchanged continuation/Rust recurrence |

The continuation asset and Rust engine require no format or runtime changes:
`.` remains an ordinary deterministic emitted-text edge with exact counts and
replay-addressed local proof provenance.
