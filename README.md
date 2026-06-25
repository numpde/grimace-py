# South Star branch

South Star is a writer-state model for exact SMILES generation.

The branch treats a writer as a live transition system rather than as a
post-filter over completed strings. A retained state records the graph prefix,
ring labels, open and closed closures, branch obligations, text choices, and
stereo residual factors. A token is legal only when the transition that emits
it produces a coherent successor.

The working goal is a single authority for writer behavior:

```text
prepared facts + policy + retained writer state
    -> terminalization evidence
    -> legal next-token transitions
    -> graph-policy blockers
    -> execution-capability evidence
```

Connected declared components enter this live transition engine. Unsupported
policy is reported at the exact frontier where the required operation or
relation envelope is encountered. There is no topology-profile admission,
tree-only initial classifier, or recursive reachable-set preflight on the
runtime path.

Snapshots are structural records. Validation checks that a cursor is coherent
with the prepared facts and retained history; it does not independently decide
writer support. Checked choices, advance, count, stream, and resume all enforce
the same live blockers.

The recursive reachability audit is diagnostic instrumentation only. It records
reachable blockers and capability uses for review; it is not an admission
layer.

South Star remains work in progress. The branch is about making graph syntax,
closure lifecycles, and stereo constraints share one inspectable state model,
so future widening can be expressed as local relations instead of special-case
enumeration.
