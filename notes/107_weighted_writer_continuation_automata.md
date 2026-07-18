# Weighted writer continuation automata

The online count owner is a minimized acyclic automaton compiled only from the
counts-disabled checked writer frontier. Local branch and terminal artifacts
remain the semantic proof authorities; their certificate identities live in a
separate provenance index and never participate in continuation-node identity.

## Ownership split

The canonical node signature contains only future online behavior: primitive
terminal weight and, for each emitted text, primitive immediate multiplicity,
normalized successor scale, and the successor signature digest. Cursor, state,
branch, lifecycle, and transition identities are provenance. Exact signature
terms are compared before nodes are merged, so a digest collision cannot merge
different continuations.

Cursor weights are normalized by their greatest common divisor. Support is
invariant under this scale; immediate, terminal, and completion multiplicities
scale linearly. The compiler checks the raw and primitive counts-disabled
frontiers whenever it encounters a non-unit scale.

The runtime core is intentionally not a durable artifact in this package. The
legacy count path and all existing schemas and budgets remain unchanged.

## Baseline

The initial global envelope baseline exposed eight budgetless identity digests
in the existing directional-ring coupling envelope. After threading its
existing work budget through those calls, the count-DAG, frontier-count,
support-string, support-image, support-artifact, envelope-consistency,
work-budget, and boundary group passed 229 tests in 154.646 seconds.

## Shared-ring sizing

The probes patch the legacy count envelope, count DAG, and support-string
enumerator to fail if invoked.

| Source | Raw cursors | Primitive cursors | Semantic nodes | Edges | Depth | Core bytes | Provenance bytes | Compile time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full shared-ring root | 19,595 | 19,595 | 2,101 | 2,843 | 27 | 1,457,372 | 445,722,466 | 1,908.970 s |
| reduced pre-opening | 19,594 | 19,594 | 2,100 | 2,842 | 26 | 1,456,755 | 445,693,207 | 1,979.299 s |
| reduced pre-pair | 8 | 8 | 5 | 6 | 4 | 3,390 | 152,266 | 0.331 s |

The full-root support and completion counts are both exactly 3,744. Its largest
equivalence class contains 3,744 primitive cursors and its maximum out-degree
is five.

The provenance measurement includes the in-memory cursor terms needed to
recompute and lazily select local branch or terminal proofs. It is deliberately
outside the compact online core; chunking or serialization belongs to the next
package.

## Field authorities

| Compiled field | Independent authority |
| --- | --- |
| terminal availability | counts-disabled terminal supports |
| terminal multiplicity | terminal-support parent weights |
| emitted text | text projection |
| immediate multiplicity | live deterministic choice |
| successor cursor | text projection |
| successor scale | normalized successor weights |
| support count | automaton recurrence |
| completion count | weighted automaton recurrence |
| node equivalence | exact future-signature equality |
| branch provenance | checked branch certificates |
| terminal provenance | checked terminal-support identities |
| root cursor | source snapshot |
