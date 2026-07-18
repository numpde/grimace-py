# Weighted writer continuation automata

The online count owner is a minimized acyclic automaton compiled only from the
counts-disabled checked writer frontier. Local branch and terminal artifacts
remain the semantic proof authorities; their certificate identities live in a
separate provenance index and never participate in continuation-node identity.

## Ownership split

The canonical node signature contains only future online behavior: primitive
terminal weight and, for each emitted text, primitive immediate multiplicity,
normalized successor scale, and the exact already-interned child semantic-class
identity. Cursor, state, branch, lifecycle, and transition identities are
provenance. Signature digests are derived integrity labels only; even a forced
digest collision cannot merge different continuations.

Cursor weights are normalized by their greatest common divisor. Support is
invariant under this scale; immediate, terminal, and completion multiplicities
scale linearly. The compiler checks the raw and primitive counts-disabled
frontiers whenever it encounters a non-unit scale.

Durable node IDs are assigned bottom-up by depth and exact signature after
minimization, so traversal order cannot affect the serialized core. The legacy
count path and all existing schemas and budgets remain unchanged.

## Durable replay-addressed asset

`writer_continuation_asset` v1 stores the source snapshot and compact core in
content-addressed chunks. Provenance chunks contain only cursor digests,
normalization scales, node IDs, canonical predecessor edge IDs, projection and
branch digests, and terminal-support digests. No cursor or writer-state term is
serialized outside the single source snapshot.

The manifest commits to ordered chunk descriptors rather than repeating the
records. Chunks are canonical JSON capped at 4 MB and are written through
temporary files before the verified directory is atomically renamed. Online
loading reads only the manifest and core chunk. A proof cursor additionally
tracks the raw cursor digest; local branch and terminal artifacts are rebuilt
on demand by replaying the canonical predecessor path through the
counts-disabled frontier.

Authority stays deliberately split:

- structural verification owns format, content addressing, canonical IDs,
  recurrence arithmetic, and graph/index consistency;
- live verification owns genuine cursor, projection, branch, and terminal
  identities;
- existing local artifact verifiers own chemistry and transition semantics.

## Baseline

The initial global envelope baseline exposed eight budgetless identity digests
in the existing directional-ring coupling envelope. After threading its
existing work budget through those calls, the count-DAG, frontier-count,
support-string, support-image, support-artifact, envelope-consistency,
work-budget, and boundary group passed 229 tests initially in 154.646 seconds
and remained green after durable-asset integration in 145.630 seconds.

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

The legacy provenance measurement includes complete raw and primitive cursor
terms. The durable asset replaces those terms with replay addresses; its exact
chunk sizing is recorded below after the full-root proceed gate.

| Durable source | Source bytes | Core chunk | Raw records | Primitive records | Edge records | Terminal records | Compact provenance | Chunks | Largest chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full shared-ring root | 31,786 | 874,597 | 6,547,216 | 4,294,108 | 10,656,137 | 1,171,970 | 22,669,431 | 10 | 3,999,960 |
| reduced pre-opening | 32,142 | 874,216 | 6,546,746 | 4,293,887 | 10,655,591 | 1,171,970 | 22,668,194 | 10 | 3,999,960 |
| reduced pre-pair | 32,405 | 1,733 | 3,026 | 2,061 | 4,416 | 1,350 | 10,853 | 6 | 32,405 |

The full compact provenance is 5.1 percent of the former 445,722,466-byte
embedded-cursor representation. Its largest individual record is 746 bytes.
The full manifest is 7,325 bytes.
The full root retains 2,101 semantic nodes, 2,843 semantic edges, and exact
support and completion counts of 3,744. All three sources remain under the
25 MB core, 50,000-edge, 64 MB compact-provenance, 4 MB chunk, and 1 MB
manifest gates without changing a production work budget.

## Durable field authorities

| Durable field | Independent authority |
| --- | --- |
| core node signature | exact child semantic-class IDs |
| signature digest | canonical exact signature |
| node ID | canonical bottom-up numbering |
| counts | automaton recurrences |
| root raw cursor | source snapshot |
| raw cursor node and scale | live reconstruction and normalization |
| predecessor edge | live text advance |
| projection digest | counts-disabled text projection |
| branch digests | checked live branch certificates |
| terminal digests | checked live terminal supports |
| finalized cursor digest | live terminal projection |
| local proof | existing branch or terminal artifact verifier |

## Runtime field authorities

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
