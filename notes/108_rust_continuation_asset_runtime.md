# Rust continuation asset runtime

The continuation asset remains the durable authority for the online language,
while Rust owns execution of its already verified compact core. Python verifies
the manifest and core chunk before copying the root cursor, canonical nodes,
choices, and exact counts into immutable Rust-owned storage. Provenance, source
snapshots, prepared molecules, live frontiers, and chemistry terms do not cross
that boundary.

## Runtime ownership

Rust independently rejects invalid topology, unreachable nodes, cycles,
noncanonical node order, duplicate semantic classes, unsorted choices,
nonpositive scales, terminal-weight disagreement, and every support or
completion recurrence mismatch. Semantic equality uses exact child node IDs;
signature digests remain Python-verified integrity labels rather than an
equality authority. Counts and multiplicities use `BigUint`, including at the
PyO3 boundary, so runtime values are not narrowed to a machine integer.

The explicit `MolToSmilesContinuationDecoder.from_asset()` route does not
replace molecule-based decoders. Its state contains one shared immutable Rust
core, a Rust cursor, and the emitted-text sequence. Terminality comes directly
from the node flag and therefore remains independent of whether the node also
has choices. Snapshot prefixes are derived from the committed emitted-text
sequence rather than serialized as a second authority.

Proof-capable states additionally carry the asset's raw-cursor identity. Rust
advance and replay-addressed provenance advance are checked against one another
at every token. Existing branch-transition and terminalization artifacts remain
the only local semantic proof vocabularies and are reconstructed lazily by
Python. Asset publication has already reconstructed and verified every exact
branch and terminal locator structurally, live, and against facts before the
bundle becomes visible; lazy reconstruction retains the same proof boundary
for later callers.

## Field authorities

| Runtime field | Authority |
| --- | --- |
| asset identity | verified manifest digest |
| node and edge topology | verified core plus Rust topology checks |
| emitted text | canonical Rust token table |
| terminal availability | core node |
| support and completion counts | Rust recurrence checks |
| probability values | exact Rust integer counts |
| prefix | emitted-text sequence |
| core cursor | Rust node ID and completion scale |
| proof cursor | asset edge provenance |
| branch semantics | branch-transition artifact verifier |
| EOS semantics | terminalization artifact verifier |

## Validation and sizing

The focused cross-language suite compares every node and edge in a durable
fixture, exercises counts above `2**128`, a node that is both terminal and has
an outgoing choice, coherent core and snapshot mutations, core identity, and
lazy branch/EOS proof reconstruction. The small tetra, single/shared acyclic
directional, simple directional-ring, and non-single directional-ring support
images agree exactly with the counts-disabled writer.

The full shared-ring asset retained 2,101 nodes, 2,843 edges, and exact root
support/completion counts of 3,744. On the development host, verified Python
core opening took 75.6 ms and Python-to-Rust copying plus independent checking
took 102.0 ms after signature-digest replay was enabled. The measured
Rust-owned core was 547,158 bytes. Median-style
loop averages were 0.20 microseconds for root choices, 0.14 microseconds for
advance, 0.11 microseconds for count lookup, 0.20 microseconds for exact
probabilities, and 0.97 microseconds for decoder copy. Traversing and sorting
all 3,744 strings took 145 ms and produced SHA-256
`3892256f7910381403433cabd6314a4680011156948e29cac7d7383038da1a0b`.

These wall-clock values are diagnostics only; the hard runtime gate is an
immutable Rust resident core below 16 MB with no provenance or source-snapshot
access in core-only mode.
