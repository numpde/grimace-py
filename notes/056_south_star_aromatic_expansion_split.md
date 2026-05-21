# South Star Aromatic Expansion Split

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 173: Split next aromatic expansion family`

## Goal

Split aromatic expansion into subfamilies before admitting more support. The
South Star boundary should keep the aromatic text policy modular: molecule
facts, atom text, bond text, traversal, parse-back evidence, and output-order
deduplication must remain inspectable.

## Candidate Families

### 1. Markerless Aromatic Branches

Examples:

- `c1ccccc1C`
- `c1ccncc1C`
- `c1ccoc1C`

This was the best next implementation slice before `South Star 176`. At that
point, the support gate blocked these as `aromatic_ring_surface` because the
active aromatic contract only admitted unbranched markerless aromatic
monocycles. A local probe that replaced only the support-gate report with an
empty report showed that the existing graph-native traversal and renderer
already produced parse-back-correct support for these examples:

- `c1ccccc1C`: `60` outputs, all accepted by the current semantic oracle;
- `c1ccncc1C`: `120` outputs, all accepted by the current semantic oracle;
- `c1ccoc1C`: `80` outputs, all accepted by the current semantic oracle.

That does not prove the domain, but it shows the implementation path is likely
small and principled: extend the aromatic text policy from "ring atoms cover
all atoms" to "one markerless aromatic monocycle with acyclic non-aromatic
branches whose atoms and bonds already satisfy supported text policy."

### 2. Fused Aromatic Systems

Examples:

- naphthalene-like fused aromatic rings;
- fused heteroaromatic systems.

This should not be bundled with branches. It changes the ring-system contract:
multiple aromatic rings, shared aromatic atoms/bonds, and closure-edge-set
choice need explicit proof obligations. Even if the renderer can emit
lowercase aromatic text, support completeness is a ring-system question.

### 3. Modified Aromatic Atom Text

Examples:

- `c1cc[nH]c1`;
- charged or mapped aromatic atoms.

This is primarily an atom-text policy expansion. A gate-bypass probe still
fails on `[nH]` with the current atom-text boundary:
`South Star aromatic atom text currently requires unmodified atoms`. That is a
clean blocker and should not be mixed with aromatic branch traversal.

### 4. Aromatic Directional Overlays

Example:

- `c1ccccc1` with an aromatic bond carrying slash/backslash direction metadata.

This requires a named constraint family. It is not an atom/bond text tweak,
because the current `aromatic_text_policy` deliberately records
`undecided_aromatic_directional_overlay`.

## Original Recommendation

Implement markerless aromatic branches next.

The implementation should be narrow:

- keep `aromatic_text_policy` as the active policy;
- admit only one sanitized aromatic monocycle plus acyclic branches;
- require aromatic ring atoms to remain unmodified lowercase aromatic tokens;
- require aromatic ring bonds to use elided aromatic bond text;
- require branch atoms and branch bonds to satisfy existing non-aromatic text
  policies;
- keep fused aromatic systems, modified aromatic atoms, and aromatic
  directional overlays gated;
- pin at least benzene-methyl, pyridine-methyl, and furan-methyl fixtures;
- update the readiness matrix, benchmark artifact, docs, and package-readiness
  runner in the same slice.

This entered as its own implementation Backlog row. It was not a Decision:
the probe and current policy split made it the least-regret next aromatic
extension.

## Follow-up

`South Star 176` implemented this slice narrowly:

- the support gate now admits one markerless aromatic monocycle plus supported
  acyclic non-aromatic branches;
- toluene, methyl-pyridine, and methyl-furan fixtures pin the expanded support;
- fused aromatic systems, modified aromatic atom text, and aromatic
  directional overlays remain separate gated families.
