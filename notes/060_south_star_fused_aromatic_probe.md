# South Star Fused Aromatic Probe

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 179: Probe fused aromatic ring-system obligations`

## Question

Can fused aromatic systems reuse the existing graph-native ring-system spine,
or do they need a new traversal model before support should be widened?

This note is a probe result. It does not change runtime support.

## Probe

The probe script is:

`tmp/exploration/aromatic_ring_system/001_probe_fused_aromatic_spine.py`

It deliberately bypasses the South Star support gate and calls private
ring-system traversal helpers directly. That is not a runtime path. The
purpose is to test whether existing pieces can already express:

- aromatic atom text;
- elided aromatic bond text;
- fused-ring closure-edge-set choices;
- traversal and branch events;
- first-occurrence output deduplication;
- RDKit parse-back and South Star semantic-oracle acceptance.

Witnesses:

- naphthalene: `c1ccc2ccccc2c1`;
- quinoline-like fused heteroaromatic: `c1ccc2ncccc2c1`;
- benzofuran-like fused heteroaromatic: `c1ccc2occc2c1`.

## Result

The existing spine can render parse-back-correct candidate supports for all
three witnesses when the fail-fast gate is bypassed:

| Witness | Atoms | Bonds | Rings | Closure edge sets | Traversals | Support | Semantic rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naphthalene | 10 | 11 | 2 | 35 | 1368 | 639 | 0 |
| quinoline-like | 10 | 11 | 2 | 35 | 1368 | 1368 | 0 |
| benzofuran-like | 9 | 10 | 2 | 29 | 944 | 944 | 0 |

The current support gate still correctly rejects them with:

- `aromatic_ring_surface`;
- `fused_or_polycyclic_ring`;
- `ring_molecule`.

That rejection is a policy boundary, not evidence that traversal/rendering is
incapable.

## Interpretation

Fused aromatic support looks implementable as a ring-system policy expansion,
not as a new renderer architecture.

The existing reusable pieces already appear to cover the mechanical surface:

- `_supported_polycyclic_closure_edge_sets` can choose spanning-tree closure
  edge sets for these molecules;
- aromatic atom text renders through the existing atom-text policy for
  unmodified aromatic atoms;
- aromatic bonds render through the existing elided aromatic bond-text policy;
- first-occurrence deduplication handles symmetric duplicate traversals.

The missing proof obligation is support completeness for the named fused
aromatic family. The implementation should not simply delete support-gate
categories. It should name the family and pin the closure-edge-set, atom-text,
bond-text, parse-back, and deduplication obligations.

## Recommendation

Open a separate implementation row for a narrow fused aromatic slice.

Initial scope:

- unmodified sanitized fused aromatic systems;
- no aromatic directional overlays;
- no modified aromatic atom text such as `[nH]`;
- no metal/dative/query surfaces;
- start with naphthalene, quinoline-like, and benzofuran-like fixtures.

Expected implementation shape:

1. Add a named support predicate for supported fused aromatic ring systems.
2. Route that predicate through the existing polycyclic ring traversal spine.
3. Add a unified-reference proof helper or extend the existing polycyclic
   proof helper so aromatic bond/atom obligations are explicit.
4. Pin fixtures for the three witnesses.
5. Update support-gate tests so modified aromatic atoms and aromatic
   directional overlays remain gated.
6. Refresh readiness counts, benchmark artifact, docs, and notes.

Do not bundle this with modified aromatic atom text. That is a separate
aromatic atom-text policy slice.
