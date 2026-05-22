# South Star 49-Row Frontier Checkpoint

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 178: Re-rank frontier after 49-row checkpoint`

## Purpose

This checkpoint re-ranks the South Star frontier after `South Star 176` and
`South Star 177`. It records the current private semantic surface and maps the
next Backlog order without changing runtime behavior.

## Current Private Surface

The checked semantic fixture surface is now:

- exact first-domain directional cases: `5`;
- expanded-support cases: `44`;
- total benchmark/readiness rows: `49`;
- public API blocker cases in the current readiness matrix: `0`;
- temporary-witness-backed current cases: `0`;
- regression-witness-backed current cases: `0`.

The newest supported families are:

- markerless aromatic monocycles with supported acyclic branches, pinned by
  toluene, methyl-pyridine, and methyl-furan;
- bracket-only non-organic atom text in the first non-metal slice, pinned by
  `[SiH3]C` and `[SeH]`.

Both families are private, fixture-backed, and routed through the shared
unified-reference/readiness machinery. They do not change the public
`MolToSmilesEnum` RDKit writer-parity API.

## Remaining Frontier

The remaining frontier is still broad enough that `MolToSmilesEnumS` should
stay private:

- broader aromatic systems outside the narrow fused-ring slice;
- broader aromatic atom-symbol breadth such as `[se]`;
- aromatic directional overlays;
- broader ordinary atom text beyond the first `Si`/`Se` slice;
- ordinary bond text beyond single/double/triple/aromatic, especially `$`;
- metal and dative chemistry;
- query atoms and query bonds;
- ring/tetrahedral and polycyclic/stereo interactions not yet covered;
- under-specified non-token stereo facts.

These are not one problem. Each needs its own molecule-fact contract,
renderer obligations, parse-back relation, and support-completeness story.

## Re-Ranked Next Work

1. Probe fused aromatic ring-system obligations.

   This is strategically highest because it tests whether the current
   graph-native ring-system spine generalizes from non-aromatic polycycles and
   markerless aromatic monocycles to fused aromatic systems. It should be a
   probe/note first, not an implementation commit, because support completeness
   depends on closure-edge-set choice and shared aromatic atoms/bonds.

2. Probe modified aromatic atom text.

   This is separate from fused rings. A witness such as `c1cc[nH]c1` mainly
   stresses bracket aromatic atom text and aromatic/Kekule parse-back policy.
   It should not be bundled with fused traversal.

3. Probe quadruple bond text.

   `C$C` is likely a small bond-text policy expansion, but it is less
   strategically important than aromatic breadth. It still needs an explicit
   OpenSMILES/RDKit parse-back statement before implementation.

4. Defer metals, dative bonds, and query semantics.

   These are semantic-universe changes, not renderer breadth. They should stay
   gated until the package needs them as a distinct product surface.

## Backlog Mapping

The current Backlog rows match this order:

- `South Star 179: Probe fused aromatic ring-system obligations`;
- `South Star 180: Probe modified aromatic atom text`;
- `South Star 181: Probe quadruple bond text policy`.

If any probe shows the target is implementable without weakening the
one-truth spine, it should spawn a separate implementation row. If it exposes a
semantic-policy choice, convert the probe row to `Decision` or open a narrower
Decision row before implementation.

## Export Posture

The export recommendation remains private continuation. The new rows reduce
frontier breadth, but they do not yet make the semantic enumerator broad enough
for a stable public API. An experimental export still needs an explicit
Decision row.

## Follow-Up After South Star 182

`South Star 182` promotes the first fused aromatic subfamily from frontier to
the checked private surface. The current counts are now:

- exact first-domain directional cases: `5`;
- expanded-support cases: `53`;
- total benchmark/readiness rows: `58`.

The new supported family is narrow: unmodified sanitized fused aromatic ring
systems with lowercase aromatic atom text, elided aromatic bond text, no
directional overlays, and no modified aromatic atom text.

## Follow-Up After South Star 184

`South Star 184` promotes the first modified-aromatic atom-text subfamily from
frontier to the checked private surface. The admitted cases cover pinned
bracket-aromatic nitrogen forms (`[nH]`, `[15nH]`, `[n:7]`, `[nH+]`, and
`[n+]([O-])`) through the shared monocycle/branch traversal spine, typed
atom-text obligations, elided aromatic bond text, parse-back evidence, and
first-occurrence deduplication.

The remaining aromatic frontier is therefore broader aromatic atom-symbol
breadth such as `[se]`, aromatic directional overlays, and broader aromatic
systems outside the currently named fused-ring and modified-atom-text slices.

## Follow-Up After South Star 186

`South Star 186` promotes the narrow ordinary quadruple-bond text case
`C$C` from frontier to checked private surface. The remaining bond-type
frontier is no longer ordinary `$` text; it is query/unspecified, dative, and
other nonstandard bond semantics.

The current counts are now:

- exact first-domain directional cases: `5`;
- expanded-support cases: `54`;
- total benchmark/readiness rows: `59`.

## Follow-Up After South Star 187

`South Star 187` promotes the first bracket-only aromatic selenium text case
`[se]1cccc1` from frontier to checked private surface. The remaining aromatic
element frontier is broader than selenium: tellurium also needs atom-symbol
admission, while the tested arsenic and silicon SMILES examples sanitize/write
as non-aromatic Kekule bracket atoms rather than the same aromatic text slice.

The current counts are now:

- exact first-domain directional cases: `5`;
- expanded-support cases: `57`;
- total benchmark/readiness rows: `62`.

`South Star 191` promotes the narrow tellurium text case `[te]1cccc1` from
frontier to checked private surface. Selenium and tellurium now share the
bracket-only aromatic element-text proof path; mapped variants remain separate
modifier-composition work.

`South Star 192` promotes mapped selenium `[se:7]1cccc1` as the first
bracket-only aromatic element modifier-composition case. Mapped tellurium and
other selenium modifiers remain separate frontier slices.
