# Writer Marker-Slot Quotient

Branch: `stereo-constraint-model`

Date: 2026-05-19

## Purpose

The `github3967_part2_directional_ring_closure_canonical` witness shows that
the next fix is not another local deferred-token patch.  The missing RDKit
writer spelling is parser-equivalent to the intended molecule, but it is
rejected by the current strict token-phase and marker-row model.

This note fixes the next target boundary: preserve strict semantic token-phase
state, and add a separately named RDKit writer marker-slot quotient if runtime
support needs to admit RDKit spellings that are equivalent at the emitted
marker-slot level but not represented by the current local row path.

## Evidence

Commits:

- `3b90c79`: exposed per-attempt token-flip row survival diagnostics.
- `7500a10`: exposed base token-phase state for those attempts.
- `5aee100`: pinned the github3967 token-phase quotient boundary in tests.
- `95792a2`: exposed emitted marker slots for deferred attempts.
- `b02b3d4`: measured base marker-row survival before candidate flip
  filtering.

For prefix `C1=CC/C=C2\C3=C`, the RDKit-needed terminal candidate is `\`.
The current diagnostic facts are:

- emitted marker slots are exactly the RDKit target slots:
  `(5, "/"), (9, "\\"), (13, "\\")`;
- those slots are in the pinned parser-equivalent minimal marker-slot set;
- strict semantic token-phase state is already forced to `stored`;
- the candidate requires `flipped`, leaving zero candidate token-phase rows;
- even without the candidate-flip filter, applying the candidate marker events
  to the base token-phase rows leaves zero marker rows.

So the failure is not only "candidate flip filter too strict".  It is also not
a row-local marker placement acceptance under the existing base state.  The
current row model represents one local spelling basis; RDKit's emitted spelling
uses a different marker-slot basis for the same parser-backed stereo
assignment.

## Decision

Do not broaden semantic token-phase assignments to make this pass.

Semantic token-phase state should continue to mean: which local token/phase
choices preserve the intended stereo assignment under the model's chosen
carrier and marker-placement representation.

RDKit writer parity needs an additional writer-policy layer that can say:

1. this emitted marker-slot set is parser-equivalent for the intended molecule
   and skeleton;
2. this emitted marker-slot set is the one RDKit's writer emits for the current
   traversal/rooting regime;
3. accepting it does not license arbitrary parser-equivalent spellings.

That layer is a quotient over marker-slot representations, not a mutation of
the semantic token-phase facts.

## Rejected Shortcuts

Broaden token-phase rows
: Rejected.  It would blur semantic state with writer spelling policy and make
  token-phase facts stop meaning one thing.

Bypass the candidate token-flip filter
: Rejected.  `b02b3d4` shows the base marker rows also reject the emitted RDKit
  marker slots.

Accept by completed-string parse equivalence
: Rejected for runtime.  The public API targets exact RDKit writer support and
  online decoding; post-hoc completed-string repair violates both boundaries.

Special-case github3967
: Rejected.  A compatibility case without a named writer-policy predicate would
  recreate the procedural patch problem this branch is trying to remove.

## Target Shape

The support boundary should have two clearly separated predicates:

- semantic row support: current carrier, token-phase, marker-row, and
  observation facts leave a nonempty semantic state;
- RDKit writer marker-slot quotient: current emitted marker slots are accepted
  because they are a pinned/witnessed RDKit writer spelling inside the
  parser-equivalent marker-slot class for the same semantic state.

The quotient must be online.  It may use facts available at the current
frontier:

- emitted marker slots so far;
- candidate marker slot and marker;
- direction-erased skeleton prefix context;
- component and side identity;
- outstanding no-marker observations;
- parser-equivalent marker-slot basis compiled or precomputed from the
  prepared molecule.

It must not use a completed-output repair phase.

## Next Slices

1. Add a diagnostic object that names quotient eligibility separately from
   semantic row survival.  For github3967 it should report:
   `semantic_row_accepts=false`, `marker_slot_quotient_candidate=true`, and
   `rdkit_writer_target_slots=true`.
2. Move parser-equivalent marker-slot computation out of the known-gap test
   body into a small test helper so more witnesses can reuse it without
   duplicating RDKit parser logic.
3. Add at least one more pinned witness, if available, where RDKit emits a
   parser-equivalent marker-slot basis that differs from the current row-local
   basis.  If no second witness exists, record that explicitly.
4. Only after the quotient diagnostics are fixture-backed, prototype a runtime
   support path that consults the quotient as a named RDKit writer-policy
   fallback after semantic row support rejects a candidate.
5. Before enabling that path broadly, require:
   exact public invariants, pinned RDKit parity, known-gap movement only for
   explicitly promoted cases, and no growth of non-target same-skeleton
   support.

