# South Star Excision Plan

Branch: `south-star`

Date: 2026-05-19

Base: forked from `stereo-constraint-model` at `cca357e`
(`Define South Star semantic enumeration`).

## Purpose

This note maps how to turn the current stereo-constraint-model branch into a
South Star branch: a principled semantic/maximally annotated stereo enumerator,
not an RDKit writer-parity runtime.

The word "excise" should be interpreted narrowly. We should remove or demote
North Star writer-parity rules from the South Star runtime path, but we should
not blindly delete useful infrastructure that happens to mention RDKit. RDKit
may remain useful as an input molecule provider, parser, and semantic
round-trip checker. What must leave the core is RDKit serializer policy.

## What Is North Star Bound

North Star-bound code exists to reproduce RDKit's
`canonical=False, doRandom=True` writer support exactly.

This includes:

- RDKit local/traversal writer layers in `StereoConstraintLayer`;
- helpers named `rdkit_writer_*`, `rdkit_marker_*`, or
  `rdkit_traversal_writer_*`;
- `RdkitTokenFlipAdjustmentObservations` and any token flip adjustment that
  exists only because RDKit chooses a traversal-conditioned spelling;
- marker-row filtering that treats `NoMarker` as an ordinary RDKit writer event;
- ring-closure marker projection that mirrors RDKit marker movement;
- committed-token parity filters whose purpose is rejecting RDKit-extra
  strings such as the diene witnesses;
- pinned RDKit exact-string fixture suites as required runtime tests;
- serializer-regression extraction, coverage, known quirks, and known gaps when
  they assert RDKit writer-string membership.

These can remain as archived comparison evidence, but not as South Star runtime
authority.

## What Is South Star Bound

South Star-bound code describes graph/stereo semantics and online support.

Keep or adapt:

- RDKit-to-prepared-graph input preparation, as long as it is named as input
  interop rather than writer policy;
- graph traversal/walker machinery that is not RDKit serializer-specific;
- component decomposition for stereo constraints;
- side domains and carrier assignment domains;
- direction-token algebra and flip/composition helpers;
- `StereoConstraintState` as the survivor-state idea;
- semantic layer facts and assignment filtering;
- parser-backed round-trip checks for graph and stereo assignment;
- Z3/tmp explorations as design evidence, not runtime dependencies;
- fixtures that encode semantic expectations rather than RDKit writer support.

The core semantic invariant is: every emitted prefix is supported by at least
one surviving semantic assignment, and every complete string parses to the
intended graph and stereo assignment.

## Target Runtime Shape

The South Star runtime should have one primary support boundary:

`SemanticFrontierSupportState`

It should consume:

- selected or observed traversal edges;
- carrier observations;
- emitted slash/backslash tokens;
- explicit no-marker observations only when the maximal-marker policy says an
  omission is a semantic decision;
- pending semantic marker obligations;
- ring/branch grammar facts needed to interpret emitted edge basis.

It should return:

- survivor semantic assignment ids or domains;
- forced carrier choices when unique;
- legal directional tokens at the current boundary;
- required marker obligations;
- terminal acceptance or a named semantic rejection reason.

It should not consume RDKit writer-policy facts.

## Maximal-Marker Replacement For RDKit Marker Rows

The current `StereoMarkerPlacementRow` is North Star shaped because it stores
subsets of neighbors that may receive visible markers and filters those subsets
against RDKit `MarkerPlaced` / `NoMarker` observations.

South Star should replace this with a semantic obligation model:

- determine which carrier edge observations are required by surviving semantic
  assignments;
- require a marker when the grammar reaches the edge boundary and the edge is
  required;
- branch on `/` and `\` only when multiple semantic assignments survive;
- reject omission when a required marker could have been emitted;
- carry an obligation only when the required marker cannot yet be emitted at the
  current grammar boundary.

This removes RDKit's "which valid markers should be suppressed as redundant?"
question from the semantic layer.

## Staged Excision

### Phase 0: Freeze The Baseline

Goal: make sure the branch starts from a known state.

Actions:

- Run `cargo test --lib`.
- Run `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`.
- Record any existing failures before deleting policy code.
- Do not edit runtime logic in this phase.

Exit criteria:

- Baseline behavior is known.
- The first South Star tests can distinguish semantic support from RDKit
  parity support.

### Phase 1: Add A Semantic Test Harness First

Goal: create a non-RDKit-parity target before removing RDKit-policy code.

Actions:

- Add `tests/semantic_stereo/` or `tests/south_star/`.
- Add tiny exact semantic cases: isolated alkene, oxime, conjugated diene
  witness, and one ring-closure witness.
- For each case, assert parser round-trip graph/stereo equivalence.
- Assert that known South Star-only strings are accepted by semantic tests even
  if RDKit writer parity does not emit them.
- Keep RDKit parity tests outside this harness.

Exit criteria:

- There is at least one test where South Star support is intentionally broader
  than RDKit writer support.
- Failing this test points to semantic support, not RDKit parity.

### Phase 2: Split The Constraint Layers

Goal: stop treating RDKit writer layers as peers of semantic state in the South
Star runtime.

Actions:

- Introduce a South Star-only semantic query path that uses only the semantic
  layer.
- Rename or wrap generic-looking support-boundary APIs so semantic and RDKit
  policy paths are not confused.
- Make `RdkitLocalWriter` and `RdkitTraversalWriter` unavailable to the South
  Star runtime path.
- Keep RDKit layers only under comparison diagnostics, if still useful.

Exit criteria:

- South Star support-state construction has no RDKit writer-policy inputs.
- RDKit layers are not needed for semantic test acceptance.

### Phase 3: Replace RDKit Marker Events With Semantic Obligations

Goal: remove RDKit marker-placement/minimization from semantic token support.

Actions:

- Add `SemanticMarkerObligation` or equivalent.
- Add obligation creation from surviving semantic assignments.
- Add obligation discharge when an edge boundary emits a directional marker.
- Reject omission of a required marker.
- Keep `StereoMarkerPlacementRow` only for RDKit comparison diagnostics or
  delete it if no diagnostics need it.

Exit criteria:

- `NoMarker` is no longer a normal support-shaping event in the South Star
  runtime.
- Legal slash/backslash tokens come from semantic survivor assignments.

### Phase 4: Delete RDKit Token-Flip Adjustments From The Semantic Path

Goal: make token orientation a semantic equation, not an RDKit spelling repair.

Actions:

- Remove `RdkitTokenFlipAdjustmentObservations` from South Star support.
- Replace "adjustment" facts with explicit semantic basis facts:
  component orientation, edge basis, emitted token basis, and carrier side.
- Keep legacy adjustment only in comparison diagnostics until tests no longer
  depend on it.

Exit criteria:

- A token flip is forced only by semantic assignment state and emitted-edge
  basis.
- RDKit-specific token adjustment names are absent from South Star runtime
  support.

### Phase 5: Remove RDKit Parity Gates From The South Star Test Path

Goal: prevent RDKit fixtures from defining South Star correctness.

Actions:

- Exclude `tests/run_pinned_rdkit_parity.py` from the South Star required test
  loop.
- Keep RDKit fixture tests only as optional comparison tests.
- Add a South Star runner that exercises semantic fixtures and parser
  round-trip checks.
- Document that RDKit exact-string failures are not South Star failures unless
  they reveal a semantic parse/assignment error.

Exit criteria:

- South Star CI/test command is independent of RDKit exact writer fixtures.
- RDKit comparison remains available but does not gate semantic support.

### Phase 6: Remove Or Quarantine Serializer-Mining Tooling

Goal: keep the South Star branch focused.

Actions:

- Move RDKit serializer miner scripts, upstream-source coverage, known quirks,
  and known gaps into optional comparison tooling or leave them untouched but
  outside the South Star test runner.
- Do not delete useful historical fixtures until the semantic harness is strong
  enough to replace their diagnostic role.

Exit criteria:

- The required South Star test surface does not depend on RDKit source snapshots
  or serializer membership fixtures.

### Phase 7: Simplify Public Surface Last

Goal: avoid API churn before the semantic core exists.

Actions:

- Keep current Python public API until the South Star internal query works.
- Then consider explicit internal or public names such as
  `MolToSmilesSemanticEnum`, `MolToSmilesSouthStarEnum`, or a mode flag only if
  the distinction is unavoidable.
- Do not overload `MolToSmilesEnum` silently with semantic support if users
  expect RDKit writer parity.

Exit criteria:

- The API makes it impossible to mistake South Star semantic support for RDKit
  writer support.

## Concrete First Slice

The first implementation slice should be deliberately small:

1. Add a South Star semantic test harness with one isolated alkene/oxime and one
   known RDKit-discrepant diene witness.
2. Add a semantic support query that runs only the existing semantic layer and
   reports survivor assignments for those cases.
3. Add an internal diagnostic function if needed; do not alter public runtime
   support yet.
4. Prove at least one South Star-only spelling parses to the intended graph and
   stereo assignment.
5. Only then begin cutting RDKit marker-event support out of the semantic path.

This first slice should not delete large RDKit fixture trees. It should create
the positive semantic target that makes later deletion safe.

## Deletion Rules

Delete only when all are true:

- a South Star semantic test covers the behavior formerly guarded by the
  North Star path;
- the code being deleted is clearly RDKit writer-policy, not graph/stereo
  semantics;
- optional comparison diagnostics no longer need the helper;
- `cargo test --lib` and the South Star semantic runner pass.

Quarantine instead of delete when:

- the helper is useful for comparing South Star support to RDKit support;
- the code also contains generic graph-preparation logic;
- the behavior is not yet covered by semantic fixtures.

## Expected End State

The `south-star` branch should end with:

- a semantic support-state boundary as the runtime source of truth;
- maximal directional-marker obligations instead of RDKit marker suppression;
- parser-backed semantic fixtures;
- optional RDKit comparison diagnostics;
- no production support decisions that depend on RDKit writer-policy layers;
- clear naming that separates semantic SMILES/stereo support from RDKit writer
  parity.
