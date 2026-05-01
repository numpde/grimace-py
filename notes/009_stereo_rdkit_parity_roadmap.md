# Stereo RDKit Parity Roadmap

Branch: `stereo-constraint-model`

## Purpose

This note expands the straight-line plan for turning the current stereo
constraint-model work into exact RDKit writer parity.

The target is not merely to make the known failing strings match. The target
is a maintainable implementation where every stereo writer decision has a
clear layer:

- `Semantic`: chemically/language-valid stereo carrier choices.
- `RdkitLocalWriter`: local RDKit writer cleanup behavior.
- `RdkitTraversalWriter`: RDKit serializer policy tied to traversal, ring
  labels, branch context, and marker placement.

The public contract remains RDKit writer parity for the supported runtime:
`canonical=False`, `doRandom=True`, current RDKit writer convention, and the
supported writer flag subset.

## Non-Negotiable Constraints

- Enumeration and decoding remain online. Do not add a terminal cleanup pass
  that rewrites finished SMILES.
- Deferred choices are acceptable only if they are represented as online
  constraints and do not bias support distribution.
- Complexity should stay in the current shape: small per-component constraint
  state, not global postprocessing over complete supports.
- RDKit-specific behavior must be named as RDKit-specific. Do not collapse it
  into the semantic SMILES layer.
- Runtime behavior changes require a pinned RDKit-version witness and a small
  model-level explanation before implementation.
- Z3 remains exploratory only. Production and CI must not depend on Z3.

## Current Baseline

The current branch has already improved the situation structurally, but not
yet by changing runtime output support.

Implemented:

- `StereoConstraintModel` scaffold in Rust.
- Version-pinned stereo model fixtures under
  `tests/fixtures/stereo_constraint_model/2026.03.1.json`.
- Diagnostic `_stereo_constraint_output_facts` path.
- Explicit local RDKit writer hazard layer.
- Pinned marker-sequence and marker-slot mismatch evidence.
- Marker-slot diagnostics in `directional_spelling["marker_slots"]`.

For the current minimal witness:

- Grimace current support size: `28`.
- RDKit sampled support size: `28`.
- Exact string overlap: `12`.
- Direction-erased skeleton overlap: `28`.
- Marker sequence mismatches: `4`.
- Test-side ring-closure marker move explains marker sequence parity.
- After that candidate move, exact overlap is `18/28`.
- The remaining `10` residual differences are marker-slot placement gaps.

Interpretation:

The known witness no longer looks like a broad support-shape problem. It is a
RDKit serializer spelling problem: where slash/backslash markers are placed on
an otherwise matching direction-erased skeleton.

## Target Architecture

At startup, Grimace should build a static stereo model for each independent
stereo component:

- endpoint side domains
- candidate carrier edges
- selected-carrier variables
- token orientation or phase variables
- local writer exclusions
- traversal-policy hooks that can consume online facts

During traversal, the walker should emit facts into that model:

- selected carrier side and neighbor
- emitted graph edge
- emitted token orientation
- tree edge vs ring open vs ring close
- ring label identity
- atom output context
- branch context
- bracket atom / atom-token context where it matters to RDKit spelling

The model should answer two questions:

- Can this partial assignment still complete under a layer?
- If a directional marker is required now or later, where is the RDKit writer
  policy allowed to place it?

The second question is the new piece. It should be stated in terms of graph
and traversal facts, not finished-string surgery.

## Phase 1: Provenance Diagnostics

Goal:

Expose enough runtime provenance to explain each emitted `/` or `\` marker
without changing enumeration.

Implementation:

- Add a diagnostic-only trace structure for directional markers.
- For each emitted directional marker, record:
  - marker slot in the direction-erased skeleton
  - marker token
  - emitted graph edge `(begin_idx, end_idx)`
  - canonical edge
  - bond index
  - component index
  - side ids touched by that edge
  - selected side, if unique
  - emitted role: tree edge, ring open, ring close, branch edge, or deferred
  - ring label for ring open/close
  - atom being emitted when the marker becomes visible
  - whether the marker is before a bracket atom or before/after a ring label
- Keep the diagnostic API private under `_core`.

Likely files:

- `rust/src/rooted_stereo.rs`
- `tests/integration/test_stereo_constraint_model.py`

Tests:

- Existing pinned stereo model test must still pass.
- Add assertions that every marker slot in `directional_spelling` has exactly
  one provenance row.
- Assert provenance rows reconstruct the current marker-slot list.
- Assert provenance uses graph ids, not only string-local context.

Exit criteria:

- The 4 marker-sequence moves and 10 residual slot moves can be described with
  provenance facts rather than raw string positions.
- No runtime support changes.

Expected commit:

- `Expose stereo marker provenance diagnostics`

## Phase 2: Fact-Based Classification

Goal:

Replace string-local residual classifications with graph/traversal classes.

Implementation:

- In tests, classify each mismatch by provenance:
  - carrier emitted on ring-open edge but RDKit places marker before the
    matching ring-close label
  - carrier emitted before a ring atom but RDKit places marker before bracket
    atom
  - branch-child marker placement differences
  - any remaining unclassified cases
- Move the current residual context counts to stronger provenance counts.
- Keep the fixture as the source of expected cases, but avoid encoding the
  rule as skeleton strings.

Likely files:

- `tests/helpers/stereo_constraint_model.py`
- `tests/fixtures/stereo_constraint_model/2026.03.1.json`
- `tests/integration/test_stereo_constraint_model.py`

Tests:

- Assert every residual slot transition maps to exactly one provenance class.
- Assert there are no unclassified residual transitions.
- Assert current candidate classes account for `10/10` residual slot gaps.

Exit criteria:

- We can state the minimal witness rule in traversal terms.
- The fixture still pins exact strings as evidence, but the test explanation
  uses graph/traversal facts.

Expected commit:

- `Classify stereo marker gaps by traversal provenance`

## Phase 3: Constraint Vocabulary

Goal:

Add explicit model vocabulary for traversal writer facts without using it to
prune or respell runtime output yet.

Implementation:

- Add typed fact variants to `StereoConstraintFact`, or a sibling
  traversal-fact structure, for:
  - `CarrierEdgeEmitted`
  - `RingCarrierOpened`
  - `RingCarrierClosed`
  - `DirectionalMarkerPlaced`
  - `BranchChildEntered`
- Add a `RdkitTraversalWriter` projection function that can consume a complete
  fact set and compute predicted marker placement for the pinned witness.
- Keep the projection diagnostic-only.

Likely files:

- `rust/src/bond_stereo_constraints.rs`
- `rust/src/rooted_stereo.rs`

Tests:

- Rust unit tests with small synthetic component facts.
- Python integration test asserts projection agrees with the pinned witness.
- Projection must distinguish `Semantic`, `RdkitLocalWriter`, and
  `RdkitTraversalWriter`.

Exit criteria:

- The model can express the known marker-placement policy without hard-coded
  skeleton strings.
- Runtime support remains unchanged.

Expected commit:

- `Add traversal stereo fact vocabulary`

## Phase 4: Diagnostic RDKit Spelling Projection

Goal:

Produce the RDKit spelling projection for the minimal witness as a diagnostic
artifact.

Implementation:

- Add an internal diagnostic that returns:
  - current Grimace SMILES
  - current marker facts
  - predicted RDKit marker slots
  - projected RDKit-style SMILES
  - layer/classification that caused each move
- Use graph/traversal facts from Phase 3, not string replacement.

Tests:

- For the minimal witness, projected support equals RDKit sampled support:
  `28/28`.
- Current support remains unchanged until the next phase.
- Projection must preserve parse identity for every projected string.

Exit criteria:

- We have a graph/traversal-derived projection that closes the known witness
  gap in diagnostics.
- Any projection disagreement is pinned as a new residual category.

Expected commit:

- `Project RDKit stereo marker placement diagnostically`

## Phase 5: Online Runtime Integration

Goal:

Make the walker emit RDKit-style marker placement online, not by repairing
completed SMILES.

Implementation:

- Identify the current emission points that produce the marker too early.
- Replace local marker emission with a deferred online fact where needed.
- Emit the marker when the corresponding RDKit traversal-policy condition is
  reached.
- Preserve decoder behavior: next-token support must come from the same online
  state machine.
- Do not add a terminal rewrite pass.

Likely risky areas:

- `defer_coupled_component_phase_if_begin_side_is_unresolved`
- `commit_coupled_component_phase_from_deferred_part`
- `forced_shared_candidate_neighbor`
- `emitted_edge_part_generic`
- `emitted_isolated_edge_part`
- ring open/close action construction in `rooted_stereo`

Tests:

- Minimal witness exact support equals pinned RDKit support.
- Decoder can walk every pinned RDKit output.
- Token inventory agrees with the exact support.
- Existing non-stereo and atom-stereo tests remain unchanged.

Exit criteria:

- The minimal witness exact support changes from `12/28` overlap to exact
  `28/28` RDKit parity.
- No post-hoc cleanup exists.
- Complexity stays local to the active traversal state.

Expected commit:

- `Emit RDKit stereo marker placement online`

## Phase 6: Replace Suspicious Heuristics

Goal:

Remove or shrink old heuristic stereo code only after model rules explain the
same behavior.

Implementation:

- For each suspicious helper, write a behavior-preserving test around the
  model explanation before editing it.
- Replace one helper at a time.
- Keep commit boundaries small.

Suspicious helpers already marked in code:

- `defer_coupled_component_phase_if_begin_side_is_unresolved`
- `commit_coupled_component_phase_from_deferred_part`
- `should_defer_unknown_two_candidate_side_commit`
- `forced_shared_candidate_neighbor`
- `emitted_edge_part_generic`
- `emitted_isolated_edge_part`

Tests:

- Full focused stereo suite after each helper replacement.
- Known stereo gap tests.
- Pinned exact RDKit support tests.

Exit criteria:

- Old heuristics are either removed or reduced to thin model adapters.
- Each remaining heuristic has a named reason and a pinned witness.

Expected commits:

- `Replace deferred stereo phase heuristic`
- `Replace forced shared stereo carrier heuristic`
- `Replace isolated stereo token repair heuristic`

## Phase 7: Corpus Expansion

Goal:

Avoid overfitting the minimal witness.

Implementation:

- Re-run the RDKit serializer triage against:
  - reduced porphyrin witness
  - known stereo gaps
  - RDKit upstream serializer cases
  - adversarial ring/branch/bracket variants
- Promote new cases into pinned fixtures only when they are small and
  explanatory.
- Keep RDKit version keying strict.

Tests:

- Add exact support fixtures where support is small enough.
- Add sampled/projection fixtures where exact support is too large.
- Add known-quirk fixtures for RDKit behavior that is strange but mirrored.

Exit criteria:

- No new class of marker-placement gap appears in the small corpus.
- Larger cases either pass or have pinned, classified residuals.

Expected commit:

- `Expand pinned stereo traversal corpus`

## Phase 8: CI and Release Gate

Goal:

Make stereo parity hard to regress.

Implementation:

- Keep fast exact fixtures in normal CI.
- Keep larger sampled/miner checks opt-in or scheduled.
- Add a short note in docs if runtime behavior changes in a release.

Required gates before merging/release:

- `cargo fmt`
- `cargo test`
- focused pinned RDKit fixture tests
- public decoder/token inventory tests
- known stereo gap tests
- full Python test suite if runtime behavior changes

Exit criteria:

- CI covers the minimal witness exact parity.
- Release notes clearly distinguish:
  - behavior changes
  - diagnostic/test-only changes
  - RDKit-specific compatibility scope

## Implementation Rules

- Do not change runtime support in the same commit that adds a new diagnostic
  fact. First observe, then classify, then implement.
- Do not introduce fixture fields unless a test consumes them.
- Do not hard-code skeleton strings in runtime code.
- Do not silently broaden the public contract beyond the supported RDKit
  writer regime.
- Any runtime mismatch fix must improve an exact pinned parity assertion.
- If a rule cannot be stated in graph/traversal terms, keep it in
  `tmp/exploration/` until it can.

## Immediate Next Commit

The next commit should be Phase 1:

`Expose stereo marker provenance diagnostics`

Minimum useful payload:

- Extend marker diagnostics with graph edge and ring/branch role provenance.
- Assert every marker in the minimal witness has a provenance row.
- Keep runtime output unchanged.

This is the necessary bridge from string-slot observations to a real
`RdkitTraversalWriter` rule.
