# Token-Flip Replacement Alternatives

Branch: `stereo-constraint-model`

## Context

The current branch has a useful boundary: deferred token support is now
answered by `StereoConstraintState`, and diagnostics expose the input facts
used by the remaining procedural `inferred_component_token_flip` helper.

The remaining problem is narrower than before. The model can already consume
token-flip facts and filter token-phase assignments. It cannot yet derive
those token-flip facts from online traversal observations. That derivation is
still procedural and uses component phase, begin atom, selected begin side,
first emitted candidate, selected token, and RDKit token-adjustment rules.

The next design question is where that derivation should live.

## Alternative A: Keep A Token-Flip Fact Adapter

Shape:

- Keep computing one `StereoTokenFlipFact` per runtime component.
- Rename the current procedural path as an adapter, not a model decision.
- Feed the fact into `StereoConstraintState`.
- Make diagnostics and tests prove every adapter-produced fact has named input
  facts and is forced by the model.

Advantages:

- Smallest behavior-preserving step.
- Keeps current support unchanged.
- Provides a narrow seam for deleting old branches incrementally.

Problems:

- Still derives a final token flip procedurally.
- Remains partly circular: the model validates a fact produced by old logic.
- Does not express why a traversal observation permits or forbids token-phase
  rows.

Verdict:

Useful as a temporary bridge only. It should not be the final replacement.

## Alternative B: Add Typed Token-Observation Facts

Shape:

- Add facts that describe traversal observations before they collapse to a
  token flip:
  - runtime component;
  - begin atom;
  - begin side;
  - selected carrier neighbor;
  - selected carrier base token;
  - first emitted carrier candidate, if known;
  - component phase if known;
  - named RDKit writer adjustment facts.
- Add model APIs that filter token-phase assignment ids by those facts.
- Keep the old inferred flip in shadow mode and assert equivalence.

Advantages:

- Moves from "computed answer" to "observed constraints".
- Preserves the existing compact token-phase table.
- Lets each suspicious branch become a named fact-to-filter rule.
- Makes RDKit-specific adjustments searchable as writer facts instead of
  anonymous local XORs.

Problems:

- The token-phase table currently stores only final token flips, so filtering
  by richer observations may still need a small derived rule layer.
- Component phase and first-emission timing are not yet first-class model
  variables.

Verdict:

Best next implementation step. It is principled enough to reduce dual truth,
but small enough to review and keep behavior unchanged.

## Alternative C: Expand Token-Phase Rows

Shape:

- Change `StereoTokenPhaseAssignment` from `{neighbor_assignment_id,
  token_flips}` to a richer row containing:
  - neighbor assignment id;
  - runtime component token flips;
  - compatible component phases;
  - compatible begin sides or begin atoms;
  - compatible first-emitted candidate states;
  - named writer-adjustment state.
- Online facts directly filter these rows.

Advantages:

- More mathematically direct.
- Reduces procedural conversion code.
- Makes token phase a true assignment dimension rather than a final bit.

Problems:

- Larger product table.
- Harder to review because it changes model construction and runtime
  filtering at once.
- Some fields are traversal-time facts, not static molecule facts, so forcing
  them into startup rows may blur the boundary with marker obligations.

Verdict:

Likely part of the final shape, but too large for the next slice.

## Alternative D: One Unified Stereo Assignment Table

Shape:

- Build one per-component table covering carrier choices, token phases,
  traversal-token constraints, and marker obligations.
- Walker state stores only remaining row ids plus pending marker obligations.

Advantages:

- Cleanest single-source-of-truth story.
- Avoids adapters between carrier state and token-phase state.

Problems:

- Too big a jump from current code.
- Marker obligations are online and timing-dependent; including them too early
  risks either overfitting current witnesses or hiding algorithmic complexity.

Verdict:

Long-term direction, not the next implementation.

## Alternative E: Native Propagator Instead Of Explicit Rows

Shape:

- Store variables and constraints directly.
- Apply local propagation after each traversal fact.
- Use explicit rows only for small components or as a test oracle.

Advantages:

- Avoids row explosion if larger components appear.
- Closer to the Z3 exploration language.

Problems:

- More machinery than current witnesses justify.
- Harder to audit than explicit assignment ids.
- Premature until explicit rows fail on real cases.

Verdict:

Keep as a fallback design if explicit component rows become too large.

## Recommended Path

Use Alternative B first, with Alternative A only as a transition seam.

The next implementation slice should:

1. Add a typed `StereoTokenObservationFact` enum beside
   `StereoTokenFlipFact`.
2. Add a model query that filters token-phase assignment ids by token
   observation facts.
3. Populate observation facts from the same inputs currently exposed in
   `token_flip_inference_inputs`.
4. In diagnostics, show:
   - old inferred token flip;
   - token observation facts;
   - token-phase rows before observations;
   - token-phase rows after observations;
   - forced token flip after observations.
5. Assert in tests that the observation-filtered forced flip equals the old
   inferred flip for all current outputs.
6. Only then replace one procedural branch at a time with the observation
   filter.

This keeps the branch behavior-preserving while moving the source of truth
from "compute a flip" toward "filter assignments by typed online facts".

## Empirical Check

`tmp/exploration/stereo_assignment/029_compare_token_flip_replacement_shapes.py`
compares the alternatives against current pinned stereo-constraint diagnostics
for RDKit 2026.03.1.

Current result:

- 45,168 output-component observations have inferred token flips.
- All observed rows come from two pinned cases:
  `minimal_nonstereo_double_hazard` and
  `reduced_porphyrin_traversal_coupling`.
- The only completed-output branch currently exercised is
  `isolated_selected_begin_side`.
- The adapter shape is internally consistent: the model forced flip matches
  the adapter-produced inferred flip in 45,168/45,168 rows.
- The existing token-phase dimension is sufficient for these witnesses:
  token-phase rows reduce exactly 2x after the inferred token constraint in
  45,168/45,168 rows.
- The required observation shape is stable in current fixtures:
  component phase, component begin atom, begin side, selected begin neighbor,
  selected begin token, and RDKit token-flip adjustment.

Interpretation:

The current fixtures do not justify expanding startup token-phase rows yet.
They justify adding typed token-observation facts first, because the existing
token-phase table already has the needed final flip dimension. The missing
piece is a principled fact-to-filter query that derives the final flip from
named observations instead of from `inferred_component_token_flip`.

## Implementation Plan

Target: introduce token-observation filtering in shadow mode. This should not
change runtime support yet.

### Commit 1: Model-Side Observation Types

Files:

- `rust/src/bond_stereo_constraints.rs`

Work:

- Add `StereoTokenObservationFact` beside `StereoTokenFlipFact`.
- Start with the observed stable shape:
  - runtime component id;
  - component phase as stored/flipped;
  - selected begin-side token `/` or `\`;
  - RDKit token-flip adjustment as a named boolean fact.
- Keep the type intentionally minimal. Do not encode molecule names, case ids,
  or branch names.
- Add helper conversion that maps one complete observation fact to the implied
  `StereoTokenFlip` for the current `isolated_selected_begin_side` branch.

Tests:

- Rust unit tests for stored/flipped phase, `/`/`\` selected token, and
  adjustment true/false.
- Rust unit tests that invalid token strings are rejected.

Exit criteria:

- The model has a typed representation of the currently observed token
  inference input shape.
- No runtime code uses it yet.

### Commit 2: Token-Phase Filtering Query

Files:

- `rust/src/bond_stereo_constraints.rs`

Work:

- Add a query parallel to
  `token_phase_assignment_ids_for_neighbor_assignment_ids`, for example
  `token_phase_assignment_ids_for_observation_facts`.
- The query should:
  - validate that every observation references a runtime component inside the
    queried model component;
  - convert each observation to an implied token flip;
  - reuse the existing token-flip filtering machinery rather than duplicating
    row filtering.
- Conflicting observations for the same runtime component should return an
  empty assignment set.
- Unknown runtime components should be an explicit error, matching
  `StereoTokenFlipFact` validation.

Tests:

- Existing fixture-like two-runtime-component token-phase unit tests should
  cover:
  - one observation forces one runtime component;
  - two observations force both runtime components;
  - conflicting observations empty the state;
  - out-of-component observations error.

Exit criteria:

- Observation facts can filter token-phase rows without calling
  `inferred_component_token_flip`.

### Commit 3: Diagnostic Fact Extraction

Files:

- `rust/src/rooted_stereo.rs`

Work:

- Add a helper that emits `StereoTokenObservationFact`s from
  `token_flip_inference_inputs` for rows whose branch is currently
  `isolated_selected_begin_side`.
- Keep unsupported branches explicit: emit no observation fact and expose a
  diagnostic reason such as `unsupported_observation_branch`.
- Add Python diagnostic fields under each `component_token_phase` row:
  - `token_observation_facts`;
  - `token_observation_assignment_count_before`;
  - `token_observation_assignment_count_after`;
  - `token_observation_forced_flip`;
  - `token_observation_matches_inferred_flip`.
- Do not route runtime behavior through these fields yet.

Tests:

- Extend `tests/integration/test_stereo_constraint_model.py`:
  - every current inferred row has one observation fact;
  - observation-filtered forced flip equals legacy inferred flip;
  - before/after assignment counts match the existing 2x reduction;
  - branch coverage remains explicit, currently only
    `isolated_selected_begin_side`.

Exit criteria:

- Diagnostics prove the observation path explains all current completed-output
  inferred flips.

### Commit 4: State Boundary Adapter

Files:

- `rust/src/bond_stereo_constraints.rs`
- `rust/src/rooted_stereo.rs`

Work:

- Add a `StereoConstraintState::from_observation_facts` or equivalent adapter
  that accepts carrier facts plus token-observation facts.
- In diagnostics, show both:
  - state from legacy token-flip facts;
  - state from token-observation facts.
- Assert equivalence in tests for current pinned witnesses.

Tests:

- Integration tests should compare forced token flips and remaining
  token-phase assignment ids between legacy-fact state and observation-fact
  state.

Exit criteria:

- There is a model-state construction path that does not require
  `StereoTokenFlipFact`s from the legacy inferred helper for the current
  branch.

### Commit 5: Replace One Branch Internally

Files:

- `rust/src/rooted_stereo.rs`

Work:

- For `isolated_selected_begin_side`, replace direct procedural flip
  calculation with:
  - construct typed observation fact;
  - query model;
  - read forced token flip.
- Keep the old calculation in shadow mode for this branch and error if it
  disagrees.
- Leave other branches untouched.

Tests:

- Full Rust tests.
- `tests.integration.test_stereo_constraint_model`.
- Pinned RDKit parity target used on this branch.

Exit criteria:

- One current production branch is model-derived while behavior remains
  unchanged.

## Risks And Boundaries

- Current completed-output fixtures exercise only one branch. Do not claim the
  whole token inference helper has been replaced until tests hit the coupled
  and all-single branches.
- `rdkit_component_token_flip_adjustment` remains suspicious. In this plan it
  becomes a named observation fact first; deriving or replacing it is a later
  slice.
- Component phase and begin atom are still walker fields. This plan observes
  them, but does not yet replace their source of truth.
- If the observation fact enum grows to mirror the old helper branch for
  branch, stop and reconsider Alternative C.

## Guardrails

- Do not add molecule-specific or fixture-specific branches.
- Do not hide RDKit behavior behind generic names. If a rule is an RDKit writer
  adjustment, name it as such.
- Do not make completed-string projection part of this phase.
- Do not make Z3 a runtime or CI dependency.
- Keep old procedural inference only as an equivalence oracle while the new
  observation path is being introduced.
