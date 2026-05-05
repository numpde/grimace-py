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

## Guardrails

- Do not add molecule-specific or fixture-specific branches.
- Do not hide RDKit behavior behind generic names. If a rule is an RDKit writer
  adjustment, name it as such.
- Do not make completed-string projection part of this phase.
- Do not make Z3 a runtime or CI dependency.
- Keep old procedural inference only as an equivalence oracle while the new
  observation path is being introduced.
