# Stereo CSP Fact-Log Plan

Branch: `stereo-constraint-model`

## Decision

Use a typed CSP/factor model as the long-term target, reached through a
fact-log oracle first. The immediate implementation should not add more
loosely related shadow helpers. It should introduce a single state boundary
that can answer: given the current typed facts, which stereo assignments remain
possible?

## Alternatives Considered

Continuing with separate shadow structs is safe, but it risks producing a
parallel hierarchy of temporary state objects. It would make the migration
reviewable in small pieces, but the abstraction boundary would remain unclear.

A monolithic explicit assignment table per component is mathematically clean:
each row represents carrier choices, token phase, and eventually marker
obligations. The risk is premature product expansion. Marker placement has an
online timing dimension, so forcing it into a static row too early may obscure
the real algorithm.

A typed CSP/factor model is the best final shape. Variables remain separate,
constraints are explicit, and runtime facts filter domains. It can be
implemented first as recomputation from a fact log, then optimized with compact
row ids or bitsets without changing semantics.

Directly replacing current heuristics is too risky. `normalize_component_token_flips`
and shared-carrier repair still contain behavior we do not fully explain with
model state.

Cloning RDKit behavior procedurally would likely improve parity fastest, but
it preserves the problem this branch is meant to solve.

## Chosen Shape

Introduce a unified `StereoConstraintState` built from:

- a model layer;
- carrier/traversal facts grouped by model component;
- token-flip facts keyed by runtime stereo component.

For now, the state computes:

- remaining carrier assignment ids per model component;
- remaining token-phase assignment ids per model component;
- forced carrier choices;
- forced token flips.

This remains a shadow oracle. The existing walker is still authoritative until
the state is complete enough to replace procedural logic.

## Immediate Implementation

1. Add typed token-flip facts.
2. Add strict validation for token-flip facts: unknown runtime components and
   facts attached to the wrong model component must not be silently ignored.
3. Add `StereoConstraintState` as the unified carrier plus token-phase state.
4. Add Rust unit tests for multi-runtime-component filtering, invalid facts,
   empty state, and forced token flips.
5. Expose the new state in diagnostics only after the Rust boundary is stable.

## Later Migration

After the state boundary exists:

1. Make diagnostics show old procedural values beside `StereoConstraintState`.
2. Refactor `normalize_component_token_flips` into an adapter over the state.
3. Replace shared-carrier repair with forced-neighbor queries.
4. Add marker obligations as first-class online facts.
5. Collapse old walker stereo fields once all decisions are explained by the
   state.
