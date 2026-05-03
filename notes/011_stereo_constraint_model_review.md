# Stereo Constraint Model Review

Branch: `stereo-constraint-model`

## Critical Review

The current direction is sound, but the branch is still mostly diagnostic.
The biggest issue is dual truth: walker state still owns procedural stereo
fields, while the constraint model owns carrier and token-assignment domains.
Until those converge, tests can prove observations, but they do not yet enforce
the target algorithm.

Specific risks:

- `resolved_selected_neighbors_from_fields` is still an unconditional repair
  hook, not a model query.
- `normalize_component_token_flips` still validates after successor
  construction, so invalid branches can still exist transiently.
- Token-phase support is currently helper-query based, not a first-class state
  object like `StereoAssignmentState`.
- `token_phase_assignment_ids_for_neighbor_assignment_ids` silently ignores
  token constraints for runtime components not in the model component; that
  should become explicit rejection or empty-state behavior.
- Current integration tests use old runtime inference to validate the new
  token-phase model, so they are useful but partially circular.

## Alternatives

Continue with shadow state first. This is the safest and most principled path:
build `StereoTokenPhaseState`, prove it mirrors current runtime, then make the
runtime consult it.

Replace `normalize_component_token_flips` directly. This is too risky because
it mixes algorithm work with behavior changes and makes failures hard to
interpret.

Build one large product assignment state containing carriers, token flips, and
marker obligations immediately. This is principled, but too large a jump for
reviewable implementation and debugging.

Use a fact-log recomputation model only. This is simple and DRY, but likely too
slow and noisy unless later backed by compact state.

Use runtime Z3. This is not appropriate for production. Z3 remains useful for
exploration, but it is the wrong dependency and performance profile for the
runtime library.

## Plan

The next commit-sized slice should cover steps 1-5.

1. Add `StereoTokenPhaseState` beside `StereoAssignmentState`, storing
   remaining token-phase assignment ids per model component.

2. Harden model APIs: unknown runtime-component token constraints should not
   be silently ignored; return empty state or an explicit error.

3. Add unit tests in `bond_stereo_constraints.rs` for multi-runtime-component
   token phase state, invalid constraints, forced token flip, and empty-state
   behavior.

4. Expose `token_phase_assignment_state` in `_stereo_constraint_output_facts`,
   parallel to the existing carrier assignment state.

5. Add integration tests proving nonempty token-phase state for all current
   witnesses, and proving the minimal witness realizes all four token-flip
   tuples through one merged model component.

6. Add a shadow transition helper: given current carrier state plus an inferred
   or chosen token flip, produce the next token-phase state. Do not change
   runtime behavior yet.

7. Refactor `normalize_component_token_flips` to call the shadow helper and
   assert equivalence with current behavior.

8. Only after equivalence is pinned, replace `normalize_component_token_flips`
   logic with token-phase state filtering.

9. Then move to shared-carrier repair replacement: make
   `resolved_selected_neighbors_from_fields` an adapter over forced-neighbor
   queries from assignment state.
