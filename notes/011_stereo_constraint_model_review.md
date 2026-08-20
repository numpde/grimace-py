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

## Long-Term Plan Toward Final Shape

The final shape should have one source of truth for stereo decisions. The
walker should perform graph traversal and output construction; the constraint
model should decide which stereo assignments remain possible, which choices
are forced, and which marker obligations are still pending.

The intended final runtime state is:

- graph traversal state: visited atoms, pending rings, labels, output prefix,
  and action stack;
- stereo assignment state: remaining assignment ids per model component;
- marker obligation state: pending directional marker obligations that have
  been created but not yet discharged;
- no independent procedural stereo fields that can contradict the assignment
  state.

RDKit-specific behavior should live in named writer layers or named projection
rules. It should not appear as anonymous local repairs inside traversal.

### Phase 1: Make Shadow State Complete

Goal: represent everything the current walker already knows as model state,
without changing output behavior.

Work:

- Add `StereoTokenPhaseState` and make token-flip constraints explicit.
- Add model-side APIs for filtering by carrier facts, token-flip facts, and
  traversal facts.
- Expose carrier and token-phase state in diagnostics in a uniform shape.
- Ensure every current output has nonempty semantic, RDKit-local, and
  RDKit-traversal state where expected.
- Make invalid or out-of-component constraints fail explicitly instead of
  being ignored.

Exit criteria:

- Diagnostics can reconstruct current selected neighbors and token flips from
  model state.
- The old procedural fields and new model state agree for all pinned witnesses.
- No runtime behavior changes are required for this phase.

### Phase 2: Replace Token-Phase Normalization

Goal: eliminate post-construction token-flip repair/checking.

Work:

- Route deferred token choices through `StereoTokenPhaseState`.
- Make `normalize_component_token_flips` a thin adapter over token-phase
  filtering.
- Reject inconsistent token choices by empty assignment state, not by late
  procedural comparison.
- Add tests that intentionally create conflicting token choices and verify
  they are pruned at the model boundary.

Exit criteria:

- `normalize_component_token_flips` contains no independent stereochemical
  logic.
- Token support for deferred markers is computed from forced token-phase state
  or branches over remaining token-phase assignments.

### Phase 3: Replace Shared-Carrier Repair

Goal: remove forced shared-edge selection from local repair hooks.

Work:

- Encode shared-carrier coupling in allowed assignment domains.
- Replace `forced_shared_candidate_neighbor` with assignment-state queries.
- Replace `resolved_selected_neighbors_from_fields` with forced-neighbor
  reads from model state.
- Keep compatibility adapters only while tests prove equivalence.

Exit criteria:

- Shared carrier behavior is explained by component assignment filtering.
- The old shared-edge repair hooks are deleted or contain no decision logic.

### Phase 4: Add Marker Obligations

Goal: make directional marker placement online and state-driven, without
finished-string cleanup.

Work:

- Define marker obligation records: model component, runtime component,
  side, carrier edge, raw marker token, traversal event, and eligible output
  slots.
- Create obligations when a carrier edge is selected but marker placement is
  not yet forced.
- Discharge obligations only at legal online output boundaries.
- Branch when multiple marker placements remain possible.
- Fail terminal states with undischarged obligations.

Exit criteria:

- Ring-marker projection diagnostics become runtime behavior.
- Marker placement is not a post-hoc string transform.
- RDKit-specific marker movement is represented as a named writer-layer rule.

### Phase 5: Collapse Walker Stereo Fields

Goal: remove the dual-truth state shape.

Work:

- Replace `stereo_selected_neighbors` with forced carrier assignments or
  pending carrier facts.
- Replace `stereo_component_token_flips` with token-phase assignment state.
- Replace `stereo_component_phases`, `stereo_component_begin_atoms`, and
  `stereo_first_emitted_candidates` with typed traversal facts or assignment
  dimensions.
- Keep only traversal data in walker state and stereo data in model state.

Exit criteria:

- The walker cannot construct a state whose stereo fields contradict each
  other, because those fields no longer exist independently.
- Completion-cache keys use assignment-state ids and obligation state, not
  procedural stereo vectors.

### Phase 6: Separate Principled Core From RDKit Compatibility

Goal: make it obvious which constraints are molecular/semantic and which are
RDKit writer quirks.

Work:

- Keep semantic OpenSMILES-like constraints in the semantic layer.
- Keep RDKit-local writer exclusions in a named RDKit-local layer.
- Keep traversal-order and marker-position quirks in a named RDKit-traversal
  layer.
- Document any behavior that is believed to be an RDKit quirk but still
  mirrored.

Exit criteria:

- The same model can answer semantic-support questions and RDKit-support
  questions by selecting a layer.
- RDKit-specific decisions are searchable by layer name, not scattered through
  traversal.

### Phase 7: Generalize Beyond Current Witnesses

Goal: avoid solving only the current failing examples.

Work:

- Expand pinned RDKit-versioned fixtures around conjugated systems, rings,
  shared carriers, terminal carriers, branch/ring ordering, and dative or
  metal-adjacent outliers.
- Add adversarial generated cases that vary root, branch order, ring order,
  and slash/backslash carrier placement.
- Keep exact support tests pinned by RDKit version.
- Keep object-equivalence investigations separate from exact RDKit serializer
  parity tests.

Exit criteria:

- New witnesses add data, not bespoke runtime branches.
- Regression failures identify either a missing constraint dimension, a missing
  writer-layer rule, or a documented RDKit quirk.

### Final Acceptance Criteria

The branch is ready to merge only when:

- all stereo pruning is assignment-state filtering;
- all directional marker placement is online obligation handling;
- there is no finished-string cleanup pass;
- old suspicious helpers are deleted or reduced to model adapters;
- RDKit-specific behavior is isolated in named layers;
- pinned parity tests pass for the supported RDKit version;
- the test suite contains at least one witness for each replacement target:
  token phase, shared carrier coupling, marker obligation, and RDKit traversal
  quirk.
