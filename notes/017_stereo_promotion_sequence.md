# Stereo Decision Promotion Sequence

Branch: `stereo-constraint-model`

## Purpose

This note scopes the umbrella task "Promote procedural stereo decisions one at
a time" into implementation-sized slices. The goal is not to remove every
procedural path at once. The goal is to move one support-shaping decision at a
time behind the production constraint boundary from
`notes/010_stereo_algorithm_target.md`.

The classification basis is `notes/016_procedural_stereo_decision_inventory.md`.

## Ordering Principles

- Promote facts before deleting helpers.
- Prefer boundaries that already have typed observations or model state.
- Do not promote RDKit writer-policy quirks until their policy decision rows
  are resolved.
- Avoid carrier-only promotion for shared-carrier cases; previous experiments
  showed that joined carrier + token + marker/observation state is required.
- Every code slice should leave the runtime behavior unchanged unless the slice
  also adds or updates exact parity fixtures proving the intended change.

## First Concrete Slices

### 1. Promote deferred phase/begin-side facts

Current procedural surface:

- `deferred_component_phase_constraint_for_unresolved_begin_side`
- `defer_component_phase_for_unresolved_begin_side`
- `commit_deferred_component_phase_constraints_from_selected_begin_sides`
- `process_children_edge_update`

Target:

- Represent phase and begin-side commitments as typed support-boundary facts.
- Keep walker vectors as compatibility storage only while assertions prove the
  fact-derived state agrees.
- Make deferred phase/begin-side state visible beside token-observation facts.

Why first:

- These fields are already named as observation inputs.
- This reduces dual truth without needing unresolved RDKit writer-policy
  decisions.

### 2. Split generic carrier commitment from edge emission

Current procedural surface:

- `emitted_edge_part_generic`
- `row_state_carrier_obligation_neighbor`
- `deferred_carrier_choice_constraint_for_row_state`
- `deferred_carrier_choice_constraints_block_neighbor`

Target:

- Extract carrier observation/fact production from edge text construction.
- Make edge emission consume carrier decisions from a support-boundary query.
- Keep existing deferred carrier-choice facts as the compatibility bridge.

Why second:

- It removes support-shaping logic from emission, but can build on the
  already-persisted `CarrierSelectionBlocked` fact.

### 3. Collapse deferred token commit into support-state facts

Current procedural surface:

- `deferred_token_support_from_constraint_state`
- `deferred_candidate_survives_marker_rows`
- `commit_deferred_token_choice`
- `assert_component_token_flip_boundary_invariants`

Target:

- Keep support computation model-backed.
- Replace post-commit mutation of `stereo_component_token_flips` with a typed
  committed-token fact or support-state transition.
- Demote `assert_component_token_flip_boundary_invariants` to a temporary
  equivalence assertion.

Why third:

- Deferred token support is already one of the cleaner model-backed paths.
- This should reduce procedural state without needing marker/ring policy
  promotion.

### 4. Replace shared-carrier post-resolution with a joined survivor query

Current procedural surface:

- `resolved_selected_neighbors_from_fields`
- `resolved_selected_neighbors`
- `shared_carrier_resolution_to_py`

Target:

- Ask the support boundary which carrier choices remain after carrier facts,
  token observations, and marker/domain facts are all applied.
- Only then remove the post-hoc shared-edge repair.

Why later:

- Carrier-only replacement already lost valid coupled-diene support.
- This must wait until the joined support query exists.

## Explicitly Blocked Until Decision Rows Are Resolved

- `rdkit_token_flip_adjustment_observation_from_state`: blocked on the RDKit
  token-flip adjustment policy decision.
- `rdkit_ring_closure_projected_marker_slots` and runtime promotion of marker
  placement/ring projection: blocked on the RDKit marker-placement promotion
  policy decision.

## Review Check

A promotion slice is complete only when:

- one procedural decision has a named fact/query boundary;
- old logic is either deleted or retained only as a compatibility assertion;
- relevant diagnostics expose both the new fact/state and any temporary legacy
  comparison;
- tests cover the promoted boundary on pinned stereo witnesses.
