# Procedural Stereo Decision Inventory

Branch: `stereo-constraint-model`

## Purpose

This note inventories the remaining stereo support-shaping decisions that still
live outside the intended production constraint boundary. It does not classify
or replace them. The next task should classify each item as principled
stereo/graph constraint, RDKit writer policy, temporary adapter/shadow
assertion, or obsolete repair.

The target boundary is the one recorded in
`notes/010_stereo_algorithm_target.md`: typed online facts narrow one support
state, and traversal code asks that state which next tokens or forced emissions
are legal.

## Current Runtime Support-Shaping Surfaces

### Shared-carrier post-resolution

Code:

- `rust/src/rooted_stereo.rs`: `resolved_selected_neighbors_from_fields`
- `rust/src/rooted_stereo.rs`: `resolved_selected_neighbors`
- `rust/src/rooted_stereo.rs`: `shared_carrier_resolution_to_py`

Current decision:

- `resolved_selected_neighbors_from_fields` forces both sides of each
  ambiguous shared-edge group to the shared edge after inspecting
  `stereo_first_emitted_candidates`.
- `resolved_selected_neighbors` then applies deferred carrier-choice blocking
  by reopening a selected neighbor to `-1` when the persisted blocked-neighbor
  state says that choice is not currently allowed.

Adjacent model state:

- `StereoAssignmentState::forced_neighbor` can already report forced carriers
  from assignment survivors.
- Prior assignment-first experiments show carrier rows alone are insufficient:
  the safe replacement must include token observations and marker/domain facts
  before this becomes runtime truth.

### Generic edge emission and carrier commitment

Code:

- `rust/src/rooted_stereo.rs`: `emitted_edge_part_generic`
- `rust/src/rooted_stereo.rs`: `row_state_carrier_obligation_neighbor`
- `rust/src/rooted_stereo.rs`: `deferred_carrier_choice_constraint_for_row_state`
- `rust/src/rooted_stereo.rs`: `deferred_carrier_choice_constraints_block_neighbor`
- `rust/src/rooted_stereo.rs`: `should_defer_carrier_commit_for_constraint`

Current decision:

- Edge emission still chooses candidate carrier neighbors while building the
  emitted part.
- It records first-emitted candidates, updates selected neighbors and
  orientations, skips candidates blocked by deferred carrier-choice state, and
  may add new deferred carrier-choice constraints before continuing.
- `row_state_carrier_obligation_neighbor` consults row-state information to
  force a carrier locally when only one viable obligation neighbor remains.

Adjacent model state:

- `DeferredCarrierChoiceConstraint` is persisted in walker state and participates
  in transition identity.
- `StereoConstraintFact::CarrierSelectionBlocked` represents the persisted
  blocked-neighbor state in `resolved_constraint_state_from_walker_state`.
- The actual choice/skip logic is still interleaved with edge emission.

### Deferred component phase and begin-side commitment

Code:

- `rust/src/rooted_stereo.rs`: `deferred_component_phase_constraint_for_unresolved_begin_side`
- `rust/src/rooted_stereo.rs`: `defer_component_phase_for_unresolved_begin_side`
- `rust/src/rooted_stereo.rs`: `commit_deferred_component_phase_constraints_from_selected_begin_sides`
- `rust/src/rooted_stereo.rs`: `process_children_edge_update`

Current decision:

- If the begin side is unresolved, phase and begin-atom information may be
  deferred while an edge is processed.
- Later, after selected begin sides are available, the deferred phase/begin
  state is committed back into `stereo_component_phases` and
  `stereo_component_begin_atoms`.

Adjacent model state:

- Phase and begin-side observations are now named inputs for token observation
  facts.
- They are still stored as walker vectors and committed procedurally during
  traversal, not owned by a single support-state boundary.

### Isolated token-basis facts

Code:

- `rust/src/rooted_stereo.rs`: `isolated_component_token_basis_fact_from_row_state`
- `rust/src/rooted_stereo.rs`: `updated_isolated_component_token_basis_facts_from_row_state`
- `rust/src/rooted_stereo.rs`: `isolated_component_stored_token_from_token_state`
- `rust/src/rooted_stereo.rs`: `emitted_isolated_edge_part`
- `rust/src/bond_stereo_constraints.rs`: `StereoTokenBasisFact`

Current decision:

- Isolated component slash/backslash basis is now represented as
  `StereoTokenBasisFact` and persisted in walker/completion state.
- The fact is still derived from row state at the isolated edge-emission
  boundary, then consumed by stored-token rendering.

Adjacent model state:

- This is cleaner than the previous inline token override because rendering
  consumes a persisted fact.
- The remaining support-shaping decision is where the token-basis fact should
  be produced in the final model boundary.

### RDKit token-flip adjustment and inferred token observations

Code:

- `rust/src/rooted_stereo.rs`: `rdkit_token_flip_adjustment_observation_from_state`
- `rust/src/rooted_stereo.rs`: `component_token_inference_inputs`
- `rust/src/rooted_stereo.rs`: `component_token_constraint_from_state`
- `rust/src/rooted_stereo.rs`: `inferred_component_token_flip`
- `rust/src/rooted_stereo.rs`: `legacy_procedural_inferred_component_token_flip`

Current decision:

- Runtime token constraints are classified as known committed token flips,
  inferred token observations, or no token constraint.
- `inferred_component_token_flip` returns the observation-derived value.
- The legacy procedural inference function remains as an equivalence oracle
  and still encodes the old branch logic for the currently pinned branch
  shapes.
- `rdkit_token_flip_adjustment_observation_from_state` is a local
  RDKit-observed observation builder derived from root, begin-side orientation,
  first-emitted candidate, and adjacent two-candidate context.

Adjacent model state:

- `StereoTokenObservationFact` and
  `StereoConstraintState::from_facts_and_token_observations` consume the typed
  observation path.
- Diagnostics expose the observation inputs.
- The adjustment is named, but not yet explained as a model-owned writer-policy
  fact.

### Deferred directional-token support and commit

Code:

- `rust/src/rooted_stereo.rs`: `deferred_token_support_from_constraint_state`
- `rust/src/rooted_stereo.rs`: `deferred_candidate_survives_marker_rows`
- `rust/src/rooted_stereo.rs`: `marker_event_for_deferred_component_token`
- `rust/src/rooted_stereo.rs`: `commit_deferred_token_choice`
- `rust/src/rooted_stereo.rs`: `assert_component_token_flip_boundary_invariants`

Current decision:

- Deferred directional-token support is mostly routed through
  `resolved_constraint_state_from_walker_state`.
- Candidate slash/backslash choices are then checked against marker-row
  survivor state before being exposed as legal tokens.
- Committing a deferred token still mutates walker token-flip state and then
  asserts that the model boundary explains the committed flips.

Adjacent model state:

- This is one of the cleaner runtime boundaries today: support comes from the
  constraint state plus marker-row survivor checks.
- The remaining dual truth is that committed token choices still update
  `stereo_component_token_flips`, with model explanation enforced afterward.

### Marker-event traces, marker-row survivor state, and ring projection

Code:

- `rust/src/rooted_stereo.rs`: `record_directional_marker_trace`
- `rust/src/rooted_stereo.rs`: `record_marker_event_traces_for_edge`
- `rust/src/rooted_stereo.rs`: `marker_event_facts_by_component`
- `rust/src/rooted_stereo.rs`: `shadow_marker_event_facts_by_component`
- `rust/src/rooted_stereo.rs`: `marker_obligation_domains_by_component`
- `rust/src/rooted_stereo.rs`: `marker_row_survivor_component_state`
- `rust/src/rooted_stereo.rs`: `rdkit_ring_closure_projected_marker_slots`
- `rust/src/bond_stereo_constraints.rs`: marker-placement row/filter APIs

Current decision:

- Traversal records positive marker events and no-marker events for diagnostics
  and row-survivor filtering.
- Marker placement state can filter rows by observed events and project
  surviving token/neighbor assignments.
- Ring-closure marker projection remains a diagnostic RDKit writer-policy rule,
  not a runtime emission rule.

Adjacent model state:

- Marker-placement rows and event filters are implemented and tested.
- The production runtime still emits markers through traversal/deferred-token
  paths rather than a single marker-obligation state.

### Completion/cache identity still carries procedural stereo vectors

Code:

- `rust/src/rooted_stereo.rs`: `RootedConnectedStereoWalkerStateData`
- `rust/src/rooted_stereo.rs`: `StereoCompletionKey`
- `rust/src/rooted_stereo.rs`: `cmp_stereo_state_structure`

Current decision:

- Completion identity still includes procedural vectors:
  `stereo_selected_neighbors`, `stereo_selected_orientations`,
  `stereo_first_emitted_candidates`, `stereo_component_phases`,
  `stereo_component_begin_atoms`, `stereo_component_token_flips`,
  `deferred_carrier_choice_constraints`, and `stereo_token_basis_facts`.

Adjacent model state:

- Several of these fields now have corresponding typed facts or model queries.
- They have not yet collapsed into a single assignment/domain state plus
  marker-obligation state.

## Immediate Follow-Up

The classification task should keep "what it is" separate from "how it is
stored." A row may be the right representation for a given item, but the
classification below is about ownership of the support-shaping decision.

## Classification

| Surface | Primary ownership | Current status | Discussion before implementation? |
| --- | --- | --- | --- |
| Shared-carrier post-resolution | Principled carrier-coupling constraint, with RDKit traversal observations needed before it is safe as runtime truth | Temporary repair/adapter | No new decision required; implementation must use joined carrier + token + marker/observation state, not carrier rows alone |
| Generic edge emission and carrier commitment | Mixed: principled carrier choice plus online RDKit traversal timing | Temporary adapter with support-shaping logic inside emission | No new decision required; split into typed fact production and support-state queries |
| Deferred component phase and begin-side commitment | Principled token/phase observation generated by traversal order | Temporary adapter over walker vectors | No new decision required; promote phase/begin-side facts into the support boundary before removing vectors |
| Isolated token-basis facts | RDKit writer-policy spelling basis layered over valid isolated stereo assignments | Transitional fact; cleaner than inline repair but still produced at edge emission | Small design choice remains: decide the upstream fact source, but no broad discussion required |
| RDKit token-flip adjustment and inferred token observations | RDKit writer policy plus typed token-observation facts | Mixed boundary: observation path is primary; legacy procedural path is shadow oracle | Yes. The adjustment should become a named writer-policy fact only after its source rule and pinned witnesses are explicit |
| Deferred directional-token support and commit | Online support query plus temporary walker mutation of committed token flips | Mostly model-backed adapter with post-commit assertion | No new decision required; collapse committed token flips into support state once replacement state exists |
| Marker-event traces, marker-row survivor state, and ring projection | RDKit writer policy for visible marker placement and movement | Diagnostic/model support exists; runtime marker placement is not yet a single obligation state | Yes. Ring projection and no-marker events should be promoted only with explicit RDKit-policy naming and witness coverage |
| Completion/cache identity procedural vectors | Implementation scaffolding for current walker state | Temporary dual-truth state | No new decision required; remove incrementally as each vector is replaced by support-state identity |

## Discussion-Required Decisions

Two areas should be treated as decisions before broad implementation:

1. `rdkit_token_flip_adjustment_observation_from_state`: this is already named
   as RDKit behavior, but the exact writer-policy rule is still encoded as
   local observation-building logic. Before promoting it, record which pinned
   witnesses require each branch and whether the adjustment belongs to root
   policy, first-emitted-candidate policy, adjacent two-candidate policy, or a
   combined rule.

2. Marker placement and ring projection: marker-row survivor state is useful,
   but runtime promotion must not blur principled stereo constraints with RDKit
   string-placement quirks. Before making projection runtime behavior, record
   the supported RDKit-policy rule, the exact online event that creates the
   obligation, and the legal slots that can discharge it.

Everything else can proceed as implementation slices once it is backed by the
single support boundary described in `notes/010_stereo_algorithm_target.md`.
