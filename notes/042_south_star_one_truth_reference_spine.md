# South Star One-Truth Reference Spine Inventory

Branch: `south-star`

Date: 2026-05-21

## Purpose

This note scopes `South Star 84A`.

The correction is not "add better feature oracles." The correction is to make
one shared South Star reference spine visible in code, then fold current
feature-local witnesses into it.

The spine should own the common vocabulary:

- molecule facts;
- semantic stereo components;
- traversal events;
- marker slots;
- ring closures;
- typed constraints;
- solved assignments;
- renderer inputs and diagnostics.

Feature-specific modules may derive facts or constraints. They should not own
separate support universes.

## Current Inventory

### Existing Shared Facts

These are already close to the desired shared layer:

- `python/grimace/_south_star/molecule_facts.py`
  - `SouthStarMoleculeFacts`
  - `SouthStarAtomTextFact`
  - `SouthStarBondTextFact`
  - `SouthStarGraphTopologyFacts`
  - `SouthStarRingSystemFacts`
- `python/grimace/_south_star/components.py`
  - `SouthStarSourceStereoFeature`
  - `SouthStarSemanticStereoComponent`
  - `SouthStarComponentExtraction`
  - `SouthStarComponentCoupling`
- `python/grimace/_south_star/tetrahedral.py`
  - `SouthStarTetrahedralCenterFact`
  - `extract_tetrahedral_center_facts`

These should stay as fact extractors. They should feed the reference spine, not
render strings directly.

### Existing Shared Policy And State

These are also close to shared infrastructure:

- `python/grimace/_south_star/annotation_policy.py`
  - `SemanticCarrierOpportunity`
  - `EmittedEdgeBasis`
  - `SurvivingSemanticAssignment`
  - `AnnotationPolicyDecision`
  - `MaximalEligibleCarrierAnnotationPolicy`
- `python/grimace/_south_star/component_support_state.py`
  - `SouthStarComponentSupportState`
  - `SouthStarComponentMarkerAssignment`
  - `SouthStarComponentMarkerSupport`
  - `SouthStarComponentComplexitySnapshot`

This layer decides which semantic assignments survive under an annotation
policy. It should not be duplicated in witness helpers.

### Runtime Traversal And Rendering Records

`python/grimace/_south_star/enum_s.py` currently owns records that are more
general than the enum prototype:

- `_CarrierContext`
- `SouthStarMarkerSlot`
- `SouthStarMarkerSlotAssignment`
- `SouthStarRingClosure`
- `SouthStarTraversalEvent`
- `SouthStarTreeTraversal`
- `_TraversalFragment`
- `_CombinedMarkerAssignment`
- `_SupportGeneration`
- `SouthStarEnumSGenerationDiagnostics`
- `SouthStarEnumSPrototypeResult`

The first five are spine vocabulary and should move to a shared record module.
`SouthStarTreeTraversal` should likely become a shared traversal object, or a
thin runtime wrapper around one. `_TraversalFragment`, `_CombinedMarkerAssignment`,
and `_SupportGeneration` are implementation-local assembly records unless later
work proves they are useful diagnostic records.

`SouthStarEnumSGenerationDiagnostics` and `SouthStarEnumSPrototypeResult` are
API/prototype result records. They should remain close to `enum_s.py` until the
public package boundary is decided.

### First-Domain Witness Duplication

`tests/helpers/south_star_first_domain_oracle.py` deliberately duplicates the
tree traversal and marker rendering model:

- `_OracleCarrierContext`
- `_OracleMarkerSlot`
- `_OracleFragment`
- `_CombinedGraphAssignment`
- `_oracle_atom_variants`
- `_oracle_branch_variants`
- `_oracle_child_variants`
- `_bond_token_and_slot`
- `_render_fragment`
- `_oriented_marker`
- `_carrier_contexts_by_edge`
- `_carrier_contexts_for_edge`

This helper is useful as an independent witness, but it should not remain a
second model vocabulary. The carrier-context, marker-slot, and fragment/event
records should be replaced by or adapted to the shared spine. The traversal
generation can remain witness code during the migration, but it should emit the
same record language as the runtime path.

### Expanded-Domain Witness Duplication

`tests/helpers/south_star_expanded_domain_oracles.py` contains several local
mini-worlds:

- saturated ring traversal:
  - `_RingToken`
  - `_RingFragment`
  - `_tree_fragments_with_blocked_edge`
  - `_branch_fragments`
  - `_child_fragments`
  - `_render_with_ring_digit`
- disconnected composition:
  - `SouthStarDisconnectedCompositionOracleResult`
  - `_independent_fragment_supports_for_case`
- ring stereo:
  - `SouthStarRingStereoOracleResult`
  - `shared_ring_stereo_monocycle_support_for_case`
- tetrahedral support:
  - `_single_star_tetrahedral_fact`
  - `_render_tetrahedral_center_root`
  - `_render_tetrahedral_ligand_root`
  - `_render_center_with_ordered_ligands`
  - `_tetrahedral_center_text`
  - `_emitted_tetrahedral_ligand_order`

The saturated ring and ring-stereo records overlap directly with the runtime
traversal/event/slot vocabulary. Disconnected composition overlaps with
`python/grimace/_south_star/fragments.py`. Tetrahedral witness code overlaps
with `SouthStarTetrahedralCenterFact` and should eventually become
atom-stereo obligations over traversal events, not a separate renderer.

### Constraint And Solver Records

`python/grimace/_south_star/marker_equations.py` and
`python/grimace/_south_star/parity_solver.py` are already the right direction:

- `SouthStarFeatureCarrierTerm`
- `SouthStarMarkerSlotParityEquation`
- `SouthStarParitySolverAssignment`
- `SouthStarParitySolverDiagnostic`
- `SouthStarParitySolverResult`

The issue is that `marker_equations.py` imports traversal records from
`enum_s.py`. That makes the enum prototype look like the owner of the reference
language. The dependency should be inverted: both `enum_s.py` and
`marker_equations.py` should import shared traversal and marker-slot records
from the spine module.

The ring-stereo monocycle witness now renders the shared EnumS traversal records
and projects the shared `SouthStarMarkerSlotParityEquation` records. It is no
longer an independent support universe, but it is still temporary witness
evidence until the broader unified-reference promotion gate decides what can be
treated as package-ready.

### Fragment Composition Records

`python/grimace/_south_star/fragments.py` already owns reusable disconnected
composition records:

- `SouthStarFragmentSupport`
- `AllFragmentOrderPolicy`
- `SouthStarDisconnectedCompositionResult`
- `compose_disconnected_fragment_supports`

`SouthStarDisconnectedCompositionOracleResult` should not survive as a parallel
result type unless it carries witness-only metadata unavailable from the shared
result. Prefer using or extending the shared result.

## Target Shared Record Module

The first extraction target should be a small private module, provisionally:

```text
python/grimace/_south_star/reference_model.py
```

Better names are acceptable during implementation, but the module should remain
private and should not imply public API stability.

Initial contents should be records only, not generation logic:

- `SouthStarCarrierContext`
  - replaces `_CarrierContext` and `_OracleCarrierContext`;
- `SouthStarMarkerSlot`
  - replaces the current runtime type plus `_OracleMarkerSlot`;
- `SouthStarMarkerSlotAssignment`
  - keeps the distinction between slot-level assignment and component-level
    graph assignment;
- `SouthStarRingClosure`
  - moves out of `enum_s.py`;
- `SouthStarTraversalEvent`
  - replaces runtime and ring-stereo local event records;
- `SouthStarTraversalFragment`
  - shared event tuple for recursive traversal builders;
- `SouthStarTraversal`
  - generalizes `SouthStarTreeTraversal` without baking in "tree" once rings
    and disconnected fragments are present.

Do not move the solver, policies, molecule facts, or generation algorithm into
this module. The point is a shared language, not a monolith.

Also do not move rendering behavior into this module. The current
`SouthStarTreeTraversal.render()` method is convenient prototype API, but it
couples the traversal record to the renderer. The shared record should be inert:
it may carry events and assignments, but rendering should stay in a renderer
function/module that consumes the record.

The initial module should have a narrow import surface:

- `dataclasses`;
- standard-library typing if needed;
- `Edge` from `annotation_policy.py`.

It should not import RDKit, `enum_s.py`, `marker_equations.py`,
`parity_solver.py`, test helpers, fixture loaders, or support gates. If that
boundary feels too small during implementation, that is a signal that logic is
being moved too early.

## Dependency Direction

After `84B`, the intended dependency direction is:

```text
molecule_facts/components/tetrahedral
        |
        v
reference_model records  <--- annotation policy types
        |
        v
traversal builders -> constraint builders -> solvers -> renderer
        |
        v
enum_s prototype result

test witnesses may import the shared records, but shared records must not import
test witnesses.
```

`reference_model.py` is not the one-truth solver. It is the record vocabulary
that prevents every feature witness from inventing its own traversal/event/slot
language.

## Planned Follow-Up File Moves

### `South Star 84B`

Create `python/grimace/_south_star/reference_model.py`.

Move the shared record definitions out of `enum_s.py` or introduce compatible
aliases first:

- `_CarrierContext` -> `SouthStarCarrierContext`
- `SouthStarMarkerSlot`
- `SouthStarMarkerSlotAssignment`
- `SouthStarRingClosure`
- `SouthStarTraversalEvent`
- `_TraversalFragment` -> `SouthStarTraversalFragment`
- `SouthStarTreeTraversal` -> compatibility wrapper or alias around an inert
  shared traversal record

Update imports in:

- `python/grimace/_south_star/enum_s.py`
- `python/grimace/_south_star/marker_equations.py`
- tests that import `SouthStarMarkerSlot` or `SouthStarTreeTraversal`

Acceptance:

- no support changes;
- `reference_model.py` has no RDKit, `enum_s.py`, solver, or test-helper
  imports;
- shared traversal records have no `.render()` behavior;
- compatibility aliases/wrappers keep existing callers stable;
- focused South Star tests still pass.

### `South Star 84C`

Change the first-domain witness helper to emit or adapt shared records:

- `tests/helpers/south_star_first_domain_oracle.py`
- `tests/south_star/test_first_domain_completeness.py`
- `tests/south_star/test_marker_slot_equations.py`
- `tests/south_star/test_parity_solver.py`

The helper may keep independent traversal search, but local record types should
not duplicate the shared vocabulary. If a local adapter remains, name it as
witness-only glue.

Acceptance:

- first-domain fixture support is unchanged;
- the helper's independent search still exists as witness evidence;
- duplicated carrier-context and marker-slot dataclasses are gone or are
  narrow adapters around shared records;
- assignment-level tests still compare runtime/custom/Z3/reference results.

### `South Star 84D`

Route expanded witnesses through the same records:

- `tests/helpers/south_star_expanded_domain_oracles.py`
- `tests/south_star/test_expanded_support_fixtures.py`
- `python/grimace/_south_star/enum_s.py`
- `python/grimace/_south_star/tetrahedral.py`
- `python/grimace/_south_star/fragments.py`

Expected cleanups:

- replace `_RingToken` / `_RingFragment` with traversal events/fragments;
- replace `SouthStarDisconnectedCompositionOracleResult` with
  `SouthStarDisconnectedCompositionResult`;
- express tetrahedral output checks as facts/obligations over traversal events.

Acceptance:

- current expanded fixtures remain witnesses, not authorities;
- nonstereo ring traversal uses shared event/fragment records even before ring
  stereo constraints are fully unified;
- tetrahedral witness code may remain independent search, but any emitted
  ligand-order obligation is named as a constraint-family concept rather than
  hidden in renderer text assembly.

### `South Star 84E`

Reclassify fixture authority metadata:

- `tests/helpers/south_star_domain_manifest.py`
- `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json`
- `tests/fixtures/south_star_exact_first_domain/first_domain_v1.json`
- `tests/south_star/test_package_readiness.py`
- `docs/enum-s.md`

Support labels should distinguish:

- unified-reference-backed support;
- temporary witness oracle;
- regression fixture only;
- semantic conformance evidence;
- RDKit parseability evidence.

The readiness matrix should not make `independent_*_oracle` names look like
final authorities.

The promotion gate now tracks a separate candidate state: a fixture case can be
generated by the shared fact/event/constraint/renderer path and still remain
blocked from public-package promotion if its manifest authority is temporary
witness evidence or graph-native regression evidence. Promoting a case requires
both shared-pipeline generation and an explicit unified-reference authority.

### `South Star 84F`

Add guardrails against new permanent mini-oracles:

- `tests/south_star/test_domain_manifest.py`
- `tests/south_star/test_dependency_boundaries.py`
- `AGENTS.md` or `docs/enum-s.md` if the rule needs to be visible to agents.

Acceptance: adding a new `independent_*_oracle` support authority without a
fold-in note should fail review/tests.

## Temporary Witnesses

The following helpers remain useful during migration:

- `independent_first_domain_support_for_case`
- `independent_saturated_monocycle_support_for_case`
- `independent_nonstereo_monocycle_support_for_case`
- `independent_disconnected_composition_support_for_case`
- `shared_ring_stereo_monocycle_support_for_case`
- `independent_tetrahedral_atom_stereo_support_for_case`

They should be treated as witness scaffolding. Their long-term value is to
prove the shared model did not lose a known case, not to define separate
domains forever.

## Risks To Avoid

- Moving algorithmic generation into `reference_model.py` and creating a
  hidden monolith.
- Renaming records without deleting duplicate local semantics.
- Treating the first-domain oracle as "legacy" and losing independent witness
  coverage before the shared model is validated.
- Folding tetrahedral rendering into the directional marker parity solver
  before atom-stereo obligations are named separately.
- Allowing fixture `support_authority` strings to imply that per-domain helpers
  are final correctness sources.

## Review Checklist For This Slice

- The inventory names exact files and duplicated records/functions.
- The target shared records are concrete enough for `84B`.
- The follow-up order preserves current support while reducing duplication.
- No behavior change is proposed in this slice.
- No new support authority is introduced.
