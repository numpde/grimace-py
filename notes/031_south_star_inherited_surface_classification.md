# South Star Inherited Surface Classification

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 05: Classify inherited stereo surfaces`

## Purpose

This note classifies the stereo machinery inherited from
`stereo-constraint-model` against the South Star split:

- semantic facts and constraints;
- annotation-policy boundary;
- RDKit writer-policy behavior;
- comparison/diagnostic evidence;
- reusable scaffolding;
- obsolete or later-removable surfaces.

This is a classification checkpoint, not a deletion plan. The rule remains:
classify first, test second, quarantine or delete only after behavior is covered.

## Boundary Used For Classification

South Star now has a test-side annotation-policy boundary:

- `tests/helpers/south_star_annotation_policy.py`
- `tests/south_star/test_annotation_policy_boundary.py`

The boundary consumes policy-neutral facts:

- semantic carrier opportunities;
- emitted-edge basis;
- surviving semantic assignments and marker options.

It returns policy output:

- whether a marker is required;
- which slash/backslash markers are allowed.

The first policy module is maximal eligible-carrier annotation. This is a
policy choice behind a boundary, not the definition of semantic support itself.

## Classification Table

| Surface | Current Location | Classification | South Star Handling |
| --- | --- | --- | --- |
| `StereoSideInfo`, side domains, carrier choices | `rust/src/bond_stereo_constraints.rs` | reusable semantic scaffolding | Keep as input to semantic facts unless later replaced by a cleaner graph-side model. |
| `StereoConstraintLayer::Semantic` | `rust/src/bond_stereo_constraints.rs` | semantic surface | Keep conceptually, but verify it is not polluted by writer-policy row filtering. |
| `StereoConstraintLayer::RdkitLocalWriter` | `rust/src/bond_stereo_constraints.rs` | RDKit writer-policy | Keep only for North Star runtime or explicit comparison. Do not import into South Star semantic runner. |
| `StereoConstraintLayer::RdkitTraversalWriter` | `rust/src/bond_stereo_constraints.rs` | RDKit writer-policy | Same as local writer; useful for comparison, not semantic authority. |
| `StereoTokenPhaseAssignment`, `StereoTokenFlip`, `StereoDirectionToken` | `rust/src/bond_stereo_constraints.rs` | likely reusable semantic scaffolding | Keep as token algebra if separated from RDKit adjustment facts. |
| `RdkitTokenFlipAdjustmentObservations` | `rust/src/bond_stereo_constraints.rs` | RDKit writer-policy | Comparison/debug only for South Star. It names RDKit traversal-conditioned spelling behavior. |
| `StereoTokenObservationFact` | `rust/src/bond_stereo_constraints.rs` | mixed | Contains useful observation vocabulary but currently includes RDKit adjustment. Split before reuse. |
| `StereoMarkerPlacementRow` | `rust/src/bond_stereo_constraints.rs` | ambiguous, RDKit-shaped representation | Do not treat as South Star target shape. It may be reusable as finite assignment scaffolding, but marker subsets/no-marker behavior are writer-shaped. |
| `StereoMarkerEventFact::MarkerPlaced` | `rust/src/bond_stereo_constraints.rs` | mixed | Visible marker facts are semantic if expressed through annotation policy; current event filtering is RDKit-shaped. |
| `StereoMarkerEventFact::NoMarker` | `rust/src/bond_stereo_constraints.rs` | RDKit writer-policy unless proven otherwise | South Star maximal eligible-carrier should not start by treating omission as a semantic observation. |
| `StereoConstraintState` | `rust/src/bond_stereo_constraints.rs` | reusable state idea, mixed fields | Keep as an idea; split carrier/token semantic survivors from writer-policy marker-row survivors before reuse. |
| marker-row filtering helpers | `rust/src/bond_stereo_constraints.rs` | RDKit writer-policy / comparison | Do not use as South Star support authority. They answer which RDKit-shaped marker rows survive events. |
| `canonical_edge` / edge normalization | `rust/src/bond_stereo_constraints.rs` | semantic utility | Reusable. South Star test helper already mirrors this as `normalized_edge`. |
| `RootedConnectedStereoWalkerStateData` stereo fields | `rust/src/rooted_stereo.rs` | mixed runtime state | Useful facts exist here, but the state also carries RDKit marker traces and writer quotient facts. Do not expose wholesale as South Star state. |
| `stereo_selected_neighbors` | `rust/src/rooted_stereo.rs` | mixed | Selected carrier facts can be semantic; repair/forced-neighbor machinery must be classified before reuse. |
| `committed_component_token_flips` | `rust/src/rooted_stereo.rs` | mixed | Token facts may be semantic; committed parity assertions are North Star guardrails until recast through the annotation-policy boundary. |
| `marker_event_traces` and `directional_marker_traces` | `rust/src/rooted_stereo.rs` | comparison/debug | Useful to compare current runtime behavior, not semantic authority. |
| `writer_marker_slot_quotient_acceptance_facts` | `rust/src/rooted_stereo.rs` | RDKit writer-policy / comparison | Keep out of South Star support. |
| `assert_committed_component_token_flips_match_boundary_observations` | `rust/src/rooted_stereo.rs` | North Star guardrail | Do not carry into South Star except as optional comparison during migration. |
| pinned RDKit exact support fixtures | `tests/fixtures/rdkit_exact_small_support/` | RDKit parity authority | Excluded from South Star runner. Useful for comparison only. |
| pinned writer-membership fixtures | `tests/fixtures/rdkit_writer_membership/` | RDKit parity authority | Excluded from South Star runner. Useful for South Star-vs-RDKit comparison tasks only. |
| stereo-constraint-model fixtures | `tests/fixtures/stereo_constraint_model/` | diagnostic evidence | Useful to understand inherited machinery; not South Star semantic fixtures. |
| known stereo gaps | `tests/fixtures/rdkit_known_stereo_gaps/` | comparison evidence | Useful for later divergence classification, not initial semantic pass/fail. |
| South Star semantic fixtures | `tests/fixtures/south_star_semantics/` | semantic evidence | This is the starting SSoT for South Star witnesses. |

## Immediate Consequences

- South Star tests should continue to import `rdkit` only as a parser/semantic
  checker, not as a writer oracle.
- South Star tests should not import `tests.rdkit_serialization`,
  `tests.helpers.pinned_rdkit_fixtures`, or
  `tests.helpers.stereo_constraint_model`.
- Runtime internals should not be exposed wholesale to South Star diagnostics.
  The next query should return policy-neutral semantic facts and separate
  annotation-policy decisions.
- `NoMarker` should stay out of South Star semantic support unless a later
  policy explicitly makes omission meaningful.
- Maximal eligible-carrier policy should be tested through the boundary, not by
  asserting marker-row behavior.

## Follow-Up Classification Questions

- Which fields in `RootedConnectedStereoWalkerStateData` are purely semantic
  facts, and which are writer-policy trace state?
- Can the existing carrier assignment IDs be reused directly as
  `SurvivingSemanticAssignment` ids, or do they need a cleaner adapter?
- Are token-phase assignments policy-neutral, or do they already encode RDKit
  basis choices?
- Can marker rows be demoted to comparison/debug without losing useful
  diagnostic visibility?
- What is the smallest runtime query that can expose semantic carrier
  opportunities without exposing RDKit writer layers?

## Minimal Next Step

For `South Star 06`, add a diagnostic query at the test/helper level first:

1. Load a South Star semantic fixture.
2. Return policy-neutral semantic facts for that fixture.
3. Apply the annotation-policy boundary separately.
4. Keep RDKit writer output and current Grimace parity output out of the
   semantic result.

Only after that should runtime-backed diagnostics be considered.
