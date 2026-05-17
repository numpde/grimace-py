# Visible-Marker Carrier-Selection Audit

Branch: `stereo-constraint-model`

## Purpose

This note inventories runtime paths that still treat a visible slash/backslash
marker as evidence for a selected carrier edge. That interpretation is not
valid in general: RDKit can place a visible marker on a complement or bridge
candidate while the selected stereo carrier for the component is a different
edge.

## Current Shortcuts

### Selected-carrier directional marker trace

Code:

- `rust/src/rooted_stereo.rs`: `selected_carrier_directional_marker_side`
- `rust/src/rooted_stereo.rs`: `record_directional_marker_trace`
- `rust/src/rooted_stereo.rs`: `marker_event_facts_by_component`
- `rust/src/rooted_stereo.rs`: `traversal_constraint_facts_by_component`

Current behavior:

- `record_directional_marker_trace` records a primary directional marker trace
  only when the visible marker edge is the currently selected carrier for a
  side.
- `marker_event_facts_by_component` converts those traces into
  `StereoMarkerEventFact::MarkerPlaced`.
- `traversal_constraint_facts_by_component` also emits
  `CarrierEdgeEmitted`/`DirectionalMarkerPlaced` facts from those selected-side
  traces.

Assessment:

- This is an explicit compatibility path for the old runtime.
- It is safe only when visible marker edge and selected carrier edge coincide.
- It cannot explain complement-candidate marker placement.

### Deferred-token marker-event candidate

Code:

- `rust/src/rooted_stereo.rs`: `marker_event_for_deferred_component_token`
- `rust/src/rooted_stereo.rs`: `deferred_candidate_survives_marker_rows`
- `rust/src/rooted_stereo.rs`: `deferred_token_support_from_constraint_state`

Current behavior:

- Deferred token support simulates the candidate marker as a row-filtering
  event only if the deferred edge is the selected carrier edge for the runtime
  component.

Assessment:

- This is the same selected-carrier shortcut in the deferred-token path.
- It should not be the final source of truth for marker-placement support.
- Replacement should ask marker-placement rows whether the visible edge is a
  valid marker placement for the component, independent of whether it is the
  selected carrier.

### Raw token basis for selected carrier edge

Code:

- `rust/src/rooted_stereo.rs`: `raw_token_for_deferred_edge`

Current behavior:

- The raw token for a deferred edge is computed from active selected carrier
  sides on that edge.

Assessment:

- This is related but not the same bug. Token basis for a selected carrier is a
  real variable in the model; the problem is using visible marker placement to
  infer selected carrier placement.
- Keep this path separate from marker-placement event routing during the next
  refactor.

## Existing Shadow Path

Code:

- `rust/src/rooted_stereo.rs`: `record_marker_event_traces_for_edge`
- `rust/src/rooted_stereo.rs`: `shadow_marker_event_facts_by_component`

Current behavior:

- The shadow trace records marker and no-marker events for every side candidate
  touching the physical edge, not just the selected carrier side.

Important caveat:

- This is not a drop-in runtime replacement. Existing integration coverage for
  `reduced_porphyrin_traversal_coupling` shows that direct all-candidate marker
  events overconstrain marker-placement rows, while the obligation-coalesced
  shadow state remains nonempty.

## Replacement Rule

Do not replace the selected-carrier shortcut with raw all-candidate marker
events. The next runtime path should route visible marker support through
marker-placement row filtering with explicit positive marker events, negative
no-marker events, and obligation coalescing where the exact placement may be
deferred.

The existing Backlog task `Promote marker placement row filtering to runtime`
is the implementation follow-up for this audit.
