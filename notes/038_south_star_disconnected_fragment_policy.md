# South Star Disconnected Fragment Policy

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 38: Plan disconnected molecule policy`

## Purpose

Disconnected molecule support should be a fragment-composition policy, not an
accidental inheritance of RDKit writer fragment ordering.

The public Grimace runtime currently targets RDKit writer parity, so its
fragment behavior is RDKit-shaped. South Star is different: it needs a
principled semantic language first, with any writer-like ordering policy named
separately.

## Current Boundary

The private South Star path originally failed fast on disconnected molecules:

- `south_star_support_gate_report()` reported `disconnected_molecule`;
- graph-native `EnumS` required exactly one connected component;
- the first-domain oracle required exactly one connected component.

That was the right temporary boundary. The next step was not to remove the gate
blindly; it was to introduce an explicit fragment composition layer.

## Implementation Checkpoint

The private `south-star` branch now has the first explicit fragment-composition
layer:

- disconnected molecules are split into connected fragments;
- each fragment is enumerated independently through the connected South Star
  path;
- rendered fragment supports are composed with the named all-fragment-orders
  policy;
- disconnected markerless and disconnected stereo-fragment cases are pinned in
  `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json`;
- the connected traversal path still rejects already-disconnected inputs.

The current policy is intentionally semantic and not RDKit writer parity. Later
package work can add selectable fragment-order policies without changing the
connected-fragment enumerator.

## Required Concept Split

Fragment semantics:
: Each connected component has its own graph semantics, stereo components,
  traversal choices, marker slots, equations, and rendered local strings.

Composition policy:
: A policy decides how local fragment strings are joined with `.`. This is not
  a chemistry/stereo constraint; it is a language-ordering choice.

Output ordering policy:
: The order in which complete disconnected strings are returned is separate
  from the set of strings in the language.

Writer parity:
: RDKit fragment order, rooted-fragment handling, and writer-specific fragment
  traversal are not South Star defaults. They may be added later as an explicit
  writer policy, not as hidden behavior.

## Semantic Target

For disconnected input with fragments `F1, F2, ..., Fn`, the semantic support
should be the product of fragment-local supports and the chosen fragment
composition policy.

If the first South Star composition policy is "all fragment permutations", then
support is:

```text
{ join(".", p1, ..., pn)
  for each fragment-order permutation P
  for each local spelling pi in support(P[i]) }
```

If the first policy is "input fragment order only", then the same local product
is used but the fragment order is fixed.

Both are principled. Neither should be confused with RDKit writer parity.

## Recommended First Policy

Use an explicit `AllFragmentOrderPolicy` for South Star semantics.

Reasons:

- it is simple to define mathematically;
- it does not smuggle in RDKit's writer/rooting behavior;
- it makes fragment independence visible in support-size tests;
- a later fixed-order policy can be added as a separate policy without changing
  connected-fragment enumeration.

This does mean support can multiply by `n!` for `n` distinguishable fragments.
That is acceptable for a semantic enumerator, and it can be controlled later by
policy selection.

## Implementation Sketch

Introduce a small fragment-composition boundary outside connected traversal:

1. Parse the molecule into connected fragment molecules while preserving atom
   index mapping metadata.
2. Run the existing connected South Star enumerator independently for each
   fragment.
3. Keep each fragment's complexity snapshot separate.
4. Combine local supports with a named fragment order policy.
5. Return a disconnected result that exposes:
   - fragment count;
   - per-fragment output counts;
   - fragment order policy name;
   - total support count.

The connected enumerator should not learn about `.`. The composer owns `.`.

## First Fixtures

Start with cases that do not require new graph syntax:

1. `CC.O`: two simple fragments, no stereo.
2. `F/C=C/Cl.O`: one directional fragment plus one simple fragment.
3. `F/C=C/Cl.O.N`: three fragments, proving permutation multiplication.
4. `F/C=C/Cl.F/C=C\\Cl`: two stereo fragments, proving independent component
   products are local to their fragment before composition.

For each fixture, assert:

- each connected fragment is accepted by the existing first-domain gate;
- disconnected composition is the only newly exercised layer;
- support size equals the product of local support sizes times the fragment
  order count for the selected policy;
- public RDKit-parity support is not used as the expected South Star set.

## Guardrails

- Keep `disconnected_molecule` fail-fast in the connected enumerator.
- Do not pass a disconnected molecule directly into connected traversal.
- Do not use RDKit's emitted fragment order as the default South Star policy.
- Do not collapse fragment-local complexity snapshots into one opaque global
  count.
- Do not parse rendered disconnected strings as the construction mechanism.

## Minimal Implementation Path

1. Add a private fragment composition module under `grimace._south_star`.
2. Add a no-stereo two-fragment unit test for composition from already-rendered
   fragment support sets.
3. Wire connected fragment enumeration only after the pure composer is tested.
4. Add directional-stereo fragment fixtures once composition is independent of
   connected traversal internals.
5. Repeat the complexity checkpoint with connected and disconnected rows shown
   separately.
