# Component-boundary proof accounting

## Owned relation

`component_boundary_transition` is owned by one exact DOT branch. Its proof is
the conjunction of the completed-component graph partition, the producer-free
local-order closure equation, and complete equality with the expected successor
writer state. Neither lifecycle flags nor aggregate obligation-family status
confer relation credit.

The support-artifact verifier records the digest of each successfully replayed
DOT branch. It reports the relation only when the number of distinct replayed
digests equals the number of component-boundary branches. The count-free branch
artifact reports the relation only when its own branch digest is in that set.

## Shared local-order equation

DOT and EOS now call the same producer-free state equation. It has three exact
outcomes:

- `already_closed_noop`: stereo state is byte-for-byte unchanged and no
  lifecycle or residual evidence exists;
- `record_only`: only the active local-order record closes, the residual
  snapshot is unchanged, and one raw lifecycle is reconstructed;
- `tetra_residual`: the tetra parity restriction, propagation, discharge,
  projection, successor snapshot, raw lifecycle, and residual work are
  reconstructed from facts and the source state.

The serialized transition term is evidence to compare, not an execution
authority. The equation executes a fresh `ResidualStore` transition.

## Component isolation

Residual variables, assignments, factor keys, factor scopes, and directional
carrier models are assigned to components through facts. Cross-component
factors reject. After DOT, no residual identity may belong to a completed
component. Exact successor-state equality proves that future-component state is
otherwise unchanged.

Completed-prefix replay also requires earlier local-order records to be closed,
allows only the current active record to remain open before DOT, validates
reusable label uniqueness, and checks closure labels and endpoint choices
against the serialized writer policy.

## Baseline correction

The mandatory legacy gate exposed two pre-existing hydrogen-coherence failures
in tetra bracket-text matching. The matcher previously accepted an implicit H
when either the atom fact or the site occurrence declared it. It now requires
the two facts to agree, matching `MoleculeFacts.validate()` ownership.
