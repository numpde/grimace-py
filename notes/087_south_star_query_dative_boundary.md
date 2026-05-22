# South Star Query And Dative Boundary

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 240: Harden query and dative boundaries`

## Purpose

Pin the current query and dative surfaces as intentional South Star
fail-fast boundaries, not accidental missing cases.

The current South Star contract is fixed-molecule semantic enumeration. Query
inputs and coordination/dative inputs require additional semantic models before
they can be admitted.

## Boundary Witnesses

### `C~C`

Observed categories:

- `query_bond`;
- `unsupported_bond_type`.

Reason:

- the unspecified/query bond does not denote one fixed bond-order graph under
  the current South Star contract;
- admitting it would require a query-product or uncertainty model, not a local
  bond-text renderer tweak.

### `N->[O]`

Observed categories:

- `dative_bond`;
- `unsupported_bond_type`.

Reason:

- dative/coordination bonds need separate semantic modeling;
- the boundary is not only about metal atoms, because `N->[O]` is non-metal and
  still outside ordinary fixed-molecule bond-text semantics.

### `[NH3]->[Cu]`

Observed categories include:

- `dative_bond`;
- `metal_atom`;
- `unsupported_atom_text`;
- `unsupported_bond_type`.

Reason:

- this combines coordination bonding with metal atom text and should remain a
  distinct broader boundary from the non-metal dative witness.

## Tests Added

`tests.south_star.test_support_gates` now pins the two small frontier witnesses
directly:

- `C~C` remains a query/unsupported-bond boundary;
- `N->[O]` remains a non-metal dative/unsupported-bond boundary and does not
  depend on the `metal_atom` blocker.

## Future Admission Requirement

Do not admit query or dative support by adding renderer cases. Admission would
need a named semantic model:

- query bonds/atoms need a query-product or fixed-instantiation contract;
- dative bonds need a coordination semantics contract that covers graph,
  charge/valence, parse-back identity, and any known serializer quirks.

