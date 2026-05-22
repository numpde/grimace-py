# South Star 1 Direction

Branch at creation: `south-star`

Date: 2026-05-22

Context:

- `notes/091_smiles_exact_support.tex`
- `notes/092_south_star_formal_model_alignment.md`

## Purpose

Define the next branch direction before pruning the current exploratory South
Star work.

The current `south-star` branch has accumulated useful evidence, but it also
contains export-readiness work, feature-local witnesses, broad fixture
expansion, and package-facing scaffolding that predate the clarified formal
model. Preserve that branch as `south-star-0`. Start `south-star-1` as a
smaller formal-model branch.

## Direction

`south-star-1` should implement South Star as the finite model from
`091_smiles_exact_support.tex`:

```text
finite attributed graph-traversal grammar
+ finite-domain constraints over syntax slots
```

The branch should be judged by whether code maps directly to that formulation.
The target vocabulary is:

- fixed molecule facts;
- traversal skeletons;
- typed syntax slots;
- finite parser relations;
- semantic constraints, including no-accidental-stereo constraints;
- presentation-policy predicates;
- solver assignments;
- pure rendering;
- support as `image(render)`.

## Keep Criteria

Keep South Star code only when it is strictly aligned with the formal model.

Aligned work:

- fact extraction that is independent of a feature-local expected-support
  fixture;
- shared records for traversal events, syntax slots, constraints, assignments,
  and renderer inputs;
- explicit annotation-policy interfaces;
- parser-defined potential stereo site records;
- named no-accidental-stereo constraint records;
- tests that check formal-model invariants directly;
- notes that define the formal direction or explain why previous work is being
  archived.

## Remove Criteria

Remove or archive from `south-star-1` anything not strictly aligned:

- public `MolToSmilesEnumS` export-readiness work;
- package-readiness gates for a provisional API;
- release-note checklist work for exporting `MolToSmilesEnumS`;
- broad fixture-growth work whose authority is expected strings rather than the
  formal model;
- feature-local mini-oracles that define their own support universe;
- performance artifacts for the previous private enumerator;
- docs that present the old implementation as package-ready or near export;
- exploratory scripts not required for the formal model branch;
- temporary witnesses whose concepts have not yet been mapped to the shared
  grammar/slot/constraint vocabulary.

This does not mean the old work was useless. It means it belongs in
`south-star-0` until a specific concept is reintroduced through the formal
spine.

## Immediate Shape Of `south-star-1`

The first branch state should be intentionally small:

- keep the formal notes;
- keep only minimal private modules/tests needed to express the shared
  vocabulary;
- remove the previous `MolToSmilesEnumS` readiness/export path;
- remove feature-domain fixtures and tests that would pressure the branch back
  into case-by-case expansion;
- add back behavior only by first naming its molecule facts, traversal
  skeletons, syntax slots, parser relation, and constraints.

## Non-Goals

- Do not export `MolToSmilesEnumS`.
- Do not preserve previous readiness gates for continuity.
- Do not keep broad expected-support fixtures as South Star authority.
- Do not keep mini-oracles merely because they pass.
- Do not chase RDKit writer parity in this branch.

