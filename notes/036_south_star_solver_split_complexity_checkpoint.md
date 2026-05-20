# South Star Solver-Split Complexity Checkpoint

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 36: Repeat complexity checkpoint after solver split`

## Scope

This checkpoint repeats the earlier seed measurement after the South Star
implementation was moved behind the private `grimace._south_star` boundary and
split into explicit component, marker-slot equation, solver, and renderer
pieces.

This is not a package performance claim. It is a structural checkpoint: the
question is whether the implementation still has the intended
component-factorized complexity shape, or whether it has regressed into one
global stereo-row filter.

Current implementation limits remain:

- one connected component only;
- acyclic traversal only;
- graph-native directional double-bond marker semantics only;
- no ring closures, disconnected fragments, aromatic output, or tetrahedral
  atom stereo;
- Python prototype implementation, not the final Rust runtime path.

## Measured Command

```sh
PYTHONPATH=python:. python3 - <<'PY'
from statistics import median
from time import perf_counter

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import (
    mol_to_smiles_enum_s_graph_native_for_case,
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from tests.helpers.south_star_comparison import _grimace_public_parity_support
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


def event_skeleton(traversal):
    return tuple(
        (
            event.kind,
            event.text,
            event.edge,
            event.begin_atom_idx,
            event.end_atom_idx,
            event.begin_parent_idx,
            None
            if event.marker_slot is None
            else (
                event.marker_slot.edge,
                event.marker_slot.begin_atom_idx,
                event.marker_slot.end_atom_idx,
                event.marker_slot.begin_parent_idx,
                event.marker_slot.syntax_position,
            ),
        )
        for event in traversal.events
    )


for case in load_south_star_semantic_cases():
    mol_to_smiles_enum_s_graph_native_for_case(case)

for case in load_south_star_semantic_cases():
    state = SouthStarComponentSupportState.from_case(case)
    snapshot = state.complexity_snapshot()
    local_assignments = " x ".join(
        str(estimate.estimated_local_assignment_count)
        for estimate in snapshot.local_assignment_estimates
    ) or "1"
    runs = []
    for _ in range(7):
        t0 = perf_counter()
        result = mol_to_smiles_enum_s_graph_native_for_case(case)
        runs.append((perf_counter() - t0) * 1000)
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    marker_slot_counts = tuple(
        sum(1 for event in traversal.events if event.marker_slot is not None)
        for traversal in traversals
    )
    equation_counts = tuple(
        len(marker_slot_parity_equations_for_traversal(state, traversal))
        for traversal in traversals
    )
    marker_slots = f"{min(marker_slot_counts)}-{max(marker_slot_counts)}"
    equations = f"{min(equation_counts)}-{max(equation_counts)}"
    print(
        case.case_id,
        snapshot.component_count,
        local_assignments,
        snapshot.estimated_product_size,
        len({event_skeleton(traversal) for traversal in traversals}),
        len(traversals),
        marker_slots,
        equations,
        len(result.outputs),
        len(_grimace_public_parity_support(case.source_smiles)),
        f"{median(runs):.3f}",
    )
PY
```

The event skeleton count ignores marker signs but preserves marker-slot
positions.

## Snapshot

| case | components | local assignments | product | event skeletons | traversals | marker slots | equations | EnumS outputs | parity outputs | median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolated_alkene_z | 1 | 2 | 2 | 6 | 12 | 2-2 | 2-2 | 12 | 6 | 1.391 |
| linear_diene_same_phase | 1 | 2 | 2 | 6 | 12 | 3-3 | 3-3 | 6 | 5 | 2.458 |
| branched_substituted_alkene | 1 | 2 | 2 | 16 | 32 | 3-3 | 3-3 | 32 | 16 | 4.135 |
| hetero_imine_carrier | 1 | 2 | 2 | 6 | 12 | 2-2 | 2-2 | 12 | 6 | 1.693 |
| independent_two_alkenes | 2 | 2 x 2 | 4 | 14 | 56 | 4-4 | 4-4 | 56 | 14 | 9.782 |
| linear_diene_opposite_phase | 1 | 2 | 2 | 6 | 12 | 3-3 | 3-3 | 6 | 5 | 1.829 |
| same_side_alternate_carriers | 1 | 2 | 2 | 16 | 32 | 3-3 | 3-3 | 32 | 16 | 3.041 |

`parity outputs` is the current public RDKit-writer-parity support size for the
same source molecule. It is included only as a scale comparison, not as a target
for South Star equality.

## Interpretation

The solver split preserved the intended stereo-complexity shape on the current
fixture corpus:

- each single directional-stereo component still has two local assignments:
  source marker orientation and global flip;
- coupled diene fixtures remain one component with two assignments, not one
  independent axis per stereo feature;
- independent components multiply explicitly: the two-alkene fixture reports
  `2 x 2 = 4`;
- marker-slot equation counts are local to the rendered traversal skeleton and
  match the number of emitted marker slots in every current fixture;
- no checkpoint row indicates a global assignment explosion across unrelated
  stereo facts.

The larger `EnumS outputs` counts are expected relative to the seed checkpoint:
the implementation now enumerates all atom roots and branch-order variants for
the connected acyclic domain. That growth is traversal-language growth, not
evidence that stereo constraints are being handled by a late global filter.

The independent-two-alkene row is the most important guardrail here. It shows
component-local assignment multiplication (`2 x 2`) while traversal skeletons
and final strings expand separately. That is the intended separation between
semantic stereo state and traversal/rendering support.

## Risks

This checkpoint is still small. It does not yet test whether the same
factorization survives:

- ring-closure syntax;
- disconnected fragment composition;
- tetrahedral atom stereo;
- aromatic and bracket-atom rendering;
- large molecules with several independent and coupled stereo components.

The timing numbers are also prototype timings. They include Python traversal,
equation construction, and solver calls. They are useful only as a local
regression smell test while the architecture is still moving.

## Follow-Up

The next South Star slices should keep this split intact:

- ring support should add new traversal events and marker-slot locations, not a
  post-render string repair;
- disconnected support should compose independent fragment policies separately
  from stereo component products;
- tetrahedral atom stereo should be introduced as a separate constraint family,
  not mixed into directional double-bond carrier equations;
- future checkpoints should add larger fixtures that distinguish traversal
  growth from semantic stereo assignment growth.
