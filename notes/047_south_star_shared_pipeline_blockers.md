# South Star Shared-Pipeline Blocker Audit

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 129a: Enumerate shared-pipeline blockers`

## Summary

The `not_generated_by_shared_pipeline` blocker currently overstates the actual
structural problem.

All twelve blocked cases below have:

- `generation_basis == south_star_graph_native_equation_solved_tree_traversal`;
- graph-native outputs equal to the pinned expected support;
- no missing readiness coverage stage.

They are classified as not shared-pipeline-generated because
`south_star_unified_reference_promotion_checks()` currently uses a narrow
feature-area eligibility predicate, `SOUTH_STAR_SHARED_PIPELINE_RING_FEATURE_AREAS`,
for expanded support cases. That predicate no longer matches the implementation
shape after atom-text, disconnected-fragment, ring-stereo, and tetrahedral
slices were routed through the graph-native path.

So the immediate fix is not feature implementation. The immediate fix is to
replace the ring-only eligibility predicate with an explicit per-feature
shared-pipeline eligibility model.

This audit does not promote any support authority to `unified-reference-backed`.
The separate authority blocker remains valid for all checked cases.

## Blocked Cases

| case id | feature area | current authority | missing stage | structural fix |
| --- | --- | --- | --- | --- |
| `ring_stereo_monocycle_cyclooctene` | `ring_stereo_monocycle` | `temporary_witness_ring_stereo_monocycle_shared_records` | none | fix shared-pipeline eligibility classification |
| `markerless_disconnected_ring_and_atom` | `disconnected_markerless_fragments` | `temporary_witness_disconnected_composition_shared_records` | none | fix shared-pipeline eligibility classification |
| `disconnected_stereo_fragment_and_atom` | `disconnected_stereo_fragments` | `temporary_witness_disconnected_composition_shared_records` | none | fix shared-pipeline eligibility classification |
| `explicit_bracket_hydrogen_h2` | `explicit_bracket_hydrogen` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `radical_atom_text_hydrogen` | `radical_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `radical_atom_text_methyl` | `radical_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `radical_atom_text_oxygen` | `radical_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `charged_atom_text_chloride` | `charged_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `charged_atom_text_ammonium` | `charged_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `charged_atom_text_methylammonium` | `charged_atom_text` | `graph_native_regression_witness_with_semantic_parseback` | none | fix shared-pipeline eligibility classification |
| `implicit_h_tetrahedral_center` | `tetrahedral_atom_stereo` | `temporary_witness_tetrahedral_atom_stereo` | none | fix shared-pipeline eligibility classification |
| `quaternary_tetrahedral_center` | `tetrahedral_atom_stereo` | `temporary_witness_tetrahedral_atom_stereo` | none | fix shared-pipeline eligibility classification |

## Coverage Shape

The blocked cases split into three readiness shapes:

- atom-text-only cases: molecule facts, traversal, solver assignment, renderer,
  and semantic evidence are covered; observation and constraint-family stages
  are correctly `not_required`;
- disconnected composition cases: fragment traversal and renderer stages are
  covered, but the eligibility predicate does not recognize disconnected
  feature areas;
- tetrahedral and ring-stereo cases: observation and constraint-family stages
  are covered through renderer inputs or marker slots, but eligibility does not
  recognize those feature areas.

This means there is no evidence here for a missing molecule-facts,
traversal-event, solver-assignment, renderer-input, or semantic-evidence
implementation task. The evidence points to readiness metadata and promotion
criteria.

## Follow-Up

Create one implementation task:

- replace the ring-only shared-pipeline eligibility predicate with explicit
  feature-area eligibility metadata, probably near the South Star domain
  manifest rather than inside `test_package_readiness.py`.

That task should keep the authority blocker untouched. A case can be generated
by the shared pipeline and still not be public-package-ready if its support
authority is temporary or regression-only.

