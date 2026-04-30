from __future__ import annotations

import math
import unittest

from rdkit import rdBase

from grimace import _core, _runtime
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import public_enum_support, supported_public_kwargs
from tests.helpers.stereo_constraint_model import (
    load_pinned_stereo_constraint_model_cases,
)


SUPPORTED_STEREO_FLAGS = _runtime.MolToSmilesFlags(
    isomeric_smiles=True,
    kekule_smiles=False,
    rooted_at_atom=-1,
    canonical=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    do_random=True,
    ignore_atom_map_numbers=False,
)


def _effective_layer_assignment_count(
    *,
    layer: dict[str, object],
    semantic_assignment_count: int,
) -> int:
    assignment_count = layer["assignment_count"]
    if assignment_count is None:
        return semantic_assignment_count
    if type(assignment_count) is not int:
        raise AssertionError(f"unexpected layer assignment count: {assignment_count!r}")
    return assignment_count


class StereoConstraintModelFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_stereo_constraint_model_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned stereo-constraint-model corpus for RDKit "
                f"{rdBase.rdkitVersion}"
            )

    def test_native_model_shape_matches_pinned_witnesses(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            summary = _core._stereo_constraint_model_summary(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_component_count,
                    summary["component_count"],
                )
                self.assertEqual(case.expected_side_count, summary["side_count"])
                self.assertEqual(
                    case.expected_component_side_domain_sizes,
                    tuple(
                        tuple(component["side_domain_sizes"])
                        for component in summary["components"]
                    ),
                )
                self.assertEqual(
                    case.expected_component_domain_assignment_counts,
                    tuple(
                        component["domain_assignment_count"]
                        for component in summary["components"]
                    ),
                )
                self.assertEqual(
                    case.expected_semantic_assignment_count,
                    math.prod(case.expected_component_domain_assignment_counts),
                )
                layers_by_name = {
                    layer["layer"]: layer
                    for component in summary["components"]
                    for layer in component["layers"]
                }
                self.assertEqual(
                    case.expected_rdkit_local_writer_assignment_count,
                    _effective_layer_assignment_count(
                        layer=layers_by_name["rdkit_local_writer"],
                        semantic_assignment_count=case.expected_semantic_assignment_count,
                    )
                )
                modeled_traversal_count = _effective_layer_assignment_count(
                    layer=layers_by_name["rdkit_traversal_writer"],
                    semantic_assignment_count=case.expected_semantic_assignment_count,
                )
                if (
                    case.expected_rdkit_traversal_writer_assignment_count
                    == case.expected_rdkit_local_writer_assignment_count
                ):
                    self.assertEqual(
                        case.expected_rdkit_traversal_writer_assignment_count,
                        modeled_traversal_count,
                    )
                else:
                    self.assertEqual(
                        case.expected_rdkit_local_writer_assignment_count,
                        modeled_traversal_count,
                    )

    def test_pinned_layer_counts_are_ordered_by_contract_strength(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertLessEqual(
                    case.expected_rdkit_local_writer_assignment_count,
                    case.expected_semantic_assignment_count,
                )
                self.assertLessEqual(
                    case.expected_rdkit_traversal_writer_assignment_count,
                    case.expected_rdkit_local_writer_assignment_count,
                )

    def test_current_runtime_support_count_matches_pinned_witnesses(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_grimace_runtime_support_count,
                    len(
                        public_enum_support(
                            mol,
                            **supported_public_kwargs(
                                isomericSmiles=True,
                                rootedAtAtom=-1,
                            ),
                        )
                    ),
                )

    def test_output_fact_diagnostic_maps_minimal_witness_to_runtime_support(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "minimal_nonstereo_double_hazard"
        )
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)

        rows = _core._stereo_constraint_output_facts(prepared)
        diagnostic_outputs = frozenset(row["smiles"] for row in rows)
        runtime_outputs = public_enum_support(
            mol,
            **supported_public_kwargs(
                isomericSmiles=True,
                rootedAtAtom=-1,
            ),
        )

        self.assertEqual(runtime_outputs, diagnostic_outputs)
        self.assertEqual(
            case.expected_grimace_runtime_support_count,
            len(diagnostic_outputs),
        )
        self.assertGreater(len(rows), len(diagnostic_outputs))
        self.assertTrue(
            all(row["resolved_layer_completions"]["semantic"] for row in rows)
        )
        self.assertEqual(
            {False, True},
            {
                row["resolved_layer_completions"]["rdkit_local_writer"]
                for row in rows
            },
        )


if __name__ == "__main__":
    unittest.main()
