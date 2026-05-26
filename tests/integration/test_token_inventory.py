from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest.mock import patch

import grimace._runtime as _runtime
from grimace._reference.dataset import load_default_connected_nonstereo_molecule_cases
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph,
)
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    exact_token_inventory_via_decoder,
    public_token_inventory,
    public_token_inventory_superset,
    prepared_input_variants,
    supported_public_kwargs,
)


@dataclass(frozen=True, slots=True)
class InventoryCase:
    name: str
    smiles: str
    rooted_at_atom: int
    cid: str | None = None
    required_nonstereo: frozenset[str] = frozenset()
    required_stereo: frozenset[str] = frozenset()
    min_nonstereo_token_count: int = 0
    min_stereo_token_count: int = 0


def _complete_nonstereo_prepared_graph(atom_count: int) -> PreparedSmilesGraph:
    neighbors = tuple(
        tuple(neighbor_idx for neighbor_idx in range(atom_count) if neighbor_idx != atom_idx)
        for atom_idx in range(atom_count)
    )
    bond_pairs = tuple(
        (begin_idx, end_idx)
        for begin_idx in range(atom_count)
        for end_idx in range(begin_idx + 1, atom_count)
    )
    return PreparedSmilesGraph(
        schema_version=PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
        surface_kind=CONNECTED_NONSTEREO_SURFACE,
        policy_name="test_static_superset",
        policy_digest="test_static_superset",
        rdkit_version="test",
        identity_smiles="",
        atom_count=atom_count,
        bond_count=len(bond_pairs),
        atom_atomic_numbers=(6,) * atom_count,
        atom_is_aromatic=(False,) * atom_count,
        atom_isotopes=(0,) * atom_count,
        atom_formal_charges=(0,) * atom_count,
        atom_total_hs=(0,) * atom_count,
        atom_radical_electrons=(0,) * atom_count,
        atom_map_numbers=(0,) * atom_count,
        atom_tokens=("C",) * atom_count,
        neighbors=neighbors,
        neighbor_bond_tokens=tuple(("",) * len(row) for row in neighbors),
        bond_pairs=bond_pairs,
        bond_kinds=("SINGLE",) * len(bond_pairs),
        writer_do_isomeric_smiles=False,
        writer_kekule_smiles=False,
        writer_all_bonds_explicit=False,
        writer_all_hs_explicit=False,
        writer_ignore_atom_map_numbers=False,
        identity_parse_with_rdkit=False,
        identity_canonical=False,
        identity_do_isomeric_smiles=False,
        identity_kekule_smiles=False,
        identity_rooted_at_atom=-1,
        identity_all_bonds_explicit=False,
        identity_all_hs_explicit=False,
        identity_do_random=True,
        identity_ignore_atom_map_numbers=False,
    )


class TokenInventoryTests(unittest.TestCase):
    SHARED_DEMANDING_CASES = (
        InventoryCase(
            name="aspirin_all_roots",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            rooted_at_atom=-1,
            required_nonstereo=frozenset({"(", ")", "1", "=", "C", "O", "c"}),
            required_stereo=frozenset({"(", ")", "1", "=", "C", "O", "c"}),
            min_nonstereo_token_count=7,
            min_stereo_token_count=7,
        ),
        InventoryCase(
            name="stereo_atom",
            smiles="F[C@H](Cl)Br",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"(", ")", "Br", "C", "Cl", "F"}),
            required_stereo=frozenset({"(", ")", "Br", "Cl", "F", "[C@H]"}),
            min_nonstereo_token_count=6,
            min_stereo_token_count=7,
        ),
        InventoryCase(
            name="bond_stereo_all_roots",
            smiles="C/C=C/C(=O)O",
            rooted_at_atom=-1,
            required_nonstereo=frozenset({"(", ")", "=", "C", "O"}),
            required_stereo=frozenset({"(", ")", "/", "\\", "=", "C", "O"}),
            min_nonstereo_token_count=5,
            min_stereo_token_count=7,
        ),
        InventoryCase(
            name="disconnected_all_roots",
            smiles="[Na+].C#N",
            rooted_at_atom=-1,
            required_nonstereo=frozenset({".", "#", "C", "N", "[Na+]"}),
            required_stereo=frozenset({".", "#", "C", "N", "[Na+]"}),
            min_nonstereo_token_count=5,
            min_stereo_token_count=5,
        ),
        InventoryCase(
            name="disconnected_duplicate_text_choices",
            smiles="[Na+].CC",
            rooted_at_atom=0,
            required_nonstereo=frozenset({".", "C", "[Na+]"}),
            required_stereo=frozenset({".", "C", "[Na+]"}),
            min_nonstereo_token_count=3,
            min_stereo_token_count=3,
        ),
        InventoryCase(
            name="dataset_long_sulfonamide",
            cid="3488",
            smiles="COC1=C(C=C(C=C1)Cl)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"1", "2", "3", "Cl", "S", "N", "c"}),
            required_stereo=frozenset({"1", "2", "3", "Cl", "S", "N", "c"}),
            min_nonstereo_token_count=12,
            min_stereo_token_count=12,
        ),
        InventoryCase(
            name="dataset_long_heteroaromatic",
            cid="3440",
            smiles="C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"o", "Cl", "S", "2", "1"}),
            required_stereo=frozenset({"o", "Cl", "S", "2", "1"}),
            min_nonstereo_token_count=12,
            min_stereo_token_count=12,
        ),
        InventoryCase(
            name="dataset_long_nitro",
            cid="4485",
            smiles="CC1=C(C(C(=C(N1)C)C(=O)OC)C2=CC=CC=C2[N+](=O)[O-])C(=O)OC",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"[N+]", "[O-]", "2", "1", "c"}),
            required_stereo=frozenset({"[N+]", "[O-]", "2", "1", "c"}),
            min_nonstereo_token_count=11,
            min_stereo_token_count=11,
        ),
        InventoryCase(
            name="dataset_multi_center_steroid",
            cid="5757",
            smiles="C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"1", "2", "3", "4", "O", "c"}),
            required_stereo=frozenset({"1", "2", "3", "4", "O", "[C@]", "[C@@H]"}),
            min_nonstereo_token_count=9,
            min_stereo_token_count=13,
        ),
        InventoryCase(
            name="dataset_long_bond_stereo",
            cid="445639",
            smiles="CCCCCCCC/C=C\\\\CCCCCCCC(=O)O",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"(", ")", "=", "C", "O"}),
            required_stereo=frozenset({"(", ")", "/", "\\", "=", "C", "O"}),
            min_nonstereo_token_count=5,
            min_stereo_token_count=7,
        ),
        InventoryCase(
            name="dataset_azide_and_atom_stereo",
            cid="35370",
            smiles="CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO)N=[N+]=[N-]",
            rooted_at_atom=0,
            required_nonstereo=frozenset({"[N+]", "[N-]", "[nH]", "n", "1", "2"}),
            required_stereo=frozenset({"[N+]", "[N-]", "[nH]", "[C@H]", "[C@@H]", "1", "2"}),
            min_nonstereo_token_count=13,
            min_stereo_token_count=15,
        ),
    )

    def test_token_inventory_matches_exact_decoder_inventory(self) -> None:
        # Use one shared demanding molecule set for both branches. On the
        # nonstereo surface, Grimace should follow RDKit and drop stereo
        # annotations instead of rejecting the molecule outright.
        for case in self.SHARED_DEMANDING_CASES:
            for isomeric_smiles, required_tokens, min_token_count in (
                (False, case.required_nonstereo, case.min_nonstereo_token_count),
                (True, case.required_stereo, case.min_stereo_token_count),
            ):
                with self.subTest(
                    case=case.name,
                    cid=case.cid,
                    isomeric_smiles=isomeric_smiles,
                ):
                    mol = parse_smiles(case.smiles)
                    expected = exact_token_inventory_via_decoder(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=case.rooted_at_atom,
                            isomericSmiles=isomeric_smiles,
                        ),
                    )

                    for token in required_tokens:
                        self.assertIn(token, expected)
                    self.assertGreaterEqual(len(expected), min_token_count)

                    self.assertEqual(
                        expected,
                        public_token_inventory(
                            mol,
                            **supported_public_kwargs(
                                rootedAtAtom=case.rooted_at_atom,
                                isomericSmiles=isomeric_smiles,
                            ),
                        ),
                    )

    def test_token_inventory_superset_contains_exact_inventory(self) -> None:
        for case in self.SHARED_DEMANDING_CASES:
            for isomeric_smiles in (False, True):
                with self.subTest(
                    case=case.name,
                    cid=case.cid,
                    isomeric_smiles=isomeric_smiles,
                ):
                    mol = parse_smiles(case.smiles)
                    kwargs = supported_public_kwargs(
                        rootedAtAtom=case.rooted_at_atom,
                        isomericSmiles=isomeric_smiles,
                    )
                    exact = set(public_token_inventory(mol, **kwargs))
                    superset = set(public_token_inventory_superset(mol, **kwargs))

                    self.assertLessEqual(exact, superset)

    def test_token_inventory_superset_contains_exact_inventory_on_curated_matrix(self) -> None:
        cases = (
            ("single_atom", "C", (0,)),
            ("linear_chain", "CCCC", (0, 1, 3)),
            ("branch", "CC(C)O", (0, 1, 3)),
            ("small_ring", "C1CC1", (0, 1)),
            ("aromatic", "c1ccncc1", (0, 3)),
            ("charged", "C[N+](=O)[O-]", (0, 1, 3)),
            ("dative", "[NH3][Cu]", (0, 1)),
            ("atom_stereo", "F[C@H](Cl)Br", (0, 1)),
            ("bond_stereo", "F/C=C\\Cl", (0, 1, 3)),
            ("disconnected", "[Na+].CC", (0, 1, 2)),
        )
        flag_cases = (
            {"isomericSmiles": False},
            {"isomericSmiles": True},
            {"isomericSmiles": False, "allBondsExplicit": True},
            {"isomericSmiles": True, "allHsExplicit": True},
        )

        for case_name, smiles, roots in cases:
            mol = parse_smiles(smiles)
            for rooted_at_atom in (-1, *roots):
                for flag_kwargs in flag_cases:
                    kwargs = supported_public_kwargs(
                        rootedAtAtom=rooted_at_atom,
                        **flag_kwargs,
                    )
                    with self.subTest(
                        case=case_name,
                        smiles=smiles,
                        rooted_at_atom=rooted_at_atom,
                        flags=flag_kwargs,
                    ):
                        exact = set(public_token_inventory(mol, **kwargs))
                        superset = set(public_token_inventory_superset(mol, **kwargs))
                        self.assertLessEqual(exact, superset)

    def test_token_inventory_superset_contains_exact_inventory_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(
            limit=16,
            max_smiles_length=14,
        )
        self.assertEqual(16, len(cases))

        for case in cases:
            mol = parse_smiles(case.smiles)
            roots = (-1, 0)
            for rooted_at_atom in roots:
                kwargs = supported_public_kwargs(
                    rootedAtAtom=rooted_at_atom,
                    isomericSmiles=False,
                )
                with self.subTest(
                    cid=case.cid,
                    smiles=case.smiles,
                    rooted_at_atom=rooted_at_atom,
                ):
                    exact = set(public_token_inventory(mol, **kwargs))
                    superset = set(public_token_inventory_superset(mol, **kwargs))
                    self.assertLessEqual(exact, superset)

    def test_token_inventory_superset_does_not_instantiate_decoder(self) -> None:
        mol = parse_smiles("CC(C)O")

        with (
            patch.object(
                _runtime,
                "MolToSmilesDecoder",
                side_effect=AssertionError("superset should not walk decoder states"),
            ),
            patch.object(
                _runtime,
                "_exact_token_inventory_from_decoder",
                side_effect=AssertionError("superset should not call exact inventory"),
            ),
        ):
            superset = public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=-1,
                    isomericSmiles=False,
                ),
            )

        self.assertLessEqual({"(", ")", "C", "O"}, set(superset))

    def test_token_inventory_superset_rejects_prepared_writer_flag_mismatch(self) -> None:
        mol = parse_smiles("CC#N")
        prepared = prepared_input_variants(
            mol,
            **supported_public_kwargs(
                rootedAtAtom=0,
                isomericSmiles=False,
                allBondsExplicit=False,
            ),
        )[1][1]

        with self.assertRaisesRegex(
            ValueError,
            "writer flags do not match",
        ):
            public_token_inventory_superset(
                prepared,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    allBondsExplicit=True,
                ),
            )

    def test_token_inventory_superset_includes_static_stereo_variants(self) -> None:
        cases = (
            ("[13C@H](F)(Cl)Br", frozenset({"[13C@H]", "[13C@@H]"})),
            ("[Si@H](F)(Cl)Br", frozenset({"[Si@H]", "[Si@@H]"})),
            ("[C@H:7](F)(Cl)Br", frozenset({"[C@H:7]", "[C@@H:7]"})),
        )

        for smiles, required_tokens in cases:
            with self.subTest(smiles=smiles):
                mol = parse_smiles(smiles)
                superset = set(
                    public_token_inventory_superset(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=0,
                            isomericSmiles=True,
                        ),
                    )
                )

                self.assertLessEqual(required_tokens, superset)

    def test_token_inventory_superset_handles_nonstereo_graphs_without_stereo_fields(self) -> None:
        mol = parse_smiles("CCO")
        exact = set(
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            )
        )
        superset = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            )
        )

        self.assertLessEqual(exact, superset)
        self.assertNotIn("/", superset)
        self.assertNotIn("\\", superset)

    def test_token_inventory_superset_avoids_trivial_branch_extras_for_rooted_linear_chain(self) -> None:
        mol = parse_smiles("CCO")
        exact = set(
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            )
        )
        superset = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            )
        )

        self.assertLessEqual(exact, superset)
        self.assertNotIn("(", exact)
        self.assertNotIn("(", superset)

    def test_token_inventory_superset_keeps_all_roots_branch_candidates_for_linear_chain(self) -> None:
        mol = parse_smiles("CCO")
        exact = set(
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=-1,
                    isomericSmiles=False,
                ),
            )
        )
        superset = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=-1,
                    isomericSmiles=False,
                ),
            )
        )

        self.assertIn("(", exact)
        self.assertIn("(", superset)
        self.assertLessEqual(exact, superset)

    def test_token_inventory_superset_includes_both_dative_directions(self) -> None:
        mol = parse_smiles("[NH3][Cu]")
        root_zero = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            )
        )
        root_one = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=False,
                ),
            )
        )

        self.assertLessEqual({"->", "<-"}, root_zero)
        self.assertEqual(root_zero, root_one)

    def test_token_inventory_superset_includes_directional_tokens_for_explicit_bond_stereo(self) -> None:
        mol = parse_smiles("F/C=C\\Cl")
        exact = set(
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    allBondsExplicit=True,
                ),
            )
        )
        superset = set(
            public_token_inventory_superset(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    allBondsExplicit=True,
                ),
            )
        )

        self.assertLessEqual(exact, superset)
        self.assertLessEqual({"/", "\\"}, superset)

    def test_token_inventory_superset_formats_large_static_ring_labels(self) -> None:
        prepared = _complete_nonstereo_prepared_graph(atom_count=6)
        superset = set(
            public_token_inventory_superset(
                prepared,
                **supported_public_kwargs(
                    rootedAtAtom=-1,
                    isomericSmiles=False,
                ),
            )
        )

        self.assertIn("%10", superset)

    def test_token_inventory_includes_branch_tokens_for_rooted_degree_two_branch_point(self) -> None:
        mol = parse_smiles("C(CO)O")
        expected = exact_token_inventory_via_decoder(
            mol,
            **supported_public_kwargs(
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
        )

        self.assertEqual(
            expected,
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=0,
                    isomericSmiles=False,
                ),
            ),
        )
        self.assertIn("(", expected)
        self.assertIn(")", expected)

    def test_token_inventory_matches_exact_decoder_for_bracketed_stereo_atom_tokens(self) -> None:
        cases = (
            ("[13C@H](F)(Cl)Br", frozenset({"[13C@H]", "[13C@@H]"})),
            ("[Si@H](F)(Cl)Br", frozenset({"[Si@H]", "[Si@@H]"})),
            ("[C@H:7](F)(Cl)Br", frozenset({"[C@H:7]", "[C@@H:7]"})),
        )

        for smiles, required_tokens in cases:
            with self.subTest(smiles=smiles):
                mol = parse_smiles(smiles)
                expected = exact_token_inventory_via_decoder(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=0,
                        isomericSmiles=True,
                    ),
                )
                actual = public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=0,
                        isomericSmiles=True,
                    ),
                )

                self.assertEqual(expected, actual)
                for token in required_tokens:
                    self.assertIn(token, expected)

    def test_token_inventory_treats_omitted_and_minus_one_as_same_all_roots_mode(self) -> None:
        cases = (
            ("CC(=O)Oc1ccccc1C(=O)O", False),
            ("F/C=C\\Cl", True),
            ("[Na+].C#N", False),
        )

        for smiles, isomeric_smiles in cases:
            with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                mol = parse_smiles(smiles)
                omitted = public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        isomericSmiles=isomeric_smiles,
                    ),
                )
                explicit_minus_one = public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=isomeric_smiles,
                    ),
                )

                self.assertEqual(omitted, explicit_minus_one)


if __name__ == "__main__":
    unittest.main()
