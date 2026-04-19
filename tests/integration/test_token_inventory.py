from __future__ import annotations

from dataclasses import dataclass
import unittest

import grimace
from grimace import _runtime
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class InventoryCase:
    name: str
    smiles: str
    rooted_at_atom: int | None
    cid: str | None = None
    required_nonstereo: frozenset[str] = frozenset()
    required_stereo: frozenset[str] = frozenset()
    min_nonstereo_token_count: int = 0
    min_stereo_token_count: int = 0


class TokenInventoryTests(unittest.TestCase):
    SHARED_DEMANDING_CASES = (
        InventoryCase(
            name="aspirin_all_roots",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            rooted_at_atom=None,
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
            rooted_at_atom=None,
            required_nonstereo=frozenset({"(", ")", "=", "C", "O"}),
            required_stereo=frozenset({"(", ")", "/", "\\", "=", "C", "O"}),
            min_nonstereo_token_count=5,
            min_stereo_token_count=7,
        ),
        InventoryCase(
            name="disconnected_all_roots",
            smiles="[Na+].C#N",
            rooted_at_atom=None,
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

    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def _exact_inventory_via_decoder(
        self,
        smiles: str,
        *,
        rooted_at_atom: int | None,
        isomeric_smiles: bool,
    ) -> tuple[str, ...]:
        mol = parse_smiles(smiles)
        roots = range(mol.GetNumAtoms()) if rooted_at_atom is None else (rooted_at_atom,)
        inventory: set[str] = set()

        for root_idx in roots:
            decoder = _runtime.MolToSmilesDecoder(
                mol,
                rooted_at_atom=root_idx,
                isomeric_smiles=isomeric_smiles,
                canonical=False,
                do_random=True,
            )
            stack = [decoder._state]
            while stack:
                state = stack.pop()
                grouped_successors = _runtime._grouped_choice_successor_states(state)
                inventory.update(text for text, _ in grouped_successors)
                stack.extend(successor for _, successor in grouped_successors)

        return tuple(sorted(inventory))

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
                    expected = self._exact_inventory_via_decoder(
                        case.smiles,
                        rooted_at_atom=case.rooted_at_atom,
                        isomeric_smiles=isomeric_smiles,
                    )

                    for token in required_tokens:
                        self.assertIn(token, expected)
                    self.assertGreaterEqual(len(expected), min_token_count)

                    self.assertEqual(
                        expected,
                        grimace.MolToSmilesTokenInventory(
                            parse_smiles(case.smiles),
                            rootedAtAtom=case.rooted_at_atom,
                            isomericSmiles=isomeric_smiles,
                            canonical=False,
                            doRandom=True,
                        ),
                    )

    def test_token_inventory_includes_branch_tokens_for_rooted_degree_two_branch_point(self) -> None:
        expected = self._exact_inventory_via_decoder(
            "C(CO)O",
            rooted_at_atom=0,
            isomeric_smiles=False,
        )

        self.assertEqual(
            expected,
            grimace.MolToSmilesTokenInventory(
                parse_smiles("C(CO)O"),
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
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
                expected = self._exact_inventory_via_decoder(
                    smiles,
                    rooted_at_atom=0,
                    isomeric_smiles=True,
                )
                actual = grimace.MolToSmilesTokenInventory(
                    parse_smiles(smiles),
                    rootedAtAtom=0,
                    isomericSmiles=True,
                    canonical=False,
                    doRandom=True,
                )

                self.assertEqual(expected, actual)
                for token in required_tokens:
                    self.assertIn(token, expected)


if __name__ == "__main__":
    unittest.main()
