from __future__ import annotations

import unittest

import grimace
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


class TokenInventoryTests(unittest.TestCase):
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
            stack = [
                grimace.MolToSmilesDecoder(
                    mol,
                    rootedAtAtom=root_idx,
                    isomericSmiles=isomeric_smiles,
                    canonical=False,
                    doRandom=True,
                )
            ]
            while stack:
                decoder = stack.pop()
                inventory.update(decoder.next_tokens)
                for token in decoder.next_tokens:
                    next_state = decoder.copy()
                    next_state.advance(token)
                    stack.append(next_state)

        return tuple(sorted(inventory))

    def test_token_inventory_matches_exact_decoder_inventory(self) -> None:
        cases = (
            {
                "name": "aspirin_all_roots",
                "cid": None,
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "rooted_at_atom": None,
                "isomeric_smiles": False,
                "required_tokens": {"(", ")", "1"},
            },
            {
                "name": "stereo_atom_tokens",
                "cid": None,
                "smiles": "F[C@H](Cl)Br",
                "rooted_at_atom": 0,
                "isomeric_smiles": True,
                "required_tokens": {"[C@H]"},
            },
            {
                "name": "dataset_nonstereo_long",
                "cid": "3488",
                "smiles": "COC1=C(C=C(C=C1)Cl)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
                "rooted_at_atom": 0,
                "isomeric_smiles": False,
                "required_tokens": {"1", "2", "3", "Cl", "S"},
            },
            {
                "name": "dataset_stereo_long",
                "cid": "5743",
                "smiles": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4([C@]3([C@H](C[C@@]2([C@]1(C(=O)CO)O)C)O)F)C",
                "rooted_at_atom": 0,
                "isomeric_smiles": True,
                "required_tokens": {"4", "F", "[C@H]", "[C@@H]", "[C@]", "[C@@]"},
            },
        )

        for case in cases:
            with self.subTest(case=case["name"], cid=case["cid"]):
                smiles = case["smiles"]
                rooted_at_atom = case["rooted_at_atom"]
                isomeric_smiles = case["isomeric_smiles"]
                expected = self._exact_inventory_via_decoder(
                    smiles,
                    rooted_at_atom=rooted_at_atom,
                    isomeric_smiles=isomeric_smiles,
                )

                for token in case["required_tokens"]:
                    self.assertIn(token, expected)

                self.assertEqual(
                    expected,
                    grimace.MolToSmilesTokenInventory(
                        parse_smiles(smiles),
                        rootedAtAtom=rooted_at_atom,
                        isomericSmiles=isomeric_smiles,
                        canonical=False,
                        doRandom=True,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
