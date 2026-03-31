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

    def test_token_inventory_matches_exact_decoder_inventory_for_aspirin_all_roots(self) -> None:
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        expected = self._exact_inventory_via_decoder(
            smiles,
            rooted_at_atom=None,
            isomeric_smiles=False,
        )

        self.assertIn("(", expected)
        self.assertIn(")", expected)
        self.assertIn("1", expected)

        self.assertEqual(
            expected,
            grimace.MolToSmilesTokenInventory(
                parse_smiles(smiles),
                rootedAtAtom=None,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
            ),
        )

    def test_token_inventory_matches_exact_decoder_inventory_for_stereo_atom_tokens(self) -> None:
        smiles = "F[C@H](Cl)Br"
        expected = self._exact_inventory_via_decoder(
            smiles,
            rooted_at_atom=0,
            isomeric_smiles=True,
        )

        self.assertIn("[C@H]", expected)

        self.assertEqual(
            expected,
            grimace.MolToSmilesTokenInventory(
                parse_smiles(smiles),
                rootedAtAtom=0,
                isomericSmiles=True,
                canonical=False,
                doRandom=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
