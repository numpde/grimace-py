from __future__ import annotations

from dataclasses import dataclass
import unittest

import grimace
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class InventoryCase:
    name: str
    smiles: str
    rooted_at_atom: int | None
    cid: str | None = None
    required_tokens: frozenset[str] = frozenset()


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
        # Use one shared molecule set for both branches. The nonstereo surface
        # rejects molecules with atom or bond stereochemistry, so the shared
        # anchors need to stay within the common supported subset.
        cases = (
            InventoryCase(
                name="aspirin_all_roots",
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                rooted_at_atom=None,
                required_tokens=frozenset({"(", ")", "1"}),
            ),
            InventoryCase(
                name="dataset_long",
                cid="3488",
                smiles="COC1=C(C=C(C=C1)Cl)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
                rooted_at_atom=0,
                required_tokens=frozenset({"1", "2", "3", "Cl", "S"}),
            ),
            InventoryCase(
                name="dataset_long_heteroaromatic",
                cid="3440",
                smiles="C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl",
                rooted_at_atom=0,
                required_tokens=frozenset({"o", "Cl", "S", "2"}),
            ),
            InventoryCase(
                name="dataset_long_nitro",
                cid="4485",
                smiles="CC1=C(C(C(=C(N1)C)C(=O)OC)C2=CC=CC=C2[N+](=O)[O-])C(=O)OC",
                rooted_at_atom=0,
                required_tokens=frozenset({"[N+]", "[O-]", "2"}),
            ),
        )

        for case in cases:
            for isomeric_smiles in (False, True):
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

                    for token in case.required_tokens:
                        self.assertIn(token, expected)

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


if __name__ == "__main__":
    unittest.main()
