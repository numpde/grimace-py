from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    public_enum_support,
    public_token_inventory,
    supported_public_kwargs,
)


PINNED_RDKIT_VERSION = "2026.03.1"


class SerializerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if rdBase.rdkitVersion != PINNED_RDKIT_VERSION:
            raise unittest.SkipTest(
                f"serializer regressions are pinned to RDKit {PINNED_RDKIT_VERSION}, "
                f"got {rdBase.rdkitVersion}"
            )

    def test_ignore_atom_map_numbers_rooted_surface(self) -> None:
        # Regression for the bug where ignoreAtomMapNumbers=True still leaked
        # atom-map text through the prepared-graph/reference writer surface.
        mol = parse_smiles("[CH3:7]C")

        rooted_expectations = (
            (0, {"[CH3:7]C"}, ("C", "[CH3:7]"), {"CC"}, ("C",)),
            (1, {"C[CH3:7]"}, ("C", "[CH3:7]"), {"CC"}, ("C",)),
        )
        for (
            rooted_at_atom,
            expected_kept_support,
            expected_kept_inventory,
            expected_ignored_support,
            expected_ignored_inventory,
        ) in rooted_expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_kept_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_kept_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )

    def test_rooted_atom_on_disconnected_fragment(self) -> None:
        # Regression for RDKit Github 8328: rootedAtAtom on multi-fragment
        # molecules must only root the fragment that contains the atom while
        # leaving other fragments in their own rooted-random support space.
        mol = parse_smiles("[C:1][C:2].[N:3]([C:4])=[O:5]")

        expectations = (
            (
                3,
                {
                    "[C:1][C:2].[C:4][N:3]=[O:5]",
                    "[C:2][C:1].[C:4][N:3]=[O:5]",
                },
                (".", "=", "[C:1]", "[C:2]", "[C:4]", "[N:3]", "[O:5]"),
            ),
            (
                4,
                {
                    "[C:1][C:2].[O:5]=[N:3][C:4]",
                    "[C:2][C:1].[O:5]=[N:3][C:4]",
                },
                (".", "=", "[C:1]", "[C:2]", "[C:4]", "[N:3]", "[O:5]"),
            ),
        )
        for rooted_at_atom, expected_support, expected_inventory in expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
