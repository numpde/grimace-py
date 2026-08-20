import unittest

from grimace._reference import (
    RootedConnectedNonStereoWalker,
    RootedConnectedNonStereoWalkerState,
    enumerate_rooted_connected_nonstereo_smiles_support,
    enumerate_rooted_connected_stereo_smiles_support,
    enumerate_rooted_nonstereo_smiles_support,
    enumerate_rooted_smiles_support,
    validate_rooted_connected_nonstereo_smiles_support,
    validate_rooted_connected_stereo_smiles_support,
    validate_rooted_nonstereo_smiles_support,
    validate_rooted_smiles_support,
)


class InternalReferenceExportsTests(unittest.TestCase):
    def test_connected_nonstereo_exports_are_available(self) -> None:
        self.assertTrue(callable(enumerate_rooted_connected_nonstereo_smiles_support))
        self.assertTrue(callable(validate_rooted_connected_nonstereo_smiles_support))
        self.assertTrue(callable(RootedConnectedNonStereoWalker))
        self.assertTrue(hasattr(RootedConnectedNonStereoWalkerState, "prefix"))

    def test_connected_stereo_exports_are_available(self) -> None:
        self.assertTrue(callable(enumerate_rooted_connected_stereo_smiles_support))
        self.assertTrue(callable(validate_rooted_connected_stereo_smiles_support))

    def test_alias_exports_point_at_connected_nonstereo_branch(self) -> None:
        self.assertIs(
            enumerate_rooted_smiles_support,
            enumerate_rooted_connected_nonstereo_smiles_support,
        )
        self.assertIs(
            enumerate_rooted_nonstereo_smiles_support,
            enumerate_rooted_connected_nonstereo_smiles_support,
        )
        self.assertIs(
            validate_rooted_smiles_support,
            validate_rooted_connected_nonstereo_smiles_support,
        )
        self.assertIs(
            validate_rooted_nonstereo_smiles_support,
            validate_rooted_connected_nonstereo_smiles_support,
        )


if __name__ == "__main__":
    unittest.main()
