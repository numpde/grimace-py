from __future__ import annotations

import unittest

from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


class CoreExtensionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_core_objects_construct_and_advance(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("CCO")
        prepared = _runtime.prepare_smiles_graph(
            mol,
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
        walker = _runtime.make_nonstereo_walker(prepared, 0)
        state = walker.initial_state()

        self.assertEqual(prepared.atom_count, kernel_prepared.atom_count)
        self.assertTrue(walker.next_token_support(state))
        self.assertEqual(["C"], walker.next_choice_texts(state))
        self.assertEqual("C", walker.advance_choice(state, 0).prefix)

    def test_core_decoder_reports_branching_prefix(self) -> None:
        from grimace import _runtime

        prepared = _runtime.prepare_smiles_graph(
            parse_smiles("CC(=O)Oc1ccccc1C(=O)O"),
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )
        decoder = CORE_MODULE.RootedConnectedNonStereoDecoder(
            prepared,
            0,
        )

        while decoder.prefix() != "CC(=O)Oc1c" and not decoder.is_terminal():
            decoder.advance_token(decoder.next_token_support()[0])

        self.assertEqual("CC(=O)Oc1c", decoder.prefix())
        self.assertEqual(["(", "c"], decoder.next_token_support())

    def test_runtime_factories_select_correct_core_types(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("F[C@H](Cl)Br")

        nonstereo_walker = _runtime.make_nonstereo_walker(mol, 0)
        stereo_walker = _runtime.make_stereo_walker(mol, 0)

        self.assertIsInstance(
            nonstereo_walker,
            CORE_MODULE.RootedConnectedNonStereoWalker,
        )
        self.assertIsInstance(
            stereo_walker,
            CORE_MODULE.RootedConnectedStereoWalker,
        )

        self.assertEqual(
            ["F"],
            nonstereo_walker.next_token_support(nonstereo_walker.initial_state()),
        )
        self.assertEqual(
            ["F"],
            stereo_walker.next_token_support(stereo_walker.initial_state()),
        )

    def test_runtime_decoder_factory_selects_correct_core_type(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("F/C=C\\Cl")
        nonstereo_flags = _runtime.MolToSmilesFlags(
            isomeric_smiles=False,
            rooted_at_atom=0,
            canonical=False,
            do_random=True,
        )
        stereo_flags = _runtime.MolToSmilesFlags(
            isomeric_smiles=True,
            rooted_at_atom=0,
            canonical=False,
            do_random=True,
        )

        nonstereo_decoder = _runtime._make_decoder(mol, nonstereo_flags)
        stereo_decoder = _runtime._make_decoder(mol, stereo_flags)

        self.assertIsInstance(
            nonstereo_decoder,
            CORE_MODULE.RootedConnectedNonStereoDecoder,
        )
        self.assertIsInstance(
            stereo_decoder,
            CORE_MODULE.RootedConnectedStereoDecoder,
        )

        self.assertEqual(["F"], nonstereo_decoder.next_token_support())
        self.assertEqual(["F"], stereo_decoder.next_token_support())

    def test_nonstereo_core_decoder_supports_all_roots_frontier(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("CCO")
        prepared = _runtime.prepare_smiles_graph(
            mol,
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )

        decoder = CORE_MODULE.RootedConnectedNonStereoDecoder(prepared, -1)

        self.assertEqual("", decoder.prefix())
        self.assertEqual(["C", "O"], decoder.next_token_support())
        self.assertEqual(["C", "C", "O"], decoder.next_choice_texts())

    def test_runtime_factories_reject_prepared_surface_mismatch(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("F[C@H](Cl)Br")
        nonstereo_flags = _runtime.MolToSmilesFlags(
            isomeric_smiles=False,
            rooted_at_atom=0,
            canonical=False,
            do_random=True,
        )
        stereo_flags = _runtime.MolToSmilesFlags(
            isomeric_smiles=True,
            rooted_at_atom=0,
            canonical=False,
            do_random=True,
        )
        nonstereo_prepared = _runtime.prepare_smiles_graph(mol, flags=nonstereo_flags)
        stereo_prepared = _runtime.prepare_smiles_graph(mol, flags=stereo_flags)

        with self.assertRaisesRegex(ValueError, "surface_kind"):
            _runtime.make_stereo_walker(nonstereo_prepared, 0)
        with self.assertRaisesRegex(ValueError, "surface_kind"):
            _runtime.make_nonstereo_walker(stereo_prepared, 0)


if __name__ == "__main__":
    unittest.main()
