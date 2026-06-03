from __future__ import annotations

import unittest

import grimace
import grimace._deviation as _deviation
import grimace._runtime as _runtime
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)
from tests.helpers.public_runtime import (
    assert_public_entrypoints_equivalent,
    assert_public_entrypoints_raise,
    supported_public_kwargs,
)
from tests.helpers.mols import parse_smiles


PUBLIC_EXPORT_NAMES = (
    "MolToSmilesChoice",
    "MolToSmilesDecoder",
    "MolToSmilesDeterminizedDecoder",
    "MolToSmilesDeviation",
    "MolToSmilesEnum",
    "MolToSmilesTokenInventory",
    "MolToSmilesTokenInventorySuperset",
)
ABSENT_TOP_LEVEL_NAMES = (
    "MolToSmilesSupport",
    "ReferencePolicy",
    "MOL_TO_SMILES_OPTIONS",
    "coerce_public_options",
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
)


class PythonApiSmokeTests(unittest.TestCase):
    def test_top_level_api_exposes_only_final_runtime_surface(self) -> None:
        for name in PUBLIC_EXPORT_NAMES:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(grimace, name)))

        for name in ABSENT_TOP_LEVEL_NAMES:
            with self.subTest(name=name):
                self.assertFalse(hasattr(grimace, name))

    def test_public_decoder_attributes_are_runtime_backed_at_import(self) -> None:
        self.assertIs(grimace.MolToSmilesChoice, _runtime.MolToSmilesChoice)
        self.assertIs(grimace.SmilesDeviation, _deviation.SmilesDeviation)

        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("CCO"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )
        determinized_decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("CCO"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        for public_decoder in (decoder, determinized_decoder):
            with self.subTest(decoder_type=type(public_decoder).__name__):
                self.assertFalse(hasattr(public_decoder, "next_tokens"))
                self.assertFalse(hasattr(public_decoder, "advance"))

        self.assertEqual("", decoder.prefix)
        self.assertIsInstance(decoder.is_terminal, bool)
        self.assertIsInstance(decoder.next_choices, tuple)
        self.assertTrue(decoder.next_choices)
        self.assertIsInstance(decoder.next_choices[0], grimace.MolToSmilesChoice)
        self.assertIsInstance(
            decoder.next_choices[0].next_state,
            grimace.MolToSmilesDecoder,
        )
        self.assertIsInstance(
            determinized_decoder.next_choices[0].next_state,
            grimace.MolToSmilesDeterminizedDecoder,
        )

    def test_public_api_rejects_unsupported_flag_combination(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "canonical=False"):
            tuple(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                )
            )
        with self.assertRaisesRegex(NotImplementedError, "doRandom=True"):
            tuple(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    canonical=False,
                )
            )

    def test_public_api_rejects_unsupported_input_type(self) -> None:
        kwargs = supported_public_kwargs(rootedAtAtom=0, isomericSmiles=False)
        assert_public_entrypoints_raise(
            self,
            object(),
            kwargs=kwargs,
            expected_exception=TypeError,
            expected_regex="Unsupported molecule/prepared type",
        )
        with self.assertRaisesRegex(TypeError, "Unsupported molecule/prepared type"):
            grimace.MolToSmilesDeviation(object(), "C", **kwargs)

    def test_public_api_treats_any_negative_root_like_rdkit_unrooted_mode(self) -> None:
        mol = parse_smiles("CCO")
        for provided_root in (-1, -2, -3):
            with self.subTest(provided_root=provided_root):
                assert_public_entrypoints_equivalent(
                    self,
                    mol,
                    provided_kwargs={
                        "rootedAtAtom": provided_root,
                        "canonical": False,
                        "doRandom": True,
                    },
                    expected_kwargs={
                        "rootedAtAtom": -1,
                        "canonical": False,
                        "doRandom": True,
                    },
                )

    def test_public_api_coerces_boolean_rooted_at_atom_like_rdkit(self) -> None:
        mol = parse_smiles("CCO")
        for provided_root, expected_root in ((False, 0), (True, 1)):
            with self.subTest(provided_root=provided_root, expected_root=expected_root):
                assert_public_entrypoints_equivalent(
                    self,
                    mol,
                    provided_kwargs={
                        "rootedAtAtom": provided_root,
                        "canonical": False,
                        "doRandom": True,
                    },
                    expected_kwargs={
                        "rootedAtAtom": expected_root,
                        "canonical": False,
                        "doRandom": True,
                    },
                )

    def test_public_api_rejects_non_integral_rooted_at_atom_values(self) -> None:
        mol = parse_smiles("CCO")
        for rooted_at_atom in (-0.2, 1.0, "1"):
            with self.subTest(rooted_at_atom=rooted_at_atom):
                assert_public_entrypoints_raise(
                    self,
                    mol,
                    kwargs=supported_public_kwargs(rootedAtAtom=rooted_at_atom),
                    expected_exception=TypeError,
                    expected_regex=(
                        "rootedAtAtom to follow RDKit's Python binding and be an integer"
                    ),
                )

    def test_public_api_rejects_none_rooted_at_atom(self) -> None:
        mol = parse_smiles("CCO")
        assert_public_entrypoints_raise(
            self,
            mol,
            kwargs=supported_public_kwargs(rootedAtAtom=None),
            expected_exception=TypeError,
            expected_regex="rootedAtAtom to follow RDKit's Python binding and be an integer",
        )

    def test_public_api_coerces_none_boolean_flags_like_rdkit(self) -> None:
        mol = parse_smiles("CCO")
        coercion_cases = (
            ("canonical", None, False),
            ("isomericSmiles", None, False),
            ("kekuleSmiles", None, False),
            ("allBondsExplicit", None, False),
            ("allHsExplicit", None, False),
            ("ignoreAtomMapNumbers", None, False),
        )

        for flag_name, provided_value, coerced_value in coercion_cases:
            kwargs = {
                "rootedAtAtom": 0,
                "canonical": False,
                "doRandom": True,
                flag_name: provided_value,
            }
            expected_kwargs = {
                "rootedAtAtom": 0,
                "canonical": False,
                "doRandom": True,
                flag_name: coerced_value,
            }
            with self.subTest(flag_name=flag_name, provided_value=provided_value):
                assert_public_entrypoints_equivalent(
                    self,
                    mol,
                    provided_kwargs=kwargs,
                    expected_kwargs=expected_kwargs,
                )

        with self.assertRaisesRegex(NotImplementedError, "doRandom=True"):
            tuple(
                grimace.MolToSmilesEnum(
                    mol,
                    rootedAtAtom=0,
                    canonical=False,
                    doRandom=None,
                )
            )

    def test_public_api_coerces_integral_boolean_flags_like_rdkit(self) -> None:
        mol = parse_smiles("CCO")
        coercion_cases = (
            ("canonical", 0, False),
            ("doRandom", 1, True),
            ("isomericSmiles", 1),
            ("kekuleSmiles", 1),
            ("allBondsExplicit", 1),
            ("allHsExplicit", 1),
            ("ignoreAtomMapNumbers", 1),
        )

        for raw_case in coercion_cases:
            if len(raw_case) == 3:
                flag_name, provided_value, coerced_value = raw_case
            else:
                flag_name, provided_value = raw_case
                coerced_value = bool(provided_value)
            kwargs = {
                "rootedAtAtom": 0,
                "canonical": False,
                "doRandom": True,
                flag_name: provided_value,
            }
            expected_kwargs = {
                "rootedAtAtom": 0,
                "canonical": False,
                "doRandom": True,
                flag_name: coerced_value,
            }
            with self.subTest(flag_name=flag_name, provided_value=provided_value):
                assert_public_entrypoints_equivalent(
                    self,
                    mol,
                    provided_kwargs=kwargs,
                    expected_kwargs=expected_kwargs,
                )

    def test_public_api_rejects_non_integral_boolean_flag_values(self) -> None:
        mol = parse_smiles("CCO")
        invalid_cases = (
            ("canonical", 0.0),
            ("doRandom", 1.0),
            ("isomericSmiles", 1.0),
            ("kekuleSmiles", "no"),
            ("allBondsExplicit", 1.0),
            ("allHsExplicit", "no"),
            ("ignoreAtomMapNumbers", 1.0),
        )

        for flag_name, invalid_value in invalid_cases:
            with self.subTest(flag_name=flag_name, invalid_value=invalid_value):
                assert_public_entrypoints_raise(
                    self,
                    mol,
                    kwargs=supported_public_kwargs(rootedAtAtom=0, **{flag_name: invalid_value}),
                    expected_exception=TypeError,
                    expected_regex=(
                        f"{flag_name} to follow RDKit's Python binding and be a bool, int, or None"
                    ),
                )

    def test_public_api_reports_out_of_range_root_consistently_for_connected_molecules(self) -> None:
        mol = parse_smiles("CCO")
        assert_public_entrypoints_raise(
            self,
            mol,
            kwargs=supported_public_kwargs(rootedAtAtom=99),
            expected_exception=IndexError,
            expected_regex="root_idx out of range",
        )

    def test_public_api_reports_out_of_range_root_consistently_for_disconnected_molecules(self) -> None:
        mol = parse_smiles("C.C")
        assert_public_entrypoints_raise(
            self,
            mol,
            kwargs=supported_public_kwargs(rootedAtAtom=99),
            expected_exception=IndexError,
            expected_regex="root_idx out of range",
            included_entrypoints=("enum", "decoder", "inventory", "inventory_superset"),
        )

    def test_internal_runtime_bridge_accepts_reference_prepared_graph(self) -> None:
        import grimace._runtime_graphs as _runtime_graphs
        from grimace._runtime_inputs import MolToSmilesFlags

        reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
            parse_smiles("CCO"),
            surface_kind=CONNECTED_NONSTEREO_SURFACE,
            isomeric_smiles=False,
        )
        prepared = _runtime_graphs.prepare_smiles_graph(
            reference_prepared,
            flags=MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )

        self.assertEqual(reference_prepared.to_dict(), prepared.to_dict())

    def test_top_level_runtime_nonstereo_surface_smoke(self) -> None:
        import grimace._runtime as _runtime

        mol = parse_smiles("CCO")
        expected = _runtime.enumerate_rooted_connected_nonstereo_smiles_support(
            mol,
            0,
        )

        support_from_enum = set(
            grimace.MolToSmilesEnum(
                mol,
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
            )
        )
        support = _runtime.mol_to_smiles_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=False,
            canonical=False,
            do_random=True,
        )

        self.assertEqual(support_from_enum, support)
        self.assertEqual(expected, support)

    def test_top_level_runtime_stereo_surface_smoke(self) -> None:
        import grimace._runtime as _runtime

        mol = parse_smiles("F/C=C\\Cl")
        expected = _runtime.enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
        )
        support_from_enum = set(
            grimace.MolToSmilesEnum(
                mol,
                rootedAtAtom=0,
                isomericSmiles=True,
                canonical=False,
                doRandom=True,
            )
        )
        support = _runtime.mol_to_smiles_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=True,
            canonical=False,
            do_random=True,
        )

        self.assertEqual(support_from_enum, support)
        self.assertEqual(expected, support)


if __name__ == "__main__":
    unittest.main()
