from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import grimace
from grimace._reference import (
    DEFAULT_MOLECULE_SOURCE_PATH,
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
    build_core_exact_sets_artifact,
    write_core_exact_sets_artifact,
)
from grimace._reference.prepared_graph import prepare_smiles_graph_from_mol_to_smiles_kwargs
from tests.helpers.public_runtime import (
    assert_public_entrypoints_equivalent,
    assert_public_entrypoints_raise,
    supported_public_kwargs,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


class PythonApiSmokeTests(unittest.TestCase):
    def test_top_level_api_exposes_only_final_runtime_surface(self) -> None:
        self.assertTrue(callable(grimace.MolToSmilesChoice))
        self.assertTrue(callable(grimace.MolToSmilesDecoder))
        self.assertTrue(callable(grimace.MolToSmilesDeterminizedDecoder))
        self.assertTrue(callable(grimace.MolToSmilesEnum))
        self.assertTrue(callable(grimace.MolToSmilesTokenInventory))
        self.assertFalse(hasattr(grimace, "MolToSmilesSupport"))
        self.assertFalse(hasattr(grimace, "ReferencePolicy"))
        self.assertFalse(hasattr(grimace, "enumerate_rooted_connected_nonstereo_smiles_support"))
        self.assertFalse(hasattr(grimace, "enumerate_rooted_connected_stereo_smiles_support"))
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
        self.assertFalse(hasattr(decoder, "next_tokens"))
        self.assertFalse(hasattr(decoder, "advance"))
        self.assertIsInstance(determinized_decoder.next_choices[0].next_state, grimace.MolToSmilesDeterminizedDecoder)
        if CORE_MODULE is None:
            with self.assertRaises(ImportError):
                tuple(
                    grimace.MolToSmilesEnum(
                        parse_smiles("CCO"),
                        rootedAtAtom=0,
                        isomericSmiles=False,
                        canonical=False,
                        doRandom=True,
                    )
                )
            return

        from grimace import _runtime

        self.assertEqual(
            _runtime.enumerate_rooted_connected_nonstereo_smiles_support(
                parse_smiles("CCO"),
                0,
            ),
            set(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    canonical=False,
                    doRandom=True,
                )
            ),
        )
        self.assertEqual(
            set(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    canonical=False,
                    doRandom=True,
                )
            ),
            _runtime.mol_to_smiles_support(
                parse_smiles("CCO"),
                rooted_at_atom=0,
                isomeric_smiles=False,
                canonical=False,
                do_random=True,
            ),
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
                    expected_exception=NotImplementedError,
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
            expected_exception=NotImplementedError,
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
                    expected_exception=NotImplementedError,
                    expected_regex=(
                        f"{flag_name} to follow RDKit's Python binding and be a bool, int, or None"
                    ),
                    included_entrypoints=("enum",),
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
            included_entrypoints=("enum", "decoder", "inventory"),
        )

    def test_reference_defaults_load_from_installed_package_layout(self) -> None:
        self.assertTrue(DEFAULT_MOLECULE_SOURCE_PATH.is_file())
        self.assertTrue(DEFAULT_RDKIT_RANDOM_POLICY_PATH.is_file())
        self.assertTrue(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH.is_file())

        policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)
        connected_nonstereo_policy = ReferencePolicy.from_path(
            DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH
        )

        self.assertEqual("rdkit_random_v1", policy.policy_name)
        self.assertEqual("rdkit_random_connected_nonstereo_v1", connected_nonstereo_policy.policy_name)
        artifact = build_core_exact_sets_artifact(policy, limit=1)
        connected_nonstereo_artifact = build_core_exact_sets_artifact(
            connected_nonstereo_policy,
            limit=1,
        )
        self.assertEqual(1, artifact["case_count"])
        self.assertEqual(1, connected_nonstereo_artifact["case_count"])
        self.assertEqual(
            "grimace/_reference/_data/reference/rdkit_random/branches/general/policies/rdkit_random_v1.json",
            artifact["policy_path"],
        )
        self.assertEqual(
            "grimace/_reference/_data/top_100000_CIDs.tsv.gz",
            artifact["source_path"],
        )
        self.assertEqual(
            "grimace/_reference/_data/top_100000_CIDs.tsv.gz",
            artifact["input_source"]["path"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                output_path = write_core_exact_sets_artifact(policy, limit=1)
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(
                Path(temp_dir)
                / "grimace_reference_artifacts"
                / "rdkit_random"
                / "branches"
                / "general"
                / "snapshots"
                / policy.policy_name
                / policy.digest()
                / "core_exact_sets.json",
                output_path,
            )
            self.assertTrue(output_path.is_file())

    def test_internal_runtime_bridge_accepts_reference_prepared_graph(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

        reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
            parse_smiles("CCO"),
            surface_kind=_runtime.CONNECTED_NONSTEREO_SURFACE,
            isomeric_smiles=False,
        )
        prepared = _runtime.prepare_smiles_graph(
            reference_prepared,
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )

        self.assertEqual(reference_prepared.to_dict(), prepared.to_dict())

    def test_top_level_runtime_nonstereo_surface_smoke(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

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
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

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
