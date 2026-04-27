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
        with self.assertRaisesRegex(NotImplementedError, "rootedAtAtom == -1 or rootedAtAtom >= 0"):
            tuple(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=-2,
                    canonical=False,
                    doRandom=True,
                )
            )
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

    def test_public_api_reports_out_of_range_root_consistently_for_disconnected_molecules(self) -> None:
        mol = parse_smiles("C.C")

        entrypoints = (
            (
                "enum",
                lambda: tuple(
                    grimace.MolToSmilesEnum(
                        mol,
                        rootedAtAtom=99,
                        canonical=False,
                        doRandom=True,
                    )
                ),
            ),
            (
                "decoder",
                lambda: grimace.MolToSmilesDecoder(
                    mol,
                    rootedAtAtom=99,
                    canonical=False,
                    doRandom=True,
                ),
            ),
            (
                "inventory",
                lambda: grimace.MolToSmilesTokenInventory(
                    mol,
                    rootedAtAtom=99,
                    canonical=False,
                    doRandom=True,
                ),
            ),
        )

        for name, call in entrypoints:
            with self.subTest(entrypoint=name):
                with self.assertRaisesRegex(IndexError, "root_idx out of range"):
                    call()

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
