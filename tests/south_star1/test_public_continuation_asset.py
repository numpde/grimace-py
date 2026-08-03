"""Public RDKit-to-certified-continuation-asset product contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import grimace
from rdkit import Chem

from grimace._south_star1 import writer_continuation_asset
from grimace._south_star1 import writer_count_dag_envelope
from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_snapshot
from grimace._south_star1 import writer_support
from grimace._south_star1 import writer_support_artifact_envelope

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.public_continuation_asset import (
    prepare_public_continuation_molecule,
)

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.qualification_plan import FAST_ACCEPTED_CASES
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASES
from tests.south_star1.qualification_plan import (
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import (
    build_slow_qualification_candidate,
    certify_slow_qualification_candidate,
    require_slow_qualification_candidate,
    require_slow_qualification_asset,
)


class PublicContinuationAssetTest(unittest.TestCase):
    def _assert_accepted_cases_build_through_public_api(self, cases) -> None:
        for case in cases:
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                path = Path(directory) / "asset"
                digest = grimace.BuildMolToSmilesContinuationAsset(
                    Chem.MolFromSmiles(case.smiles),
                    path,
                    rootedAtAtom=case.rooted_at_atom,
                )
                decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    expected_manifest_digest=digest,
                )
                support = _decoder_support(decoder)
                self.assertEqual(decoder.support_count, case.expected_support_count)
                self.assertEqual(
                    decoder.completion_count,
                    case.expected_completion_count,
                )
                self.assertEqual(_support_digest(support), case.expected_support_digest)
                successor = decoder.next_choices[0].next_state
                resumed = grimace.MolToSmilesContinuationDecoder.from_snapshot(
                    path,
                    successor.snapshot(),
                )
                self.assertEqual(resumed.cache_key(), successor.cache_key())

    def test_fast_accepted_cases_build_through_public_api(self) -> None:
        self._assert_accepted_cases_build_through_public_api(FAST_ACCEPTED_CASES)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_build_through_public_api(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = build_slow_qualification_candidate(case)
                required = require_slow_qualification_candidate(case)
                self.assertEqual(required.manifest_digest, cached.manifest_digest)
                self.assertTrue(cached.candidate_path.is_dir())
                self.assertTrue(cached.metadata_path.is_file())

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_certify_public_candidates(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                certify_slow_qualification_candidate(case)
                require_slow_qualification_asset(case)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_run_public_runtime(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = require_slow_qualification_asset(case)
                mol = Chem.MolFromSmiles(case.smiles)
                with (
                    patch.object(
                        grimace,
                        "BuildMolToSmilesContinuationAsset",
                        side_effect=AssertionError("public asset build invoked"),
                    ),
                    patch.object(
                        writer_continuation_asset,
                        "write_writer_continuation_asset",
                        side_effect=AssertionError("asset writer invoked"),
                    ),
                    patch.object(
                        grimace,
                        "VerifyMolToSmilesContinuationAsset",
                        side_effect=AssertionError("whole-asset recertification invoked"),
                    ),
                    patch.object(
                        writer_support_artifact_envelope,
                        "writer_support_artifact_envelope_for_snapshot",
                        side_effect=AssertionError("rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_support_artifact_envelope,
                        "_writer_support_artifact_envelope_for_snapshot_with_count_envelope",
                        side_effect=AssertionError("cached rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_frontier_count_envelope,
                        "writer_frontier_count_envelope_for_snapshot",
                        side_effect=AssertionError("count envelope invoked"),
                    ),
                    patch.object(
                        writer_count_dag_envelope,
                        "writer_count_certificate_dag_envelope_for_product",
                        side_effect=AssertionError("count DAG invoked"),
                    ),
                    patch.object(
                        writer_snapshot,
                        "_iter_writer_snapshot_certified_support_strings",
                        side_effect=AssertionError("support strings materialized"),
                    ),
                    patch.object(
                        writer_support,
                        "enumerate_prepared_writer_shaped_support",
                        side_effect=AssertionError("legacy support enumeration invoked"),
                    ),
                ):
                    decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                        cached.asset_path,
                        expected_manifest_digest=cached.manifest_digest,
                    )
                    support = _decoder_support(decoder)
                    self.assertEqual(decoder.support_count, case.expected_support_count)
                    self.assertEqual(decoder.completion_count, case.expected_completion_count)
                    self.assertEqual(_support_digest(support), case.expected_support_digest)
                    self.assertEqual(
                        sum(item.numerator for item in decoder.exact_probabilities()),
                        decoder.completion_count,
                    )
                    successor = decoder.next_choices[0].next_state
                    resumed = grimace.MolToSmilesContinuationDecoder.from_snapshot(
                        cached.asset_path,
                        successor.snapshot(),
                    )
                    self.assertEqual(resumed.cache_key(), successor.cache_key())

    def test_renumbered_stereo_builds_the_same_mapped_root_language(self) -> None:
        cases = (
            ("[C@H](F)(Cl)Br", 0, None),
            ("C[C@H](F)Cl", 1, None),
            ("[C@H](F)(Cl)Br.O", 0, (3, 2, 1, 0, 4)),
            ("F/C=C/Cl", 0, None),
            ("F/C=C\\Cl", 0, None),
        )
        for source, root, requested_order in cases:
            with self.subTest(source=source), TemporaryDirectory() as directory:
                mol = Chem.MolFromSmiles(source)
                order = requested_order or tuple(reversed(range(mol.GetNumAtoms())))
                renumbered = Chem.RenumberAtoms(mol, list(order))
                original = _build_support(
                    mol,
                    root=root,
                    path=Path(directory) / "original",
                )
                changed = _build_support(
                    renumbered,
                    root=order.index(root),
                    path=Path(directory) / "renumbered",
                )
                self.assertEqual(changed, original)

    def test_repeated_builds_are_byte_identical(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("c1ccccc1.O")
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_digest = grimace.BuildMolToSmilesContinuationAsset(mol, first)
            second_digest = grimace.BuildMolToSmilesContinuationAsset(mol, second)
            self.assertEqual(second_digest, first_digest)
            self.assertEqual(_bundle_bytes(second), _bundle_bytes(first))

    def test_blocked_default_molecules_keep_typed_dispositions(self) -> None:
        cases = tuple(
            case
            for case in BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES
            if case.extraction_profile == "graph_no_potential_sites"
        )
        for case in cases:
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                mol = Chem.MolFromSmiles(case.smiles)
                before = mol.ToBinary()
                path = Path(directory) / "asset"
                with self.assertRaises(grimace.SouthStarError) as raised:
                    grimace.BuildMolToSmilesContinuationAsset(
                        mol,
                        path,
                        rootedAtAtom=case.rooted_at_atom,
                    )
                self.assertIs(raised.exception.kind, case.blocker_error_kind)
                self.assertFalse(path.exists())
                self.assertEqual(tuple(Path(directory).glob(".asset.*")), ())
                self.assertEqual(mol.ToBinary(), before)

    def test_unsupported_public_flags_and_roots_are_typed_and_transactional(self) -> None:
        cases = (
            ({"canonical": True}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"doRandom": False}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"kekuleSmiles": True}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"allBondsExplicit": True}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"allHsExplicit": True}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"ignoreAtomMapNumbers": True}, grimace.SouthStarErrorKind.UNSUPPORTED_POLICY),
            ({"rootedAtAtom": -2}, grimace.SouthStarErrorKind.INVALID_FACTS),
            ({"rootedAtAtom": 99}, grimace.SouthStarErrorKind.INVALID_FACTS),
        )
        for kwargs, kind in cases:
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as directory:
                mol = Chem.MolFromSmiles("CCO")
                before = mol.ToBinary()
                path = Path(directory) / "asset"
                with self.assertRaises(grimace.SouthStarError) as raised:
                    grimace.BuildMolToSmilesContinuationAsset(mol, path, **kwargs)
                self.assertIs(raised.exception.kind, kind)
                self.assertFalse(path.exists())
                self.assertEqual(tuple(Path(directory).glob(".asset.*")), ())
                self.assertEqual(mol.ToBinary(), before)

    def test_stereo_suppression_is_typed_and_transactional(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
            before = mol.ToBinary()
            path = Path(directory) / "asset"
            with self.assertRaises(grimace.SouthStarError) as raised:
                grimace.BuildMolToSmilesContinuationAsset(
                    mol,
                    path,
                    isomericSmiles=False,
                    rootedAtAtom=0,
                )
            self.assertIs(
                raised.exception.kind,
                grimace.SouthStarErrorKind.UNSUPPORTED_POLICY,
            )
            self.assertFalse(path.exists())
            self.assertEqual(tuple(Path(directory).glob(".asset.*")), ())
            self.assertEqual(mol.ToBinary(), before)

    def test_unrequested_potential_stereo_uses_graph_only_extraction(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            digest = grimace.BuildMolToSmilesContinuationAsset(
                Chem.MolFromSmiles("C1=CC1"),
                path,
                rootedAtAtom=0,
            )
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                expected_manifest_digest=digest,
            )
            self.assertEqual(decoder.support_count, 3)
            self.assertEqual(decoder.completion_count, 3)

    def test_ambiguous_specified_stereo_rejects_transactionally(self) -> None:
        source = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        editable = Chem.RWMol()
        for atom in source.GetAtoms():
            editable.AddAtom(Chem.Atom(atom))
        for bond in source.GetBonds():
            editable.AddBond(
                bond.GetEndAtomIdx(),
                bond.GetBeginAtomIdx(),
                bond.GetBondType(),
            )
        ambiguous = editable.GetMol()
        Chem.SanitizeMol(ambiguous)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            with self.assertRaises(grimace.SouthStarError) as raised:
                grimace.BuildMolToSmilesContinuationAsset(ambiguous, path)
            self.assertIs(
                raised.exception.kind,
                grimace.SouthStarErrorKind.UNSUPPORTED_STEREO,
            )
            self.assertFalse(path.exists())
            self.assertEqual(tuple(Path(directory).glob(".asset.*")), ())

    def test_remote_coupled_tetra_surface_enters_public_preparation(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)([C@](F)(Cl)Br)[C@@](F)(Cl)Br")
        prepared = prepare_public_continuation_molecule(
            mol,
            writer_surface=SouthStarWriterSurface(),
            runtime_options=SouthStarRuntimeOptions(
                rooted_at_atom=0,
                serialization_language=SerializationLanguageMode.WRITER_SHAPED,
            ),
        )
        self.assertEqual(
            tuple(int(site.center) for site in prepared.facts.stereo.tetrahedral),
            (0, 2, 6),
        )

    def test_public_builder_does_not_invoke_legacy_materialization(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch(
                    "grimace._runtime.mol_to_smiles_enum",
                    side_effect=AssertionError("legacy decoder invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_support_artifact_envelope.writer_support_artifact_envelope_for_snapshot",
                    side_effect=AssertionError("rich support invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_frontier_count_envelope.writer_frontier_count_envelope_for_snapshot",
                    side_effect=AssertionError("count envelope invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product",
                    side_effect=AssertionError("count DAG invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_snapshot._iter_writer_snapshot_certified_support_strings",
                    side_effect=AssertionError("support materialization invoked"),
                ),
            ):
                digest = grimace.BuildMolToSmilesContinuationAsset(
                    Chem.MolFromSmiles("F/C=C/Cl"),
                    Path(directory) / "asset",
                    rootedAtAtom=0,
                )
            self.assertEqual(len(digest), 64)

    def test_returned_digest_binds_decoder(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            digest = grimace.BuildMolToSmilesContinuationAsset(
                Chem.MolFromSmiles("CCO"),
                path,
            )
            grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                expected_manifest_digest=digest,
            )
            with self.assertRaisesRegex(
                grimace.SouthStarError,
                "continuation_asset_manifest_digest_mismatch",
            ):
                grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    expected_manifest_digest="0" * 64,
                )


def _build_support(mol, *, root: int, path: Path):
    digest = grimace.BuildMolToSmilesContinuationAsset(
        mol,
        path,
        rootedAtAtom=root,
    )
    decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
        path,
        expected_manifest_digest=digest,
    )
    return _decoder_support(decoder), decoder.support_count, decoder.completion_count


def _decoder_support(decoder) -> tuple[str, ...]:
    pending = [decoder]
    support: list[str] = []
    while pending:
        state = pending.pop()
        if state.is_terminal:
            support.append(state.prefix)
        pending.extend(choice.next_state for choice in state.next_choices)
    return tuple(sorted(support))


def _support_digest(support: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            support,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _bundle_bytes(path: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (str(item.relative_to(path)), item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


if __name__ == "__main__":
    unittest.main()
