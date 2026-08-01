"""Vertical product contract for RDKit-ingested specified stereo."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rdkit import rdBase

from grimace import MolToSmilesContinuationDecoder
from grimace import SouthStarError
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_continuation_asset import open_writer_continuation_core
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_consistency
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_for_prepared,
)
from grimace._south_star1.writer_continuation_asset import write_writer_continuation_asset
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support import enumerate_prepared_writer_shaped_support
from tests.helpers.rdkit_south_star_stereo_audit import load_pinned_south_star_stereo_audit_cases
from tests.south_star1.default_writer_capability_ledger import DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.default_writer_qualification_shards import FAST_ACCEPTED_CASES
from tests.south_star1.default_writer_qualification_shards import SLOW_COUPLED_CASES
from tests.south_star1.default_writer_qualification_shards import (
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import require_slow_qualification_asset


class WriterDefaultStereoAuditFixtureTest(unittest.TestCase):
    QUALIFICATION_CASES = FAST_ACCEPTED_CASES
    USE_CACHED_SLOW_ASSETS = False

    def setUp(self) -> None:
        if self.QUALIFICATION_CASES is not FAST_ACCEPTED_CASES and self._testMethodName not in {
            "test_ledger_stereo_pinning_has_fixture_coverage",
            "test_every_local_stereo_case_passes_full_asset_semantic_replay",
        }:
            self.skipTest("case-specific stereo audit is outside the slow shard")

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = None if cls.USE_CACHED_SLOW_ASSETS else TemporaryDirectory()
        all_fixture_cases = load_pinned_south_star_stereo_audit_cases(rdBase.rdkitVersion)
        allowed = {case.name for case in cls.QUALIFICATION_CASES}
        cls.fixture_cases = tuple(item for item in all_fixture_cases if item.name in allowed)
        cls.ledger = {item.name: item for item in DEFAULT_WRITER_CAPABILITY_CASES}
        cls.assets = {}
        cls.facts = {}
        cls.prepared = {}
        cls.snapshots = {}
        cls.options = {}
        with (
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
            for item in cls.fixture_cases:
                options = SouthStarRuntimeOptions(
                    rooted_at_atom=cls.ledger[item.name].rooted_at_atom,
                    serialization_language=SerializationLanguageMode.WRITER_SHAPED,
                )
                facts = ordinary_molecule_facts_from_smiles(
                    item.source_smiles,
                    cls.ledger[item.name].extraction_options,
                )
                prepared = prepare_south_star_mol_from_facts(
                    facts,
                    writer_surface=SouthStarWriterSurface(),
                )
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=initial_writer_frontier_cursor(prepared, options),
                )
                if cls.USE_CACHED_SLOW_ASSETS:
                    cached = require_slow_qualification_asset(cls.ledger[item.name])
                    cls.assets[item.name] = open_writer_continuation_core(
                        cached.asset_path
                    )
                else:
                    path = Path(cls.temporary.name) / item.name
                    write_writer_continuation_asset(
                        path=path,
                        prepared=prepared,
                        snapshot=snapshot,
                    )
                    cls.assets[item.name] = open_writer_continuation_core(path)
                cls.facts[item.name] = facts
                cls.prepared[item.name] = prepared
                cls.snapshots[item.name] = snapshot
                cls.options[item.name] = options

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def test_fixture_ledger_and_full_runtime_agree(self) -> None:
        tetra_comparison = facts_are_isomorphic(
            self.facts["tetra_plus"],
            tetrahedral_facts(),
        )
        self.assertTrue(tetra_comparison.isomorphic, tetra_comparison.reason)
        self.assertFalse(
            facts_are_isomorphic(
                self.facts["directional_opposite"],
                directional_facts(),
                compare_stereo=False,
            ).isomorphic
        )
        for item in self.fixture_cases:
            with self.subTest(case=item.name):
                ledger = self.ledger[item.name]
                asset = self.assets[item.name]
                self.assertEqual(item.source_smiles, ledger.smiles)
                self.assertEqual(item.extraction_profile, ledger.extraction_profile)
                self.assertEqual(item.support_count, ledger.expected_support_count)
                self.assertEqual(item.completion_count, ledger.expected_completion_count)
                self.assertEqual(item.sorted_support_sha256, ledger.expected_support_digest)
                self.assertTrue(
                    verify_writer_continuation_asset_consistency(asset.path).accepted
                )
                live = verify_writer_continuation_asset_for_prepared(
                    prepared=self.prepared[item.name],
                    asset=asset,
                )
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(live.structurally_verified)
                self.assertTrue(live.live_replay_complete)
                self.assertEqual(live.branch_locator_count, live.branch_proof_count)
                self.assertEqual(
                    live.terminal_locator_count, live.terminal_proof_count
                )
                self.assertEqual(live.unchecked_obligation_families, ())

                decoder = MolToSmilesContinuationDecoder.from_asset(asset.path)
                self.assertEqual(decoder.support_count, item.support_count)
                self.assertEqual(decoder.completion_count, item.completion_count)
                self.assertEqual(_decoder_support(decoder), item.expected_support)
                self.assertEqual(
                    sum(value.numerator for value in decoder.exact_probabilities()),
                    decoder.completion_count,
                )
                advanced = decoder.next_choices[0].next_state
                resumed = MolToSmilesContinuationDecoder.from_snapshot(
                    asset.path,
                    advanced.snapshot(),
                )
                self.assertEqual(resumed.cache_key(), advanced.cache_key())
                if item.target_class == "directional":
                    residual = self.snapshots[item.name].cursor.weighted_states[0][0].stereo_state.residual_snapshot
                    variables = tuple(var for var, _domain in residual.domains)
                    self.assertEqual(len(variables), 2)
                    self.assertEqual(
                        {var.kind for var in variables},
                        {"directional_site_carrier"},
                    )
                    self.assertEqual(
                        {factor.key.kind for factor in residual.factors},
                        {"directional_site", "directional_bond_emission"},
                    )

    def test_ledger_stereo_pinning_has_fixture_coverage(self) -> None:
        pinned_ledger = {
            case.name
            for case in self.ledger.values()
            if case.name in {item.name for item in self.fixture_cases}
            if case.extraction_profile == "specified_stereo_closure"
            and case.expected_rdkit_audit_version_pinned
        }
        fixture_names = {item.name for item in self.fixture_cases}

        self.assertEqual(pinned_ledger, fixture_names)

        for item in self.fixture_cases:
            with self.subTest(case=item.name):
                ledger = self.ledger[item.name]
                self.assertTrue(ledger.expected_rdkit_audit_version_pinned)
                self.assertEqual(
                    ledger.extraction_profile,
                    "specified_stereo_closure",
                )

    def test_every_local_stereo_case_passes_full_asset_semantic_replay(self) -> None:
        for item in self.fixture_cases:
            with self.subTest(case=item.name):
                verification = verify_writer_continuation_asset_for_prepared(
                    prepared=self.prepared[item.name],
                    asset=self.assets[item.name],
                )
                self.assertTrue(verification.accepted, verification.unchecked_obligation_families)
                self.assertTrue(verification.live_replay_complete)
                self.assertEqual(
                    verification.branch_locator_count,
                    verification.branch_proof_count,
                )
                self.assertEqual(
                    verification.terminal_locator_count,
                    verification.terminal_proof_count,
                )
                self.assertEqual(verification.unchecked_obligation_families, ())

    def test_polarities_are_disjoint_and_reparse_only_to_their_source(self) -> None:
        pairs = (
            ("tetra_plus", "tetra_minus"),
            ("directional_opposite", "directional_together"),
            (
                "remote_coupled_tetrahedral_a",
                "remote_coupled_tetrahedral_b",
            ),
        )
        remote_pairs = (("remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"),)
        fixture_by_name = {item.name: item for item in self.fixture_cases}
        for left, right in pairs:
            self.assertTrue(
                set(fixture_by_name[left].expected_support).isdisjoint(
                    fixture_by_name[right].expected_support
                )
            )
            self.assertNotEqual(
                fixture_by_name[left].sorted_support_sha256,
                fixture_by_name[right].sorted_support_sha256,
            )
            for name, opposite in ((left, right), (right, left)):
                for text in fixture_by_name[name].expected_support:
                    reparsed = ordinary_molecule_facts_from_smiles(
                        text,
                        self.ledger[name].extraction_options,
                    )
                    self.assertTrue(
                        facts_are_isomorphic(self.facts[name], reparsed).isomorphic,
                        (name, text),
                    )
                    self.assertFalse(
                        facts_are_isomorphic(self.facts[opposite], reparsed).isomorphic,
                        (opposite, text),
                    )
        for left, right in remote_pairs:
            self.assertTrue(
                set(fixture_by_name[left].expected_support).isdisjoint(
                    fixture_by_name[right].expected_support
                )
            )
            self.assertEqual(fixture_by_name[left].support_count, 216)
            self.assertEqual(fixture_by_name[left].completion_count, 216)
            self.assertEqual(fixture_by_name[right].support_count, 216)
            self.assertEqual(fixture_by_name[right].completion_count, 216)

    def test_directional_reference_pair_and_target_transform_together(self) -> None:
        facts = self.facts["directional_opposite"]
        site = facts.stereo.directional[0]
        transformed = replace(
            site,
            target=DirectionalValue.TOGETHER,
            reference_pair=(site.left_ligands[1], site.right_ligands[0]),
        )
        detached = replace(
            transformed,
            target=site.target,
        )
        self.assertEqual(
            _support_for_replaced_directional_site(
                facts,
                transformed,
                self.options["directional_opposite"],
            ),
            _support_for_replaced_directional_site(
                facts,
                site,
                self.options["directional_opposite"],
            ),
        )
        self.assertEqual(
            _support_for_replaced_directional_site(
                facts,
                detached,
                self.options["directional_opposite"],
            ),
            set(
                next(
                    item.expected_support
                    for item in self.fixture_cases
                    if item.name == "directional_together"
                )
            ),
        )

    def test_proof_decoder_binds_prepared_at_construction(self) -> None:
        with self.assertRaisesRegex(SouthStarError, "proof_molecule_required"):
            MolToSmilesContinuationDecoder.from_asset(
                self.assets["tetra_plus"].path,
                proof_capable=True,
            )
        for asset_name, prepared_name in (
            ("tetra_plus", "tetra_minus"),
            ("tetra_minus", "tetra_plus"),
            ("directional_opposite", "directional_together"),
            ("directional_together", "directional_opposite"),
            ("zero_h_tetrahedral", "tetra_plus"),
            ("adjacent_specified_tetrahedral", "tetra_plus"),
            ("remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"),
            ("remote_coupled_tetrahedral_b", "remote_coupled_tetrahedral_a"),
            ("disconnected_tetra_oxygen", "tetra_plus"),
            ("disconnected_directional_oxygen", "directional_opposite"),
        ):
            with self.subTest(asset=asset_name, prepared=prepared_name):
                with self.assertRaisesRegex(Exception, "prepared_identity_mismatch"):
                    MolToSmilesContinuationDecoder.from_asset(
                        self.assets[asset_name].path,
                        proof_capable=True,
                        prepared=self.prepared[prepared_name],
                    )

class WriterDefaultStereoAuditSlowTest(WriterDefaultStereoAuditFixtureTest):
    QUALIFICATION_CASES = None
    USE_CACHED_SLOW_ASSETS = True

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("SOUTH_STAR1_RUN_SLOW") != "1":
            raise unittest.SkipTest("set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases")
        cls.QUALIFICATION_CASES = selected_slow_qualification_cases()
        with (
            patch(
                "grimace.BuildMolToSmilesContinuationAsset",
                side_effect=AssertionError("public asset build invoked"),
            ),
            patch(
                "grimace._south_star1.writer_continuation_asset.write_writer_continuation_asset",
                side_effect=AssertionError("asset writer invoked"),
            ),
        ):
            super().setUpClass()


def _decoder_support(decoder) -> tuple[str, ...]:
    pending = [decoder]
    support = []
    while pending:
        state = pending.pop()
        if state.is_terminal:
            support.append(state.prefix)
        pending.extend(choice.next_state for choice in state.next_choices)
    return tuple(sorted(support))


def _support_for_replaced_directional_site(facts, site, options) -> set[str]:
    changed = replace(facts, stereo=replace(facts.stereo, directional=(site,)))
    prepared = prepare_south_star_mol_from_facts(
        changed,
        writer_surface=SouthStarWriterSurface(),
    )
    return set(
        enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=options,
        ).strings
    )


if __name__ == "__main__":
    unittest.main()
