"""Vertical product contract for RDKit-ingested specified stereo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rdkit import rdBase

from grimace import MolToSmilesContinuationDecoder
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_branch_transition_artifact import verify_writer_branch_transition_artifact_envelope
from grimace._south_star1.writer_branch_transition_artifact_checker import verify_writer_branch_transition_artifact_consistency
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import verify_writer_branch_transition_artifact_for_facts
from grimace._south_star1.writer_continuation_asset import branch_transition_artifact_from_continuation_asset
from grimace._south_star1.writer_continuation_asset import open_writer_continuation_core
from grimace._south_star1.writer_continuation_asset import terminalization_artifact_from_continuation_asset
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_consistency
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_live
from grimace._south_star1.writer_continuation_asset import write_writer_continuation_asset
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support import enumerate_prepared_writer_shaped_support
from grimace._south_star1.writer_terminalization_artifact import verify_writer_terminalization_artifact_envelope
from grimace._south_star1.writer_terminalization_artifact_checker import verify_writer_terminalization_artifact_consistency
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import verify_writer_terminalization_artifact_for_facts
from tests.helpers.rdkit_south_star_stereo_audit import load_pinned_south_star_stereo_audit_cases
from tests.south_star1.default_writer_capability_ledger import DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class WriterDefaultStereoAuditFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        cls.fixture_cases = load_pinned_south_star_stereo_audit_cases(
            rdBase.rdkitVersion
        )
        cls.ledger = {item.name: item for item in DEFAULT_WRITER_CAPABILITY_CASES}
        cls.options = SouthStarRuntimeOptions(
            rooted_at_atom=0,
            serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        )
        cls.assets = {}
        cls.facts = {}
        cls.prepared = {}
        cls.snapshots = {}
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
                    runtime_options=cls.options,
                    cursor=initial_writer_frontier_cursor(prepared, cls.options),
                )
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

    @classmethod
    def tearDownClass(cls) -> None:
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
                self.assertEqual(item.sorted_support_sha256, ledger.expected_support_digest)
                self.assertTrue(
                    verify_writer_continuation_asset_consistency(asset.path).accepted
                )
                live = verify_writer_continuation_asset_live(
                    prepared=self.prepared[item.name],
                    asset=asset,
                    full=True,
                )
                self.assertTrue(live.accepted, live.reason)

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

    def test_every_local_stereo_proof_replays_facts_bound(self) -> None:
        expected_operations = {
            "tetra_plus": {
                "tetrahedral atom-token restriction",
                "tetrahedral local-order factor closure",
            },
            "tetra_minus": {
                "tetrahedral atom-token restriction",
                "tetrahedral local-order factor closure",
            },
            "directional_opposite": {"directional carrier-mark restriction"},
            "directional_together": {"directional carrier-mark restriction"},
        }
        for item in self.fixture_cases:
            asset = self.assets[item.name]
            prepared = self.prepared[item.name]
            facts = self.facts[item.name]
            operations = set()
            for edge in asset.records("edge_records"):
                for certificate_digest in edge.branch_certificate_digests:
                    branch = branch_transition_artifact_from_continuation_asset(
                        prepared=prepared,
                        asset=asset,
                        source_raw_cursor_digest=edge.source_raw_cursor_digest,
                        emitted_text=edge.emitted_text,
                        branch_certificate_digest=certificate_digest,
                    )
                    structural = verify_writer_branch_transition_artifact_consistency(branch)
                    live = verify_writer_branch_transition_artifact_envelope(
                        prepared=prepared,
                        artifact=branch,
                    )
                    offline = verify_writer_branch_transition_artifact_for_facts(
                        facts=facts,
                        runtime_options=self.options,
                        artifact=branch,
                    )
                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(offline.accepted, offline.reason)
                    self.assertEqual(offline.unchecked_obligation_families, ())
                    operations.update(offline.semantically_replayed_operations)
            for terminal in asset.records("terminal_records"):
                for support_digest in terminal.terminal_support_identity_digests:
                    proof = terminalization_artifact_from_continuation_asset(
                        prepared=prepared,
                        asset=asset,
                        source_raw_cursor_digest=terminal.source_raw_cursor_digest,
                        terminal_support_identity_digest=support_digest,
                    )
                    structural = verify_writer_terminalization_artifact_consistency(proof)
                    live = verify_writer_terminalization_artifact_envelope(
                        prepared=prepared,
                        artifact=proof,
                    )
                    offline = verify_writer_terminalization_artifact_for_facts(
                        facts=facts,
                        runtime_options=self.options,
                        artifact=proof,
                    )
                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(offline.accepted, offline.reason)
                    self.assertEqual(offline.unchecked_obligation_families, ())
                    operations.update(offline.semantically_replayed_operations)
            self.assertLessEqual(expected_operations[item.name], operations)

    def test_polarities_are_disjoint_and_reparse_only_to_their_source(self) -> None:
        pairs = (("tetra_plus", "tetra_minus"), ("directional_opposite", "directional_together"))
        fixture_by_name = {item.name: item for item in self.fixture_cases}
        for left, right in pairs:
            self.assertTrue(
                set(fixture_by_name[left].expected_support).isdisjoint(
                    fixture_by_name[right].expected_support
                )
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
            _support_for_replaced_directional_site(facts, transformed, self.options),
            _support_for_replaced_directional_site(facts, site, self.options),
        )
        self.assertEqual(
            _support_for_replaced_directional_site(facts, detached, self.options),
            set(
                next(
                    item.expected_support
                    for item in self.fixture_cases
                    if item.name == "directional_together"
                )
            ),
        )

    def test_proof_decoder_binds_prepared_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "proof_prepared_required"):
            MolToSmilesContinuationDecoder.from_asset(
                self.assets["tetra_plus"].path,
                proof_capable=True,
            )
        for asset_name, prepared_name in (
            ("tetra_plus", "tetra_minus"),
            ("tetra_minus", "tetra_plus"),
            ("directional_opposite", "directional_together"),
            ("directional_together", "directional_opposite"),
        ):
            with self.subTest(asset=asset_name, prepared=prepared_name):
                with self.assertRaisesRegex(Exception, "prepared_identity_mismatch"):
                    MolToSmilesContinuationDecoder.from_asset(
                        self.assets[asset_name].path,
                        proof_capable=True,
                        prepared=self.prepared[prepared_name],
                    )


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
