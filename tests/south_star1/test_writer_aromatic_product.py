"""Pinned ordinary aromatic product contract."""

from __future__ import annotations

from dataclasses import replace
import unittest

from rdkit import rdBase

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_support import enumerate_prepared_writer_shaped_support
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from tests.helpers.rdkit_south_star_aromatic_audit import (
    load_pinned_south_star_aromatic_audit_cases,
)
from tests.south_star1.default_writer_capability_ledger import (
    default_writer_cases_for_rdkit_audit,
)


class WriterAromaticProductTest(unittest.TestCase):
    def test_pinned_corpus_has_exact_support_and_aromatic_roundtrip(self) -> None:
        ledger = {
            item.name: item for item in default_writer_cases_for_rdkit_audit("aromatic")
        }
        for fixture in load_pinned_south_star_aromatic_audit_cases(
            rdBase.rdkitVersion
        ):
            with self.subTest(case=fixture.case_id):
                case = ledger[fixture.case_id]
                facts = ordinary_molecule_facts_from_smiles(case.smiles)
                prepared = prepare_south_star_mol_from_facts(
                    facts, writer_surface=SouthStarWriterSurface()
                )
                image = enumerate_prepared_writer_shaped_support(
                    prepared=prepared,
                    runtime_options=SouthStarRuntimeOptions(
                        rooted_at_atom=fixture.rooted_at_atom,
                        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
                    ),
                )
                self.assertEqual(tuple(sorted(image.strings)), fixture.expected_support)
                self.assertEqual(image.distinct_count, fixture.support_count)
                self.assertEqual(image.witness_count, fixture.completion_count)
                self.assertEqual(
                    tuple(text for text, _count in fixture.witness_multiplicities),
                    fixture.expected_support,
                )
                self.assertEqual(
                    sum(count for _text, count in fixture.witness_multiplicities),
                    fixture.completion_count,
                )
                self.assertEqual(case.expected_support_digest, fixture.sorted_support_sha256)
                for rendered in image.strings:
                    reparsed = ordinary_molecule_facts_from_smiles(rendered)
                    self.assertTrue(
                        facts_are_isomorphic(facts, reparsed).isomorphic,
                        rendered,
                    )
                    self.assertEqual(
                        sum(atom.is_aromatic for atom in reparsed.atoms),
                        sum(atom.is_aromatic for atom in facts.atoms),
                    )
                    self.assertEqual(
                        sum(bond.order is BondOrder.AROMATIC for bond in reparsed.bonds),
                        sum(bond.order is BondOrder.AROMATIC for bond in facts.bonds),
                    )

    def test_endpoint_aware_single_bond_domains(self) -> None:
        toluene = ordinary_molecule_facts_from_smiles("Cc1ccccc1")
        biphenyl = ordinary_molecule_facts_from_smiles("c1ccccc1-c1ccccc1")
        toluene_bridge = next(bond for bond in toluene.bonds if bond.order is BondOrder.SINGLE)
        biphenyl_bridge = next(bond for bond in biphenyl.bonds if bond.order is BondOrder.SINGLE)
        toluene_policy = ordinary_policy_for_facts(toluene)
        biphenyl_policy = ordinary_policy_for_facts(biphenyl)
        self.assertEqual(
            tuple(
                (item.base_text, item.permits_direction)
                for item in toluene_policy.bond_text_domain(
                    toluene, toluene_bridge.id, slot_kind="tree"
                )
            ),
            (("", True),),
        )
        self.assertEqual(
            tuple(
                (item.base_text, item.permits_direction)
                for item in biphenyl_policy.bond_text_domain(
                    biphenyl, biphenyl_bridge.id, slot_kind="tree"
                )
            ),
            (("-", False),),
        )
        semantics = OrdinarySmilesSemantics()
        self.assertFalse(
            semantics.bond_decode_ok(
                biphenyl,
                biphenyl_bridge.id,
                BondTextChoice("forged_elision", "", True),
                DirectionMark.ABSENT,
            )
        )

    def test_aromatic_modes_are_exact_policy_domains(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("c1ccccc1")
        bond = facts.bonds[0]
        for mode, expected in (
            ("elide", (("", False),)),
            ("explicit", ((":", False),)),
            ("both", (("", False), (":", False))),
        ):
            with self.subTest(mode=mode):
                policy = ordinary_policy_for_facts(
                    facts, OrdinaryPolicyOptions(aromatic_bond_mode=mode)
                )
                self.assertEqual(
                    tuple(
                        (item.base_text, item.permits_direction)
                        for item in policy.bond_text_domain(
                            facts, bond.id, slot_kind="ring_endpoint"
                        )
                    ),
                    expected,
                )
                prepared = prepare_south_star_mol_from_facts(
                    facts,
                    writer_surface=SouthStarWriterSurface(),
                    policy=policy,
                )
                options = SouthStarRuntimeOptions(
                    rooted_at_atom=0,
                    serialization_language=SerializationLanguageMode.WRITER_SHAPED,
                )
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=initial_writer_frontier_cursor(prepared, options),
                )
                artifact = writer_support_artifact_envelope_for_snapshot(
                    prepared=prepared,
                    snapshot=snapshot,
                )
                replay = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                    policy=policy,
                )
                self.assertTrue(replay.accepted, replay.reason)
                self.assertTrue(replay.offline_replay_complete, replay.reason)

    def test_adjacent_aromatic_atom_surfaces_remain_typed_blocked(self) -> None:
        for smiles in ("[nH]1cccc1", "b1ccccc1", "p1ccccc1"):
            with self.subTest(smiles=smiles):
                facts = ordinary_molecule_facts_from_smiles(smiles)
                with self.assertRaises(SouthStarError) as raised:
                    ordinary_policy_for_facts(facts)
                self.assertIn("unsupported", str(raised.exception))

    def test_aromatic_single_ring_closure_is_preparation_blocked(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("c1ccccc1")
        cyclic_single = replace(
            facts,
            bonds=(
                replace(
                    facts.bonds[0],
                    order=BondOrder.SINGLE,
                    is_aromatic=False,
                ),
                *facts.bonds[1:],
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "aromatic/aromatic single bonds that can become ring closures",
        ):
            ordinary_policy_for_facts(cyclic_single)

    def test_mapped_aromatic_source_is_ingestion_blocked(self) -> None:
        with self.assertRaisesRegex(SouthStarError, "atom-mapped.*unsupported"):
            ordinary_molecule_facts_from_smiles("c1cc[c:1]cc1")


if __name__ == "__main__":
    unittest.main()
