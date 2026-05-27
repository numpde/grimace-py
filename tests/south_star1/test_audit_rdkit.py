"""Tests for optional RDKit audit helpers."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.audit_rdkit import (
    SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES,
)
from grimace._south_star1.audit_rdkit import RoundTripFailureKind
from grimace._south_star1.audit_rdkit import SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES
from grimace._south_star1.audit_rdkit import SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES
from grimace._south_star1.audit_rdkit import audit_generated_support_with_rdkit
from grimace._south_star1.audit_rdkit import audit_generated_support_with_rdkit_smiles_source
from grimace._south_star1.audit_rdkit import audit_generated_witnesses_with_rdkit
from grimace._south_star1.audit_rdkit import classify_specified_closure_round_trips
from grimace._south_star1.audit_rdkit import summarize_rdkit_audit
from grimace._south_star1.audit_rdkit import trace_specified_closure_round_trip
from grimace._south_star1.audit_rdkit import trace_specified_closure_support_round_trips
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.support_enumeration import enumerate_exhaustive_stereo_support


class RdkitAuditTest(unittest.TestCase):
    def test_supported_audit_matrix_roundtrips_to_isomorphic_facts(self) -> None:
        for case in SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES:
            with self.subTest(case=case.name):
                results = audit_generated_support_with_rdkit(
                    Chem.MolFromSmiles(case.smiles),
                    policy_options=case.policy_options,
                    adapter_options=case.adapter_options,
                )
                summary = summarize_rdkit_audit(
                    case_name=case.name,
                    input_smiles=case.smiles,
                    results=results,
                )

                self.assertTrue(results)
                self.assertTrue(summary.ok, summary)
                self.assertEqual(summary.parsed_count, summary.support_count)
                self.assertIsNone(summary.first_failure)
                if case.max_support_size is not None:
                    self.assertLessEqual(summary.support_count, case.max_support_size)

    def test_unsupported_audit_matrix_rejects_with_expected_kind(self) -> None:
        for case in SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES:
            with self.subTest(case=case.name):
                with self.assertRaises(SouthStarError) as raised:
                    ordinary_molecule_facts_from_rdkit(
                        Chem.MolFromSmiles(case.smiles),
                        case.adapter_options,
                    )

                self.assertIs(raised.exception.kind, case.expected_error_kind)

    def test_experimental_exact_equivalence_audit_matrix_roundtrips(self) -> None:
        for case in SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES:
            with self.subTest(case=case.name):
                results = audit_generated_support_with_rdkit(
                    Chem.MolFromSmiles(case.smiles),
                    policy_options=case.policy_options,
                    adapter_options=case.adapter_options,
                )
                summary = summarize_rdkit_audit(
                    case_name=case.name,
                    input_smiles=case.smiles,
                    results=results,
                )

                self.assertTrue(results)
                self.assertTrue(summary.ok, summary)
                if case.max_support_size is not None:
                    self.assertLessEqual(summary.support_count, case.max_support_size)

    def test_audit_accepts_explicit_skeleton_slice(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        facts = ordinary_molecule_facts_from_rdkit(mol)
        policy = ordinary_policy_for_facts(facts)
        skeletons = tuple(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
            if skeleton.roots == (AtomId(0),)
        )

        results = audit_generated_support_with_rdkit(mol, skeletons=skeletons)

        self.assertTrue(results)
        self.assertTrue(all(result.ok for result in results))

    def test_small_connected_support_is_stable_under_rdkit_reparse(self) -> None:
        cases = ("[C@H](F)(Cl)Br", "C(/F)=C(\\Cl)")

        for text in cases:
            with self.subTest(text=text):
                original_support = _support_for_smiles(text)
                self.assertTrue(original_support)

                for generated in original_support:
                    reparsed_support = _support_for_smiles(generated)
                    self.assertEqual(reparsed_support, original_support, generated)

    def test_exact_tetra_support_is_stable_under_rdkit_reparse(self) -> None:
        case = SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES[0]
        original_support = _support_for_smiles(
            case.smiles,
            adapter_options=case.adapter_options,
        )
        self.assertTrue(original_support)

        for generated in original_support:
            reparsed_support = _support_for_smiles(
                generated,
                adapter_options=case.adapter_options,
            )
            self.assertEqual(reparsed_support, original_support, generated)

    def test_exact_directional_representatives_are_stable_under_rdkit_reparse(
        self,
    ) -> None:
        case = SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES[1]
        original_support = tuple(
            sorted(
                _support_for_smiles(
                    case.smiles,
                    adapter_options=case.adapter_options,
                )
            )
        )
        self.assertTrue(original_support)

        for generated in original_support[:8]:
            reparsed_support = _support_for_smiles(
                generated,
                adapter_options=case.adapter_options,
            )
            self.assertEqual(frozenset(reparsed_support), frozenset(original_support))

    def test_witness_audit_preserves_witness_context(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        facts = ordinary_molecule_facts_from_rdkit(mol)
        policy = ordinary_policy_for_facts(facts)
        skeletons = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[:1]

        results = audit_generated_witnesses_with_rdkit(mol, skeletons=skeletons)

        self.assertTrue(results)
        self.assertTrue(all(result.ok for result in results))
        self.assertTrue(all(result.witness_id for result in results))
        self.assertTrue(all(result.constraints for result in results))

    def test_specified_closure_round_trip_failure_is_classified(self) -> None:
        adapter_options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )
        original = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)([C@H](F)Cl)[C@@H](F)Cl"),
            adapter_options,
        )

        trace = trace_specified_closure_round_trip(
            "[C@H]([C@@H](F)([C@@H](F)(Cl)))(F)(Cl)",
            original_facts=original,
            extraction_options=adapter_options,
            policy_options=OrdinaryPolicyOptions(),
        )

        self.assertTrue(trace.parsed)
        self.assertIs(
            trace.failure_kind,
            RoundTripFailureKind.SPECIFIED_TETRA_RECORD_LOSS,
        )
        self.assertEqual(trace.original_tetra_status.specified, 3)
        self.assertEqual(trace.reparsed_tetra_status.specified, 2)
        self.assertEqual(len(trace.original_raw.tetrahedral), 3)
        self.assertEqual(len(trace.reparsed_raw_sanitized.tetrahedral), 2)
        self.assertIsNotNone(trace.reparsed_raw_unsanitized)
        self.assertEqual(len(trace.reparsed_raw_unsanitized.tetrahedral), 3)
        self.assertFalse(trace.isomorphic_without_potential_sites)

        summary = classify_specified_closure_round_trips((trace,))
        self.assertEqual(
            summary[RoundTripFailureKind.SPECIFIED_TETRA_RECORD_LOSS],
            1,
        )

    def test_smiles_source_ingestion_preserves_cleanup_lost_tetra(self) -> None:
        adapter_options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )
        original = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)([C@H](F)Cl)[C@@H](F)Cl"),
            adapter_options,
        )
        text = "[C@H]([C@@H](F)([C@@H](F)(Cl)))(F)(Cl)"

        mol_state_facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles(text),
            adapter_options,
        )
        source_facts = ordinary_molecule_facts_from_smiles(text, adapter_options)

        self.assertEqual(_specified_tetra_count(original), 3)
        self.assertEqual(_specified_tetra_count(mol_state_facts), 2)
        self.assertEqual(_specified_tetra_count(source_facts), 3)
        self.assertFalse(facts_are_isomorphic(original, mol_state_facts).isomorphic)
        self.assertTrue(facts_are_isomorphic(original, source_facts).isomorphic)

    def test_specified_closure_smiles_source_audit_round_trips(self) -> None:
        adapter_options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )

        results = audit_generated_support_with_rdkit_smiles_source(
            "[C@H](F)([C@H](F)Cl)[C@@H](F)Cl",
            adapter_options=adapter_options,
        )
        summary = summarize_rdkit_audit(
            case_name="specified_closure_source_audit",
            input_smiles="[C@H](F)([C@H](F)Cl)[C@@H](F)Cl",
            results=results,
        )

        self.assertEqual(len(results), 1216)
        self.assertTrue(summary.ok, summary)
        self.assertEqual(summary.parsed_count, summary.support_count)

    def test_smiles_source_and_mol_state_agree_for_directional_case(self) -> None:
        text = "C(/F)=C(\\Cl)"
        mol_state_facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles(text),
        )
        source_facts = ordinary_molecule_facts_from_smiles(text)

        self.assertTrue(facts_are_isomorphic(mol_state_facts, source_facts).isomorphic)

    @unittest.skip(
        "Known specified-closure RDKit round-trip mismatch; the classifier "
        "pins a representative sanitized raw-stereo loss.",
    )
    def test_specified_closure_full_rdkit_audit_round_trips(self) -> None:
        adapter_options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )
        results = audit_generated_support_with_rdkit(
            Chem.MolFromSmiles("[C@H](F)([C@H](F)Cl)[C@@H](F)Cl"),
            adapter_options=adapter_options,
        )

        self.assertTrue(all(result.ok for result in results))

    @unittest.skip(
        "Diagnostic for known specified-closure RDKit round-trip mismatch.",
    )
    def test_specified_closure_full_rdkit_audit_failure_summary(self) -> None:
        adapter_options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )
        original = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)([C@H](F)Cl)[C@@H](F)Cl"),
            adapter_options,
        )
        traces = trace_specified_closure_support_round_trips(
            original,
            extraction_options=adapter_options,
        )
        summary = classify_specified_closure_round_trips(traces)

        self.assertEqual(len(traces), 1216)
        self.assertEqual(
            summary,
            {RoundTripFailureKind.SPECIFIED_TETRA_RECORD_LOSS: 384},
        )


def _support_for_smiles(text: str, *, adapter_options=None) -> frozenset[str]:
    mol = Chem.MolFromSmiles(text)
    if adapter_options is None:
        facts = ordinary_molecule_facts_from_rdkit(mol)
    else:
        facts = ordinary_molecule_facts_from_rdkit(mol, adapter_options)
    image = enumerate_exhaustive_stereo_support(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )
    return frozenset(image.strings)


def _specified_tetra_count(facts) -> int:
    return sum(site.status.value == "specified" for site in facts.stereo.tetrahedral)


if __name__ == "__main__":
    unittest.main()
