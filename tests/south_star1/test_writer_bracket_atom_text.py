"""Bracket atom text support for narrow charged ammonium-class atoms."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.ids import AtomId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_atom_text_lifecycle import (
    validate_writer_bracket_atom_text_transition,
)
from grimace._south_star1.writer_atom_text_lifecycle import (
    writer_bracket_atom_text_evidence,
)
from grimace._south_star1.writer_frontier import (
    _checked_writer_frontier_branch_supports,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_state_delta_certificates import (
    validate_writer_branch_successor_state_certificate,
)
from grimace._south_star1.writer_support import (
    enumerate_prepared_writer_shaped_support,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)


_EXTRACTION = RdkitOrdinaryExtractionOptions(include_potential_sites=False)


class WriterBracketAtomTextTest(unittest.TestCase):
    def test_nh4_plus_prepares_under_default_policy(self) -> None:
        prepared = _prepare(_facts("[NH4+]"))

        self.assertEqual(
            prepared.policy.atom_text_domain_unchecked(AtomId(0))[0].render(
                TetraToken.NONE
            ),
            "[NH4+]",
        )

    def test_nh4_plus_frontier_emits_bracket_atom_text(self) -> None:
        support = _first_branch_support("[NH4+]")

        self.assertEqual(support.emitted_text, "[NH4+]")
        self.assertEqual(support.events[0].text, "[NH4+]")

    def test_nh4_plus_support_count_and_artifact_verifiers(self) -> None:
        facts = _facts("[NH4+]")
        prepared = _prepare(facts)
        runtime_state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        image = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(image.strings, ("[NH4+]",))
        self.assertEqual(image.distinct_count, 1)
        self.assertEqual(image.witness_count, 1)
        self.assertEqual(
            count_writer_runtime_support(prepared=prepared, state=runtime_state),
            1,
        )
        self.assertEqual(
            count_writer_runtime_completions(
                prepared=prepared,
                state=runtime_state,
            ),
            1,
        )

        artifact = _artifact(prepared)
        structural = verify_writer_support_artifact_consistency(artifact)
        live = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=artifact,
        )
        fact_bound = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertTrue(fact_bound.accepted, fact_bound.reason)

    def test_nh4_plus_rdkit_round_trip_audit_passes(self) -> None:
        source = _facts("[NH4+]")
        reparsed = _facts("[NH4+]")

        self.assertTrue(facts_are_isomorphic(source, reparsed).isomorphic)

    def test_tampered_charge_text_is_rejected(self) -> None:
        atom = _facts("[NH4+]").atoms[0]
        evidence = writer_bracket_atom_text_evidence(
            atom,
            rendered_text="[NH4+]",
        )

        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_mismatch"):
            validate_writer_bracket_atom_text_transition(
                atom=atom,
                rendered_text="[NH4-]",
                evidence=replace(evidence, rendered_text="[NH4-]"),
            )

    def test_tampered_hydrogen_count_is_rejected(self) -> None:
        atom = _facts("[NH4+]").atoms[0]
        evidence = writer_bracket_atom_text_evidence(
            atom,
            rendered_text="[NH4+]",
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "bracket_atom_hydrogen_count_mismatch",
        ):
            validate_writer_bracket_atom_text_transition(
                atom=atom,
                rendered_text="[NH4+]",
                evidence=replace(evidence, hydrogen_count=3),
            )

    def test_tampered_unbracketed_charged_atom_is_rejected(self) -> None:
        atom = _facts("[NH4+]").atoms[0]
        evidence = writer_bracket_atom_text_evidence(
            atom,
            rendered_text="[NH4+]",
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "bracket_atom_text_lacks_brackets",
        ):
            validate_writer_bracket_atom_text_transition(
                atom=atom,
                rendered_text="NH4+",
                evidence=replace(evidence, rendered_text="NH4+"),
            )

    def test_tampered_atom_event_text_is_rejected_by_successor_certificate(
        self,
    ) -> None:
        support = _first_branch_support("[NH4+]")
        bad_event = replace(support.events[0], text="[NH3+]")
        bad_certificate = replace(
            support.successor_state_certificate,
            events=(bad_event,),
        )

        with self.assertRaisesRegex(SouthStarError, "event_view_mismatch"):
            validate_writer_branch_successor_state_certificate(bad_certificate)

    def test_isotope_still_blocks_for_13ch4(self) -> None:
        facts = _facts("[13CH4]")

        with self.assertRaisesRegex(SouthStarError, "isotopic atoms"):
            _prepare(facts)


def _facts(smiles: str):
    return ordinary_molecule_facts_from_smiles(smiles, _EXTRACTION)


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=0,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _first_branch_support(smiles: str):
    prepared = _prepare(_facts(smiles))
    batch = _checked_writer_frontier_branch_supports(
        prepared,
        initial_writer_frontier_cursor(prepared, _writer_options()),
        include_counts=False,
    )
    self_support = batch.supports[0]
    return self_support


def _artifact(prepared):
    options = _writer_options()
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )


if __name__ == "__main__":
    unittest.main()
