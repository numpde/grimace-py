"""Exact residual-transition proof-term tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualFactorKey
from grimace._south_star1.residual_constraints import TetraLocalParity
from grimace._south_star1.residual_constraints import tetra_parity_var
from grimace._south_star1.residual_constraints import tetra_token_var
from grimace._south_star1.writer_events import WriterAtomEmitted
from grimace._south_star1.writer_events import WriterLocalOrderClosed
from grimace._south_star1.writer_residual_transition_terms import (
    WriterResidualTransitionKind,
)
from grimace._south_star1.writer_residual_transition_terms import (
    verify_writer_residual_transition_term,
)
from grimace._south_star1.writer_residual_transition_terms import (
    writer_residual_transition_term,
)
from grimace._south_star1.writer_stereo import advance_writer_stereo_state_with_evidence
from grimace._south_star1.writer_stereo import initial_writer_stereo_state
from grimace._south_star1.writer_stereo_branch_certificates import (
    WriterStereoBranchCertificateKind,
)
from grimace._south_star1.writer_stereo_branch_certificates import (
    writer_stereo_branch_certificates,
)
from tests.south_star1.helpers import tetrahedral_facts


class WriterResidualTransitionTermTest(unittest.TestCase):
    def test_tetra_certificates_compile_to_exact_replayable_terms(self) -> None:
        prepared, certificates = _tetra_certificates()
        token = writer_residual_transition_term(
            prepared=prepared,
            certificate=_certificate(
                certificates,
                WriterStereoBranchCertificateKind.TETRA_TOKEN_RESTRICTED,
            ),
        )
        local_order = writer_residual_transition_term(
            prepared=prepared,
            certificate=_certificate(
                certificates,
                WriterStereoBranchCertificateKind.TETRA_LOCAL_ORDER_RESTRICTED,
            ),
        )

        self.assertIs(
            token.kind,
            WriterResidualTransitionKind.TETRA_TOKEN_RESTRICTION,
        )
        self.assertEqual(
            token.restrictions,
            ((tetra_token_var(prepared.tetra_templates[0].site), TetraToken.ATAT),),
        )
        self.assertEqual(token.discharged_factor_keys, ())
        verify_writer_residual_transition_term(token)

        self.assertIs(
            local_order.kind,
            WriterResidualTransitionKind.TETRA_LOCAL_ORDER_RESTRICTION,
        )
        self.assertEqual(
            local_order.restrictions,
            ((
                tetra_parity_var(prepared.tetra_templates[0].site),
                TetraLocalParity.ODD,
            ),),
        )
        self.assertEqual(
            local_order.discharged_factor_keys,
            (ResidualFactorKey("tetra_site", (0,)),),
        )
        verify_writer_residual_transition_term(local_order)

    def test_tetra_term_rejects_a_coherent_wrong_restriction(self) -> None:
        prepared, certificates = _tetra_certificates()
        term = writer_residual_transition_term(
            prepared=prepared,
            certificate=_certificate(
                certificates,
                WriterStereoBranchCertificateKind.TETRA_TOKEN_RESTRICTED,
            ),
        )
        forged = replace(
            term,
            restrictions=((term.restrictions[0][0], TetraToken.AT),),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "replayed_successor_snapshot_mismatch",
        ):
            verify_writer_residual_transition_term(forged)

    def test_tetra_term_rejects_missing_factor_discharge(self) -> None:
        prepared, certificates = _tetra_certificates()
        term = writer_residual_transition_term(
            prepared=prepared,
            certificate=_certificate(
                certificates,
                WriterStereoBranchCertificateKind.TETRA_LOCAL_ORDER_RESTRICTED,
            ),
        )
        forged = replace(term, discharged_factor_keys=())

        with self.assertRaisesRegex(
            SouthStarError,
            "replayed_successor_snapshot_mismatch",
        ):
            verify_writer_residual_transition_term(forged)

    def test_tetra_term_rejects_forged_work_metrics(self) -> None:
        prepared, certificates = _tetra_certificates()
        term = writer_residual_transition_term(
            prepared=prepared,
            certificate=_certificate(
                certificates,
                WriterStereoBranchCertificateKind.TETRA_TOKEN_RESTRICTED,
            ),
        )
        forged = replace(
            term,
            work_evidence=replace(
                term.work_evidence,
                checked_candidate_rows=(
                    term.work_evidence.checked_candidate_rows + 1
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "replayed_work_evidence_mismatch",
        ):
            verify_writer_residual_transition_term(forged)


def _tetra_certificates():
    prepared = prepare_south_star_mol_from_facts(
        tetrahedral_facts(),
        writer_surface=SouthStarWriterSurface(),
    )
    events = (
        WriterAtomEmitted(
            atom=AtomId(1),
            text="F",
            tetra_token=TetraToken.NONE,
        ),
        WriterAtomEmitted(
            atom=AtomId(0),
            text="[C@@H]",
            tetra_token=TetraToken.ATAT,
            parent=AtomId(1),
            incoming_bond=BondId(0),
        ),
        WriterAtomEmitted(
            atom=AtomId(3),
            text="Br",
            tetra_token=TetraToken.NONE,
            parent=AtomId(0),
            incoming_bond=BondId(2),
        ),
        WriterAtomEmitted(
            atom=AtomId(2),
            text="Cl",
            tetra_token=TetraToken.NONE,
            parent=AtomId(0),
            incoming_bond=BondId(1),
        ),
        WriterLocalOrderClosed(atom=AtomId(0)),
    )
    outcome = advance_writer_stereo_state_with_evidence(
        prepared,
        initial_writer_stereo_state(prepared),
        events,
    )
    if outcome.state is None:
        raise AssertionError("specified tetra event sequence was rejected")
    certificates = writer_stereo_branch_certificates(
        execution_capabilities=outcome.execution_capabilities,
        stereo_lifecycle_evidence=outcome.stereo_lifecycle_evidence,
        events=events,
    )
    return prepared, certificates


def _certificate(certificates, kind):
    matches = tuple(item for item in certificates if item.kind is kind)
    if len(matches) != 1:
        raise AssertionError(f"expected one certificate for {kind.value}")
    return matches[0]


if __name__ == "__main__":
    unittest.main()
