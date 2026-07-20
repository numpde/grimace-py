"""Graph-work envelope qualification for the declared aromatic corpus."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
import unittest
from unittest.mock import patch

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_execution_evidence import (
    WriterGraphObligationWorkEnvelope,
)
from grimace._south_star1.writer_execution_evidence import (
    WriterGraphObligationWorkEvidence,
)
from grimace._south_star1.writer_execution_evidence import (
    writer_graph_obligation_work_envelope_violation,
)
from grimace._south_star1.writer_frontier import (
    _checked_writer_frontier_branch_supports,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_events import WriterBondEmitted
import grimace._south_star1.writer_execution_evidence as writer_execution_evidence


_AROMATIC_CORPUS = (
    ("benzene", "c1ccccc1"),
    ("pyridine", "n1ccccc1"),
    ("furan", "c1occc1"),
    ("thiophene", "c1sccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("toluene", "Cc1ccccc1"),
    ("biphenyl", "c1ccccc1-c1ccccc1"),
    ("disconnected_benzene", "c1ccccc1.O"),
)

_GRAPH_WORK_METRICS = tuple(
    field.name
    for field in fields(WriterGraphObligationWorkEvidence)
    if field.name not in {"operation", "component_index"}
)

_EXPECTED_MAXIMA = {
    "component_atom_count": 12,
    "component_bond_count": 13,
    "edge_obligation_count": 13,
    "residual_attachment_count": 2,
    "residual_attachment_action_count": 2,
    "boundary_incidence_count": 3,
    "closure_candidate_count": 0,
    "live_branch_return_closure_candidate_count": 0,
    "deferred_branch_return_closure_candidate_count": 0,
    "deferred_control_live_closure_candidate_count": 0,
    "unsupported_closure_candidate_count": 0,
    "open_closure_count": 2,
    "closed_closure_count": 2,
    "max_attachment_atom_count": 12,
    "max_attachment_boundary_count": 2,
    "max_attachment_cyclic_rank": 2,
}

_EXPECTED_WITNESSES = {
    "component_atom_count": (
        "biphenyl",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "component_bond_count": (
        "biphenyl",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "edge_obligation_count": (
        "biphenyl",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "residual_attachment_count": (
        "naphthalene",
        6,
        "b499ab31beb4033f72000cd36cb526f58946d517de541d52fe7cbee5146687fb",
    ),
    "residual_attachment_action_count": (
        "naphthalene",
        6,
        "b499ab31beb4033f72000cd36cb526f58946d517de541d52fe7cbee5146687fb",
    ),
    "boundary_incidence_count": (
        "biphenyl",
        6,
        "bc98764b1ca31f961060eeed7937b8e8aaebe453282eed7ed761b649fa713b03",
    ),
    "closure_candidate_count": (
        "benzene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "live_branch_return_closure_candidate_count": (
        "benzene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "deferred_branch_return_closure_candidate_count": (
        "benzene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "deferred_control_live_closure_candidate_count": (
        "benzene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "unsupported_closure_candidate_count": (
        "benzene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "open_closure_count": (
        "naphthalene",
        5,
        "a4141122e1600e797fd3eeb4974bde7c0dfdb44ba959fc9c9f4b143d3011b21d",
    ),
    "closed_closure_count": (
        "naphthalene",
        14,
        "a2f48ed12e505e17d6100740c064327e37b77ef551c7f7facc3dd6ef7184cd61",
    ),
    "max_attachment_atom_count": (
        "biphenyl",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
    "max_attachment_boundary_count": (
        "benzene",
        1,
        "aac1bc0a8d1c12f9d828501527a0db4e33f183c0e5dcf50e161f2ff9a40b0dec",
    ),
    "max_attachment_cyclic_rank": (
        "naphthalene",
        0,
        "a00eade6928be25f2748170b6937b5d4a601ee55bf019c7da164d1b84a1b0138",
    ),
}

_CHANGED_LIMIT_FIELDS = {
    "component_atom_count": "max_component_atom_count",
    "component_bond_count": "max_component_bond_count",
    "edge_obligation_count": "max_edge_obligation_count",
    "max_attachment_atom_count": "max_attachment_atom_count",
}


@dataclass(frozen=True, slots=True)
class _GraphWorkWitness:
    value: int
    case: str
    operation: str
    component_index: int
    cursor_digest: str
    token_depth: int


class WriterAromaticGraphWorkEnvelopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.characterization = _characterize_aromatic_corpus()

    def test_relaxed_live_characterization_records_every_metric(self) -> None:
        self.assertEqual(
            {
                metric: witness.value
                for metric, witness in self.characterization.items()
            },
            _EXPECTED_MAXIMA,
        )
        for metric, witness in self.characterization.items():
            with self.subTest(metric=metric):
                self.assertEqual(
                    (
                        witness.case,
                        witness.token_depth,
                        witness.cursor_digest,
                    ),
                    _EXPECTED_WITNESSES[metric],
                )
                self.assertEqual(
                    witness.operation,
                    "writer graph obligation context",
                )
                self.assertEqual(witness.component_index, 0)

    def test_requalified_limits_accept_exact_and_reject_plus_one(self) -> None:
        envelope = (
            writer_execution_evidence
            ._PUBLIC_WRITER_GRAPH_OBLIGATION_WORK_ENVELOPE
        )
        baseline = _zero_graph_work_evidence()

        for metric, limit_field in _CHANGED_LIMIT_FIELDS.items():
            with self.subTest(metric=metric):
                limit = getattr(envelope, limit_field)
                self.assertEqual(limit, _EXPECTED_MAXIMA[metric])
                assert limit is not None
                exact = replace(baseline, **{metric: limit})
                self.assertIsNone(
                    writer_graph_obligation_work_envelope_violation(exact)
                )

                over = replace(exact, **{metric: limit + 1})
                violation = writer_graph_obligation_work_envelope_violation(over)
                self.assertIsNotNone(violation)
                assert violation is not None
                self.assertEqual(violation.evidence.operation, baseline.operation)
                self.assertEqual(
                    violation.evidence.component_index,
                    baseline.component_index,
                )
                self.assertEqual(violation.metric, metric)
                self.assertEqual(violation.actual, limit + 1)
                self.assertEqual(violation.limit, limit)

    def test_default_envelope_reaches_every_checked_live_frontier(self) -> None:
        for name, smiles in _AROMATIC_CORPUS:
            with self.subTest(case=name):
                traversal = _traverse_checked_frontiers(name, smiles)
                self.assertGreater(traversal.cursor_count, 0)
                self.assertGreater(traversal.terminal_support_count, 0)

    def test_biphenyl_bridge_is_live_and_explicit(self) -> None:
        facts, prepared = _prepare("c1ccccc1-c1ccccc1")
        bridge = next(
            bond
            for bond in facts.bonds
            if bond.order is BondOrder.SINGLE
            and _atom(facts, bond.a).is_aromatic
            and _atom(facts, bond.b).is_aromatic
        )
        choices = prepared.policy.bond_text_domain(
            facts,
            bridge.id,
            slot_kind="tree",
        )
        self.assertEqual(
            tuple(
                (choice.base_text, choice.permits_direction)
                for choice in choices
            ),
            (("-", False),),
        )

        traversal = _traverse_checked_frontiers("biphenyl", None, prepared=prepared)
        self.assertIn(bridge.id, traversal.emitted_bonds)


@dataclass(frozen=True, slots=True)
class _TraversalResult:
    cursor_count: int
    terminal_support_count: int
    emitted_bonds: frozenset[object]
    evidence: tuple[
        tuple[WriterGraphObligationWorkEvidence, str, int],
        ...,
    ]


def _characterize_aromatic_corpus() -> dict[str, _GraphWorkWitness]:
    relaxed = WriterGraphObligationWorkEnvelope()
    maxima: dict[str, _GraphWorkWitness] = {}
    with patch.object(
        writer_execution_evidence,
        "_PUBLIC_WRITER_GRAPH_OBLIGATION_WORK_ENVELOPE",
        relaxed,
    ):
        for name, smiles in _AROMATIC_CORPUS:
            traversal = _traverse_checked_frontiers(name, smiles)
            for evidence, cursor_digest, token_depth in traversal.evidence:
                for metric in _GRAPH_WORK_METRICS:
                    candidate = _GraphWorkWitness(
                        value=getattr(evidence, metric),
                        case=name,
                        operation=evidence.operation,
                        component_index=evidence.component_index,
                        cursor_digest=cursor_digest,
                        token_depth=token_depth,
                    )
                    current = maxima.get(metric)
                    if current is None or candidate.value > current.value:
                        maxima[metric] = candidate
    return maxima


def _traverse_checked_frontiers(
    name: str,
    smiles: str | None,
    *,
    prepared=None,
) -> _TraversalResult:
    del name
    if prepared is None:
        _facts, prepared = _prepare(smiles)
    pending = deque([(initial_writer_frontier_cursor(prepared, _options()), 0)])
    seen = set()
    terminal_support_count = 0
    emitted_bonds = set()
    evidence = []

    while pending:
        cursor, token_depth = pending.popleft()
        if cursor in seen:
            continue
        seen.add(cursor)
        cursor_digest = _identity_digest(cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        for support in batch.supports:
            evidence.extend(
                (item, cursor_digest, token_depth)
                for item in support.graph_obligation_work_evidence
            )
            emitted_bonds.update(
                event.bond
                for event in support.events
                if isinstance(event, WriterBondEmitted)
            )
        for support in batch.terminal_supports:
            terminal_support_count += 1
            evidence.extend(
                (item, cursor_digest, token_depth)
                for item in support.graph_obligation_work_evidence
            )
        pending.extend(
            (projection.successor_cursor, token_depth + 1)
            for projection in batch.text_choice_projection_certificates
        )

    return _TraversalResult(
        cursor_count=len(seen),
        terminal_support_count=terminal_support_count,
        emitted_bonds=frozenset(emitted_bonds),
        evidence=tuple(evidence),
    )


def _prepare(smiles: str):
    facts = ordinary_molecule_facts_from_smiles(smiles)
    return facts, prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=0,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _atom(facts, atom_id):
    return next(atom for atom in facts.atoms if atom.id == atom_id)


def _zero_graph_work_evidence() -> WriterGraphObligationWorkEvidence:
    return WriterGraphObligationWorkEvidence(
        operation="aromatic graph envelope boundary test",
        component_index=7,
        component_atom_count=0,
        component_bond_count=0,
        edge_obligation_count=0,
        residual_attachment_count=0,
        residual_attachment_action_count=0,
        boundary_incidence_count=0,
        closure_candidate_count=0,
        live_branch_return_closure_candidate_count=0,
        deferred_branch_return_closure_candidate_count=0,
        deferred_control_live_closure_candidate_count=0,
        unsupported_closure_candidate_count=0,
        open_closure_count=0,
        closed_closure_count=0,
        max_attachment_atom_count=0,
        max_attachment_boundary_count=0,
        max_attachment_cyclic_rank=0,
    )


if __name__ == "__main__":
    unittest.main()
