"""Inspectable proof objects for South Star traversal witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .constraints import TraversalAssignment
from .constraints import validate_stereo_traversal_witness
from .ids import AtomId
from .ids import CarrierSlotId
from .policy import AnnotationMode
from .policy import DirectionMark
from .policy import TetraToken


class CertificateRelationKind(Enum):
    GRAPH_COVERAGE = "graph_coverage"
    ATOM_DECODE = "atom_decode"
    BOND_DECODE = "bond_decode"
    RING_LABELS = "ring_labels"
    RING_PAIR_DECODE = "ring_pair_decode"
    TETRA_SITE = "tetra_site"
    DIRECTIONAL_SITE = "directional_site"
    NO_ACCIDENTAL_STEREO = "no_accidental_stereo"


@dataclass(frozen=True, slots=True)
class RelationCertificate:
    kind: CertificateRelationKind
    subject: str
    detail: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationSelectionCertificate:
    mode: AnnotationMode
    selected_support: frozenset[CarrierSlotId]
    feasible_supports: frozenset[frozenset[CarrierSlotId]]
    selected_supports: frozenset[frozenset[CarrierSlotId]]

    def validate(self) -> None:
        if self.selected_support not in self.feasible_supports:
            raise ValueError("selected support is not feasible")
        if self.selected_support not in self.selected_supports:
            raise ValueError("selected support is not in selected_supports")
        if not self.selected_supports <= self.feasible_supports:
            raise ValueError("selected_supports must be feasible supports")

        if self.mode is AnnotationMode.SUPPORT_MAXIMAL:
            if any(self.selected_support < support for support in self.feasible_supports):
                raise ValueError("support is not inclusion-maximal")
            return

        if self.mode is AnnotationMode.CARDINALITY_MAXIMAL:
            if self.feasible_supports:
                maximum = max(len(support) for support in self.feasible_supports)
                if len(self.selected_support) != maximum:
                    raise ValueError("support is not cardinality-maximal")
            return

        if self.mode in {AnnotationMode.HARD, AnnotationMode.CANONICAL}:
            return

        raise ValueError(f"unknown annotation mode: {self.mode!r}")


@dataclass(frozen=True, slots=True)
class StereoSolutionCertificate:
    tetra_tokens: tuple[tuple[AtomId, TetraToken], ...]
    direction_marks: tuple[tuple[CarrierSlotId, DirectionMark], ...]
    relation_certificates: tuple[RelationCertificate, ...]
    annotation_certificate: AnnotationSelectionCertificate


@dataclass(frozen=True, slots=True)
class WitnessCertificate:
    witness_id: str
    rendered: str
    skeleton_key: tuple[object, ...]
    prefix_key: tuple[object, ...]
    assignment_key: tuple[object, ...]
    stereo_solution: StereoSolutionCertificate


def validate_annotation_selection_certificate(
    cert: AnnotationSelectionCertificate,
) -> None:
    cert.validate()


def validate_stereo_solution_certificate(
    csp,
    solution,
    cert: StereoSolutionCertificate,
) -> None:
    validate_annotation_selection_certificate(cert.annotation_certificate)
    if dict(cert.tetra_tokens) != solution.tetra_tokens:
        raise ValueError("certificate tetra tokens do not match solution")
    if dict(cert.direction_marks) != solution.direction_marks:
        raise ValueError("certificate direction marks do not match solution")

    for atom, token in solution.tetra_tokens.items():
        if token not in csp.tetra_domains[atom]:
            raise ValueError(f"tetra token is outside domain for atom {atom!r}")
    for relation in csp.tetra_relations:
        if solution.tetra_tokens[relation.center] not in relation.allowed_tokens:
            raise ValueError(f"tetra relation rejected site {relation.site!r}")

    for carrier, mark in solution.direction_marks.items():
        if mark not in csp.direction_domains[carrier]:
            raise ValueError(
                f"direction mark is outside domain for carrier {carrier!r}"
            )
    for relation in csp.mark_relations():
        row = tuple(solution.direction_marks[carrier] for carrier in relation.scope)
        if row not in relation.allowed_rows:
            raise ValueError(f"mark relation rejected {relation.subject!r}")


def validate_witness_certificate(
    *,
    facts,
    skeleton,
    slots,
    assignment: TraversalAssignment,
    policy,
    semantics,
    certificate: WitnessCertificate,
) -> None:
    from .stereo_csp import PresentationPrefix
    from .stereo_csp import StereoSolution
    from .stereo_csp import assignment_from_prefix_solution
    from .stereo_csp import build_stereo_csp

    validate_stereo_traversal_witness(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
    )
    if certificate.witness_id == "":
        raise ValueError("witness certificate id must be nonempty")

    prefix = PresentationPrefix(
        atom_text=dict(assignment.atom_text),
        bond_text=dict(assignment.bond_text),
        ring_labels=dict(assignment.ring_labels),
    )
    solution = StereoSolution(
        tetra_tokens=dict(assignment.tetra_tokens),
        direction_marks=dict(assignment.direction_marks),
    )
    csp = build_stereo_csp(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        prefix=prefix,
        policy=policy,
        semantics=semantics,
    )
    reconstructed = assignment_from_prefix_solution(prefix, solution)
    if reconstructed != assignment:
        raise ValueError("certificate assignment reconstruction mismatch")
    validate_stereo_solution_certificate(csp, solution, certificate.stereo_solution)


__all__ = (
    "AnnotationSelectionCertificate",
    "CertificateRelationKind",
    "RelationCertificate",
    "StereoSolutionCertificate",
    "WitnessCertificate",
    "validate_annotation_selection_certificate",
    "validate_stereo_solution_certificate",
    "validate_witness_certificate",
)
