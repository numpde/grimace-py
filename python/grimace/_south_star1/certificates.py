"""Inspectable proof objects for South Star traversal witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .constraints import TraversalAssignment
from .constraints import validate_stereo_traversal_witness
from .ids import AtomId
from .ids import CarrierSlotId
from .policy import AnnotationMode
from .policy import DirectionMark
from .policy import TetraToken
from .ring_labels import validate_bounded_ring_labels
from .slots import BondSlotKind
from .slots import carrier_slot_by_bond_slot
from .slots import ring_bond_slots_by_bond


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
    canonical_key: tuple[object, ...] | None = None
    selected_solution_key: tuple[object, ...] | None = None
    selected_solution_keys: frozenset[tuple[object, ...]] = frozenset()
    feasible_solution_count: int = 0
    selected_solution_count: int = 0

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

        if self.mode is AnnotationMode.CANONICAL:
            if self.canonical_key is None:
                raise ValueError("canonical certificate lacks canonical key")
            if self.selected_solution_key is None:
                raise ValueError("canonical certificate lacks selected solution key")
            if not self.selected_solution_keys:
                raise ValueError("canonical certificate lacks selected solution keys")
            if self.canonical_key != min(self.selected_solution_keys):
                raise ValueError("canonical key is not minimal")
            if self.selected_solution_key != self.canonical_key:
                raise ValueError("selected solution is not canonical-minimal")
            return

        if self.mode is AnnotationMode.HARD:
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
    traversal_relation_certificates: tuple[RelationCertificate, ...]
    stereo_solution: StereoSolutionCertificate


@dataclass(frozen=True, slots=True)
class SupportEnumerationManifest:
    skeleton_count: int
    prefix_count: int
    csp_count: int
    feasible_solution_count: int
    selected_solution_count: int
    witness_count: int
    support_count: int
    support_hash: str
    witness_hash: str


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
    actual_subjects = tuple(
        (relation.kind, relation.subject)
        for relation in cert.relation_certificates
    )
    expected_subjects = expected_relation_certificate_subjects(csp)
    if actual_subjects != expected_subjects:
        raise ValueError("relation certificate coverage mismatch")
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


def expected_relation_certificate_subjects(csp) -> tuple[tuple[CertificateRelationKind, str], ...]:
    out: list[tuple[CertificateRelationKind, str]] = []
    for relation in csp.tetra_relations:
        out.append((CertificateRelationKind.TETRA_SITE, f"site:{int(relation.site)}"))
    for relation in csp.mark_relations():
        out.append((_relation_kind_for_mark_relation_name(relation.name), relation.subject))
    return tuple(out)


def certify_traversal_assignment(
    *,
    facts,
    skeleton,
    slots,
    assignment: TraversalAssignment,
    policy,
    semantics,
) -> tuple[RelationCertificate, ...]:
    validate_stereo_traversal_witness(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
    )
    validate_bounded_ring_labels(policy, slots, assignment.ring_labels)
    out: list[RelationCertificate] = [
        RelationCertificate(
            kind=CertificateRelationKind.GRAPH_COVERAGE,
            subject="molecule",
            detail=(
                "atoms",
                tuple(int(atom.id) for atom in facts.atoms),
                "tree_bonds",
                tuple(sorted(int(bond) for bond in skeleton.tree_bonds)),
                "ring_bonds",
                tuple(sorted(int(bond) for bond in skeleton.ring_bonds)),
                "roots",
                tuple(int(root) for root in skeleton.roots),
            ),
        ),
        RelationCertificate(
            kind=CertificateRelationKind.RING_LABELS,
            subject="ring_endpoints",
            detail=tuple(
                sorted(
                    (int(endpoint), label.value)
                    for endpoint, label in assignment.ring_labels.items()
                )
            ),
        ),
    ]
    out.extend(_atom_decode_certificates(facts, slots, assignment, semantics))
    out.extend(_bond_decode_certificates(facts, slots, assignment, semantics))
    out.extend(_ring_pair_decode_certificates(facts, slots, assignment, semantics))
    return tuple(out)


def _atom_decode_certificates(
    facts,
    slots,
    assignment: TraversalAssignment,
    semantics,
) -> tuple[RelationCertificate, ...]:
    incident_texts: dict[AtomId, list[object]] = {
        atom.id: [] for atom in facts.atoms
    }
    for slot in slots.bond_slots:
        incident_texts[slot.written_from].append(assignment.bond_text[slot.id])
        if slot.written_to is not None:
            incident_texts[slot.written_to].append(assignment.bond_text[slot.id])

    out: list[RelationCertificate] = []
    for atom in facts.atoms:
        token = assignment.tetra_tokens[atom.id]
        atom_text = assignment.atom_text[atom.id]
        if not semantics.atom_decode_ok(
            facts,
            atom.id,
            atom_text,
            token,
            tuple(incident_texts[atom.id]),
        ):
            raise ValueError(f"atom decode relation rejected atom {atom.id!r}")
        out.append(
            RelationCertificate(
                kind=CertificateRelationKind.ATOM_DECODE,
                subject=f"atom:{int(atom.id)}",
                detail=("text", atom_text.name, "token", token.value),
            )
        )
    return tuple(out)


def _bond_decode_certificates(
    facts,
    slots,
    assignment: TraversalAssignment,
    semantics,
) -> tuple[RelationCertificate, ...]:
    carrier_by_slot = carrier_slot_by_bond_slot(slots)
    out: list[RelationCertificate] = []
    for slot in slots.bond_slots:
        if slot.kind is not BondSlotKind.TREE:
            continue
        carrier = carrier_by_slot[slot.id]
        mark = assignment.direction_marks[carrier.id]
        bond_text = assignment.bond_text[slot.id]
        if not semantics.bond_decode_ok(
            facts,
            slot.bond,
            bond_text,
            mark,
        ):
            raise ValueError(f"bond decode relation rejected slot {slot.id!r}")
        out.append(
            RelationCertificate(
                kind=CertificateRelationKind.BOND_DECODE,
                subject=f"bond_slot:{int(slot.id)}",
                detail=(
                    "bond",
                    int(slot.bond),
                    "text",
                    bond_text.name,
                    "mark",
                    mark.value,
                ),
            )
        )
    return tuple(out)


def _ring_pair_decode_certificates(
    facts,
    slots,
    assignment: TraversalAssignment,
    semantics,
) -> tuple[RelationCertificate, ...]:
    carrier_by_slot = carrier_slot_by_bond_slot(slots)
    out: list[RelationCertificate] = []
    for bond, (left, right) in ring_bond_slots_by_bond(slots).items():
        left_carrier = carrier_by_slot[left.id]
        right_carrier = carrier_by_slot[right.id]
        left_mark = assignment.direction_marks[left_carrier.id]
        right_mark = assignment.direction_marks[right_carrier.id]
        if not semantics.ring_pair_decode_ok(
            facts,
            bond,
            assignment.bond_text[left.id],
            left_mark,
            assignment.bond_text[right.id],
            right_mark,
        ):
            raise ValueError(f"ring-pair decode relation rejected bond {bond!r}")
        out.append(
            RelationCertificate(
                kind=CertificateRelationKind.RING_PAIR_DECODE,
                subject=f"bond:{int(bond)}",
                detail=(
                    "slots",
                    (int(left.id), int(right.id)),
                    "marks",
                    (left_mark.value, right_mark.value),
                ),
            )
        )
    return tuple(out)


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
    expected_traversal = certify_traversal_assignment(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
    )
    if certificate.traversal_relation_certificates != expected_traversal:
        raise ValueError("traversal relation certificate coverage mismatch")

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


def witness_certificate_to_jsonable(cert: WitnessCertificate) -> dict[str, object]:
    return {
        "witness_id": cert.witness_id,
        "rendered": cert.rendered,
        "skeleton_key": _jsonable(cert.skeleton_key),
        "prefix_key": _jsonable(cert.prefix_key),
        "assignment_key": _jsonable(cert.assignment_key),
        "traversal_relation_certificates": [
            _relation_to_jsonable(relation)
            for relation in cert.traversal_relation_certificates
        ],
        "stereo_solution": _stereo_solution_cert_to_jsonable(cert.stereo_solution),
    }


def witness_certificate_from_jsonable(data: Mapping[str, object]) -> WitnessCertificate:
    stereo = data["stereo_solution"]
    if not isinstance(stereo, Mapping):
        raise TypeError("stereo_solution must be a mapping")
    return WitnessCertificate(
        witness_id=str(data["witness_id"]),
        rendered=str(data["rendered"]),
        skeleton_key=_tuple_from_jsonable(data["skeleton_key"]),
        prefix_key=_tuple_from_jsonable(data["prefix_key"]),
        assignment_key=_tuple_from_jsonable(data["assignment_key"]),
        traversal_relation_certificates=tuple(
            _relation_from_jsonable(item)
            for item in _require_list(data["traversal_relation_certificates"])
        ),
        stereo_solution=_stereo_solution_cert_from_jsonable(stereo),
    )


def manifest_to_jsonable(manifest: SupportEnumerationManifest) -> dict[str, object]:
    return {
        "skeleton_count": manifest.skeleton_count,
        "prefix_count": manifest.prefix_count,
        "csp_count": manifest.csp_count,
        "feasible_solution_count": manifest.feasible_solution_count,
        "selected_solution_count": manifest.selected_solution_count,
        "witness_count": manifest.witness_count,
        "support_count": manifest.support_count,
        "support_hash": manifest.support_hash,
        "witness_hash": manifest.witness_hash,
    }


def manifest_from_jsonable(data: Mapping[str, object]) -> SupportEnumerationManifest:
    return SupportEnumerationManifest(
        skeleton_count=int(data["skeleton_count"]),
        prefix_count=int(data["prefix_count"]),
        csp_count=int(data["csp_count"]),
        feasible_solution_count=int(data["feasible_solution_count"]),
        selected_solution_count=int(data["selected_solution_count"]),
        witness_count=int(data["witness_count"]),
        support_count=int(data["support_count"]),
        support_hash=str(data["support_hash"]),
        witness_hash=str(data["witness_hash"]),
    )


def _relation_kind_for_mark_relation_name(name: str) -> CertificateRelationKind:
    if name == "tree_bond_decode":
        return CertificateRelationKind.BOND_DECODE
    if name == "ring_pair_decode":
        return CertificateRelationKind.RING_PAIR_DECODE
    if name == "directional_site":
        return CertificateRelationKind.DIRECTIONAL_SITE
    if name == "no_accidental_stereo":
        return CertificateRelationKind.NO_ACCIDENTAL_STEREO
    return CertificateRelationKind.NO_ACCIDENTAL_STEREO


def _relation_to_jsonable(relation: RelationCertificate) -> dict[str, object]:
    return {
        "kind": relation.kind.value,
        "subject": relation.subject,
        "detail": _jsonable(relation.detail),
    }


def _relation_from_jsonable(data: object) -> RelationCertificate:
    if not isinstance(data, Mapping):
        raise TypeError("relation certificate must be a mapping")
    return RelationCertificate(
        kind=CertificateRelationKind(str(data["kind"])),
        subject=str(data["subject"]),
        detail=_tuple_from_jsonable(data["detail"]),
    )


def _stereo_solution_cert_to_jsonable(
    cert: StereoSolutionCertificate,
) -> dict[str, object]:
    return {
        "tetra_tokens": [
            [int(atom), token.value]
            for atom, token in cert.tetra_tokens
        ],
        "direction_marks": [
            [int(carrier), mark.value]
            for carrier, mark in cert.direction_marks
        ],
        "relation_certificates": [
            _relation_to_jsonable(relation)
            for relation in cert.relation_certificates
        ],
        "annotation_certificate": _annotation_cert_to_jsonable(
            cert.annotation_certificate,
        ),
    }


def _stereo_solution_cert_from_jsonable(
    data: Mapping[str, object],
) -> StereoSolutionCertificate:
    return StereoSolutionCertificate(
        tetra_tokens=tuple(
            (AtomId(int(item[0])), TetraToken(str(item[1])))
            for item in _require_list(data["tetra_tokens"])
        ),
        direction_marks=tuple(
            (CarrierSlotId(int(item[0])), DirectionMark(int(item[1])))
            for item in _require_list(data["direction_marks"])
        ),
        relation_certificates=tuple(
            _relation_from_jsonable(item)
            for item in _require_list(data["relation_certificates"])
        ),
        annotation_certificate=_annotation_cert_from_jsonable(
            _require_mapping(data["annotation_certificate"]),
        ),
    )


def _annotation_cert_to_jsonable(
    cert: AnnotationSelectionCertificate,
) -> dict[str, object]:
    return {
        "mode": cert.mode.value,
        "selected_support": _support_to_jsonable(cert.selected_support),
        "feasible_supports": [
            _support_to_jsonable(support)
            for support in sorted(
                cert.feasible_supports,
                key=lambda support: tuple(sorted(int(c) for c in support)),
            )
        ],
        "selected_supports": [
            _support_to_jsonable(support)
            for support in sorted(
                cert.selected_supports,
                key=lambda support: tuple(sorted(int(c) for c in support)),
            )
        ],
        "canonical_key": None
        if cert.canonical_key is None
        else _jsonable(cert.canonical_key),
        "selected_solution_key": None
        if cert.selected_solution_key is None
        else _jsonable(cert.selected_solution_key),
        "selected_solution_keys": [
            _jsonable(key)
            for key in sorted(cert.selected_solution_keys, key=repr)
        ],
        "feasible_solution_count": cert.feasible_solution_count,
        "selected_solution_count": cert.selected_solution_count,
    }


def _annotation_cert_from_jsonable(
    data: Mapping[str, object],
) -> AnnotationSelectionCertificate:
    canonical_key = data["canonical_key"]
    selected_solution_key = data["selected_solution_key"]
    return AnnotationSelectionCertificate(
        mode=AnnotationMode(str(data["mode"])),
        selected_support=_support_from_jsonable(data["selected_support"]),
        feasible_supports=frozenset(
            _support_from_jsonable(item)
            for item in _require_list(data["feasible_supports"])
        ),
        selected_supports=frozenset(
            _support_from_jsonable(item)
            for item in _require_list(data["selected_supports"])
        ),
        canonical_key=None
        if canonical_key is None
        else _tuple_from_jsonable(canonical_key),
        selected_solution_key=None
        if selected_solution_key is None
        else _tuple_from_jsonable(selected_solution_key),
        selected_solution_keys=frozenset(
            _tuple_from_jsonable(item)
            for item in _require_list(data["selected_solution_keys"])
        ),
        feasible_solution_count=int(data["feasible_solution_count"]),
        selected_solution_count=int(data["selected_solution_count"]),
    )


def _support_to_jsonable(support: frozenset[CarrierSlotId]) -> list[int]:
    return sorted(int(carrier) for carrier in support)


def _support_from_jsonable(data: object) -> frozenset[CarrierSlotId]:
    return frozenset(CarrierSlotId(int(item)) for item in _require_list(data))


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, frozenset):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, Enum):
        return value.value
    return value


def _tuple_from_jsonable(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(
            _tuple_from_jsonable(item) if isinstance(item, list) else item
            for item in value
        )
    raise TypeError(f"expected JSON list for tuple value: {value!r}")


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping: {value!r}")
    return value


__all__ = (
    "AnnotationSelectionCertificate",
    "CertificateRelationKind",
    "RelationCertificate",
    "StereoSolutionCertificate",
    "SupportEnumerationManifest",
    "WitnessCertificate",
    "certify_traversal_assignment",
    "expected_relation_certificate_subjects",
    "manifest_from_jsonable",
    "manifest_to_jsonable",
    "validate_annotation_selection_certificate",
    "validate_stereo_solution_certificate",
    "validate_witness_certificate",
    "witness_certificate_from_jsonable",
    "witness_certificate_to_jsonable",
)
