"""Online stereo witness enumeration for South Star 1."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetraValue
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .online_decisions import DecisionPathFilter
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_traversal import OnlineAtomEvent
from .online_traversal import OnlineBranchClose
from .online_traversal import OnlineBranchOpen
from .online_traversal import OnlineDotEvent
from .online_traversal import OnlineRingEndpointEvent
from .online_traversal import OnlineTraversalTrace
from .online_traversal import OnlineTreeBondEvent
from .online_traversal import iter_online_traversal_traces
from .online_traversal import trace_to_skeleton_like_key
from .policy import AnnotationMode
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .online_render_sink import OnlineRenderSink
from .online_render_sink import OnlineStringBuffer
from .residual_constraints import DirectionalCarrierResidual
from .residual_constraints import DirectionalResidualFactor
from .residual_constraints import ResidualStore
from .residual_constraints import direction_var
from .semantics import ParserSemantics
from .stereo_templates import DirectionalTemplate
from .stereo_templates import StereoTemplateBundle
from .stereo_templates import TetraTemplate
from .stereo_templates import build_stereo_templates


@dataclass(frozen=True, slots=True)
class OnlineWitness:
    rendered: str
    traversal_key: tuple[object, ...]
    annotation_count: int


@dataclass(frozen=True, slots=True)
class OnlineBondSlot:
    id: int
    bond: BondId
    kind: Literal["tree", "ring_endpoint"]
    written_from: AtomId
    written_to: AtomId | None
    ring_endpoint_id: int | None
    syntax_position: int


@dataclass(frozen=True, slots=True)
class OnlineRingEndpointSlot:
    id: int
    bond: BondId
    atom: AtomId
    other_atom: AtomId
    bond_slot: int
    syntax_position: int


@dataclass(frozen=True, slots=True)
class OnlineCarrierSlot:
    id: int
    bond_slot: int
    bond: BondId
    written_from: AtomId
    written_to: AtomId | None


@dataclass(frozen=True, slots=True)
class OnlineSlotView:
    bond_slots: tuple[OnlineBondSlot, ...]
    carrier_slots: tuple[OnlineCarrierSlot, ...]
    ring_endpoints: tuple[OnlineRingEndpointSlot, ...]


@dataclass(frozen=True, slots=True)
class _PrefixChoice:
    atom_text: dict[AtomId, AtomTextChoice]
    bond_text: dict[int, BondTextChoice]
    ring_labels: dict[int, RingLabel]


@dataclass(frozen=True, slots=True)
class _DirectionalCandidate:
    marks: tuple[tuple[int, DirectionMark], ...]
    support: frozenset[int]
    annotation_count: int
    decision_path: OnlineDecisionPath | None


def iter_online_stereo_witness_strings(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> Iterator[str]:
    for witness in iter_online_stereo_witnesses(
        facts=facts,
        policy=policy,
        semantics=semantics,
    ):
        yield witness.rendered


def iter_online_stereo_witnesses(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> Iterator[OnlineWitness]:
    yield from iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=OnlineStringBuffer,
    )


def iter_online_stereo_witnesses_with_sink(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    sink_factory: Callable[[], OnlineRenderSink],
    decision_recorder: OnlineDecisionRecorder | None = None,
    decision_filter: DecisionPathFilter | None = None,
    templates: StereoTemplateBundle | None = None,
    rooted_at_atom: AtomId | None = None,
) -> Iterator[OnlineWitness]:
    facts.validate()
    policy.validate_for_facts(facts)
    _validate_annotation_mode(policy)
    templates = templates if templates is not None else build_stereo_templates(facts)

    for trace in iter_online_traversal_traces(
        facts=facts,
        policy=policy,
        rooted_at_atom=rooted_at_atom,
    ):
        decision_checkpoint = _push_decision(
            decision_recorder,
            decision_filter,
            OnlineDecision("traversal", (trace_to_skeleton_like_key(trace),)),
        )
        if decision_checkpoint is None:
            continue
        try:
            yield from _iter_witnesses_for_trace(
                facts=facts,
                policy=policy,
                semantics=semantics,
                templates=templates,
                trace=trace,
                sink_factory=sink_factory,
                decision_recorder=decision_recorder,
                decision_filter=decision_filter,
            )
        finally:
            _rollback_decision(decision_recorder, decision_checkpoint)


def online_local_tetra_order(
    *,
    facts: MoleculeFacts,
    trace: OnlineTraversalTrace,
    atom: AtomId,
    template: TetraTemplate,
) -> tuple[OccurrenceId, ...]:
    occurrence_by_atom = _neighbor_occurrences_by_atom(facts, template)
    implicit_h = tuple(
        occurrence.id
        for occurrence in facts.ligand_occurrences
        if occurrence.site == template.site and occurrence.kind is LigandKind.IMPLICIT_H
    )
    parent = dict(trace.parent)[atom]
    order: list[OccurrenceId] = []
    if parent is not None and parent in occurrence_by_atom:
        order.append(occurrence_by_atom[parent])
    for event in _local_events_for_atom(trace, atom):
        if isinstance(event, OnlineTreeBondEvent):
            occurrence = occurrence_by_atom.get(event.written_to)
        elif isinstance(event, OnlineRingEndpointEvent):
            occurrence = occurrence_by_atom.get(event.other_atom)
        else:
            continue
        if occurrence is not None:
            order.append(occurrence)
    order.extend(implicit_h)
    return tuple(order)


def directional_templates_for_carrier(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    carrier: OnlineCarrierSlot,
) -> tuple[DirectionalTemplate, ...]:
    return tuple(
        template
        for template in templates.directional
        if carrier.bond != template.center_bond
        and carrier.bond in _directional_template_substituent_bonds(facts, template)
    )


def online_slot_view_for_trace(trace: OnlineTraversalTrace) -> OnlineSlotView:
    return _slot_view_for_trace(trace)


def online_slot_key(slots: OnlineSlotView) -> tuple[object, ...]:
    return (
        (),
        tuple(
            (
                slot.id,
                int(slot.bond),
                slot.kind,
                int(slot.written_from),
                None if slot.written_to is None else int(slot.written_to),
                slot.syntax_position,
                slot.ring_endpoint_id,
            )
            for slot in slots.bond_slots
        ),
        tuple(
            (
                endpoint.id,
                int(endpoint.bond),
                int(endpoint.atom),
                int(endpoint.other_atom),
                endpoint.bond_slot,
                endpoint.syntax_position,
            )
            for endpoint in slots.ring_endpoints
        ),
        tuple(
            (
                carrier.id,
                carrier.bond_slot,
                int(carrier.bond),
                int(carrier.written_from),
                None if carrier.written_to is None else int(carrier.written_to),
            )
            for carrier in slots.carrier_slots
        ),
    )


def iter_online_ring_label_assignments(
    *,
    trace: OnlineTraversalTrace,
    policy: SmilesPolicy,
) -> Iterator[dict[int, RingLabel]]:
    slots = _slot_view_for_trace(trace)
    if not slots.ring_endpoints:
        yield {}
        return
    intervals = _ring_intervals(slots)
    labels = policy.ring_labels
    out: dict[int, RingLabel] = {}
    chosen: list[tuple[int, int, RingLabel]] = []

    def active_labels_at(position: int) -> set[RingLabel]:
        return {
            label
            for start, end, label in chosen
            if start < position < end
        }

    def rec(index: int) -> Iterator[dict[int, RingLabel]]:
        if index == len(intervals):
            yield dict(out)
            return
        endpoint_1, endpoint_2, start, end = intervals[index]
        active = active_labels_at(start)
        candidates = tuple(label for label in labels if label not in active)
        if policy.least_free_ring_labels:
            if not candidates:
                return
            candidates = (min(candidates, key=lambda label: label.value),)
        for label in candidates:
            out[endpoint_1] = label
            out[endpoint_2] = label
            chosen.append((start, end, label))
            yield from rec(index + 1)
            chosen.pop()
            del out[endpoint_1]
            del out[endpoint_2]

    yield from rec(0)


def _iter_witnesses_for_trace(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    templates: StereoTemplateBundle,
    trace: OnlineTraversalTrace,
    sink_factory: Callable[[], OnlineRenderSink],
    decision_recorder: OnlineDecisionRecorder | None,
    decision_filter: DecisionPathFilter | None,
) -> Iterator[OnlineWitness]:
    slots = _slot_view_for_trace(trace)
    atom_domains = tuple(
        (atom.id, policy.atom_text_domain(facts, atom.id))
        for atom in facts.atoms
    )
    bond_domains = tuple(
        (
            slot.id,
            policy.bond_text_domain(facts, slot.bond, slot_kind=slot.kind),
        )
        for slot in slots.bond_slots
    )

    for ring_labels in iter_online_ring_label_assignments(trace=trace, policy=policy):
        decision_checkpoint = _push_decision(
            decision_recorder,
            decision_filter,
            OnlineDecision("ring_labels", _ring_label_decision_value(ring_labels)),
        )
        if decision_checkpoint is None:
            continue
        try:
            for atom_text in _dict_product(atom_domains):
                atom_decision_checkpoint = _push_decision(
                    decision_recorder,
                    decision_filter,
                    OnlineDecision("atom_text", _atom_text_decision_value(atom_text)),
                )
                if atom_decision_checkpoint is None:
                    continue
                try:
                    tetra_tokens = _forced_tetra_tokens(
                        facts=facts,
                        trace=trace,
                        templates=templates,
                        atom_text=atom_text,
                    )
                    if tetra_tokens is None:
                        continue
                    for bond_text in _dict_product(bond_domains):
                        bond_decision_checkpoint = _push_decision(
                            decision_recorder,
                            decision_filter,
                            OnlineDecision("bond_text", _bond_text_decision_value(bond_text)),
                        )
                        if bond_decision_checkpoint is None:
                            continue
                        try:
                            prefix = _PrefixChoice(
                                atom_text=atom_text,
                                bond_text=bond_text,
                                ring_labels=ring_labels,
                            )
                            candidates = tuple(
                                _iter_directional_candidates(
                                    facts=facts,
                                    policy=policy,
                                    semantics=semantics,
                                    templates=templates,
                                    trace=trace,
                                    slots=slots,
                                    prefix=prefix,
                                    tetra_tokens=tetra_tokens,
                                    sink_factory=sink_factory,
                                    decision_recorder=decision_recorder,
                                    decision_filter=decision_filter,
                                )
                            )
                            if policy.annotation_mode is AnnotationMode.SUPPORT_MAXIMAL:
                                candidates = tuple(_support_maximal_candidates(candidates))
                            for candidate in candidates:
                                rendered = _render_directional_candidate(
                                    trace=trace,
                                    slots=slots,
                                    prefix=prefix,
                                    tetra_tokens=tetra_tokens,
                                    candidate=candidate,
                                    sink_factory=sink_factory,
                                    decision_recorder=decision_recorder,
                                )
                                if rendered is None:
                                    continue
                                yield OnlineWitness(
                                    rendered=rendered,
                                    traversal_key=trace_to_skeleton_like_key(trace),
                                    annotation_count=candidate.annotation_count,
                                )
                        finally:
                            _rollback_decision(decision_recorder, bond_decision_checkpoint)
                finally:
                    _rollback_decision(decision_recorder, atom_decision_checkpoint)
        finally:
            _rollback_decision(decision_recorder, decision_checkpoint)


def _iter_directional_candidates(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    templates: StereoTemplateBundle,
    trace: OnlineTraversalTrace,
    slots: OnlineSlotView,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, TetraToken],
    sink_factory: Callable[[], OnlineRenderSink],
    decision_recorder: OnlineDecisionRecorder | None,
    decision_filter: DecisionPathFilter | None,
) -> Iterator[_DirectionalCandidate]:
    del policy, sink_factory
    carrier_by_slot = {carrier.bond_slot: carrier for carrier in slots.carrier_slots}
    domains = _direction_domains(
        facts=facts,
        templates=templates,
        slots=slots,
        prefix=prefix,
    )
    carrier_models_by_site = _directional_models_by_site(
        facts=facts,
        templates=templates,
        slots=slots,
    )

    carriers = tuple(carrier.id for carrier in slots.carrier_slots)
    store = ResidualStore()
    for carrier_id in carriers:
        store.add_var(direction_var(carrier_id), domains[carrier_id])
    factor_ids = []
    for template in templates.directional:
        scope_models = carrier_models_by_site[template.site]
        factor = DirectionalResidualFactor(
            scope=tuple(scope_models),
            status=template.status,
            target=template.target,
            carrier_models=scope_models,
        )
        factor_ids.append(store.add_factor(factor))

    marks: dict[int, DirectionMark] = {}

    def rec(index: int) -> Iterator[_DirectionalCandidate]:
        if index == len(carriers):
            if not _bond_decode_ok(
                facts=facts,
                semantics=semantics,
                slots=slots,
                prefix=prefix,
                marks=marks,
            ):
                return
            if not all(store.close_factor(factor_id) for factor_id in factor_ids):
                return
            support = frozenset(
                carrier_id
                for carrier_id, mark in marks.items()
                if mark is not DirectionMark.ABSENT
            )
            yield _DirectionalCandidate(
                marks=tuple(sorted(marks.items())),
                support=support,
                annotation_count=len(support),
                decision_path=decision_recorder.path() if decision_recorder is not None else None,
            )
            return

        carrier_id = carriers[index]
        var = direction_var(carrier_id)
        for mark in domains[carrier_id]:
            decision_checkpoint = _push_decision(
                decision_recorder,
                decision_filter,
                OnlineDecision("direction_mark", (carrier_id, mark.name)),
            )
            if decision_checkpoint is None:
                continue
            checkpoint = store.checkpoint()
            try:
                if store.assign(var, mark):
                    marks[carrier_id] = mark
                    yield from rec(index + 1)
                    del marks[carrier_id]
            finally:
                store.rollback(checkpoint)
                _rollback_decision(decision_recorder, decision_checkpoint)

    yield from rec(0)


def _render_directional_candidate(
    *,
    trace: OnlineTraversalTrace,
    slots: OnlineSlotView,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, TetraToken],
    candidate: _DirectionalCandidate,
    sink_factory: Callable[[], OnlineRenderSink],
    decision_recorder: OnlineDecisionRecorder | None,
) -> str | None:
    previous_decision_path = decision_recorder.path() if decision_recorder is not None else None
    if decision_recorder is not None and candidate.decision_path is not None:
        decision_recorder.restore_path(candidate.decision_path)
    sink = sink_factory()
    checkpoint = sink.checkpoint()
    try:
        if not _render_online_to_sink(
            trace=trace,
            slots=slots,
            prefix=prefix,
            tetra_tokens=tetra_tokens,
            marks=dict(candidate.marks),
            sink=sink,
        ):
            sink.rollback(checkpoint)
            return None
        if not sink.complete():
            sink.rollback(checkpoint)
            return None
        rendered = sink.value()
        sink.rollback(checkpoint)
        return rendered
    finally:
        if decision_recorder is not None and previous_decision_path is not None:
            decision_recorder.restore_path(previous_decision_path)


def _slot_view_for_trace(trace: OnlineTraversalTrace) -> OnlineSlotView:
    bond_slots: list[OnlineBondSlot] = []
    carrier_slots: list[OnlineCarrierSlot] = []
    ring_endpoints: list[OnlineRingEndpointSlot] = []
    syntax_position = 0
    for event in trace.events:
        if isinstance(event, OnlineTreeBondEvent):
            slot_id = len(bond_slots)
            bond_slots.append(
                OnlineBondSlot(
                    id=slot_id,
                    bond=event.bond,
                    kind="tree",
                    written_from=event.written_from,
                    written_to=event.written_to,
                    ring_endpoint_id=None,
                    syntax_position=syntax_position,
                )
            )
            carrier_slots.append(
                OnlineCarrierSlot(
                    id=len(carrier_slots),
                    bond_slot=slot_id,
                    bond=event.bond,
                    written_from=event.written_from,
                    written_to=event.written_to,
                )
            )
            syntax_position += 1
            continue
        if isinstance(event, OnlineRingEndpointEvent):
            slot_id = len(bond_slots)
            endpoint_id = len(ring_endpoints)
            if event.syntax_position != syntax_position:
                raise ValueError("online ring endpoint syntax position mismatch")
            bond_slots.append(
                OnlineBondSlot(
                    id=slot_id,
                    bond=event.bond,
                    kind="ring_endpoint",
                    written_from=event.at,
                    written_to=event.other_atom,
                    ring_endpoint_id=endpoint_id,
                    syntax_position=syntax_position,
                )
            )
            ring_endpoints.append(
                OnlineRingEndpointSlot(
                    id=endpoint_id,
                    bond=event.bond,
                    atom=event.at,
                    other_atom=event.other_atom,
                    bond_slot=slot_id,
                    syntax_position=syntax_position,
                )
            )
            carrier_slots.append(
                OnlineCarrierSlot(
                    id=len(carrier_slots),
                    bond_slot=slot_id,
                    bond=event.bond,
                    written_from=event.at,
                    written_to=event.other_atom,
                )
            )
            syntax_position += 1
    return OnlineSlotView(
        bond_slots=tuple(bond_slots),
        carrier_slots=tuple(carrier_slots),
        ring_endpoints=tuple(ring_endpoints),
    )


def _forced_tetra_tokens(
    *,
    facts: MoleculeFacts,
    trace: OnlineTraversalTrace,
    templates: StereoTemplateBundle,
    atom_text: dict[AtomId, AtomTextChoice],
) -> dict[AtomId, TetraToken] | None:
    template_by_center = {template.center: template for template in templates.tetrahedral}
    out: dict[AtomId, TetraToken] = {}
    for atom in (atom.id for atom in facts.atoms):
        template = template_by_center.get(atom)
        if template is None:
            token = TetraToken.NONE
        else:
            token = _forced_tetra_token(
                facts=facts,
                trace=trace,
                atom=atom,
                template=template,
            )
            if token is None:
                return None
        if not atom_text[atom].permits(token):
            return None
        out[atom] = token
    return out


def _forced_tetra_token(
    *,
    facts: MoleculeFacts,
    trace: OnlineTraversalTrace,
    atom: AtomId,
    template: TetraTemplate,
) -> TetraToken | None:
    if template.status is SiteStatus.UNSPECIFIED:
        return TetraToken.NONE
    local_order = online_local_tetra_order(
        facts=facts,
        trace=trace,
        atom=atom,
        template=template,
    )
    if set(local_order) != set(template.reference_order):
        return None
    if len(local_order) != len(template.reference_order):
        return None
    is_even = _is_even_permutation(
        tuple(template.reference_order.index(item) for item in local_order)
    )
    if template.target is TetraValue.PLUS:
        return TetraToken.AT if is_even else TetraToken.ATAT
    if template.target is TetraValue.MINUS:
        return TetraToken.ATAT if is_even else TetraToken.AT
    return None


def _direction_domains(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    slots: OnlineSlotView,
    prefix: _PrefixChoice,
) -> dict[int, tuple[DirectionMark, ...]]:
    eligible = _eligible_marker_carriers(facts=facts, templates=templates, slots=slots)
    domains: dict[int, tuple[DirectionMark, ...]] = {}
    for carrier in slots.carrier_slots:
        choice = prefix.bond_text[carrier.bond_slot]
        if carrier.id in eligible and choice.permits_direction:
            domains[carrier.id] = (
                DirectionMark.ABSENT,
                DirectionMark.FWD,
                DirectionMark.REV,
            )
        else:
            domains[carrier.id] = (DirectionMark.ABSENT,)
    return domains


def _eligible_marker_carriers(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    slots: OnlineSlotView,
) -> frozenset[int]:
    out: set[int] = set()
    for template in templates.directional:
        if template.status is not SiteStatus.SPECIFIED:
            continue
        substituent_bonds = _directional_template_substituent_bonds(facts, template)
        for carrier in slots.carrier_slots:
            if carrier.bond != template.center_bond and carrier.bond in substituent_bonds:
                out.add(carrier.id)
    return frozenset(out)


def _directional_models_by_site(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    slots: OnlineSlotView,
) -> dict[object, dict[object, DirectionalCarrierResidual]]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    out: dict[object, dict[object, DirectionalCarrierResidual]] = {}
    for template in templates.directional:
        left_reference, right_reference = _directional_reference_pair(template)
        left_by_bond = _neighbor_ligands_by_bond(
            occurrence_by_id,
            template.left_ligands,
        )
        right_by_bond = _neighbor_ligands_by_bond(
            occurrence_by_id,
            template.right_ligands,
        )
        models: dict[object, DirectionalCarrierResidual] = {}
        for carrier in slots.carrier_slots:
            if carrier.bond == template.center_bond:
                continue
            var = direction_var(carrier.id)
            if carrier.bond in left_by_bond:
                models[var] = DirectionalCarrierResidual(
                    var=var,
                    side="left",
                    orientation=_carrier_orientation(carrier, template.left_endpoint),
                    ligand_factor=_ligand_factor(
                        left_by_bond[carrier.bond],
                        reference=left_reference,
                        side_ligands=template.left_ligands,
                    ),
                )
                continue
            if carrier.bond in right_by_bond:
                models[var] = DirectionalCarrierResidual(
                    var=var,
                    side="right",
                    orientation=_carrier_orientation(carrier, template.right_endpoint),
                    ligand_factor=_ligand_factor(
                        right_by_bond[carrier.bond],
                        reference=right_reference,
                        side_ligands=template.right_ligands,
                    ),
                )
        out[template.site] = models
    return out


def _bond_decode_ok(
    *,
    facts: MoleculeFacts,
    semantics: ParserSemantics,
    slots: OnlineSlotView,
    prefix: _PrefixChoice,
    marks: dict[int, DirectionMark],
) -> bool:
    carriers_by_slot = {carrier.bond_slot: carrier for carrier in slots.carrier_slots}
    tree_slots = tuple(slot for slot in slots.bond_slots if slot.kind == "tree")
    ring_slots_by_bond: dict[BondId, list[OnlineBondSlot]] = {}
    for slot in slots.bond_slots:
        if slot.kind == "ring_endpoint":
            ring_slots_by_bond.setdefault(slot.bond, []).append(slot)
    for slot in tree_slots:
        carrier = carriers_by_slot[slot.id]
        if not semantics.bond_decode_ok(
            facts,
            slot.bond,
            prefix.bond_text[slot.id],
            marks[carrier.id],
        ):
            return False
    for bond, endpoints in ring_slots_by_bond.items():
        if len(endpoints) != 2:
            return False
        left, right = endpoints
        left_carrier = carriers_by_slot[left.id]
        right_carrier = carriers_by_slot[right.id]
        if not semantics.ring_pair_decode_ok(
            facts,
            bond,
            prefix.bond_text[left.id],
            marks[left_carrier.id],
            prefix.bond_text[right.id],
            marks[right_carrier.id],
        ):
            return False
    return True


def _render_online_to_sink(
    *,
    trace: OnlineTraversalTrace,
    slots: OnlineSlotView,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, TetraToken],
    marks: dict[int, DirectionMark],
    sink: OnlineRenderSink,
) -> bool:
    bond_slot_by_event: dict[tuple[object, ...], OnlineBondSlot] = {}
    for slot in slots.bond_slots:
        key = (
            slot.kind,
            slot.bond,
            slot.written_from,
            slot.written_to,
            slot.ring_endpoint_id,
        )
        bond_slot_by_event[key] = slot
    ring_label_by_endpoint = {
        endpoint.id: prefix.ring_labels[endpoint.id]
        for endpoint in slots.ring_endpoints
    }
    for event in trace.events:
        if isinstance(event, OnlineAtomEvent):
            text = prefix.atom_text[event.atom].render(tetra_tokens[event.atom])
            if not sink.append(text, token_text=text):
                return False
            continue
        if isinstance(event, OnlineTreeBondEvent):
            key = ("tree", event.bond, event.written_from, event.written_to, None)
            text = _render_bond_slot(bond_slot_by_event[key], prefix, marks)
            if not sink.append(text, token_text=text or None):
                return False
            continue
        if isinstance(event, OnlineRingEndpointEvent):
            slot = next(
                item
                for item in slots.bond_slots
                if item.kind == "ring_endpoint"
                and item.bond == event.bond
                and item.written_from == event.at
                and item.written_to == event.other_atom
                and item.syntax_position == event.syntax_position
            )
            text = _render_bond_slot(slot, prefix, marks)
            if not sink.append(text, token_text=text or None):
                return False
            if slot.ring_endpoint_id is None:
                raise ValueError("ring endpoint slot lacks endpoint id")
            label_text = ring_label_by_endpoint[slot.ring_endpoint_id].text()
            if not sink.append(label_text, token_text=label_text):
                return False
            continue
        if isinstance(event, OnlineBranchOpen):
            if not sink.append("(", token_text="("):
                return False
            continue
        if isinstance(event, OnlineBranchClose):
            if not sink.append(")", token_text=")"):
                return False
            continue
        if isinstance(event, OnlineDotEvent):
            if not sink.append(".", token_text="."):
                return False
            continue
        raise TypeError(event)
    return True


def _render_bond_slot(
    slot: OnlineBondSlot,
    prefix: _PrefixChoice,
    marks: dict[int, DirectionMark],
) -> str:
    choice = prefix.bond_text[slot.id]
    mark = marks[slot.id]
    if mark is DirectionMark.ABSENT:
        return choice.base_text
    if not choice.permits_direction:
        raise ValueError("direction mark is outside bond render policy")
    if mark is DirectionMark.FWD:
        return "/"
    if mark is DirectionMark.REV:
        return "\\"
    raise ValueError(f"unknown direction mark: {mark!r}")


def _support_maximal_candidates(
    candidates: tuple[_DirectionalCandidate, ...],
) -> Iterator[_DirectionalCandidate]:
    for candidate in candidates:
        if any(candidate.support < other.support for other in candidates):
            continue
        yield candidate


def _push_decision(
    recorder: OnlineDecisionRecorder | None,
    path_filter: DecisionPathFilter | None,
    decision: OnlineDecision,
) -> int | None:
    if recorder is None:
        return 0
    checkpoint = recorder.checkpoint()
    recorder.push(decision)
    if path_filter is not None and not path_filter.allows_prefix(recorder.path()):
        recorder.rollback(checkpoint)
        return None
    return checkpoint


def _rollback_decision(
    recorder: OnlineDecisionRecorder | None,
    checkpoint: int,
) -> None:
    if recorder is not None:
        recorder.rollback(checkpoint)


def _ring_label_decision_value(ring_labels: dict[int, RingLabel]) -> tuple[object, ...]:
    return tuple(sorted((endpoint, label.value) for endpoint, label in ring_labels.items()))


def _atom_text_decision_value(
    atom_text: dict[AtomId, AtomTextChoice],
) -> tuple[object, ...]:
    return tuple(sorted((int(atom), choice.name) for atom, choice in atom_text.items()))


def _bond_text_decision_value(
    bond_text: dict[int, BondTextChoice],
) -> tuple[object, ...]:
    return tuple(sorted((slot_id, choice.name) for slot_id, choice in bond_text.items()))


def _dict_product(domains):
    if not domains:
        yield {}
        return
    keys = tuple(key for key, _ in domains)
    value_domains = tuple(values for _, values in domains)
    if any(not values for values in value_domains):
        return
    for values in product(*value_domains):
        yield dict(zip(keys, values, strict=True))


def _local_events_for_atom(
    trace: OnlineTraversalTrace,
    atom: AtomId,
) -> tuple[object, ...]:
    out = []
    for event in trace.events:
        if isinstance(event, OnlineTreeBondEvent) and event.written_from == atom:
            out.append(event)
        elif isinstance(event, OnlineRingEndpointEvent) and event.at == atom:
            out.append(event)
    return tuple(out)


def _neighbor_occurrences_by_atom(
    facts: MoleculeFacts,
    template: TetraTemplate,
) -> dict[AtomId, OccurrenceId]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    out = {}
    for occurrence_id in template.ligand_occurrences:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.atom is None:
            raise ValueError("neighbor occurrence lacks atom")
        out[occurrence.atom] = occurrence.id
    return out


def _directional_template_substituent_bonds(
    facts: MoleculeFacts,
    template: DirectionalTemplate,
) -> frozenset[BondId]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    return _directional_template_substituent_bonds_from_occurrences(
        template,
        occurrence_by_id,
    )


def _directional_template_substituent_bonds_from_occurrences(
    template: DirectionalTemplate,
    occurrence_by_id,
) -> frozenset[BondId]:
    if occurrence_by_id is None:
        raise ValueError("directional substituent bonds require ligand occurrences")
    bonds: set[BondId] = set()
    for ligand_id in template.left_ligands + template.right_ligands:
        occurrence = occurrence_by_id[ligand_id]
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            if occurrence.bond is None:
                raise ValueError("neighbor occurrence lacks bond")
            bonds.add(occurrence.bond)
    return frozenset(bonds)


def _neighbor_ligands_by_bond(
    occurrence_by_id,
    ligand_ids: tuple[OccurrenceId, ...],
) -> dict[BondId, OccurrenceId]:
    out = {}
    for ligand_id in ligand_ids:
        occurrence = occurrence_by_id[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            raise ValueError("neighbor occurrence lacks bond")
        out[occurrence.bond] = ligand_id
    return out


def _directional_reference_pair(
    template: DirectionalTemplate,
) -> tuple[OccurrenceId, OccurrenceId]:
    if template.reference_pair is not None:
        return template.reference_pair
    if template.status is SiteStatus.SPECIFIED:
        raise ValueError("specified directional site lacks reference pair")
    return (
        min(template.left_ligands, key=int),
        min(template.right_ligands, key=int),
    )


def _carrier_orientation(carrier: OnlineCarrierSlot, endpoint: AtomId) -> Literal[-1, 1]:
    if carrier.written_from == endpoint:
        return 1
    if carrier.written_to == endpoint:
        return -1
    raise ValueError("carrier is not incident to directional endpoint")


def _ligand_factor(
    occurrence: OccurrenceId,
    *,
    reference: OccurrenceId,
    side_ligands: tuple[OccurrenceId, ...],
) -> Literal[-1, 1]:
    if occurrence == reference:
        return 1
    if occurrence not in side_ligands:
        raise ValueError("occurrence is not on directional side")
    return -1


def _ring_intervals(slots: OnlineSlotView) -> tuple[tuple[int, int, int, int], ...]:
    by_bond: dict[BondId, list[OnlineRingEndpointSlot]] = {}
    for endpoint in slots.ring_endpoints:
        by_bond.setdefault(endpoint.bond, []).append(endpoint)
    intervals = []
    for endpoints in by_bond.values():
        if len(endpoints) != 2:
            raise ValueError("ring bond does not have two endpoints")
        left, right = sorted(endpoints, key=lambda item: item.syntax_position)
        intervals.append((left.id, right.id, left.syntax_position, right.syntax_position))
    return tuple(sorted(intervals, key=lambda item: (item[2], item[0])))


def _is_even_permutation(indices: tuple[int, ...]) -> bool:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return inversions % 2 == 0


def _validate_annotation_mode(policy: SmilesPolicy) -> None:
    if policy.annotation_mode in {AnnotationMode.HARD, AnnotationMode.SUPPORT_MAXIMAL}:
        return
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        "online enumerator currently supports HARD and SUPPORT_MAXIMAL annotation modes",
    )


__all__ = (
    "OnlineBondSlot",
    "OnlineCarrierSlot",
    "OnlineRingEndpointSlot",
    "OnlineSlotView",
    "OnlineWitness",
    "directional_templates_for_carrier",
    "iter_online_ring_label_assignments",
    "iter_online_stereo_witness_strings",
    "iter_online_stereo_witnesses",
    "iter_online_stereo_witnesses_with_sink",
    "online_local_tetra_order",
    "online_slot_key",
    "online_slot_view_for_trace",
)
