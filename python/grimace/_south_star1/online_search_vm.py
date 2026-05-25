"""Explicit event-level online search VM for South Star witness enumeration."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations
from itertools import permutations
from itertools import product
from typing import Literal

from .facts import DirectionalValue
from .facts import LigandKind
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetraValue
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_render_sink import OnlineRenderSink
from .online_render_sink import OnlineStringBuffer
from .online_traversal import OnlineAtomEvent
from .online_traversal import OnlineBranchClose
from .online_traversal import OnlineBranchOpen
from .online_traversal import OnlineDotEvent
from .online_traversal import OnlineRingEndpointEvent
from .online_traversal import OnlineTreeBondEvent
from .policy import AnnotationMode
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .residual_constraints import DirectionalCarrierResidual
from .residual_constraints import DirectionalResidualFactor
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import direction_var
from .semantics import ParserSemantics
from .skeleton import ChildRole
from .stereo_templates import DirectionalTemplate
from .stereo_templates import StereoTemplateBundle
from .stereo_templates import TetraTemplate
from .stereo_templates import build_stereo_templates


@dataclass(frozen=True, slots=True)
class ComponentRootChoiceFrame:
    component_index: int
    atom: AtomId
    cursor: int


@dataclass(frozen=True, slots=True)
class SpanningTreeChoiceFrame:
    component_index: int
    tree_bonds: tuple[BondId, ...]
    cursor: int


@dataclass(frozen=True, slots=True)
class ParentOrientationFrame:
    parent: tuple[tuple[AtomId, AtomId | None], ...]


@dataclass(frozen=True, slots=True)
class LocalEventOrderChoiceFrame:
    atom: AtomId
    order_key: tuple[object, ...]
    cursor: int


@dataclass(frozen=True, slots=True)
class EventLoopFrame:
    trace_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RingLabelChoiceFrame:
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class AtomTextChoiceFrame:
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class BondTextChoiceFrame:
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class DirectionMarkChoiceFrame:
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ProductChoiceFrame:
    choice: Literal["atom_text", "bond_text", "direction_mark"]
    domain_count: int
    cursor: int


@dataclass(frozen=True, slots=True)
class CompletionFrame:
    trace_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RenderCursorFrame:
    cursor: "_RenderCursor"


OnlineFramePayload = (
    ComponentRootChoiceFrame
    | SpanningTreeChoiceFrame
    | ParentOrientationFrame
    | LocalEventOrderChoiceFrame
    | EventLoopFrame
    | RingLabelChoiceFrame
    | AtomTextChoiceFrame
    | BondTextChoiceFrame
    | DirectionMarkChoiceFrame
    | ProductChoiceFrame
    | CompletionFrame
    | RenderCursorFrame
)


@dataclass(frozen=True, slots=True)
class OnlineSearchFrame:
    payload: OnlineFramePayload


@dataclass(frozen=True, slots=True)
class OnlineSearchSnapshot:
    traversal_state: object
    residual_snapshot: object
    ring_state: object
    output_snapshot: object
    decision_snapshot: object
    frame_stack: tuple[OnlineSearchFrame, ...]


@dataclass(frozen=True, slots=True)
class RenderContinuationPayloadShape:
    render_resume_continuation_count: int = 0
    max_render_piece_count: int = 0
    total_render_piece_count: int = 0
    max_remaining_render_piece_count: int = 0
    total_remaining_render_piece_count: int = 0
    max_render_payload_chars: int = 0
    total_render_payload_chars: int = 0
    render_cursor_count: int = 0
    max_render_program_event_count: int = 0
    total_render_program_event_count: int = 0
    max_remaining_render_event_count: int = 0
    total_remaining_render_event_count: int = 0
    max_render_program_choice_count: int = 0
    total_render_program_choice_count: int = 0


@dataclass(frozen=True, slots=True)
class OnlineStepResult:
    kind: Literal["advanced", "yield_witness", "exhausted", "rejected"]
    witness: "OnlineWitness | None" = None


@dataclass(frozen=True, slots=True)
class OnlineWitness:
    rendered: str
    traversal_key: tuple[object, ...]
    annotation_count: int


@dataclass(frozen=True, slots=True)
class OnlineResidualContinuation:
    prefix: str
    snapshot: OnlineSearchSnapshot


@dataclass(slots=True)
class MutableTraversalState:
    component_index: int = 0
    roots: list[AtomId] = field(default_factory=list)
    parent: dict[AtomId, AtomId | None] = field(default_factory=dict)
    tree_bonds: set[BondId] = field(default_factory=set)
    ring_bonds: set[BondId] = field(default_factory=set)
    visited_atoms: set[AtomId] = field(default_factory=set)
    active_atom_stack: list[AtomId] = field(default_factory=list)
    syntax_position: int = 0

    def checkpoint(self) -> object:
        return (
            self.component_index,
            tuple(self.roots),
            tuple(sorted(self.parent.items())),
            frozenset(self.tree_bonds),
            frozenset(self.ring_bonds),
            frozenset(self.visited_atoms),
            tuple(self.active_atom_stack),
            self.syntax_position,
        )

    def rollback(self, token: object) -> None:
        (
            self.component_index,
            roots,
            parent,
            tree_bonds,
            ring_bonds,
            visited_atoms,
            active_atom_stack,
            self.syntax_position,
        ) = token  # type: ignore[misc]
        self.roots = list(roots)
        self.parent = dict(parent)
        self.tree_bonds = set(tree_bonds)
        self.ring_bonds = set(ring_bonds)
        self.visited_atoms = set(visited_atoms)
        self.active_atom_stack = list(active_atom_stack)


@dataclass(slots=True)
class MutableRingState:
    endpoint_by_bond: dict[BondId, list[int]] = field(default_factory=dict)
    label_by_endpoint: dict[int, RingLabel] = field(default_factory=dict)
    open_intervals: dict[RingLabel, BondId] = field(default_factory=dict)
    next_endpoint_id: int = 0

    def checkpoint(self) -> object:
        return (
            tuple(sorted((bond, tuple(endpoints)) for bond, endpoints in self.endpoint_by_bond.items())),
            tuple(sorted((endpoint, label.value) for endpoint, label in self.label_by_endpoint.items())),
            tuple(sorted((label.value, bond) for label, bond in self.open_intervals.items())),
            self.next_endpoint_id,
        )

    def rollback(self, token: object) -> None:
        endpoint_by_bond, label_by_endpoint, open_intervals, self.next_endpoint_id = token  # type: ignore[misc]
        self.endpoint_by_bond = {
            bond: list(endpoints)
            for bond, endpoints in endpoint_by_bond
        }
        self.label_by_endpoint = {
            endpoint: RingLabel(label)
            for endpoint, label in label_by_endpoint
        }
        self.open_intervals = {
            RingLabel(label): bond
            for label, bond in open_intervals
        }

    def register_endpoint(self, *, bond: BondId, endpoint: int, label: RingLabel) -> bool:
        endpoints = self.endpoint_by_bond.setdefault(bond, [])
        if len(endpoints) >= 2:
            return False
        if label in self.open_intervals and self.open_intervals[label] != bond:
            return False
        if endpoints:
            first = endpoints[0]
            if self.label_by_endpoint[first] != label:
                return False
            self.open_intervals.pop(label, None)
        else:
            self.open_intervals[label] = bond
        endpoints.append(endpoint)
        self.label_by_endpoint[endpoint] = label
        self.next_endpoint_id = max(self.next_endpoint_id, endpoint + 1)
        return True


@dataclass(slots=True)
class OnlineSearchState:
    facts: MoleculeFacts
    policy: SmilesPolicy
    semantics: ParserSemantics
    templates: StereoTemplateBundle
    traversal: MutableTraversalState
    residual: ResidualStore
    ring: MutableRingState
    output: OnlineRenderSink
    decisions: OnlineDecisionRecorder
    frames: list[OnlineSearchFrame] = field(default_factory=list)

    def checkpoint(self) -> OnlineSearchSnapshot:
        return OnlineSearchSnapshot(
            traversal_state=self.traversal.checkpoint(),
            residual_snapshot=self.residual.value_snapshot(),
            ring_state=self.ring.checkpoint(),
            output_snapshot=self.output.checkpoint(),
            decision_snapshot=self.decisions.path(),
            frame_stack=tuple(self.frames),
        )

    def rollback(self, snapshot: OnlineSearchSnapshot) -> None:
        self.traversal.rollback(snapshot.traversal_state)
        if isinstance(snapshot.residual_snapshot, ResidualStoreValueSnapshot):
            self.residual = ResidualStore.from_value_snapshot(snapshot.residual_snapshot)
        else:
            self.residual.rollback(snapshot.residual_snapshot)  # type: ignore[arg-type]
        self.ring.rollback(snapshot.ring_state)
        self.output.rollback(snapshot.output_snapshot)
        if isinstance(snapshot.decision_snapshot, OnlineDecisionPath):
            self.decisions.restore_path(snapshot.decision_snapshot)
        else:
            self.decisions.rollback(snapshot.decision_snapshot)  # type: ignore[arg-type]
        self.frames[:] = list(snapshot.frame_stack)


@dataclass(frozen=True, slots=True)
class _Graph:
    atoms: tuple[AtomId, ...]
    bonds: dict[BondId, tuple[AtomId, AtomId]]
    components: tuple[tuple[tuple[AtomId, ...], tuple[BondId, ...]], ...]
    incident_bonds: dict[AtomId, tuple[BondId, ...]]


@dataclass(frozen=True, slots=True)
class _ChildLocalEvent:
    bond: BondId
    parent: AtomId
    child: AtomId
    role: ChildRole


@dataclass(frozen=True, slots=True)
class _RingLocalEvent:
    bond: BondId
    atom: AtomId
    other_atom: AtomId


@dataclass(frozen=True, slots=True)
class _VmTraversalTrace:
    roots: tuple[AtomId, ...]
    parent: tuple[tuple[AtomId, AtomId | None], ...]
    tree_bonds: frozenset[BondId]
    ring_bonds: frozenset[BondId]
    events: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _BondSlot:
    id: int
    bond: BondId
    kind: Literal["tree", "ring_endpoint"]
    written_from: AtomId
    written_to: AtomId | None
    ring_endpoint_id: int | None
    syntax_position: int


@dataclass(frozen=True, slots=True)
class _RingEndpointSlot:
    id: int
    bond: BondId
    atom: AtomId
    other_atom: AtomId
    bond_slot: int
    syntax_position: int


@dataclass(frozen=True, slots=True)
class _CarrierSlot:
    id: int
    bond_slot: int
    bond: BondId
    written_from: AtomId
    written_to: AtomId | None


@dataclass(frozen=True, slots=True)
class _SlotView:
    bond_slots: tuple[_BondSlot, ...]
    carrier_slots: tuple[_CarrierSlot, ...]
    ring_endpoints: tuple[_RingEndpointSlot, ...]


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
    residual_snapshot: ResidualStoreValueSnapshot
    ring_state: object
    decision_path: OnlineDecisionPath
    frame_stack: tuple[OnlineSearchFrame, ...]


@dataclass(frozen=True, slots=True)
class _RingAction:
    bond: BondId
    endpoint: int
    label: RingLabel


@dataclass(frozen=True, slots=True)
class _RenderPiece:
    text: str
    token_text: str | None
    ring_action: _RingAction | None = None


@dataclass(frozen=True, slots=True)
class _FrozenPrefixChoice:
    atom_text: tuple[tuple[AtomId, AtomTextChoice], ...]
    bond_text: tuple[tuple[int, BondTextChoice], ...]
    ring_labels: tuple[tuple[int, RingLabel], ...]


@dataclass(frozen=True, slots=True)
class _RenderProgram:
    trace: _VmTraversalTrace
    prefix: _FrozenPrefixChoice
    tetra_tokens: tuple[tuple[AtomId, TetraToken], ...]
    marks: tuple[tuple[int, DirectionMark], ...]
    annotation_count: int


@dataclass(frozen=True, slots=True)
class _RenderCursor:
    program: _RenderProgram
    event_index: int
    piece_index: int


def make_online_search_state(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    templates: StereoTemplateBundle | None = None,
    sink: OnlineRenderSink | None = None,
) -> OnlineSearchState:
    facts.validate()
    policy.validate_for_facts(facts)
    return OnlineSearchState(
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates if templates is not None else build_stereo_templates(facts),
        traversal=MutableTraversalState(),
        residual=ResidualStore(),
        ring=MutableRingState(),
        output=sink if sink is not None else OnlineStringBuffer(),
        decisions=OnlineDecisionRecorder(),
    )


class OnlineSearchVM:
    def __init__(
        self,
        *,
        facts: MoleculeFacts,
        policy: SmilesPolicy,
        semantics: ParserSemantics,
        templates: StereoTemplateBundle | None = None,
        sink_factory: Callable[[], OnlineRenderSink] | None = None,
    ) -> None:
        _validate_annotation_mode(policy)
        self.state = make_online_search_state(
            facts=facts,
            policy=policy,
            semantics=semantics,
            templates=templates,
        )
        self._sink_factory = sink_factory or OnlineStringBuffer
        self._iterator = self._run()
        self._exhausted = False

    def step(self) -> OnlineStepResult:
        if self._exhausted:
            return OnlineStepResult("exhausted")
        try:
            return OnlineStepResult("yield_witness", next(self._iterator))
        except StopIteration:
            self._exhausted = True
            return OnlineStepResult("exhausted")

    def run_until_witness_or_exhausted(self) -> OnlineWitness | None:
        result = self.step()
        if result.kind == "yield_witness":
            return result.witness
        return None

    def checkpoint(self) -> OnlineSearchSnapshot:
        return self.state.checkpoint()

    def rollback(self, snapshot: OnlineSearchSnapshot) -> None:
        self.state.rollback(snapshot)

    def _run(self) -> Iterator[OnlineWitness]:
        graph = _graph_from_facts(self.state.facts)
        yield from _iter_vm_traversals(self.state, graph, self._sink_factory)

    @classmethod
    def from_snapshot(
        cls,
        *,
        facts: MoleculeFacts,
        policy: SmilesPolicy,
        semantics: ParserSemantics,
        templates: StereoTemplateBundle | None = None,
        snapshot: OnlineSearchSnapshot,
        sink: OnlineRenderSink,
    ) -> "OnlineSearchVM":
        vm = cls(
            facts=facts,
            policy=policy,
            semantics=semantics,
            templates=templates,
            sink_factory=lambda: sink,
        )
        vm.state.output = sink
        _restore_resume_snapshot(vm.state, snapshot)
        vm._iterator = _resume_from_frames(vm.state)
        vm._exhausted = False
        return vm


def capture_residual_continuation(
    state: OnlineSearchState,
    *,
    prefix: str,
) -> OnlineResidualContinuation:
    return OnlineResidualContinuation(prefix=prefix, snapshot=state.checkpoint())


def iter_online_stereo_witness_strings_vm(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> Iterator[str]:
    for witness in iter_online_stereo_witnesses_vm(
        facts=facts,
        policy=policy,
        semantics=semantics,
    ):
        yield witness.rendered


def iter_online_stereo_witnesses_vm(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    templates: StereoTemplateBundle | None = None,
    sink_factory: Callable[[], OnlineRenderSink] | None = None,
) -> Iterator[OnlineWitness]:
    vm = OnlineSearchVM(
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates,
        sink_factory=sink_factory,
    )
    while True:
        result = vm.step()
        if result.kind == "yield_witness":
            if result.witness is None:
                raise ValueError("yield_witness result lacks witness")
            yield result.witness
            continue
        if result.kind == "exhausted":
            return


def resume_online_search_from_snapshot(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    templates: StereoTemplateBundle | None = None,
    snapshot: OnlineSearchSnapshot,
    sink: OnlineRenderSink,
) -> Iterator[OnlineWitness]:
    vm = OnlineSearchVM.from_snapshot(
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates,
        snapshot=snapshot,
        sink=sink,
    )
    while True:
        result = vm.step()
        if result.kind == "yield_witness":
            if result.witness is None:
                raise ValueError("yield_witness result lacks witness")
            yield result.witness
            continue
        if result.kind == "exhausted":
            return


def render_continuation_payload_shape(
    frame_stack: tuple[OnlineSearchFrame, ...],
) -> RenderContinuationPayloadShape:
    count = 0
    max_piece_count = 0
    total_piece_count = 0
    max_remaining_piece_count = 0
    total_remaining_piece_count = 0
    max_payload_chars = 0
    total_payload_chars = 0
    cursor_count = 0
    max_program_event_count = 0
    total_program_event_count = 0
    max_remaining_event_count = 0
    total_remaining_event_count = 0
    max_program_choice_count = 0
    total_program_choice_count = 0
    for frame in frame_stack:
        if not isinstance(frame.payload, RenderCursorFrame):
            continue
        cursor = frame.payload.cursor
        cursor_count += 1
        event_count = len(cursor.program.trace.events)
        remaining_event_count = max(0, event_count - cursor.event_index)
        choice_count = _render_program_choice_count(cursor.program)
        max_program_event_count = max(max_program_event_count, event_count)
        total_program_event_count += event_count
        max_remaining_event_count = max(max_remaining_event_count, remaining_event_count)
        total_remaining_event_count += remaining_event_count
        max_program_choice_count = max(max_program_choice_count, choice_count)
        total_program_choice_count += choice_count
    return RenderContinuationPayloadShape(
        render_resume_continuation_count=count,
        max_render_piece_count=max_piece_count,
        total_render_piece_count=total_piece_count,
        max_remaining_render_piece_count=max_remaining_piece_count,
        total_remaining_render_piece_count=total_remaining_piece_count,
        max_render_payload_chars=max_payload_chars,
        total_render_payload_chars=total_payload_chars,
        render_cursor_count=cursor_count,
        max_render_program_event_count=max_program_event_count,
        total_render_program_event_count=total_program_event_count,
        max_remaining_render_event_count=max_remaining_event_count,
        total_remaining_render_event_count=total_remaining_event_count,
        max_render_program_choice_count=max_program_choice_count,
        total_render_program_choice_count=total_program_choice_count,
    )


def _iter_vm_traversals(
    state: OnlineSearchState,
    graph: _Graph,
    sink_factory: Callable[[], OnlineRenderSink],
) -> Iterator[OnlineWitness]:
    all_bonds = frozenset(graph.bonds)
    for roots in _iter_root_choices(state, graph):
        state.traversal.roots = list(roots)
        for tree_bonds in _iter_spanning_forest_choices(state, graph):
            ring_bonds = all_bonds - tree_bonds
            state.traversal.tree_bonds = set(tree_bonds)
            state.traversal.ring_bonds = set(ring_bonds)
            parent, children_by_parent = _orient_forest_from_roots(
                graph=graph,
                roots=roots,
                tree_bonds=tree_bonds,
            )
            state.traversal.parent = dict(parent)
            state.frames.append(
                OnlineSearchFrame(ParentOrientationFrame(tuple(sorted(parent.items()))))
            )
            try:
                ring_events_by_atom = _ring_events_by_atom(graph, ring_bonds)
                local_domains = tuple(
                    (
                        atom,
                        tuple(
                            _local_event_orders(
                                atom,
                                children_by_parent[atom],
                                ring_events_by_atom[atom],
                            )
                        ),
                    )
                    for atom in graph.atoms
                )
                yield from _iter_local_order_products(
                    state=state,
                    roots=roots,
                    parent=parent,
                    tree_bonds=tree_bonds,
                    ring_bonds=ring_bonds,
                    local_domains=local_domains,
                    sink_factory=sink_factory,
                )
            finally:
                state.frames.pop()


def _iter_local_order_products(
    *,
    state: OnlineSearchState,
    roots: tuple[AtomId, ...],
    parent: dict[AtomId, AtomId | None],
    tree_bonds: frozenset[BondId],
    ring_bonds: frozenset[BondId],
    local_domains: tuple[tuple[AtomId, tuple[tuple[object, ...], ...]], ...],
    sink_factory: Callable[[], OnlineRenderSink],
) -> Iterator[OnlineWitness]:
    events_at: dict[AtomId, tuple[object, ...]] = {}

    def rec(index: int) -> Iterator[OnlineWitness]:
        if index == len(local_domains):
            event_buffer: list[object] = []
            state.traversal.syntax_position = 0
            state.traversal.visited_atoms.clear()
            for root_index, root in enumerate(roots):
                if root_index:
                    event_buffer.append(OnlineDotEvent())
                _emit_atom_subtree(state, event_buffer, events_at, root)
            trace = _VmTraversalTrace(
                roots=roots,
                parent=tuple(sorted(parent.items())),
                tree_bonds=tree_bonds,
                ring_bonds=ring_bonds,
                events=tuple(event_buffer),
            )
            yield from _iter_witnesses_for_trace(
                state=state,
                trace=trace,
                sink_factory=sink_factory,
            )
            return

        atom, orders = local_domains[index]
        for cursor, order in enumerate(orders):
            state.frames.append(
                OnlineSearchFrame(
                    LocalEventOrderChoiceFrame(
                        atom=atom,
                        order_key=_local_order_key(order),
                        cursor=cursor,
                    )
                )
            )
            events_at[atom] = order
            try:
                yield from rec(index + 1)
            finally:
                del events_at[atom]
                state.frames.pop()

    yield from rec(0)


def _iter_witnesses_for_trace(
    *,
    state: OnlineSearchState,
    trace: _VmTraversalTrace,
    sink_factory: Callable[[], OnlineRenderSink],
) -> Iterator[OnlineWitness]:
    traversal_key = _trace_key(trace)
    state.frames.append(OnlineSearchFrame(EventLoopFrame(traversal_key)))
    state.decisions.push(OnlineDecision("traversal", (traversal_key,)))
    decision_checkpoint = state.decisions.checkpoint()
    try:
        slots = _slot_view_for_trace(trace)
        atom_domains = tuple(
            (atom.id, state.policy.atom_text_domain(state.facts, atom.id))
            for atom in state.facts.atoms
        )
        bond_domains = tuple(
            (
                slot.id,
                state.policy.bond_text_domain(state.facts, slot.bond, slot_kind=slot.kind),
            )
            for slot in slots.bond_slots
        )
        for ring_labels in _iter_ring_label_assignments(state, slots):
            ring_checkpoint = state.ring.checkpoint()
            if not _install_ring_labels(state, slots, ring_labels):
                state.ring.rollback(ring_checkpoint)
                continue
            state.frames.append(
                OnlineSearchFrame(RingLabelChoiceFrame(_ring_label_decision_value(ring_labels)))
            )
            state.decisions.push(OnlineDecision("ring_labels", _ring_label_decision_value(ring_labels)))
            ring_decision_checkpoint = state.decisions.checkpoint()
            try:
                for atom_text in _iter_dict_product_frame(state, kind="atom-text-choice", domains=atom_domains):
                    state.frames.append(
                        OnlineSearchFrame(AtomTextChoiceFrame(_atom_text_decision_value(atom_text)))
                    )
                    state.decisions.push(OnlineDecision("atom_text", _atom_text_decision_value(atom_text)))
                    atom_decision_checkpoint = state.decisions.checkpoint()
                    try:
                        tetra_tokens = _forced_tetra_tokens(
                            facts=state.facts,
                            trace=trace,
                            templates=state.templates,
                            atom_text=atom_text,
                        )
                        if tetra_tokens is None:
                            continue
                        for bond_text in _iter_dict_product_frame(state, kind="bond-text-choice", domains=bond_domains):
                            state.frames.append(
                                OnlineSearchFrame(BondTextChoiceFrame(_bond_text_decision_value(bond_text)))
                            )
                            state.decisions.push(OnlineDecision("bond_text", _bond_text_decision_value(bond_text)))
                            bond_decision_checkpoint = state.decisions.checkpoint()
                            try:
                                prefix = _PrefixChoice(
                                    atom_text=atom_text,
                                    bond_text=bond_text,
                                    ring_labels=ring_labels,
                                )
                                candidates = _iter_directional_candidates(
                                    state=state,
                                    trace=trace,
                                    slots=slots,
                                    prefix=prefix,
                                )
                                if state.policy.annotation_mode is AnnotationMode.SUPPORT_MAXIMAL:
                                    candidates = tuple(_support_maximal_candidates(tuple(candidates)))
                                for candidate in candidates:
                                    rendered = _render_directional_candidate(
                                        state=state,
                                        trace=trace,
                                        slots=slots,
                                        prefix=prefix,
                                        tetra_tokens=tetra_tokens,
                                        candidate=candidate,
                                        sink_factory=sink_factory,
                                    )
                                    if rendered is None:
                                        continue
                                    yield OnlineWitness(
                                        rendered=rendered,
                                        traversal_key=traversal_key,
                                        annotation_count=candidate.annotation_count,
                                    )
                            finally:
                                state.decisions.rollback(bond_decision_checkpoint)
                                state.frames.pop()
                    finally:
                        state.decisions.rollback(atom_decision_checkpoint)
                        state.frames.pop()
            finally:
                state.decisions.rollback(ring_decision_checkpoint)
                state.frames.pop()
                state.ring.rollback(ring_checkpoint)
    finally:
        state.decisions.rollback(decision_checkpoint)
        state.frames.pop()


def _iter_directional_candidates(
    *,
    state: OnlineSearchState,
    trace: _VmTraversalTrace,
    slots: _SlotView,
    prefix: _PrefixChoice,
) -> Iterator[_DirectionalCandidate]:
    domains = _direction_domains(
        facts=state.facts,
        templates=state.templates,
        slots=slots,
        prefix=prefix,
    )
    carrier_models_by_site = _directional_models_by_site(
        facts=state.facts,
        templates=state.templates,
        slots=slots,
    )
    carriers = tuple(carrier.id for carrier in slots.carrier_slots)
    previous_store = state.residual
    try:
        mark_domains = tuple((carrier_id, domains[carrier_id]) for carrier_id in carriers)
        for marks in _iter_dict_product_frame(state, kind="direction-mark-choice", domains=mark_domains):
            store = ResidualStore()
            for carrier_id in carriers:
                store.add_var(direction_var(carrier_id), domains[carrier_id])
            factor_ids = []
            for template in state.templates.directional:
                scope_models = carrier_models_by_site[template.site]
                factor = DirectionalResidualFactor(
                    scope=tuple(scope_models),
                    status=template.status,
                    target=template.target,
                    carrier_models=scope_models,
                )
                factor_ids.append(store.add_factor(factor))
            state.residual = store
            state.frames.append(
                OnlineSearchFrame(DirectionMarkChoiceFrame(_direction_decision_value(marks)))
            )
            decision_checkpoint = state.decisions.checkpoint()
            for carrier_id in carriers:
                state.decisions.push(OnlineDecision("direction_mark", (carrier_id, marks[carrier_id].name)))
            try:
                if not all(
                    store.assign(direction_var(carrier_id), marks[carrier_id])
                    for carrier_id in carriers
                ):
                    continue
                if not _bond_decode_ok(
                    facts=state.facts,
                    semantics=state.semantics,
                    slots=slots,
                    prefix=prefix,
                    marks=marks,
                ):
                    continue
                if not all(store.close_factor(factor_id) for factor_id in factor_ids):
                    continue
                yield _DirectionalCandidate(
                    marks=tuple(sorted(marks.items())),
                    support=frozenset(
                        carrier_id
                        for carrier_id, mark in marks.items()
                        if mark is not DirectionMark.ABSENT
                    ),
                    annotation_count=sum(
                        1 for mark in marks.values() if mark is not DirectionMark.ABSENT
                    ),
                    residual_snapshot=store.value_snapshot(),
                    ring_state=state.ring.checkpoint(),
                    decision_path=state.decisions.path(),
                    frame_stack=tuple(state.frames),
                )
            finally:
                state.decisions.rollback(decision_checkpoint)
                state.frames.pop()
    finally:
        state.residual = previous_store


def _render_directional_candidate(
    *,
    state: OnlineSearchState,
    trace: _VmTraversalTrace,
    slots: _SlotView,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, TetraToken],
    candidate: _DirectionalCandidate,
    sink_factory: Callable[[], OnlineRenderSink],
) -> str | None:
    previous_output = state.output
    previous_store = state.residual
    previous_ring = state.ring.checkpoint()
    previous_decisions = state.decisions.path()
    previous_frames = tuple(state.frames)
    state.residual = ResidualStore.from_value_snapshot(candidate.residual_snapshot)
    state.ring.rollback(candidate.ring_state)
    state.decisions.restore_path(candidate.decision_path)
    state.frames[:] = list(candidate.frame_stack)
    state.output = sink_factory()
    checkpoint = state.output.checkpoint()
    state.frames.append(OnlineSearchFrame(CompletionFrame(_trace_key(trace))))
    try:
        program = _render_program(
            trace=trace,
            prefix=prefix,
            tetra_tokens=tetra_tokens,
            marks=dict(candidate.marks),
            annotation_count=candidate.annotation_count,
        )
        if not _render_program_to_sink(
            state=state,
            slots=slots,
            start_cursor=_RenderCursor(program=program, event_index=0, piece_index=0),
        ):
            state.output.rollback(checkpoint)
            return None
        if not state.output.complete():
            state.output.rollback(checkpoint)
            return None
        rendered = state.output.value()
        state.output.rollback(checkpoint)
        return rendered
    finally:
        state.frames.pop()
        state.output = previous_output
        state.residual = previous_store
        state.ring.rollback(previous_ring)
        state.decisions.restore_path(previous_decisions)
        state.frames[:] = list(previous_frames)


def _render_program(
    *,
    trace: _VmTraversalTrace,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, TetraToken],
    marks: dict[int, DirectionMark],
    annotation_count: int,
) -> _RenderProgram:
    return _RenderProgram(
        trace=trace,
        prefix=_freeze_prefix_choice(prefix),
        tetra_tokens=tuple(sorted(tetra_tokens.items(), key=lambda item: int(item[0]))),
        marks=tuple(sorted(marks.items())),
        annotation_count=annotation_count,
    )


def _freeze_prefix_choice(prefix: _PrefixChoice) -> _FrozenPrefixChoice:
    return _FrozenPrefixChoice(
        atom_text=tuple(sorted(prefix.atom_text.items(), key=lambda item: int(item[0]))),
        bond_text=tuple(sorted(prefix.bond_text.items())),
        ring_labels=tuple(sorted(prefix.ring_labels.items())),
    )


def _thaw_prefix_choice(prefix: _FrozenPrefixChoice) -> _PrefixChoice:
    return _PrefixChoice(
        atom_text=dict(prefix.atom_text),
        bond_text=dict(prefix.bond_text),
        ring_labels=dict(prefix.ring_labels),
    )


def _advance_render_cursor(
    *,
    program: _RenderProgram,
    event_index: int,
    piece_index: int,
    piece_count: int,
) -> _RenderCursor:
    if piece_index + 1 < piece_count:
        return _RenderCursor(
            program=program,
            event_index=event_index,
            piece_index=piece_index + 1,
        )
    return _RenderCursor(
        program=program,
        event_index=event_index + 1,
        piece_index=0,
    )


def _render_program_choice_count(program: _RenderProgram) -> int:
    return (
        len(program.prefix.atom_text)
        + len(program.prefix.bond_text)
        + len(program.prefix.ring_labels)
        + len(program.tetra_tokens)
        + len(program.marks)
    )


def _render_event_pieces(
    *,
    program: _RenderProgram,
    slots: _SlotView,
    event_index: int,
) -> tuple[_RenderPiece, ...]:
    prefix = _thaw_prefix_choice(program.prefix)
    marks = dict(program.marks)
    tetra_tokens = dict(program.tetra_tokens)
    bond_slot_by_event: dict[tuple[object, ...], _BondSlot] = {}
    for slot in slots.bond_slots:
        bond_slot_by_event[
            (slot.kind, slot.bond, slot.written_from, slot.written_to, slot.ring_endpoint_id)
        ] = slot
    ring_label_by_endpoint = {
        endpoint.id: prefix.ring_labels[endpoint.id]
        for endpoint in slots.ring_endpoints
    }
    event = program.trace.events[event_index]
    if isinstance(event, OnlineAtomEvent):
        text = prefix.atom_text[event.atom].render(tetra_tokens[event.atom])
        return (_RenderPiece(text=text, token_text=text),)
    if isinstance(event, OnlineTreeBondEvent):
        key = ("tree", event.bond, event.written_from, event.written_to, None)
        text = _render_bond_slot(bond_slot_by_event[key], prefix, marks)
        return (_RenderPiece(text=text, token_text=text or None),)
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
        if slot.ring_endpoint_id is None:
            raise ValueError("ring endpoint slot lacks endpoint id")
        label = ring_label_by_endpoint[slot.ring_endpoint_id]
        return (
            _RenderPiece(text=text, token_text=text or None),
            _RenderPiece(
                text=label.text(),
                token_text=label.text(),
                ring_action=_RingAction(
                    bond=slot.bond,
                    endpoint=slot.ring_endpoint_id,
                    label=label,
                ),
            ),
        )
    if isinstance(event, OnlineBranchOpen):
        return (_RenderPiece(text="(", token_text="("),)
    if isinstance(event, OnlineBranchClose):
        return (_RenderPiece(text=")", token_text=")"),)
    if isinstance(event, OnlineDotEvent):
        return (_RenderPiece(text=".", token_text="."),)
    raise TypeError(event)


def _render_program_to_sink(
    *,
    state: OnlineSearchState,
    slots: _SlotView,
    start_cursor: _RenderCursor,
) -> bool:
    program = start_cursor.program
    for event_index in range(start_cursor.event_index, len(program.trace.events)):
        pieces = _render_event_pieces(program=program, slots=slots, event_index=event_index)
        first_piece = start_cursor.piece_index if event_index == start_cursor.event_index else 0
        for piece_index in range(first_piece, len(pieces)):
            piece = pieces[piece_index]
            checkpoint = state.ring.checkpoint()
            if piece.ring_action is not None and not state.ring.register_endpoint(
                bond=piece.ring_action.bond,
                endpoint=piece.ring_action.endpoint,
                label=piece.ring_action.label,
            ):
                state.ring.rollback(checkpoint)
                return False
            next_cursor = _advance_render_cursor(
                program=program,
                event_index=event_index,
                piece_index=piece_index,
                piece_count=len(pieces),
            )
            state.frames.append(OnlineSearchFrame(RenderCursorFrame(next_cursor)))
            try:
                if not state.output.append(piece.text, token_text=piece.token_text):
                    state.ring.rollback(checkpoint)
                    return False
            finally:
                state.frames.pop()
    return True


def _restore_resume_snapshot(
    state: OnlineSearchState,
    snapshot: OnlineSearchSnapshot,
) -> None:
    state.rollback(snapshot)


def _resume_from_frames(state: OnlineSearchState) -> Iterator[OnlineWitness]:
    frame = _pop_resumable_frame(state.frames)
    if frame is None:
        return
    match frame.payload:
        case RenderCursorFrame(cursor=cursor):
            yield from _resume_render_cursor_frame(state, cursor)
        case _:
            raise TypeError(f"frame is not resumable: {frame.payload!r}")


def _resume_render_cursor_frame(
    state: OnlineSearchState,
    cursor: _RenderCursor,
) -> Iterator[OnlineWitness]:
    checkpoint = state.output.checkpoint()
    slots = _slot_view_for_trace(cursor.program.trace)
    if not _render_program_to_sink(
        state=state,
        slots=slots,
        start_cursor=cursor,
    ):
        state.output.rollback(checkpoint)
        return
    if not state.output.complete():
        state.output.rollback(checkpoint)
        return
    rendered = state.output.value()
    yield OnlineWitness(
        rendered=rendered,
        traversal_key=_trace_key(cursor.program.trace),
        annotation_count=cursor.program.annotation_count,
    )


def _pop_resumable_frame(
    frames: list[OnlineSearchFrame],
) -> OnlineSearchFrame | None:
    active_index: int | None = None
    for index in range(len(frames) - 1, -1, -1):
        frame = frames[index]
        if not isinstance(frame.payload, RenderCursorFrame):
            continue
        active_index = index
        break
    if active_index is None:
        return None
    popped = frames.pop(active_index)
    if any(isinstance(frame.payload, RenderCursorFrame) for frame in frames):
        raise AssertionError("multiple active render-cursor frames")
    return popped


def _graph_from_facts(facts: MoleculeFacts) -> _Graph:
    incident: dict[AtomId, list[BondId]] = {atom.id: [] for atom in facts.atoms}
    bonds: dict[BondId, tuple[AtomId, AtomId]] = {}
    for bond in facts.bonds:
        bonds[bond.id] = (bond.a, bond.b)
        incident[bond.a].append(bond.id)
        incident[bond.b].append(bond.id)
    return _Graph(
        atoms=tuple(atom.id for atom in facts.atoms),
        bonds=bonds,
        components=tuple((component.atoms, component.bonds) for component in facts.components),
        incident_bonds={atom: tuple(items) for atom, items in incident.items()},
    )


def _iter_root_choices(state: OnlineSearchState, graph: _Graph) -> Iterator[tuple[AtomId, ...]]:
    roots: list[AtomId] = []

    def rec(index: int) -> Iterator[tuple[AtomId, ...]]:
        if index == len(graph.components):
            yield tuple(roots)
            return
        atoms, _ = graph.components[index]
        state.traversal.component_index = index
        for cursor, atom in enumerate(atoms):
            state.frames.append(
                OnlineSearchFrame(
                    ComponentRootChoiceFrame(
                        component_index=index,
                        atom=atom,
                        cursor=cursor,
                    )
                )
            )
            roots.append(atom)
            try:
                yield from rec(index + 1)
            finally:
                roots.pop()
                state.frames.pop()

    yield from rec(0)


def _iter_spanning_forest_choices(state: OnlineSearchState, graph: _Graph) -> Iterator[frozenset[BondId]]:
    chosen: list[BondId] = []

    def rec(index: int) -> Iterator[frozenset[BondId]]:
        if index == len(graph.components):
            yield frozenset(chosen)
            return
        atoms, bonds = graph.components[index]
        for cursor, tree in enumerate(_iter_component_spanning_trees(graph, atoms, bonds)):
            state.frames.append(
                OnlineSearchFrame(
                    SpanningTreeChoiceFrame(
                        component_index=index,
                        tree_bonds=tuple(tree),
                        cursor=cursor,
                    )
                )
            )
            checkpoint = len(chosen)
            chosen.extend(tree)
            try:
                yield from rec(index + 1)
            finally:
                del chosen[checkpoint:]
                state.frames.pop()

    yield from rec(0)


def _iter_component_spanning_trees(
    graph: _Graph,
    atoms: tuple[AtomId, ...],
    bonds: tuple[BondId, ...],
) -> Iterator[tuple[BondId, ...]]:
    if len(atoms) == 1:
        if bonds:
            raise ValueError("single-atom component cannot have bonds")
        yield ()
        return
    atom_set = set(atoms)
    for candidate in combinations(bonds, len(atoms) - 1):
        candidate_set = frozenset(candidate)
        if _reachable_atoms_on_bonds(graph, atoms[0], atom_set, candidate_set) == atom_set:
            yield tuple(candidate)


def _orient_forest_from_roots(
    *,
    graph: _Graph,
    roots: tuple[AtomId, ...],
    tree_bonds: frozenset[BondId],
) -> tuple[dict[AtomId, AtomId | None], dict[AtomId, list[tuple[BondId, AtomId]]]]:
    parent: dict[AtomId, AtomId | None] = {}
    children_by_parent: dict[AtomId, list[tuple[BondId, AtomId]]] = {
        atom: [] for atom in graph.atoms
    }
    for root, (component_atoms, _) in zip(roots, graph.components, strict=True):
        if root not in component_atoms:
            raise ValueError("root is outside component")
        parent[root] = None
        stack = [root]
        while stack:
            atom = stack.pop()
            for bond in reversed(graph.incident_bonds[atom]):
                if bond not in tree_bonds:
                    continue
                left, right = graph.bonds[bond]
                neighbor = right if left == atom else left
                if parent.get(atom) == neighbor:
                    continue
                if neighbor in parent:
                    raise ValueError("tree bond set is cyclic")
                parent[neighbor] = atom
                children_by_parent[atom].append((bond, neighbor))
                stack.append(neighbor)
        if not set(component_atoms) <= set(parent):
            raise ValueError("tree bond set does not connect component")
    return parent, children_by_parent


def _ring_events_by_atom(
    graph: _Graph,
    ring_bonds: frozenset[BondId],
) -> dict[AtomId, list[_RingLocalEvent]]:
    out: dict[AtomId, list[_RingLocalEvent]] = {atom: [] for atom in graph.atoms}
    for bond in graph.bonds:
        if bond not in ring_bonds:
            continue
        left, right = graph.bonds[bond]
        out[left].append(_RingLocalEvent(bond=bond, atom=left, other_atom=right))
        out[right].append(_RingLocalEvent(bond=bond, atom=right, other_atom=left))
    return out


def _local_event_orders(
    parent: AtomId,
    children: list[tuple[BondId, AtomId]],
    rings: list[_RingLocalEvent],
) -> Iterator[tuple[object, ...]]:
    branch_children = tuple(
        _ChildLocalEvent(bond=bond, parent=parent, child=child, role=ChildRole.BRANCH)
        for bond, child in children
    )
    ring_tuple = tuple(rings)
    seen: set[tuple[object, ...]] = set()
    base = ring_tuple + branch_children
    for order in permutations(base):
        if order not in seen:
            seen.add(order)
            yield order
    for ordered_children in permutations(children):
        if not ordered_children:
            continue
        continuation = _ChildLocalEvent(
            bond=ordered_children[-1][0],
            parent=parent,
            child=ordered_children[-1][1],
            role=ChildRole.CONTINUATION,
        )
        decorations = tuple(
            _ChildLocalEvent(
                bond=bond,
                parent=parent,
                child=child,
                role=ChildRole.BRANCH,
            )
            for bond, child in ordered_children[:-1]
        )
        for prefix in permutations(ring_tuple + decorations):
            order = prefix + (continuation,)
            if order not in seen:
                seen.add(order)
                yield order


def _emit_atom_subtree(
    state: OnlineSearchState,
    event_buffer: list[object],
    events_at: dict[AtomId, tuple[object, ...]],
    atom: AtomId,
) -> None:
    state.traversal.visited_atoms.add(atom)
    state.traversal.active_atom_stack.append(atom)
    event_buffer.append(OnlineAtomEvent(atom=atom, parent=state.traversal.parent[atom]))
    for event in events_at[atom]:
        if isinstance(event, _RingLocalEvent):
            event_buffer.append(
                OnlineRingEndpointEvent(
                    bond=event.bond,
                    at=event.atom,
                    other_atom=event.other_atom,
                    syntax_position=state.traversal.syntax_position,
                )
            )
            state.traversal.syntax_position += 1
            continue
        if isinstance(event, _ChildLocalEvent):
            role = event.role
            if role is ChildRole.BRANCH:
                event_buffer.append(OnlineBranchOpen())
            event_buffer.append(
                OnlineTreeBondEvent(
                    bond=event.bond,
                    written_from=event.parent,
                    written_to=event.child,
                    role=role,
                )
            )
            state.traversal.syntax_position += 1
            _emit_atom_subtree(state, event_buffer, events_at, event.child)
            if role is ChildRole.BRANCH:
                event_buffer.append(OnlineBranchClose())
            continue
        raise TypeError(event)
    state.traversal.active_atom_stack.pop()


def _slot_view_for_trace(trace: _VmTraversalTrace) -> _SlotView:
    bond_slots: list[_BondSlot] = []
    carrier_slots: list[_CarrierSlot] = []
    ring_endpoints: list[_RingEndpointSlot] = []
    syntax_position = 0
    for event in trace.events:
        if isinstance(event, OnlineTreeBondEvent):
            slot_id = len(bond_slots)
            bond_slots.append(
                _BondSlot(
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
                _CarrierSlot(
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
            bond_slots.append(
                _BondSlot(
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
                _RingEndpointSlot(
                    id=endpoint_id,
                    bond=event.bond,
                    atom=event.at,
                    other_atom=event.other_atom,
                    bond_slot=slot_id,
                    syntax_position=syntax_position,
                )
            )
            carrier_slots.append(
                _CarrierSlot(
                    id=len(carrier_slots),
                    bond_slot=slot_id,
                    bond=event.bond,
                    written_from=event.at,
                    written_to=event.other_atom,
                )
            )
            syntax_position += 1
    return _SlotView(
        bond_slots=tuple(bond_slots),
        carrier_slots=tuple(carrier_slots),
        ring_endpoints=tuple(ring_endpoints),
    )


def _iter_ring_label_assignments(
    state: OnlineSearchState,
    slots: _SlotView,
) -> Iterator[dict[int, RingLabel]]:
    if not slots.ring_endpoints:
        yield {}
        return
    intervals = _ring_intervals(slots)
    labels = state.policy.ring_labels
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
        if state.policy.least_free_ring_labels:
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


def _install_ring_labels(
    state: OnlineSearchState,
    slots: _SlotView,
    ring_labels: dict[int, RingLabel],
) -> bool:
    del state
    ring = MutableRingState()
    for endpoint in slots.ring_endpoints:
        label = ring_labels[endpoint.id]
        if not ring.register_endpoint(
            bond=endpoint.bond,
            endpoint=endpoint.id,
            label=label,
        ):
            return False
    return not ring.open_intervals


def _forced_tetra_tokens(
    *,
    facts: MoleculeFacts,
    trace: _VmTraversalTrace,
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
            token = _forced_tetra_token(facts=facts, trace=trace, atom=atom, template=template)
            if token is None:
                return None
        if not atom_text[atom].permits(token):
            return None
        out[atom] = token
    return out


def _forced_tetra_token(
    *,
    facts: MoleculeFacts,
    trace: _VmTraversalTrace,
    atom: AtomId,
    template: TetraTemplate,
) -> TetraToken | None:
    if template.status is SiteStatus.UNSPECIFIED:
        return TetraToken.NONE
    local_order = _local_tetra_order(facts=facts, trace=trace, atom=atom, template=template)
    if set(local_order) != set(template.reference_order):
        return None
    if len(local_order) != len(template.reference_order):
        return None
    is_even = _is_even_permutation(tuple(template.reference_order.index(item) for item in local_order))
    if template.target is TetraValue.PLUS:
        return TetraToken.AT if is_even else TetraToken.ATAT
    if template.target is TetraValue.MINUS:
        return TetraToken.ATAT if is_even else TetraToken.AT
    return None


def _local_tetra_order(
    *,
    facts: MoleculeFacts,
    trace: _VmTraversalTrace,
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


def _direction_domains(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    slots: _SlotView,
    prefix: _PrefixChoice,
) -> dict[int, tuple[DirectionMark, ...]]:
    eligible = _eligible_marker_carriers(facts=facts, templates=templates, slots=slots)
    domains: dict[int, tuple[DirectionMark, ...]] = {}
    for carrier in slots.carrier_slots:
        choice = prefix.bond_text[carrier.bond_slot]
        if carrier.id in eligible and choice.permits_direction:
            domains[carrier.id] = (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV)
        else:
            domains[carrier.id] = (DirectionMark.ABSENT,)
    return domains


def _eligible_marker_carriers(
    *,
    facts: MoleculeFacts,
    templates: StereoTemplateBundle,
    slots: _SlotView,
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
    slots: _SlotView,
) -> dict[object, dict[object, DirectionalCarrierResidual]]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    out: dict[object, dict[object, DirectionalCarrierResidual]] = {}
    for template in templates.directional:
        left_reference, right_reference = _directional_reference_pair(template)
        left_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.left_ligands)
        right_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.right_ligands)
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
    slots: _SlotView,
    prefix: _PrefixChoice,
    marks: dict[int, DirectionMark],
) -> bool:
    carriers_by_slot = {carrier.bond_slot: carrier for carrier in slots.carrier_slots}
    tree_slots = tuple(slot for slot in slots.bond_slots if slot.kind == "tree")
    ring_slots_by_bond: dict[BondId, list[_BondSlot]] = {}
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


def _render_bond_slot(
    slot: _BondSlot,
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


def _iter_dict_product_frame(state: OnlineSearchState, *, kind: str, domains):
    keys = tuple(key for key, _ in domains)
    value_domains = tuple(tuple(values) for _, values in domains)
    if any(not values for values in value_domains):
        return
    for cursor, values in enumerate(product(*value_domains)):
        state.frames.append(
            OnlineSearchFrame(
                ProductChoiceFrame(
                    choice=_product_choice_name(kind),
                    domain_count=len(value_domains),
                    cursor=cursor,
                )
            )
        )
        try:
            yield dict(zip(keys, values, strict=True))
        finally:
            state.frames.pop()


def _product_choice_name(kind: str) -> Literal["atom_text", "bond_text", "direction_mark"]:
    if kind == "atom-text-choice":
        return "atom_text"
    if kind == "bond-text-choice":
        return "bond_text"
    if kind == "direction-mark-choice":
        return "direction_mark"
    raise ValueError(f"unknown product choice frame kind: {kind!r}")


def _reachable_atoms_on_bonds(
    graph: _Graph,
    start: AtomId,
    allowed_atoms: set[AtomId],
    allowed_bonds: frozenset[BondId],
) -> set[AtomId]:
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for bond in graph.incident_bonds[atom]:
            if bond not in allowed_bonds:
                continue
            left, right = graph.bonds[bond]
            neighbor = right if left == atom else left
            if neighbor in allowed_atoms and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _local_events_for_atom(trace: _VmTraversalTrace, atom: AtomId) -> tuple[object, ...]:
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


def _directional_reference_pair(template: DirectionalTemplate) -> tuple[OccurrenceId, OccurrenceId]:
    if template.reference_pair is not None:
        return template.reference_pair
    if template.status is SiteStatus.SPECIFIED:
        raise ValueError("specified directional site lacks reference pair")
    return (min(template.left_ligands, key=int), min(template.right_ligands, key=int))


def _carrier_orientation(carrier: _CarrierSlot, endpoint: AtomId) -> Literal[-1, 1]:
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


def _ring_intervals(slots: _SlotView) -> tuple[tuple[int, int, int, int], ...]:
    by_bond: dict[BondId, list[_RingEndpointSlot]] = {}
    for endpoint in slots.ring_endpoints:
        by_bond.setdefault(endpoint.bond, []).append(endpoint)
    intervals = []
    for endpoints in by_bond.values():
        if len(endpoints) != 2:
            raise ValueError("ring bond does not have two endpoints")
        left, right = sorted(endpoints, key=lambda item: item.syntax_position)
        intervals.append((left.id, right.id, left.syntax_position, right.syntax_position))
    return tuple(sorted(intervals, key=lambda item: (item[2], item[0])))


def _trace_key(trace: _VmTraversalTrace) -> tuple[object, ...]:
    events_by_atom: dict[AtomId, list[tuple[object, ...]]] = {
        atom: [] for atom, _ in trace.parent
    }
    for event in trace.events:
        if isinstance(event, OnlineTreeBondEvent):
            events_by_atom[event.written_from].append(
                ("child", int(event.bond), int(event.written_from), int(event.written_to), event.role.value)
            )
        elif isinstance(event, OnlineRingEndpointEvent):
            events_by_atom[event.at].append(
                ("ring", int(event.bond), int(event.at), int(event.other_atom))
            )
    return (
        tuple(int(root) for root in trace.roots),
        tuple(sorted((int(atom), None if parent is None else int(parent)) for atom, parent in trace.parent)),
        tuple(sorted(int(bond) for bond in trace.tree_bonds)),
        tuple(sorted(int(bond) for bond in trace.ring_bonds)),
        tuple(sorted((int(atom), tuple(events)) for atom, events in events_by_atom.items())),
    )


def _local_order_key(order: tuple[object, ...]) -> tuple[object, ...]:
    out = []
    for event in order:
        if isinstance(event, _ChildLocalEvent):
            out.append(("child", int(event.bond), int(event.child), event.role.value))
        elif isinstance(event, _RingLocalEvent):
            out.append(("ring", int(event.bond), int(event.atom), int(event.other_atom)))
    return tuple(out)


def _ring_label_decision_value(ring_labels: dict[int, RingLabel]) -> tuple[object, ...]:
    return tuple(sorted((endpoint, label.value) for endpoint, label in ring_labels.items()))


def _atom_text_decision_value(atom_text: dict[AtomId, AtomTextChoice]) -> tuple[object, ...]:
    return tuple(sorted((int(atom), choice.name) for atom, choice in atom_text.items()))


def _bond_text_decision_value(bond_text: dict[int, BondTextChoice]) -> tuple[object, ...]:
    return tuple(sorted((slot_id, choice.name) for slot_id, choice in bond_text.items()))


def _direction_decision_value(marks: dict[int, DirectionMark]) -> tuple[object, ...]:
    return tuple(sorted((carrier_id, mark.name) for carrier_id, mark in marks.items()))


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
    from .errors import SouthStarError
    from .errors import SouthStarErrorKind

    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        "online VM currently supports HARD and SUPPORT_MAXIMAL annotation modes",
    )


__all__ = (
    "MutableRingState",
    "MutableTraversalState",
    "AtomTextChoiceFrame",
    "BondTextChoiceFrame",
    "CompletionFrame",
    "ComponentRootChoiceFrame",
    "DirectionMarkChoiceFrame",
    "EventLoopFrame",
    "LocalEventOrderChoiceFrame",
    "OnlineResidualContinuation",
    "OnlineFramePayload",
    "ParentOrientationFrame",
    "ProductChoiceFrame",
    "RenderContinuationPayloadShape",
    "RenderCursorFrame",
    "RingLabelChoiceFrame",
    "SpanningTreeChoiceFrame",
    "OnlineSearchFrame",
    "OnlineSearchSnapshot",
    "OnlineSearchState",
    "OnlineSearchVM",
    "OnlineStepResult",
    "OnlineWitness",
    "capture_residual_continuation",
    "iter_online_stereo_witness_strings_vm",
    "iter_online_stereo_witnesses_vm",
    "make_online_search_state",
    "render_continuation_payload_shape",
    "resume_online_search_from_snapshot",
    "_pop_resumable_frame",
    "_resume_from_frames",
)
