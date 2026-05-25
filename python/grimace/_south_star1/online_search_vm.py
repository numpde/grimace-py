"""Explicit-frame online search VM for South Star witness enumeration."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from itertools import product

from .facts import MoleculeFacts
from .ids import AtomId
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionRecorder
from .online_render_sink import OnlineRenderSink
from .online_render_sink import OnlineStringBuffer
from .online_stereo_witness import _DirectionalSolution
from .online_stereo_witness import _PrefixChoice
from .online_stereo_witness import _atom_text_decision_value
from .online_stereo_witness import _bond_decode_ok
from .online_stereo_witness import _bond_text_decision_value
from .online_stereo_witness import _direction_domains
from .online_stereo_witness import _directional_models_by_site
from .online_stereo_witness import _forced_tetra_tokens
from .online_stereo_witness import _render_online_to_sink
from .online_stereo_witness import _ring_label_decision_value
from .online_stereo_witness import _slot_view_for_trace
from .online_stereo_witness import _support_maximal
from .online_stereo_witness import _validate_annotation_mode
from .online_stereo_witness import iter_online_ring_label_assignments
from .online_stereo_witness import OnlineWitness
from .online_traversal import iter_online_traversal_traces
from .online_traversal import trace_to_skeleton_like_key
from .policy import AnnotationMode
from .policy import DirectionMark
from .policy import SmilesPolicy
from .residual_constraints import DirectionalResidualFactor
from .residual_constraints import ResidualStore
from .residual_constraints import direction_var
from .semantics import ParserSemantics
from .stereo_templates import StereoTemplateBundle
from .stereo_templates import build_stereo_templates


@dataclass(frozen=True, slots=True)
class OnlineSearchFrame:
    kind: str
    data: tuple[object, ...]
    cursor: int = 0


@dataclass(frozen=True, slots=True)
class OnlineSearchSnapshot:
    traversal_state: object
    residual_snapshot: object
    ring_state: object
    output_snapshot: object
    decision_snapshot: object
    frame_stack: tuple[OnlineSearchFrame, ...]


@dataclass(slots=True)
class MutableTraversalState:
    current_key: tuple[object, ...] | None = None

    def checkpoint(self) -> object:
        return self.current_key

    def rollback(self, token: object) -> None:
        self.current_key = token  # type: ignore[assignment]


@dataclass(slots=True)
class MutableRingState:
    labels: tuple[tuple[int, int], ...] = ()

    def checkpoint(self) -> object:
        return self.labels

    def rollback(self, token: object) -> None:
        self.labels = token  # type: ignore[assignment]


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
            residual_snapshot=self.residual.checkpoint(),
            ring_state=self.ring.checkpoint(),
            output_snapshot=self.output.checkpoint(),
            decision_snapshot=self.decisions.checkpoint(),
            frame_stack=tuple(self.frames),
        )

    def rollback(self, snapshot: OnlineSearchSnapshot) -> None:
        self.traversal.rollback(snapshot.traversal_state)
        self.residual.rollback(snapshot.residual_snapshot)  # type: ignore[arg-type]
        self.ring.rollback(snapshot.ring_state)
        self.output.rollback(snapshot.output_snapshot)
        self.decisions.rollback(snapshot.decision_snapshot)  # type: ignore[arg-type]
        self.frames[:] = list(snapshot.frame_stack)


@dataclass(frozen=True, slots=True)
class OnlineResidualContinuation:
    prefix: str
    snapshot: OnlineSearchSnapshot


def make_online_search_state(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    sink: OnlineRenderSink | None = None,
) -> OnlineSearchState:
    facts.validate()
    policy.validate_for_facts(facts)
    return OnlineSearchState(
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=build_stereo_templates(facts),
        traversal=MutableTraversalState(),
        residual=ResidualStore(),
        ring=MutableRingState(),
        output=sink if sink is not None else OnlineStringBuffer(),
        decisions=OnlineDecisionRecorder(),
    )


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
    sink_factory: Callable[[], OnlineRenderSink] | None = None,
) -> Iterator[OnlineWitness]:
    state = make_online_search_state(
        facts=facts,
        policy=policy,
        semantics=semantics,
    )
    _validate_annotation_mode(policy)
    factory = sink_factory or OnlineStringBuffer

    for trace in iter_online_traversal_traces(facts=facts, policy=policy):
        traversal_key = trace_to_skeleton_like_key(trace)
        state.traversal.current_key = traversal_key
        state.frames.append(OnlineSearchFrame("traversal", (traversal_key,)))
        decision_checkpoint = state.decisions.checkpoint()
        state.decisions.push(OnlineDecision("traversal", (traversal_key,)))
        try:
            yield from _iter_witnesses_for_trace_vm(
                state=state,
                trace=trace,
                sink_factory=factory,
            )
        finally:
            state.decisions.rollback(decision_checkpoint)
            state.frames.pop()
            state.traversal.current_key = None


def _iter_witnesses_for_trace_vm(
    *,
    state: OnlineSearchState,
    trace,
    sink_factory: Callable[[], OnlineRenderSink],
) -> Iterator[OnlineWitness]:
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
    ring_assignments = tuple(
        iter_online_ring_label_assignments(trace=trace, policy=state.policy)
    )
    for ring_labels in _iter_sequence_frame(
        state,
        kind="ring-label-assignment",
        values=ring_assignments,
    ):
        state.ring.labels = tuple(
            sorted((endpoint, label.value) for endpoint, label in ring_labels.items())
        )
        ring_decision_checkpoint = state.decisions.checkpoint()
        state.decisions.push(OnlineDecision("ring_labels", _ring_label_decision_value(ring_labels)))
        try:
            for atom_text in _iter_dict_product_frame(
                state,
                kind="atom-text",
                domains=atom_domains,
            ):
                atom_decision_checkpoint = state.decisions.checkpoint()
                state.decisions.push(OnlineDecision("atom_text", _atom_text_decision_value(atom_text)))
                try:
                    tetra_tokens = _forced_tetra_tokens(
                        facts=state.facts,
                        trace=trace,
                        templates=state.templates,
                        atom_text=atom_text,
                    )
                    if tetra_tokens is None:
                        continue
                    for bond_text in _iter_dict_product_frame(
                        state,
                        kind="bond-text",
                        domains=bond_domains,
                    ):
                        bond_decision_checkpoint = state.decisions.checkpoint()
                        state.decisions.push(
                            OnlineDecision("bond_text", _bond_text_decision_value(bond_text))
                        )
                        try:
                            prefix = _PrefixChoice(
                                atom_text=atom_text,
                                bond_text=bond_text,
                                ring_labels=ring_labels,
                            )
                            solutions = tuple(
                                _iter_directional_solutions_vm(
                                    state=state,
                                    trace=trace,
                                    slots=slots,
                                    prefix=prefix,
                                    tetra_tokens=tetra_tokens,
                                    sink_factory=sink_factory,
                                )
                            )
                            if state.policy.annotation_mode is AnnotationMode.SUPPORT_MAXIMAL:
                                solutions = tuple(_support_maximal(solutions))
                            for solution in solutions:
                                yield OnlineWitness(
                                    rendered=solution.rendered,
                                    traversal_key=trace_to_skeleton_like_key(trace),
                                    annotation_count=solution.annotation_count,
                                )
                        finally:
                            state.decisions.rollback(bond_decision_checkpoint)
                finally:
                    state.decisions.rollback(atom_decision_checkpoint)
        finally:
            state.decisions.rollback(ring_decision_checkpoint)
            state.ring.labels = ()


def _iter_directional_solutions_vm(
    *,
    state: OnlineSearchState,
    trace,
    slots,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, object],
    sink_factory: Callable[[], OnlineRenderSink],
) -> Iterator[_DirectionalSolution]:
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
        for marks in _iter_dict_product_frame(
            state,
            kind="direction-mark",
            domains=mark_domains,
        ):
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
            decision_checkpoint = state.decisions.checkpoint()
            for carrier_id in carriers:
                state.decisions.push(
                    OnlineDecision("direction_mark", (carrier_id, marks[carrier_id].name))
                )
            try:
                if not all(
                    store.assign(direction_var(carrier_id), marks[carrier_id])
                    for carrier_id in carriers
                ):
                    continue
                rendered = _directional_solution_rendered(
                    state=state,
                    trace=trace,
                    slots=slots,
                    prefix=prefix,
                    tetra_tokens=tetra_tokens,
                    marks=marks,
                    factor_ids=factor_ids,
                    sink_factory=sink_factory,
                )
                if rendered is None:
                    continue
                yield _directional_solution(
                    rendered=rendered,
                    marks=marks,
                )
            finally:
                state.decisions.rollback(decision_checkpoint)
    finally:
        state.residual = previous_store


def _directional_solution_rendered(
    *,
    state: OnlineSearchState,
    trace,
    slots,
    prefix: _PrefixChoice,
    tetra_tokens: dict[AtomId, object],
    marks: dict[int, DirectionMark],
    factor_ids: list[int],
    sink_factory: Callable[[], OnlineRenderSink],
) -> str | None:
    if not _bond_decode_ok(
        facts=state.facts,
        semantics=state.semantics,
        slots=slots,
        prefix=prefix,
        marks=marks,
    ):
        return None
    if not all(state.residual.close_factor(factor_id) for factor_id in factor_ids):
        return None
    state.output = sink_factory()
    checkpoint = state.output.checkpoint()
    state.frames.append(OnlineSearchFrame("completion", (trace_to_skeleton_like_key(trace),)))
    try:
        if not _render_online_to_sink(
            trace=trace,
            slots=slots,
            prefix=prefix,
            tetra_tokens=tetra_tokens,  # type: ignore[arg-type]
            marks=marks,
            sink=state.output,
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


def _directional_solution(
    *,
    rendered: str,
    marks: dict[int, DirectionMark],
) -> _DirectionalSolution:
    support = frozenset(
        carrier_id
        for carrier_id, mark in marks.items()
        if mark is not DirectionMark.ABSENT
    )
    return _DirectionalSolution(
        rendered=rendered,
        support=support,
        annotation_count=len(support),
    )


def _iter_dict_product_frame(
    state: OnlineSearchState,
    *,
    kind: str,
    domains,
):
    keys = tuple(key for key, _ in domains)
    value_domains = tuple(tuple(values) for _, values in domains)
    if any(not values for values in value_domains):
        return
    for assignment in _iter_product_frame(
        state,
        kind=kind,
        values=tuple(product(*value_domains)),
    ):
        yield dict(zip(keys, assignment, strict=True))


def _iter_sequence_frame(
    state: OnlineSearchState,
    *,
    kind: str,
    values: tuple[object, ...],
):
    for value in _iter_product_frame(
        state,
        kind=kind,
        values=tuple((item,) for item in values),
    ):
        yield value[0]


def _iter_product_frame(
    state: OnlineSearchState,
    *,
    kind: str,
    values: tuple[tuple[object, ...], ...],
):
    index = 0
    while index < len(values):
        frame = OnlineSearchFrame(kind, (len(values),), index)
        state.frames.append(frame)
        try:
            yield values[index]
        finally:
            state.frames.pop()
        index += 1


def _advance_top_frame(state: OnlineSearchState, index: int) -> None:
    frame = state.frames[index]
    state.frames[index] = OnlineSearchFrame(
        frame.kind,
        frame.data,
        frame.cursor + 1,
    )


__all__ = (
    "MutableRingState",
    "MutableTraversalState",
    "OnlineResidualContinuation",
    "OnlineSearchFrame",
    "OnlineSearchSnapshot",
    "OnlineSearchState",
    "capture_residual_continuation",
    "iter_online_stereo_witness_strings_vm",
    "iter_online_stereo_witnesses_vm",
    "make_online_search_state",
)
