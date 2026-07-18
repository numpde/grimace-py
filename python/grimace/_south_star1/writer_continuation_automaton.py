"""Count-free compilation of exact weighted writer continuations."""

from __future__ import annotations

from collections import Counter
from collections import deque
from dataclasses import dataclass
from dataclasses import replace
from math import gcd
from time import perf_counter_ns

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _canonical_json
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _term
from .writer_frontier import _checked_writer_frontier_branch_supports
from .writer_frontier import WriterFrontierCursor
from .writer_snapshot import validate_writer_search_snapshot


_MAX_CANONICAL_CORE_BYTES = 25_000_000
_MAX_SEMANTIC_EDGES = 50_000


@dataclass(frozen=True, slots=True)
class WriterContinuationChoice:
    emitted_text: str
    immediate_multiplicity: int
    successor_node_id: int
    successor_scale: int
    support_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterContinuationNode:
    node_id: int
    signature_digest: str
    terminal_available: bool
    terminal_multiplicity: int
    terminal_completion_count: int
    choices: tuple[WriterContinuationChoice, ...]
    support_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterContinuationCursor:
    node_id: int
    completion_scale: int

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("continuation node ID must be nonnegative")
        if self.completion_scale <= 0:
            raise ValueError("continuation completion scale must be positive")


@dataclass(frozen=True, slots=True)
class WriterContinuationRawCursorRecord:
    raw_cursor_digest: str
    primitive_cursor_digest: str
    normalization_scale: int
    compiled_node_id: int
    token_depth: int
    predecessor_edge_id: str | None


@dataclass(frozen=True, slots=True)
class WriterContinuationPrimitiveRecord:
    primitive_cursor_digest: str
    compiled_node_id: int
    representative_raw_cursor_digest: str


@dataclass(frozen=True, slots=True)
class WriterContinuationEdgeRecord:
    edge_id: str
    source_raw_cursor_digest: str
    source_node_id: int
    emitted_text: str
    text_projection_digest: str
    branch_certificate_digests: tuple[str, ...]
    successor_raw_cursor_digest: str
    successor_node_id: int
    successor_scale: int


@dataclass(frozen=True, slots=True)
class WriterContinuationTerminalRecord:
    source_raw_cursor_digest: str
    source_node_id: int
    terminal_support_identity_digests: tuple[str, ...]
    finalized_cursor_digest: str


@dataclass(frozen=True, slots=True)
class WriterContinuationProvenance:
    source_snapshot_digest: str
    root_raw_cursor_digest: str
    raw_cursors: tuple[WriterContinuationRawCursorRecord, ...]
    primitives: tuple[WriterContinuationPrimitiveRecord, ...]
    edges: tuple[WriterContinuationEdgeRecord, ...]
    terminals: tuple[WriterContinuationTerminalRecord, ...]


@dataclass(frozen=True, slots=True)
class WriterContinuationMetrics:
    raw_cursor_count: int
    primitive_cursor_count: int
    semantic_node_count: int
    semantic_edge_count: int
    terminal_node_count: int
    maximum_depth: int
    maximum_out_degree: int
    largest_equivalence_class_membership: int
    weight_normalization_merge_count: int
    semantic_minimization_merge_count: int
    canonical_core_bytes: int
    provenance_index_bytes: int
    compile_time_ns: int
    peak_active_depth: int
    peak_primitive_memo_size: int
    peak_signature_memo_size: int


@dataclass(frozen=True, slots=True)
class WriterContinuationAutomaton:
    root: WriterContinuationCursor
    nodes: tuple[WriterContinuationNode, ...]
    provenance: WriterContinuationProvenance
    metrics: WriterContinuationMetrics


@dataclass(frozen=True, slots=True)
class WriterContinuationCore:
    root: WriterContinuationCursor
    nodes: tuple[WriterContinuationNode, ...]
    metrics: WriterContinuationMetrics


@dataclass(frozen=True, slots=True)
class WriterContinuationProbability:
    emitted_text: str | None
    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class WriterContinuationAutomatonVerification:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CompiledCursor:
    node_id: int
    scale: int
    depth: int


@dataclass(frozen=True, slots=True)
class _CursorTrace:
    primitive_cursor: WriterFrontierCursor
    normalization_scale: int
    compiled_node_id: int


class _Compiler:
    def __init__(
        self,
        *,
        prepared,
        snapshot,
        enforce_limits: bool,
        signature_digest_function,
    ) -> None:
        self.prepared = prepared
        self.snapshot = snapshot
        self.enforce_limits = enforce_limits
        self.signature_digest_function = signature_digest_function
        self.nodes: list[WriterContinuationNode] = []
        self.node_by_signature: dict[tuple[object, ...], int] = {}
        self.signature_by_node: dict[int, tuple[object, ...]] = {}
        self.primitive_memo: dict[WriterFrontierCursor, tuple[int, int]] = {}
        self.active: set[WriterFrontierCursor] = set()
        self.cursor_traces: dict[WriterFrontierCursor, _CursorTrace] = {}
        self.edge_provenance: dict[
            tuple[WriterFrontierCursor, str],
            WriterContinuationEdgeRecord,
        ] = {}
        self.terminal_provenance: dict[
            WriterFrontierCursor,
            WriterContinuationTerminalRecord,
        ] = {}
        self.node_memberships: Counter[int] = Counter()
        self.peak_active_depth = 0
        self.peak_primitive_memo_size = 0
        self.peak_signature_memo_size = 0

    def compile(self) -> WriterContinuationAutomaton:
        started = perf_counter_ns()
        validate_writer_search_snapshot(self.snapshot, prepared=self.prepared)
        compiled_root = self._compile_cursor(self.snapshot.cursor)
        internal_root = WriterContinuationCursor(
            node_id=compiled_root.node_id,
            completion_scale=compiled_root.scale,
        )
        root, nodes, canonical_ids = _canonicalize_nodes(
            root=internal_root,
            nodes=tuple(self.nodes),
            signature_digest_function=self.signature_digest_function,
        )
        provenance = self._compact_provenance(canonical_ids=canonical_ids)
        core_bytes = _canonical_size((root, nodes))
        provenance_bytes = _provenance_record_bytes(provenance)
        edge_count = sum(len(node.choices) for node in nodes)
        canonical_memberships = Counter(
            {
                canonical_ids[node_id]: count
                for node_id, count in self.node_memberships.items()
            }
        )
        if self.enforce_limits and core_bytes >= _MAX_CANONICAL_CORE_BYTES:
            _violation("continuation_canonical_core_too_large")
        if self.enforce_limits and edge_count > _MAX_SEMANTIC_EDGES:
            _violation("continuation_semantic_edge_limit_exceeded")
        metrics = WriterContinuationMetrics(
            raw_cursor_count=len(provenance.raw_cursors),
            primitive_cursor_count=len(provenance.primitives),
            semantic_node_count=len(nodes),
            semantic_edge_count=edge_count,
            terminal_node_count=sum(node.terminal_available for node in nodes),
            maximum_depth=compiled_root.depth,
            maximum_out_degree=max((len(node.choices) for node in nodes), default=0),
            largest_equivalence_class_membership=max(
                canonical_memberships.values(), default=0
            ),
            weight_normalization_merge_count=(
                sum(
                    item.normalization_scale > 1
                    for item in provenance.raw_cursors
                )
            ),
            semantic_minimization_merge_count=(
                len(provenance.primitives) - len(nodes)
            ),
            canonical_core_bytes=core_bytes,
            provenance_index_bytes=provenance_bytes,
            compile_time_ns=perf_counter_ns() - started,
            peak_active_depth=self.peak_active_depth,
            peak_primitive_memo_size=self.peak_primitive_memo_size,
            peak_signature_memo_size=self.peak_signature_memo_size,
        )
        return WriterContinuationAutomaton(
            root=root,
            nodes=nodes,
            provenance=provenance,
            metrics=metrics,
        )

    def _compact_provenance(
        self, *, canonical_ids: dict[int, int]
    ) -> WriterContinuationProvenance:
        root_digest = _identity_digest(self.snapshot.cursor)
        compact_edges: list[WriterContinuationEdgeRecord] = []
        cursor_by_digest = {
            _identity_digest(cursor): cursor for cursor in self.cursor_traces
        }
        for item in self.edge_provenance.values():
            terms = (
                item.source_raw_cursor_digest,
                item.emitted_text,
                item.text_projection_digest,
                item.branch_certificate_digests,
                item.successor_raw_cursor_digest,
            )
            compact_edges.append(
                replace(
                    item,
                    edge_id=_identity_digest(terms),
                    source_node_id=canonical_ids[item.source_node_id],
                    successor_node_id=canonical_ids[item.successor_node_id],
                )
            )
        compact_edges.sort(
            key=lambda item: (
                item.source_raw_cursor_digest,
                item.emitted_text,
                item.text_projection_digest,
            )
        )
        reachable, depths, predecessors = _canonical_predecessor_tree(
            root_raw_cursor_digest=root_digest,
            edges=tuple(compact_edges),
        )
        raw_records = []
        primitive_representatives: dict[str, list[str]] = {}
        for raw_digest in sorted(reachable):
            cursor = cursor_by_digest.get(raw_digest)
            if cursor is None:
                _violation("continuation_reachable_cursor_trace_missing")
            trace = self.cursor_traces[cursor]
            primitive_digest = _identity_digest(trace.primitive_cursor)
            primitive_representatives.setdefault(primitive_digest, []).append(
                raw_digest
            )
            raw_records.append(
                WriterContinuationRawCursorRecord(
                    raw_cursor_digest=raw_digest,
                    primitive_cursor_digest=primitive_digest,
                    normalization_scale=trace.normalization_scale,
                    compiled_node_id=canonical_ids[trace.compiled_node_id],
                    token_depth=depths[raw_digest],
                    predecessor_edge_id=predecessors.get(raw_digest),
                )
            )
        primitive_records = []
        for primitive, (node_id, _depth) in self.primitive_memo.items():
            primitive_digest = _identity_digest(primitive)
            representatives = primitive_representatives.get(primitive_digest, ())
            if not representatives:
                _violation("continuation_primitive_representative_missing")
            primitive_records.append(
                WriterContinuationPrimitiveRecord(
                    primitive_cursor_digest=primitive_digest,
                    compiled_node_id=canonical_ids[node_id],
                    representative_raw_cursor_digest=min(representatives),
                )
            )
        compact_terminals = tuple(
            sorted(
                (
                    replace(
                        item,
                        source_node_id=canonical_ids[item.source_node_id],
                    )
                    for cursor, item in self.terminal_provenance.items()
                    if _identity_digest(cursor) in reachable
                ),
                key=lambda item: item.source_raw_cursor_digest,
            )
        )
        return WriterContinuationProvenance(
            source_snapshot_digest=_identity_digest(self.snapshot),
            root_raw_cursor_digest=root_digest,
            raw_cursors=tuple(raw_records),
            primitives=tuple(
                sorted(
                    primitive_records,
                    key=lambda item: item.primitive_cursor_digest,
                )
            ),
            edges=tuple(
                item
                for item in compact_edges
                if item.source_raw_cursor_digest in reachable
            ),
            terminals=compact_terminals,
        )

    def _compile_cursor(self, raw_cursor: WriterFrontierCursor) -> _CompiledCursor:
        primitive, scale = _normalize_cursor(raw_cursor)
        scaled_batch = None
        if scale > 1:
            scaled_batch = self._check_scaling(
                raw_cursor=raw_cursor,
                primitive_cursor=primitive,
                scale=scale,
            )
        if primitive in self.primitive_memo:
            node_id, depth = self.primitive_memo[primitive]
            self._record_cursor(primitive, primitive, 1, node_id)
            self._record_cursor(raw_cursor, primitive, scale, node_id)
            if scaled_batch is not None:
                self._record_batch_provenance(
                    source_cursor=raw_cursor,
                    source_node_id=node_id,
                    batch=scaled_batch,
                )
            return _CompiledCursor(node_id=node_id, scale=scale, depth=depth)
        if primitive in self.active:
            _violation("continuation_cursor_cycle")

        self.active.add(primitive)
        self.peak_active_depth = max(self.peak_active_depth, len(self.active))
        try:
            batch = _frontier_batch(self.prepared, primitive)
            projections = tuple(batch.text_choice_projection_certificates)
            texts = tuple(projection.emitted_text for projection in projections)
            if texts != tuple(sorted(texts)) or len(texts) != len(set(texts)):
                _violation("continuation_duplicate_or_unsorted_emitted_text")
            choices: list[WriterContinuationChoice] = []
            child_depth = 0
            for projection in projections:
                _check_projection_membership(
                    source_cursor=primitive,
                    projection=projection,
                    batch=batch,
                )
                child = self._compile_cursor(projection.successor_cursor)
                successor_node = self.nodes[child.node_id]
                choices.append(
                    WriterContinuationChoice(
                        emitted_text=projection.emitted_text,
                        immediate_multiplicity=projection.immediate_multiplicity,
                        successor_node_id=child.node_id,
                        successor_scale=child.scale,
                        support_count=successor_node.support_count,
                        completion_count=(
                            child.scale * successor_node.completion_count
                        ),
                    )
                )
                child_depth = max(child_depth, child.depth)

            terminal = batch.choices.terminal
            terminal_available = terminal is not None
            terminal_multiplicity = 0 if terminal is None else terminal.multiplicity
            terminal_completion_count = terminal_multiplicity
            choice_terms = tuple(
                (
                    choice.emitted_text,
                    choice.immediate_multiplicity,
                    choice.successor_scale,
                    choice.successor_node_id,
                )
                for choice in choices
            )
            signature = (
                terminal_available,
                terminal_multiplicity,
                terminal_completion_count,
                choice_terms,
            )
            node_id = self.node_by_signature.get(signature)
            if node_id is None:
                node_id = len(self.nodes)
                node = WriterContinuationNode(
                    node_id=node_id,
                    signature_digest=self.signature_digest_function(signature),
                    terminal_available=terminal_available,
                    terminal_multiplicity=terminal_multiplicity,
                    terminal_completion_count=terminal_completion_count,
                    choices=tuple(choices),
                    support_count=(
                        int(terminal_available)
                        + sum(choice.support_count for choice in choices)
                    ),
                    completion_count=(
                        terminal_completion_count
                        + sum(choice.completion_count for choice in choices)
                    ),
                )
                self.nodes.append(node)
                self.node_by_signature[signature] = node_id
                self.signature_by_node[node_id] = signature
                self.peak_signature_memo_size = max(
                    self.peak_signature_memo_size,
                    len(self.node_by_signature),
                )
            else:
                existing = self.nodes[node_id]
                if self.signature_by_node[node_id] != signature:
                    _violation("continuation_semantic_signature_mismatch")
                if (
                    existing.support_count
                    != int(terminal_available)
                    + sum(choice.support_count for choice in choices)
                    or existing.completion_count
                    != terminal_completion_count
                    + sum(choice.completion_count for choice in choices)
                ):
                    _violation("continuation_semantic_merge_count_mismatch")

            depth = 1 + child_depth
            self.primitive_memo[primitive] = (node_id, depth)
            self.node_memberships[node_id] += 1
            self.peak_primitive_memo_size = max(
                self.peak_primitive_memo_size,
                len(self.primitive_memo),
            )
            self._record_cursor(primitive, primitive, 1, node_id)
            self._record_cursor(raw_cursor, primitive, scale, node_id)
            self._record_batch_provenance(
                source_cursor=primitive,
                source_node_id=node_id,
                batch=batch,
            )
            if scaled_batch is not None:
                self._record_batch_provenance(
                    source_cursor=raw_cursor,
                    source_node_id=node_id,
                    batch=scaled_batch,
                )
            return _CompiledCursor(node_id=node_id, scale=scale, depth=depth)
        finally:
            self.active.remove(primitive)

    def _record_cursor(
        self,
        raw_cursor: WriterFrontierCursor,
        primitive_cursor: WriterFrontierCursor,
        scale: int,
        node_id: int,
    ) -> None:
        item = _CursorTrace(
            primitive_cursor=primitive_cursor,
            normalization_scale=scale,
            compiled_node_id=node_id,
        )
        previous = self.cursor_traces.get(raw_cursor)
        if previous is not None and previous != item:
            _violation("continuation_cursor_provenance_mismatch")
        self.cursor_traces[raw_cursor] = item

    def _record_batch_provenance(
        self,
        *,
        source_cursor: WriterFrontierCursor,
        source_node_id: int,
        batch,
    ) -> None:
        source_digest = _identity_digest(source_cursor)
        for projection in batch.text_choice_projection_certificates:
            successor_compiled = self._compile_cursor(
                projection.successor_cursor
            )
            item = WriterContinuationEdgeRecord(
                edge_id="",
                source_raw_cursor_digest=source_digest,
                source_node_id=source_node_id,
                emitted_text=projection.emitted_text,
                text_projection_digest=_identity_digest(projection),
                branch_certificate_digests=tuple(
                    _identity_digest(certificate)
                    for certificate in projection.branch_certificates
                ),
                successor_raw_cursor_digest=_identity_digest(
                    projection.successor_cursor
                ),
                successor_node_id=successor_compiled.node_id,
                successor_scale=successor_compiled.scale,
            )
            key = (source_cursor, projection.emitted_text)
            previous = self.edge_provenance.get(key)
            if previous is not None and previous != item:
                _violation("continuation_edge_provenance_mismatch")
            self.edge_provenance[key] = item
        terminal = batch.choices.terminal
        if terminal is not None:
            item = WriterContinuationTerminalRecord(
                source_raw_cursor_digest=source_digest,
                source_node_id=source_node_id,
                terminal_support_identity_digests=tuple(
                    _identity_digest(support.checked_terminal_certificate)
                    for support in batch.terminal_supports
                ),
                finalized_cursor_digest=_identity_digest(
                    terminal.finalized_cursor
                ),
            )
            previous = self.terminal_provenance.get(source_cursor)
            if previous is not None and previous != item:
                _violation("continuation_terminal_provenance_mismatch")
            self.terminal_provenance[source_cursor] = item

    def _check_scaling(
        self,
        *,
        raw_cursor: WriterFrontierCursor,
        primitive_cursor: WriterFrontierCursor,
        scale: int,
    ):
        raw = _frontier_batch(self.prepared, raw_cursor)
        primitive = _frontier_batch(self.prepared, primitive_cursor)
        raw_by_text = {
            item.emitted_text: item
            for item in raw.text_choice_projection_certificates
        }
        primitive_by_text = {
            item.emitted_text: item
            for item in primitive.text_choice_projection_certificates
        }
        if tuple(sorted(raw_by_text)) != tuple(sorted(primitive_by_text)):
            _violation("continuation_cursor_scaling_mismatch")
        for emitted_text, raw_projection in raw_by_text.items():
            primitive_projection = primitive_by_text[emitted_text]
            if (
                raw_projection.immediate_multiplicity
                != scale * primitive_projection.immediate_multiplicity
                or raw_projection.successor_cursor
                != _scaled_cursor(primitive_projection.successor_cursor, scale)
            ):
                _violation("continuation_cursor_scaling_mismatch")
        raw_terminal = raw.choices.terminal
        primitive_terminal = primitive.choices.terminal
        if (raw_terminal is None) != (primitive_terminal is None):
            _violation("continuation_cursor_scaling_mismatch")
        if raw_terminal is not None and (
            raw_terminal.multiplicity != scale * primitive_terminal.multiplicity
            or raw_terminal.finalized_cursor
            != _scaled_cursor(primitive_terminal.finalized_cursor, scale)
        ):
            _violation("continuation_cursor_scaling_mismatch")
        if len(raw.terminal_supports) != len(primitive.terminal_supports):
            _violation("continuation_cursor_scaling_mismatch")
        for raw_support, primitive_support in zip(
            raw.terminal_supports,
            primitive.terminal_supports,
        ):
            if (
                raw_support.source_state != primitive_support.source_state
                or raw_support.finalized_state
                != primitive_support.finalized_state
                or raw_support.terminal_ordinal
                != primitive_support.terminal_ordinal
                or raw_support.parent_weight
                != scale * primitive_support.parent_weight
            ):
                _violation("continuation_cursor_scaling_mismatch")
        return raw


def compile_writer_continuation_automaton(
    *, prepared, snapshot, _signature_digest_function=_identity_digest
) -> WriterContinuationAutomaton:
    return _Compiler(
        prepared=prepared,
        snapshot=snapshot,
        enforce_limits=True,
        signature_digest_function=_signature_digest_function,
    ).compile()


def writer_continuation_choices(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor | None = None,
) -> tuple[WriterContinuationChoice, ...]:
    cursor = automaton.root if cursor is None else cursor
    node = _node_for_cursor(automaton, cursor)
    return tuple(
        replace(
            choice,
            immediate_multiplicity=(
                cursor.completion_scale * choice.immediate_multiplicity
            ),
            successor_scale=cursor.completion_scale * choice.successor_scale,
            completion_count=cursor.completion_scale * choice.completion_count,
        )
        for choice in node.choices
    )


def advance_writer_continuation(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor,
    emitted_text: str,
) -> WriterContinuationCursor:
    choices = tuple(
        choice
        for choice in writer_continuation_choices(automaton, cursor)
        if choice.emitted_text == emitted_text
    )
    if len(choices) != 1:
        _violation("continuation_emitted_text_not_available")
    choice = choices[0]
    return WriterContinuationCursor(
        node_id=choice.successor_node_id,
        completion_scale=choice.successor_scale,
    )


def writer_continuation_is_terminal(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor | None = None,
) -> bool:
    cursor = automaton.root if cursor is None else cursor
    return _node_for_cursor(automaton, cursor).terminal_available


def writer_continuation_support_count(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor | None = None,
) -> int:
    cursor = automaton.root if cursor is None else cursor
    return _node_for_cursor(automaton, cursor).support_count


def writer_continuation_completion_count(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor | None = None,
) -> int:
    cursor = automaton.root if cursor is None else cursor
    return cursor.completion_scale * _node_for_cursor(
        automaton, cursor
    ).completion_count


def writer_continuation_probabilities(
    automaton: WriterContinuationAutomaton,
    cursor: WriterContinuationCursor | None = None,
) -> tuple[WriterContinuationProbability, ...]:
    cursor = automaton.root if cursor is None else cursor
    node = _node_for_cursor(automaton, cursor)
    denominator = cursor.completion_scale * node.completion_count
    if denominator <= 0:
        _violation("continuation_has_no_completion")
    values = [
        WriterContinuationProbability(
            emitted_text=choice.emitted_text,
            numerator=choice.completion_count,
            denominator=denominator,
        )
        for choice in writer_continuation_choices(automaton, cursor)
    ]
    if node.terminal_available:
        values.append(
            WriterContinuationProbability(
                emitted_text=None,
                numerator=(
                    cursor.completion_scale
                    * node.terminal_completion_count
                ),
                denominator=denominator,
            )
        )
    if sum(item.numerator for item in values) != denominator:
        _violation("continuation_probability_normalization_mismatch")
    return tuple(values)


def writer_continuation_provenance_edge(
    automaton: WriterContinuationAutomaton,
    *,
    source_raw_cursor_digest: str,
    emitted_text: str,
) -> WriterContinuationEdgeRecord:
    matches = tuple(
        item
        for item in automaton.provenance.edges
        if item.source_raw_cursor_digest == source_raw_cursor_digest
        and item.emitted_text == emitted_text
    )
    if len(matches) != 1:
        _violation("continuation_provenance_edge_not_unique")
    return matches[0]


def writer_continuation_terminal_provenance(
    automaton: WriterContinuationAutomaton,
    *,
    source_raw_cursor_digest: str,
) -> WriterContinuationTerminalRecord:
    matches = tuple(
        item
        for item in automaton.provenance.terminals
        if item.source_raw_cursor_digest == source_raw_cursor_digest
    )
    if len(matches) != 1:
        _violation("continuation_terminal_provenance_not_unique")
    return matches[0]


def verify_writer_continuation_automaton_consistency(
    *,
    automaton,
    prepared=None,
    snapshot=None,
    _signature_digest_function=_identity_digest,
) -> WriterContinuationAutomatonVerification:
    try:
        _verify_internal_consistency(
            automaton,
            signature_digest_function=_signature_digest_function,
        )
        if (prepared is None) != (snapshot is None):
            _violation("continuation_live_authority_incomplete")
        if prepared is not None:
            expected = _Compiler(
                prepared=prepared,
                snapshot=snapshot,
                enforce_limits=True,
                signature_digest_function=_signature_digest_function,
            ).compile()
            if (
                automaton.root != expected.root
                or automaton.nodes != expected.nodes
                or automaton.provenance != expected.provenance
                or replace(automaton.metrics, compile_time_ns=0)
                != replace(expected.metrics, compile_time_ns=0)
            ):
                _violation("continuation_live_recompile_mismatch")
        return WriterContinuationAutomatonVerification(accepted=True)
    except SouthStarError as exc:
        return WriterContinuationAutomatonVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "continuation_verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterContinuationAutomatonVerification(
            accepted=False,
            reason=f"malformed_continuation_automaton:{type(exc).__name__}:{exc}",
        )


def _verify_internal_consistency(
    automaton: WriterContinuationAutomaton, *, signature_digest_function
) -> None:
    _verify_core_consistency(
        automaton,
        signature_digest_function=signature_digest_function,
    )
    if automaton.metrics.provenance_index_bytes != _provenance_record_bytes(
        automaton.provenance
    ):
        _violation("continuation_provenance_size_metric_mismatch")
    _verify_provenance(automaton)


def _verify_core_consistency(automaton, *, signature_digest_function) -> None:
    nodes = automaton.nodes
    if tuple(node.node_id for node in nodes) != tuple(range(len(nodes))):
        _violation("continuation_node_id_mismatch")
    if not 0 <= automaton.root.node_id < len(nodes):
        _violation("continuation_root_node_mismatch")
    if automaton.root.completion_scale <= 0:
        _violation("continuation_root_scale_mismatch")
    visiting: set[int] = set()
    visited: set[int] = set()
    depth_by_node: dict[int, int] = {}
    signatures: dict[tuple[object, ...], int] = {}

    def check(node_id: int) -> int:
        if node_id in visiting:
            _violation("continuation_cycle")
        if node_id in visited:
            return depth_by_node[node_id]
        if not 0 <= node_id < len(nodes):
            _violation("continuation_successor_node_mismatch")
        visiting.add(node_id)
        node = nodes[node_id]
        texts = tuple(choice.emitted_text for choice in node.choices)
        if texts != tuple(sorted(texts)) or len(texts) != len(set(texts)):
            _violation("continuation_duplicate_or_unsorted_emitted_text")
        child_depth = 0
        for choice in node.choices:
            child_depth = max(child_depth, check(choice.successor_node_id))
            child = nodes[choice.successor_node_id]
            if choice.successor_scale <= 0 or choice.immediate_multiplicity <= 0:
                _violation("continuation_choice_weight_mismatch")
            if choice.support_count != child.support_count:
                _violation("continuation_choice_support_count_mismatch")
            if choice.completion_count != (
                choice.successor_scale * child.completion_count
            ):
                _violation("continuation_choice_completion_count_mismatch")
        if (
            node.terminal_available != (node.terminal_multiplicity > 0)
            or node.terminal_completion_count != node.terminal_multiplicity
        ):
            _violation("continuation_terminal_weight_mismatch")
        if node.support_count != (
            int(node.terminal_available)
            + sum(choice.support_count for choice in node.choices)
        ):
            _violation("continuation_support_count_mismatch")
        if node.completion_count != (
            node.terminal_completion_count
            + sum(choice.completion_count for choice in node.choices)
        ):
            _violation("continuation_completion_count_mismatch")
        signature = _signature_for_node(node=node, nodes=nodes)
        if node.signature_digest != signature_digest_function(signature):
            _violation("continuation_signature_digest_mismatch")
        if signature in signatures:
            _violation("continuation_noncanonical_semantic_split")
        signatures[signature] = node_id
        visiting.remove(node_id)
        visited.add(node_id)
        depth_by_node[node_id] = 1 + child_depth
        return depth_by_node[node_id]

    depth = check(automaton.root.node_id)
    if len(visited) != len(nodes):
        _violation("continuation_unreachable_node")
    if automaton.metrics.maximum_depth != depth:
        _violation("continuation_depth_metric_mismatch")
    if automaton.metrics.semantic_node_count != len(nodes):
        _violation("continuation_node_metric_mismatch")
    if automaton.metrics.semantic_edge_count != sum(
        len(node.choices) for node in nodes
    ):
        _violation("continuation_edge_metric_mismatch")
    if automaton.metrics.terminal_node_count != sum(
        node.terminal_available for node in nodes
    ):
        _violation("continuation_terminal_node_metric_mismatch")
    if automaton.metrics.maximum_out_degree != max(
        (len(node.choices) for node in nodes), default=0
    ):
        _violation("continuation_out_degree_metric_mismatch")
    if automaton.metrics.canonical_core_bytes != _canonical_size(
        (automaton.root, automaton.nodes)
    ):
        _violation("continuation_core_size_metric_mismatch")
    if automaton.metrics.canonical_core_bytes >= _MAX_CANONICAL_CORE_BYTES:
        _violation("continuation_canonical_core_too_large")
    if automaton.metrics.semantic_edge_count > _MAX_SEMANTIC_EDGES:
        _violation("continuation_semantic_edge_limit_exceeded")
    _verify_canonical_node_order(nodes=nodes, depth_by_node=depth_by_node)


def _verify_provenance(automaton: WriterContinuationAutomaton) -> None:
    cursors = {
        item.raw_cursor_digest: item
        for item in automaton.provenance.raw_cursors
    }
    if len(cursors) != len(automaton.provenance.raw_cursors):
        _violation("continuation_duplicate_cursor_provenance")
    root_cursor = cursors.get(automaton.provenance.root_raw_cursor_digest)
    if (
        root_cursor is None
        or root_cursor.compiled_node_id != automaton.root.node_id
        or root_cursor.normalization_scale != automaton.root.completion_scale
    ):
        _violation("continuation_root_provenance_mismatch")
    edges_by_id = {item.edge_id: item for item in automaton.provenance.edges}
    if len(edges_by_id) != len(automaton.provenance.edges):
        _violation("continuation_duplicate_edge_id")
    for item in cursors.values():
        if (
            item.normalization_scale <= 0
            or item.token_depth < 0
            or not 0 <= item.compiled_node_id < len(automaton.nodes)
        ):
            _violation("continuation_cursor_provenance_mismatch")
        if item.raw_cursor_digest == automaton.provenance.root_raw_cursor_digest:
            if item.token_depth != 0 or item.predecessor_edge_id is not None:
                _violation("continuation_root_predecessor_mismatch")
        else:
            predecessor = edges_by_id.get(item.predecessor_edge_id)
            source = None if predecessor is None else cursors.get(
                predecessor.source_raw_cursor_digest
            )
            if (
                predecessor is None
                or source is None
                or predecessor.successor_raw_cursor_digest
                != item.raw_cursor_digest
                or source.token_depth + 1 != item.token_depth
            ):
                _violation("continuation_predecessor_mismatch")
    primitives = {
        item.primitive_cursor_digest: item
        for item in automaton.provenance.primitives
    }
    if len(primitives) != len(automaton.provenance.primitives):
        _violation("continuation_duplicate_primitive_provenance")
    for item in primitives.values():
        representative = cursors.get(item.representative_raw_cursor_digest)
        if (
            representative is None
            or representative.primitive_cursor_digest
            != item.primitive_cursor_digest
            or representative.compiled_node_id != item.compiled_node_id
        ):
            _violation("continuation_primitive_representative_mismatch")
    if {item.primitive_cursor_digest for item in cursors.values()} != set(
        primitives
    ):
        _violation("continuation_primitive_coverage_mismatch")
    memberships = Counter(item.compiled_node_id for item in primitives.values())
    if (
        automaton.metrics.raw_cursor_count != len(cursors)
        or automaton.metrics.primitive_cursor_count != len(primitives)
        or automaton.metrics.weight_normalization_merge_count
        != sum(item.normalization_scale > 1 for item in cursors.values())
        or automaton.metrics.semantic_minimization_merge_count
        != len(primitives) - len(automaton.nodes)
        or automaton.metrics.largest_equivalence_class_membership
        != max(memberships.values(), default=0)
    ):
        _violation("continuation_cursor_metric_mismatch")
    edge_keys: set[tuple[str, str]] = set()
    for item in automaton.provenance.edges:
        key = (item.source_raw_cursor_digest, item.emitted_text)
        if key in edge_keys:
            _violation("continuation_duplicate_edge_provenance")
        edge_keys.add(key)
        source = cursors.get(item.source_raw_cursor_digest)
        successor = cursors.get(item.successor_raw_cursor_digest)
        if (
            source is None
            or successor is None
            or source.compiled_node_id != item.source_node_id
            or successor.compiled_node_id != item.successor_node_id
            or successor.normalization_scale != item.successor_scale
        ):
            _violation("continuation_edge_provenance_mismatch")
        choices = tuple(
            choice
            for choice in automaton.nodes[item.source_node_id].choices
            if choice.emitted_text == item.emitted_text
        )
        if (
            len(choices) != 1
            or choices[0].successor_node_id != item.successor_node_id
            or source.normalization_scale * choices[0].successor_scale
            != item.successor_scale
            or not item.branch_certificate_digests
        ):
            _violation("continuation_edge_provenance_mismatch")
        if item.edge_id != _identity_digest(
            (
                item.source_raw_cursor_digest,
                item.emitted_text,
                item.text_projection_digest,
                item.branch_certificate_digests,
                item.successor_raw_cursor_digest,
            )
        ):
            _violation("continuation_edge_id_mismatch")
    expected_edge_keys = {
        (item.raw_cursor_digest, choice.emitted_text)
        for item in cursors.values()
        for choice in automaton.nodes[item.compiled_node_id].choices
    }
    if edge_keys != expected_edge_keys:
        _violation("continuation_edge_provenance_coverage_mismatch")
    terminal_sources: set[str] = set()
    for item in automaton.provenance.terminals:
        if item.source_raw_cursor_digest in terminal_sources:
            _violation("continuation_duplicate_terminal_provenance")
        terminal_sources.add(item.source_raw_cursor_digest)
        source = cursors.get(item.source_raw_cursor_digest)
        if (
            source is None
            or source.compiled_node_id != item.source_node_id
            or not automaton.nodes[item.source_node_id].terminal_available
            or not item.terminal_support_identity_digests
        ):
            _violation("continuation_terminal_provenance_mismatch")
    expected_terminal_sources = {
        item.raw_cursor_digest
        for item in cursors.values()
        if automaton.nodes[item.compiled_node_id].terminal_available
    }
    if terminal_sources != expected_terminal_sources:
        _violation("continuation_terminal_provenance_coverage_mismatch")


def _canonicalize_nodes(
    *, root, nodes, signature_digest_function
) -> tuple[
    WriterContinuationCursor,
    tuple[WriterContinuationNode, ...],
    dict[int, int],
]:
    depths: dict[int, int] = {}

    def depth(node_id: int) -> int:
        known = depths.get(node_id)
        if known is not None:
            return known
        value = 1 + max(
            (depth(choice.successor_node_id) for choice in nodes[node_id].choices),
            default=0,
        )
        depths[node_id] = value
        return value

    for node in nodes:
        depth(node.node_id)
    canonical_ids: dict[int, int] = {}
    canonical_signatures: dict[int, tuple[object, ...]] = {}
    next_id = 0
    for current_depth in sorted(set(depths.values())):
        at_depth = tuple(
            node for node in nodes if depths[node.node_id] == current_depth
        )
        decorated = []
        for node in at_depth:
            signature = _signature_with_child_ids(
                node=node,
                child_ids=canonical_ids,
            )
            decorated.append((signature, node.node_id))
        for signature, internal_id in sorted(decorated):
            canonical_ids[internal_id] = next_id
            canonical_signatures[internal_id] = signature
            next_id += 1
    rewritten: list[WriterContinuationNode | None] = [None] * len(nodes)
    for node in nodes:
        canonical_id = canonical_ids[node.node_id]
        choices = tuple(
            replace(
                choice,
                successor_node_id=canonical_ids[choice.successor_node_id],
            )
            for choice in node.choices
        )
        signature = canonical_signatures[node.node_id]
        rewritten[canonical_id] = replace(
            node,
            node_id=canonical_id,
            signature_digest=signature_digest_function(signature),
            choices=choices,
        )
    if any(node is None for node in rewritten):
        _violation("continuation_canonical_node_gap")
    return (
        WriterContinuationCursor(
            node_id=canonical_ids[root.node_id],
            completion_scale=root.completion_scale,
        ),
        tuple(rewritten),
        canonical_ids,
    )


def _canonical_predecessor_tree(*, root_raw_cursor_digest, edges):
    outgoing: dict[str, list[WriterContinuationEdgeRecord]] = {}
    incoming: dict[str, list[WriterContinuationEdgeRecord]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source_raw_cursor_digest, []).append(edge)
        incoming.setdefault(edge.successor_raw_cursor_digest, []).append(edge)
    depths = {root_raw_cursor_digest: 0}
    pending = deque((root_raw_cursor_digest,))
    while pending:
        source = pending.popleft()
        for edge in outgoing.get(source, ()):
            candidate = depths[source] + 1
            known = depths.get(edge.successor_raw_cursor_digest)
            if known is None or candidate < known:
                depths[edge.successor_raw_cursor_digest] = candidate
                pending.append(edge.successor_raw_cursor_digest)
    predecessors = {}
    for raw_digest, depth in depths.items():
        if raw_digest == root_raw_cursor_digest:
            continue
        candidates = tuple(
            edge
            for edge in incoming.get(raw_digest, ())
            if depths.get(edge.source_raw_cursor_digest) == depth - 1
        )
        if not candidates:
            _violation("continuation_predecessor_missing")
        predecessor = min(
            candidates,
            key=lambda edge: (
                edge.source_raw_cursor_digest,
                edge.emitted_text,
                edge.text_projection_digest,
            ),
        )
        predecessors[raw_digest] = predecessor.edge_id
    return frozenset(depths), depths, predecessors


def _verify_canonical_node_order(*, nodes, depth_by_node) -> None:
    expected_ids = []
    for depth in sorted(set(depth_by_node.values())):
        expected_ids.extend(
            node.node_id
            for node in sorted(
                (node for node in nodes if depth_by_node[node.node_id] == depth),
                key=lambda node: _signature_for_node(node=node, nodes=nodes),
            )
        )
    if tuple(expected_ids) != tuple(range(len(nodes))):
        _violation("continuation_canonical_node_order_mismatch")


def _signature_with_child_ids(*, node, child_ids) -> tuple[object, ...]:
    return (
        node.terminal_available,
        node.terminal_multiplicity,
        node.terminal_completion_count,
        tuple(
            (
                choice.emitted_text,
                choice.immediate_multiplicity,
                choice.successor_scale,
                child_ids[choice.successor_node_id],
            )
            for choice in node.choices
        ),
    )


def _frontier_batch(prepared, cursor):
    return _checked_writer_frontier_branch_supports(
        prepared,
        cursor,
        include_counts=False,
        include_frontier_certificate=True,
        include_count_certificate=False,
    )


def _normalize_cursor(
    cursor: WriterFrontierCursor,
) -> tuple[WriterFrontierCursor, int]:
    weights = tuple(weight for _state, weight in cursor.weighted_states)
    if not weights:
        return cursor, 1
    scale = weights[0]
    for weight in weights[1:]:
        scale = gcd(scale, weight)
    primitive = WriterFrontierCursor(
        weighted_states=tuple(
            (state, weight // scale)
            for state, weight in cursor.weighted_states
        )
    )
    return primitive, scale


def _scaled_cursor(cursor: WriterFrontierCursor, scale: int) -> WriterFrontierCursor:
    return WriterFrontierCursor(
        weighted_states=tuple(
            (state, scale * weight)
            for state, weight in cursor.weighted_states
        )
    )


def _check_projection_membership(*, source_cursor, projection, batch) -> None:
    if projection.source_cursor != source_cursor:
        _violation("continuation_projection_source_mismatch")
    choices = tuple(
        choice
        for choice in batch.choices.choices
        if choice.emitted_text == projection.emitted_text
    )
    if (
        len(choices) != 1
        or choices[0].successor != projection.successor_cursor
        or choices[0].immediate_multiplicity
        != projection.immediate_multiplicity
    ):
        _violation("continuation_projection_membership_mismatch")
    expected_branches = tuple(
        support.checked_branch_certificate
        for support in batch.supports
        if support.emitted_text == projection.emitted_text
    )
    if projection.branch_certificates != expected_branches:
        _violation("continuation_branch_projection_membership_mismatch")


def _signature_for_node(*, node, nodes) -> tuple[object, ...]:
    return (
        node.terminal_available,
        node.terminal_multiplicity,
        node.terminal_completion_count,
        tuple(
            (
                choice.emitted_text,
                choice.immediate_multiplicity,
                choice.successor_scale,
                choice.successor_node_id,
            )
            for choice in node.choices
        ),
    )


def _node_for_cursor(automaton, cursor) -> WriterContinuationNode:
    if not 0 <= cursor.node_id < len(automaton.nodes):
        _violation("continuation_cursor_node_mismatch")
    if cursor.completion_scale <= 0:
        _violation("continuation_cursor_scale_mismatch")
    return automaton.nodes[cursor.node_id]


def _canonical_size(value) -> int:
    return len(_canonical_json(_term(value)).encode("utf-8"))


def _provenance_record_bytes(provenance) -> int:
    return sum(
        _canonical_size(item)
        for item in (
            *provenance.raw_cursors,
            *provenance.primitives,
            *provenance.edges,
            *provenance.terminals,
        )
    )


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer continuation automaton violation: {kind}",
    )


__all__ = (
    "WriterContinuationAutomaton",
    "WriterContinuationAutomatonVerification",
    "WriterContinuationChoice",
    "WriterContinuationCore",
    "WriterContinuationCursor",
    "WriterContinuationMetrics",
    "WriterContinuationNode",
    "WriterContinuationProbability",
    "WriterContinuationEdgeRecord",
    "WriterContinuationPrimitiveRecord",
    "WriterContinuationRawCursorRecord",
    "WriterContinuationTerminalRecord",
    "advance_writer_continuation",
    "compile_writer_continuation_automaton",
    "verify_writer_continuation_automaton_consistency",
    "writer_continuation_choices",
    "writer_continuation_completion_count",
    "writer_continuation_is_terminal",
    "writer_continuation_probabilities",
    "writer_continuation_provenance_edge",
    "writer_continuation_support_count",
    "writer_continuation_terminal_provenance",
)
