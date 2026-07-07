"""Structural consistency checks for durable writer envelopes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _term
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_count_dag_envelope import count_dag_node_by_id
from .writer_count_dag_envelope import (
    validate_writer_count_certificate_dag_envelope,
)


@dataclass(frozen=True, slots=True)
class WriterEnvelopeConsistencyVerification:
    accepted: bool
    schema_name: str
    schema_version: int | None
    support_count: int | None = None
    witness_count: int | None = None
    reason: str | None = None


def verify_writer_support_image_envelope_consistency(
    envelope: object,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterEnvelopeConsistencyVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_support_image_envelope(envelope)
        assert isinstance(envelope, Mapping)
        _check_consistency_work(envelope, budget=budget)
        return WriterEnvelopeConsistencyVerification(
            accepted=True,
            schema_name=str(envelope["schema_name"]),
            schema_version=int(envelope["schema_version"]),
            support_count=int(envelope["distinct_count"]),
            witness_count=int(envelope["witness_count"]),
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterEnvelopeConsistencyVerification(
            accepted=False,
            schema_name=(
                str(envelope.get("schema_name", "unknown"))
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            schema_version=(
                envelope.get("schema_version")
                if isinstance(envelope, Mapping)
                and isinstance(envelope.get("schema_version"), int)
                else None
            ),
            reason=writer_envelope_work_reason(exc),
        )
    except _ConsistencyViolation as exc:
        return WriterEnvelopeConsistencyVerification(
            accepted=False,
            schema_name=(
                str(envelope.get("schema_name", "unknown"))
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            schema_version=(
                envelope.get("schema_version")
                if isinstance(envelope, Mapping)
                and isinstance(envelope.get("schema_version"), int)
                else None
            ),
            reason=str(exc),
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterEnvelopeConsistencyVerification(
            accepted=False,
            schema_name=(
                str(envelope.get("schema_name", "unknown"))
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            schema_version=(
                envelope.get("schema_version")
                if isinstance(envelope, Mapping)
                and isinstance(envelope.get("schema_version"), int)
                else None
            ),
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


_ADVANCE_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_snapshot",
    "emitted_text",
    "outcome_kind",
    "frontier_product_kind",
    "advance_certificate",
))
_REPLAY_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_snapshot",
    "emitted_texts",
    "outcome_kind",
    "consumed_emitted_texts",
    "remaining_emitted_texts",
    "step_advance_envelopes",
    "current_snapshot",
    "replay_certificate",
    "failed_advance_envelope",
))
_PREFIX_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_snapshot",
    "emitted_texts",
    "include_counts",
    "read_kind",
    "replay_envelope",
    "final_snapshot",
    "final_frontier_product_kind",
    "final_frontier_product",
    "prefix_read_certificate",
    "public_frontier",
    "support_count",
    "completion_count",
    "failure",
))
_COUNT_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "frontier_snapshot",
    "frontier_product",
    "support_count",
    "completion_count",
    "count_dag",
    "support_count_certificate",
    "completion_count_certificate",
    "choice_count_certificates",
    "terminal_choice_count_certificate",
    "coverage",
))
_SUPPORT_STRING_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "count_envelope",
    "string",
    "emitted_texts",
    "replay_envelope",
    "final_snapshot",
    "terminal_frontier_product",
    "terminal_frontier_projection",
    "terminal_projection",
    "support_string_certificate",
    "terminal_support_identities",
    "text_projection_chain",
))
_SUPPORT_IMAGE_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "count_envelope",
    "support_strings",
    "support_string_envelopes",
    "distinct_count",
    "witness_count",
    "support_image_certificate",
    "enumeration_coverage",
    "frontier_product",
    "checked_frontier_certificate",
    "support_count_certificate",
    "witness_count_certificate",
))


def _validate_support_image_envelope(envelope: object) -> None:
    _require_schema(envelope, "writer_support_image", 1, _SUPPORT_IMAGE_FIELDS)
    assert isinstance(envelope, Mapping)
    _require_source_kind(envelope)
    count = envelope["count_envelope"]
    _validate_count_envelope(count)
    source = _source_snapshot_identity(envelope)
    _check_common_source(
        outer=envelope,
        nested=count,
        outer_source=source,
        nested_source=count["frontier_snapshot"],
        label="count_envelope",
    )
    if envelope["frontier_product"] != count["frontier_product"]:
        _fail("frontier_product_identity_mismatch")
    if (
        envelope["checked_frontier_certificate"]
        != count["frontier_product"]["checked_frontier_certificate"]
    ):
        _fail("checked_frontier_identity_mismatch")
    if envelope["support_count_certificate"] != count["support_count_certificate"]:
        _fail("support_count_certificate_mismatch")
    if (
        envelope["witness_count_certificate"]
        != count["completion_count_certificate"]
    ):
        _fail("witness_count_certificate_mismatch")
    strings = _require_list(envelope, "support_strings")
    string_envelopes = _require_list(envelope, "support_string_envelopes")
    if len(strings) != len(string_envelopes):
        _fail("support_string_count_mismatch")
    if len(set(strings)) != len(strings):
        _fail("duplicate_support_string")
    if envelope["distinct_count"] != len(strings):
        _fail("distinct_count_mismatch")
    if envelope["distinct_count"] != count["support_count"]:
        _fail("support_count_mismatch")
    if envelope["witness_count"] != count["completion_count"]:
        _fail("witness_count_mismatch")
    for index, (string, string_envelope) in enumerate(
        zip(strings, string_envelopes, strict=True)
    ):
        _validate_support_string_envelope(
            string_envelope,
            expected_source_kind=envelope["source_kind"],
            expected_source=source,
            expected_prefix=envelope["prefix_read_envelope"],
            expected_count=count,
            expected_string=string,
            index=index,
        )
    _validate_support_image_certificate(envelope)
    _validate_enumeration_coverage(envelope, source)


def _validate_count_envelope(envelope: object) -> None:
    _require_schema(envelope, "writer_frontier_count", 1, _COUNT_FIELDS)
    assert isinstance(envelope, Mapping)
    _require_source_kind(envelope)
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] != envelope["frontier_snapshot"]:
            _fail("count_snapshot_source_mismatch")
        if envelope["prefix_read_envelope"] is not None:
            _fail("count_snapshot_has_prefix")
    else:
        prefix = envelope["prefix_read_envelope"]
        _validate_prefix_envelope(prefix)
        if prefix["read_kind"] != "readable":
            _fail("count_prefix_not_readable")
        if envelope["source_snapshot"] is not None:
            _fail("count_prefix_has_source_snapshot")
        if envelope["frontier_snapshot"] != prefix["final_snapshot"]:
            _fail("count_prefix_final_snapshot_mismatch")
    _check_prepared_identity(envelope, envelope["frontier_snapshot"])
    if envelope["frontier_product"]["kind"] != "legal":
        _fail("count_frontier_product_not_legal")
    if (
        envelope["frontier_product"]["cursor"]
        != envelope["frontier_snapshot"]["cursor"]
    ):
        _fail("count_frontier_cursor_mismatch")
    _validate_self_digest(envelope["frontier_product"], "frontier_product")
    _validate_self_digest(
        envelope["frontier_product"]["checked_frontier_certificate"],
        "checked_frontier_certificate",
    )
    try:
        validate_writer_count_certificate_dag_envelope(envelope["count_dag"])
    except (TypeError, ValueError) as exc:
        _fail(f"count_dag_invalid:{exc}")
    nodes = count_dag_node_by_id(envelope["count_dag"])
    roots = envelope["count_dag"]["roots"]
    if envelope["support_count_certificate"] != nodes[roots["support_count_root"]]:
        _fail("support_count_root_mismatch")
    if (
        envelope["completion_count_certificate"]
        != nodes[roots["completion_count_root"]]
    ):
        _fail("completion_count_root_mismatch")
    if envelope["choice_count_certificates"] != [
        nodes[node_id] for node_id in roots["choice_count_roots"]
    ]:
        _fail("choice_count_roots_mismatch")
    terminal_root = roots["terminal_choice_count_root"]
    expected_terminal = None if terminal_root is None else nodes[terminal_root]
    if envelope["terminal_choice_count_certificate"] != expected_terminal:
        _fail("terminal_choice_count_root_mismatch")
    if (
        envelope["support_count_certificate"]["support_count"]
        != envelope["support_count"]
    ):
        _fail("support_count_certificate_total_mismatch")
    if (
        envelope["completion_count_certificate"]["completion_count"]
        != envelope["completion_count"]
    ):
        _fail("completion_count_certificate_total_mismatch")
    coverage = envelope["coverage"]
    if coverage["support_count_total"] != envelope["support_count"]:
        _fail("coverage_support_count_mismatch")
    if coverage["completion_count_total"] != envelope["completion_count"]:
        _fail("coverage_completion_count_mismatch")
    _validate_count_coverage_node_references(coverage, nodes, terminal_root)


def _validate_count_coverage_node_references(
    coverage: Mapping[str, object],
    nodes: Mapping[str, Mapping[str, object]],
    terminal_root,
) -> None:
    for item in _require_list(coverage, "text_choices_covered"):
        support_node_id = item["successor_support_count_node_id"]
        completion_node_id = item["completion_count_node_id"]
        if support_node_id not in nodes:
            _fail("choice_coverage_missing_support_node")
        if completion_node_id not in nodes:
            _fail("choice_coverage_missing_completion_node")
        if item["successor_support_count_digest"] != nodes[support_node_id]["digest"]:
            _fail("choice_coverage_support_node_digest_mismatch")
        if item["completion_count_node_digest"] != nodes[completion_node_id]["digest"]:
            _fail("choice_coverage_completion_node_digest_mismatch")
    for item in _require_list(coverage, "branch_terms_covered"):
        node_id = item["successor_count_certificate_node_id"]
        if node_id not in nodes:
            _fail("branch_coverage_missing_successor_node")
        if item["successor_count_certificate_digest"] != nodes[node_id]["digest"]:
            _fail("branch_coverage_successor_digest_mismatch")
    terminal = coverage["terminal_choice_coverage"]
    if terminal is None:
        if terminal_root is not None:
            _fail("terminal_coverage_missing")
    else:
        if terminal["terminal_choice_count_node_id"] != terminal_root:
            _fail("terminal_coverage_node_mismatch")
        if terminal_root not in nodes:
            _fail("terminal_coverage_missing_node")
        if terminal["terminal_choice_count_node_digest"] != nodes[terminal_root]["digest"]:
            _fail("terminal_coverage_node_digest_mismatch")


def _validate_support_string_envelope(
    envelope: object,
    *,
    expected_source_kind: str,
    expected_source,
    expected_prefix,
    expected_count,
    expected_string,
    index: int,
) -> None:
    _require_schema(envelope, "writer_support_string", 1, _SUPPORT_STRING_FIELDS)
    assert isinstance(envelope, Mapping)
    _require_source_kind(envelope)
    if envelope["source_kind"] != expected_source_kind:
        _fail(f"support_string_source_kind_mismatch:{index}")
    source = _source_snapshot_identity(envelope)
    if source != expected_source:
        _fail(f"support_string_source_mismatch:{index}")
    if envelope["prefix_read_envelope"] != expected_prefix:
        _fail(f"support_string_prefix_mismatch:{index}")
    if envelope["count_envelope"] != expected_count:
        _fail(f"support_string_count_envelope_mismatch:{index}")
    if envelope["string"] != expected_string:
        _fail(f"support_string_order_mismatch:{index}")
    if "".join(envelope["emitted_texts"]) != envelope["string"]:
        _fail(f"support_string_emitted_texts_mismatch:{index}")
    _validate_count_envelope(envelope["count_envelope"])
    replay = envelope["replay_envelope"]
    _validate_replay_envelope(replay)
    if replay["outcome_kind"] != "advanced":
        _fail(f"support_string_replay_not_advanced:{index}")
    if replay["source_snapshot"] != source:
        _fail(f"support_string_replay_source_mismatch:{index}")
    if replay["emitted_texts"] != envelope["emitted_texts"]:
        _fail(f"support_string_replay_texts_mismatch:{index}")
    if envelope["final_snapshot"] != replay["current_snapshot"]:
        _fail(f"support_string_final_snapshot_mismatch:{index}")
    terminal = envelope["terminal_projection"]
    if terminal["source_cursor"] != envelope["final_snapshot"]["cursor"]:
        _fail(f"terminal_projection_final_cursor_mismatch:{index}")
    if terminal["terminal_support_identities"] != envelope["terminal_support_identities"]:
        _fail(f"terminal_support_identity_mismatch:{index}")
    if (
        envelope["support_string_certificate"]["terminal_projection_digest"]
        != terminal["digest"]
    ):
        _fail(f"support_string_terminal_projection_digest_mismatch:{index}")
    if envelope["support_string_certificate"]["text_projection_chain_digests"] != [
        step["text_projection"]["digest"] for step in envelope["text_projection_chain"]
    ]:
        _fail(f"support_string_text_chain_digest_mismatch:{index}")
    _validate_text_projection_chain(envelope)


def _validate_replay_envelope(envelope: object) -> None:
    _require_schema(envelope, "writer_snapshot_replay", 1, _REPLAY_FIELDS)
    assert isinstance(envelope, Mapping)
    if envelope["outcome_kind"] not in (
        "advanced",
        "invalid_emitted_text",
        "blocked",
    ):
        _fail("unknown_replay_outcome_kind")
    _check_prepared_identity(envelope, envelope["source_snapshot"])
    emitted = _require_list(envelope, "emitted_texts")
    consumed = _require_list(envelope, "consumed_emitted_texts")
    remaining = _require_list(envelope, "remaining_emitted_texts")
    steps = _require_list(envelope, "step_advance_envelopes")
    if consumed + remaining != emitted:
        _fail("replay_consumed_remaining_mismatch")
    current = envelope["source_snapshot"]
    for index, step in enumerate(steps):
        _validate_advance_envelope(step)
        if step["source_snapshot"] != current:
            _fail(f"replay_step_source_mismatch:{index}")
        if step["emitted_text"] != emitted[index]:
            _fail(f"replay_step_text_mismatch:{index}")
        if step["outcome_kind"] != "advanced":
            _fail(f"replay_step_not_advanced:{index}")
        current = step["advance_certificate"]["advanced_snapshot"]
    failed = envelope["failed_advance_envelope"]
    if envelope["outcome_kind"] == "advanced":
        if failed is not None:
            _fail("advanced_replay_has_failed_step")
        if remaining:
            _fail("advanced_replay_has_remaining_texts")
        if len(steps) != len(emitted):
            _fail("advanced_replay_step_count_mismatch")
        if envelope["current_snapshot"] != current:
            _fail("replay_current_snapshot_mismatch")
        replay_certificate = envelope["replay_certificate"]
        if replay_certificate is None:
            _fail("advanced_replay_missing_certificate")
        if replay_certificate["source_snapshot"] != envelope["source_snapshot"]:
            _fail("replay_certificate_source_mismatch")
        if replay_certificate["final_snapshot"] != current:
            _fail("replay_certificate_final_mismatch")
        if replay_certificate["emitted_texts"] != emitted:
            _fail("replay_certificate_texts_mismatch")
        return
    if failed is None:
        _fail("failed_replay_missing_failed_step")
    _validate_advance_envelope(failed)
    failed_index = len(steps)
    if failed_index >= len(emitted):
        _fail("failed_replay_index_mismatch")
    if failed["source_snapshot"] != current:
        _fail("failed_replay_source_mismatch")
    if failed["emitted_text"] != emitted[failed_index]:
        _fail("failed_replay_text_mismatch")
    if failed["outcome_kind"] != envelope["outcome_kind"]:
        _fail("failed_replay_kind_mismatch")
    if envelope["replay_certificate"] is not None:
        _fail("failed_replay_has_certificate")
    if envelope["current_snapshot"] != current:
        _fail("failed_replay_current_snapshot_mismatch")


def _validate_advance_envelope(envelope: object) -> None:
    _require_schema(envelope, "writer_snapshot_advance", 1, _ADVANCE_FIELDS)
    assert isinstance(envelope, Mapping)
    if envelope["outcome_kind"] not in (
        "advanced",
        "invalid_emitted_text",
        "blocked",
    ):
        _fail("unknown_advance_outcome_kind")
    if envelope["frontier_product_kind"] not in ("legal", "blocked"):
        _fail("unknown_advance_product_kind")
    _check_prepared_identity(envelope, envelope["source_snapshot"])
    cert = envelope["advance_certificate"]
    if cert["kind"] != envelope["outcome_kind"]:
        _fail("advance_certificate_kind_mismatch")
    if envelope["outcome_kind"] != "advanced":
        return
    if envelope["frontier_product_kind"] != "legal":
        _fail("advanced_product_kind_mismatch")
    selected = cert["selected_text_projection"]
    step = cert["step_certificate"]
    if selected["emitted_text"] != envelope["emitted_text"]:
        _fail("advance_selected_text_mismatch")
    if step["source_snapshot"] != envelope["source_snapshot"]:
        _fail("advance_step_source_snapshot_mismatch")
    if step["source_cursor"] != envelope["source_snapshot"]["cursor"]:
        _fail("advance_step_source_cursor_mismatch")
    if step["successor_cursor"] != selected["successor_cursor"]:
        _fail("advance_step_successor_cursor_mismatch")
    if cert["advanced_snapshot"] != step["advanced_snapshot"]:
        _fail("advance_advanced_snapshot_mismatch")
    if cert["advanced_snapshot"]["cursor"] != selected["successor_cursor"]:
        _fail("advance_snapshot_cursor_mismatch")
    before = _decoder_boundary_count(step["decoder_boundary_before"])
    after = _decoder_boundary_count(step["decoder_boundary_after"])
    if after != before + 1:
        _fail("advance_decoder_boundary_increment_mismatch")
    if envelope["source_snapshot"]["decoder_boundary"]["consumed_token_count"] != before:
        _fail("advance_source_boundary_mismatch")
    if cert["advanced_snapshot"]["decoder_boundary"]["consumed_token_count"] != after:
        _fail("advance_snapshot_boundary_mismatch")
    if step["text_projection_digest"] != selected["digest"]:
        _fail("advance_text_projection_digest_mismatch")


def _validate_prefix_envelope(envelope: object) -> None:
    _require_schema(envelope, "writer_snapshot_prefix_read", 1, _PREFIX_FIELDS)
    assert isinstance(envelope, Mapping)
    if envelope["read_kind"] not in (
        "readable",
        "replay_blocked",
        "invalid_emitted_text",
        "final_frontier_blocked",
    ):
        _fail("unknown_prefix_read_kind")
    _check_prepared_identity(envelope, envelope["source_snapshot"])
    replay = envelope["replay_envelope"]
    _validate_replay_envelope(replay)
    if replay["source_snapshot"] != envelope["source_snapshot"]:
        _fail("prefix_replay_source_mismatch")
    if replay["emitted_texts"] != envelope["emitted_texts"]:
        _fail("prefix_replay_texts_mismatch")
    if envelope["read_kind"] == "readable":
        if replay["outcome_kind"] != "advanced":
            _fail("readable_prefix_replay_not_advanced")
        if envelope["final_snapshot"] != replay["current_snapshot"]:
            _fail("prefix_final_snapshot_mismatch")
        if envelope["final_frontier_product_kind"] != "legal":
            _fail("readable_prefix_product_kind_mismatch")
        if envelope["prefix_read_certificate"] is None:
            _fail("readable_prefix_missing_certificate")
        if envelope["public_frontier"] is None:
            _fail("readable_prefix_missing_public_frontier")
    elif envelope["read_kind"] in ("replay_blocked", "invalid_emitted_text"):
        if envelope["final_snapshot"] is not None:
            _fail("failed_prefix_has_final_snapshot")
        if envelope["prefix_read_certificate"] is not None:
            _fail("failed_prefix_has_certificate")
    else:
        if replay["outcome_kind"] != "advanced":
            _fail("final_blocked_prefix_replay_not_advanced")
        if envelope["final_frontier_product_kind"] != "blocked":
            _fail("final_blocked_prefix_product_kind_mismatch")


def _validate_text_projection_chain(envelope: Mapping[str, object]) -> None:
    chain = _require_list(envelope, "text_projection_chain")
    replay_steps = envelope["replay_envelope"]["step_advance_envelopes"]
    emitted_texts = envelope["emitted_texts"]
    if len(chain) != len(emitted_texts):
        _fail("text_projection_chain_length_mismatch")
    if len(chain) != len(replay_steps):
        _fail("text_projection_chain_replay_length_mismatch")
    for index, (chain_step, replay_step) in enumerate(
        zip(chain, replay_steps, strict=True)
    ):
        if chain_step["step_index"] != index:
            _fail("text_projection_chain_order_mismatch")
        if chain_step["emitted_text"] != emitted_texts[index]:
            _fail("text_projection_chain_text_mismatch")
        selected = replay_step["advance_certificate"]["selected_text_projection"]
        if _text_bucket_key(chain_step["text_projection"]) != _text_bucket_key(
            selected
        ):
            _fail("text_projection_chain_projection_mismatch")
        if chain_step["source_cursor"] != replay_step["source_snapshot"]["cursor"]:
            _fail("text_projection_chain_source_cursor_mismatch")
        if chain_step["successor_cursor"] != selected["successor_cursor"]:
            _fail("text_projection_chain_successor_cursor_mismatch")


def _validate_support_image_certificate(envelope: Mapping[str, object]) -> None:
    certificate = envelope["support_image_certificate"]
    if certificate["source_snapshot"] != _source_snapshot_identity(envelope):
        _fail("support_image_certificate_source_mismatch")
    if certificate["strings"] != envelope["support_strings"]:
        _fail("support_image_certificate_strings_mismatch")
    if certificate["distinct_count"] != envelope["distinct_count"]:
        _fail("support_image_certificate_distinct_count_mismatch")
    if certificate["witness_count"] != envelope["witness_count"]:
        _fail("support_image_certificate_witness_count_mismatch")
    if (
        certificate["support_count_certificate_digest"]
        != envelope["support_count_certificate"]["digest"]
    ):
        _fail("support_image_certificate_support_digest_mismatch")
    if (
        certificate["witness_count_certificate_digest"]
        != envelope["witness_count_certificate"]["digest"]
    ):
        _fail("support_image_certificate_witness_digest_mismatch")
    if (
        certificate["checked_frontier_certificate_digest"]
        != envelope["checked_frontier_certificate"]["digest"]
    ):
        _fail("support_image_certificate_checked_digest_mismatch")
    if (
        certificate["enumeration_coverage_digest"]
        != envelope["enumeration_coverage"]["digest"]
    ):
        _fail("support_image_certificate_coverage_digest_mismatch")
    expected_string_digests = [
        item["support_string_certificate"]["digest"]
        for item in envelope["support_string_envelopes"]
    ]
    if certificate["string_certificate_digests"] != expected_string_digests:
        _fail("support_image_certificate_string_digest_mismatch")


def _validate_enumeration_coverage(
    envelope: Mapping[str, object],
    source,
) -> None:
    coverage = envelope["enumeration_coverage"]
    if coverage["source_snapshot"] != source:
        _fail("coverage_source_mismatch")
    if coverage["checked_frontier_certificate"] != envelope["checked_frontier_certificate"]:
        _fail("coverage_checked_frontier_mismatch")
    if coverage["support_count_certificate"] != envelope["support_count_certificate"]:
        _fail("coverage_support_count_certificate_mismatch")
    if coverage["distinct_count"] != envelope["distinct_count"]:
        _fail("coverage_distinct_count_mismatch")
    if coverage["support_count"] != envelope["distinct_count"]:
        _fail("coverage_support_count_mismatch")
    string_envelopes = envelope["support_string_envelopes"]
    assigned_indices: list[int] = []
    bucket_total = 0
    for bucket in coverage["text_buckets"]:
        indices = _require_list(bucket, "string_indices")
        if bucket["support_count"] != len(indices):
            _fail("text_bucket_support_count_mismatch")
        bucket_total += bucket["support_count"]
        for index in indices:
            if not isinstance(index, int) or index < 0 or index >= len(string_envelopes):
                _fail("text_bucket_index_out_of_range")
            string_envelope = string_envelopes[index]
            if not string_envelope["emitted_texts"]:
                _fail("empty_string_in_text_bucket")
            first_projection = string_envelope["text_projection_chain"][0][
                "text_projection"
            ]
            if _text_bucket_key(first_projection) != _text_bucket_key(
                bucket["text_projection"]
            ):
                _fail("text_bucket_projection_mismatch")
            if (
                string_envelope["support_string_certificate"]["digest"]
                not in bucket["string_digests"]
            ):
                _fail("text_bucket_string_digest_mismatch")
        assigned_indices.extend(indices)
    terminal = coverage["terminal_bucket"]
    empty_indices = [
        index
        for index, item in enumerate(string_envelopes)
        if not item["emitted_texts"]
    ]
    if empty_indices:
        if terminal is None:
            _fail("terminal_bucket_missing")
        if terminal["support_count"] != len(empty_indices):
            _fail("terminal_bucket_support_count_mismatch")
        if terminal["string_index"] != empty_indices[0]:
            _fail("terminal_bucket_index_mismatch")
        terminal_string = string_envelopes[empty_indices[0]]
        if terminal["string_digest"] != terminal_string["support_string_certificate"]["digest"]:
            _fail("terminal_bucket_string_digest_mismatch")
        if (
            terminal["terminal_support_identities"]
            != terminal_string["terminal_support_identities"]
        ):
            _fail("terminal_support_identity_mismatch")
        assigned_indices.extend(empty_indices)
        bucket_total += terminal["support_count"]
    elif terminal is not None:
        _fail("terminal_bucket_without_empty_string")
    if Counter(assigned_indices) != Counter(range(len(string_envelopes))):
        _fail("coverage_string_partition_mismatch")
    if bucket_total != envelope["distinct_count"]:
        _fail("coverage_bucket_total_mismatch")


def _check_consistency_work(
    envelope: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    nested_count = _count_nested_envelopes(envelope)
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="nested_envelope_count",
        actual=nested_count,
        limit=budget.max_nested_envelopes,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="consistency_node_count",
        actual=_count_json_nodes(envelope),
        limit=budget.max_consistency_nodes,
    )
    strings = _require_list(envelope, "support_strings")
    string_envelopes = _require_list(envelope, "support_string_envelopes")
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="support_string_count",
        actual=len(strings),
        limit=budget.max_support_strings,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="support_string_envelope_count",
        actual=len(string_envelopes),
        limit=budget.max_support_string_envelopes,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="total_emitted_text_bytes",
        actual=sum(
            len(text.encode("utf-8"))
            for string_envelope in string_envelopes
            for text in string_envelope["emitted_texts"]
        ),
        limit=budget.max_total_emitted_text_bytes,
    )
    coverage = envelope["enumeration_coverage"]
    bucket_count = len(coverage["text_buckets"]) + (
        0 if coverage["terminal_bucket"] is None else 1
    )
    assignment_count = sum(
        len(bucket["string_indices"])
        for bucket in coverage["text_buckets"]
    )
    terminal = coverage["terminal_bucket"]
    if terminal is not None and terminal["string_index"] is not None:
        assignment_count += 1
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="coverage_bucket_count",
        actual=bucket_count,
        limit=budget.max_coverage_buckets,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_image_consistency",
        metric="bucket_assignment_count",
        actual=assignment_count,
        limit=budget.max_bucket_assignments,
    )


def _count_nested_envelopes(value) -> int:
    if isinstance(value, Mapping):
        own = 1 if "schema_name" in value and "schema_version" in value else 0
        return own + sum(_count_nested_envelopes(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_nested_envelopes(item) for item in value)
    return 0


def _count_json_nodes(value) -> int:
    if isinstance(value, Mapping):
        return 1 + sum(_count_json_nodes(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_count_json_nodes(item) for item in value)
    return 1


def _source_snapshot_identity(envelope: Mapping[str, object]):
    if envelope["source_kind"] == "snapshot":
        return envelope["source_snapshot"]
    prefix = envelope["prefix_read_envelope"]
    _validate_prefix_envelope(prefix)
    if prefix["read_kind"] != "readable":
        _fail("prefix_source_not_readable")
    return prefix["final_snapshot"]


def _check_common_source(
    *,
    outer,
    nested,
    outer_source,
    nested_source,
    label: str,
) -> None:
    if outer["prepared_identity"] != nested["prepared_identity"]:
        _fail(f"{label}_prepared_identity_mismatch")
    if nested_source != outer_source:
        _fail(f"{label}_source_mismatch")


def _check_prepared_identity(envelope: Mapping[str, object], snapshot) -> None:
    if envelope["prepared_identity"]["digest"] != snapshot["prepared_identity_digest"]:
        _fail("prepared_identity_snapshot_digest_mismatch")


def _validate_self_digest(envelope: Mapping[str, object], label: str) -> None:
    if "digest" not in envelope:
        _fail(f"{label}_digest_missing")
    expected = _identity_digest(
        {
            key: value
            for key, value in envelope.items()
            if key != "digest"
        },
        budget=WriterEnvelopeWorkBudget(),
        operation=f"consistency.{label}.digest",
    )
    if envelope["digest"] != expected:
        _fail(f"{label}_digest_mismatch")


def _require_schema(
    envelope: object,
    schema_name: str,
    schema_version: int,
    fields: frozenset[str],
) -> None:
    if not isinstance(envelope, Mapping):
        _fail("envelope_not_mapping")
    if frozenset(envelope) != fields:
        _fail(f"{schema_name}_fields_mismatch")
    if envelope["schema_name"] != schema_name:
        _fail("unknown_schema_name")
    if envelope["schema_version"] != schema_version:
        _fail("unknown_schema_version")


def _require_source_kind(envelope: Mapping[str, object]) -> None:
    if envelope["source_kind"] not in ("snapshot", "prefix_read"):
        _fail("unknown_source_kind")
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] is None:
            _fail("snapshot_source_missing")
        if envelope["prefix_read_envelope"] is not None:
            _fail("snapshot_source_has_prefix")
    else:
        if envelope["source_snapshot"] is not None:
            _fail("prefix_source_has_source_snapshot")
        if envelope["prefix_read_envelope"] is None:
            _fail("prefix_source_missing_prefix")


def _require_list(envelope: Mapping[str, object], field: str) -> list:
    value = envelope[field]
    if not isinstance(value, list):
        _fail(f"{field}_not_list")
    return value


def _text_bucket_key(projection: Mapping[str, object]):
    return (
        projection["source_cursor"]["digest"],
        projection["emitted_text"],
        projection["successor_cursor"]["digest"],
        projection["immediate_multiplicity"],
    )


def _decoder_boundary_count(value) -> int:
    if isinstance(value, Mapping) and "consumed_token_count" in value:
        return int(value["consumed_token_count"])
    if isinstance(value, Mapping) and "fields" in value:
        for name, item in value["fields"]:
            if name == "consumed_token_count":
                return int(item)
    _fail("decoder_boundary_count_missing")


class _ConsistencyViolation(Exception):
    pass


def _fail(kind: str) -> None:
    raise _ConsistencyViolation(kind)


__all__ = (
    "WriterEnvelopeConsistencyVerification",
    "verify_writer_support_image_envelope_consistency",
)
