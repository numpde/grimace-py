"""Reusable closed terms and digest helpers for durable writer envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
import hashlib
import json

from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import WriterEnvelopeWorkViolation


def _term(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Enum):
        return {
            "__enum__": (
                f"{value.__class__.__module__}.{value.__class__.__name__}"
            ),
            "value": value.value,
        }
    if isinstance(value, (tuple, list)):
        return [_term(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return [
            *_sorted_terms((_term(item) for item in value))
        ]
    if isinstance(value, Mapping):
        return [
            [str(key), _term(item)]
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
    if is_dataclass(value):
        return {
            "__dataclass__": (
                f"{value.__class__.__module__}.{value.__class__.__name__}"
            ),
            "fields": [
                [field.name, _term(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    _envelope_violation(f"unsupported_term_type:{type(value).__name__}")


def _sorted_terms(values):
    return sorted(values, key=_canonical_json)


def _digest(term) -> str:
    return hashlib.sha256(_canonical_json(term).encode("utf-8")).hexdigest()


def _digest_bounded(term, *, budget, operation: str) -> str:
    payload = _canonical_json(term)
    limit = getattr(budget, "max_digest_term_bytes", None)
    if limit is not None and len(payload.encode("utf-8")) > limit:
        raise ValueError(
            "writer envelope work exceeded: "
            f"operation={operation!r}; metric='digest_term_bytes'; "
            f"actual={len(payload.encode('utf-8'))}; limit={limit}"
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(term) -> str:
    return json.dumps(term, sort_keys=True, separators=(",", ":"))


def _identity_digest(identity, *, budget=None, operation: str = "identity_digest") -> str:
    terms = _term(identity)
    if budget is not None:
        return _digest_terms_bounded(
            terms,
            budget=budget,
            operation=operation,
        )
    return _digest(terms)


def _digest_terms_bounded(term, *, budget, operation: str) -> str:
    try:
        return _digest_bounded(term, budget=budget, operation=operation)
    except ValueError as exc:
        raise _work_exceeded_from_digest_error(
            exc,
            operation=operation,
            budget=budget,
        ) from exc


def _identity_envelope(
    identity,
    *,
    budget=None,
    operation: str = "identity_envelope",
) -> dict[str, object]:
    terms = _term(identity)
    return {
        "terms": terms,
        "digest": (
            _digest_terms_bounded(terms, budget=budget, operation=operation)
            if budget is not None
            else _digest(terms)
        ),
    }


def _runtime_options_terms(options: SouthStarRuntimeOptions) -> dict[str, object]:
    return {
        "rooted_at_atom": options.rooted_at_atom,
        "canonical": options.canonical,
        "do_random": options.do_random,
        "serialization_language": options.serialization_language.value,
    }


def _runtime_options_from_terms(terms: Mapping[str, object]):
    return SouthStarRuntimeOptions(
        rooted_at_atom=terms["rooted_at_atom"],
        canonical=terms["canonical"],
        do_random=terms["do_random"],
        serialization_language=SerializationLanguageMode(
            terms["serialization_language"]
        ),
    )


def _decoder_boundary_terms(boundary) -> dict[str, object]:
    return {
        "consumed_token_count": boundary.consumed_token_count,
    }


def _cursor_envelope(
    cursor,
    *,
    budget=None,
    operation: str = "cursor_envelope",
) -> dict[str, object]:
    terms = _term(cursor)
    return {
        "terms": terms,
        "digest": (
            _digest_terms_bounded(terms, budget=budget, operation=operation)
            if budget is not None
            else _digest(terms)
        ),
    }


def _snapshot_identity_envelope(
    snapshot,
    *,
    budget=None,
    operation: str = "snapshot_identity_envelope",
) -> dict[str, object]:
    prepared_terms = _term(snapshot.prepared_identity)
    return {
        "serialization_language": snapshot.serialization_language.value,
        "runtime_options": _runtime_options_terms(snapshot.runtime_options),
        "prepared_identity_terms": prepared_terms,
        "prepared_identity_digest": (
            _digest_terms_bounded(
                prepared_terms,
                budget=budget,
                operation=f"{operation}.prepared_identity",
            )
            if budget is not None
            else _digest(prepared_terms)
        ),
        "cursor": _cursor_envelope(
            snapshot.cursor,
            budget=budget,
            operation=f"{operation}.cursor",
        ),
        "decoder_boundary": _decoder_boundary_terms(snapshot.decoder_boundary),
        "frame_stack_cursors": [
            _cursor_envelope(
                frame.cursor,
                budget=budget,
                operation=f"{operation}.frame_stack_cursor",
            )
            for frame in snapshot.frame_stack
        ],
        "digest": _identity_digest(
            snapshot,
            budget=budget,
            operation=operation,
        ),
    }


def _envelope_violation(kind: str) -> None:
    raise ValueError(f"unsupported writer envelope term: {kind}")


def _work_exceeded_from_digest_error(
    exc: ValueError,
    *,
    operation: str,
    budget,
) -> WriterEnvelopeWorkExceeded:
    message = str(exc)
    metric = "digest_term_bytes"
    actual = 0
    limit = budget.max_digest_term_bytes
    for item in message.split(";"):
        item = item.strip()
        if item.startswith("metric="):
            metric = item.split("=", 1)[1].strip("'")
        elif item.startswith("actual="):
            actual = int(item.split("=", 1)[1])
        elif item.startswith("limit="):
            limit = int(item.split("=", 1)[1])
    return WriterEnvelopeWorkExceeded(
        WriterEnvelopeWorkViolation(
            operation=operation,
            metric=metric,
            actual=actual,
            limit=limit,
        )
    )


__all__ = (
    "_canonical_json",
    "_cursor_envelope",
    "_decoder_boundary_terms",
    "_digest",
    "_digest_bounded",
    "_digest_terms_bounded",
    "_identity_digest",
    "_identity_envelope",
    "_runtime_options_from_terms",
    "_runtime_options_terms",
    "_snapshot_identity_envelope",
    "_term",
)
