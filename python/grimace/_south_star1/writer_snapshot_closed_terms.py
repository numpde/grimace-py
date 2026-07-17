"""Allowlisted decoding for writer snapshot cursor identity terms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import get_args, get_origin, get_type_hints

from .errors import SouthStarError, SouthStarErrorKind
from .facts import DirectionalValue, SiteStatus, TetraValue
from .policy import DirectionMark, TetraToken
from .residual_constraints import (
    DirectionalBondEmissionFactorValueSnapshot,
    DirectionalCarrierResidual,
    DirectionalNormalizedSign,
    DirectionalResidualFactorValueSnapshot,
    DirectionalSiteCarrierModel,
    DirectionalSiteFactorValueSnapshot,
    ResidualFactorKey,
    ResidualStoreValueSnapshot,
    TetraLocalParity,
    TetraResidualFactorValueSnapshot,
    TetraTokenParityFactorValueSnapshot,
    VarId,
)
from .writer_frontier import WriterFrontierCursor
from .writer_state import (
    ComponentCursor,
    ObligationStateKey,
    PendingEntryPhase,
    PendingWriterEntry,
    WriterAtomFrame,
    WriterBranchFrame,
    WriterClosedClosure,
    WriterClosureLabel,
    WriterOpenClosureEndpoint,
    WriterPolicyStateKey,
    WriterRingLabelState,
    WriterRingStateKey,
    WriterStateKey,
    WriterStereoStateKey,
)
from .writer_stereo import (
    WriterAtomOccurrenceRecord,
    WriterBondOccurrenceRecord,
    WriterLocalOrderRecord,
)


_ALLOWED_CLASSES = (
    ComponentCursor,
    DirectionalBondEmissionFactorValueSnapshot,
    DirectionalCarrierResidual,
    DirectionalNormalizedSign,
    DirectionalResidualFactorValueSnapshot,
    DirectionalSiteCarrierModel,
    DirectionalSiteFactorValueSnapshot,
    DirectionalValue,
    DirectionMark,
    ObligationStateKey,
    PendingEntryPhase,
    PendingWriterEntry,
    ResidualFactorKey,
    ResidualStoreValueSnapshot,
    SiteStatus,
    TetraLocalParity,
    TetraResidualFactorValueSnapshot,
    TetraToken,
    TetraTokenParityFactorValueSnapshot,
    TetraValue,
    VarId,
    WriterAtomFrame,
    WriterAtomOccurrenceRecord,
    WriterBondOccurrenceRecord,
    WriterBranchFrame,
    WriterClosedClosure,
    WriterClosureLabel,
    WriterFrontierCursor,
    WriterLocalOrderRecord,
    WriterOpenClosureEndpoint,
    WriterPolicyStateKey,
    WriterRingLabelState,
    WriterRingStateKey,
    WriterStateKey,
    WriterStereoStateKey,
)
_CLASS_BY_PATH = {
    f"{cls.__module__}.{cls.__qualname__}": cls for cls in _ALLOWED_CLASSES
}


def writer_frontier_cursor_from_closed_terms(term: object) -> WriterFrontierCursor:
    value = _closed_value(term, WriterFrontierCursor)
    if not isinstance(value, WriterFrontierCursor):
        _violation("cursor_type_mismatch")
    return value


def _closed_value(term: object, annotation: object = None) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if term is None or isinstance(term, (str, bool, int)):
        return term
    if isinstance(term, list):
        if origin is frozenset:
            item_type = args[0] if args else None
            return frozenset(_closed_value(item, item_type) for item in term)
        if origin is dict:
            if any(not isinstance(item, list) or len(item) != 2 for item in term):
                _violation("mapping_shape_mismatch")
            key_type, value_type = args if len(args) == 2 else (None, None)
            return {
                _closed_value(item[0], key_type): _closed_value(item[1], value_type)
                for item in term
            }
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(term) != len(args):
                _violation("tuple_shape_mismatch")
            return tuple(_closed_value(item, item_type) for item, item_type in zip(term, args))
        item_type = args[0] if origin is tuple and len(args) == 2 else None
        return tuple(_closed_value(item, item_type) for item in term)
    if not isinstance(term, Mapping):
        _violation("term_shape_mismatch")
    if "__enum__" in term:
        if set(term) != {"__enum__", "value"}:
            _violation("enum_shape_mismatch")
        cls = _allowed_class(term["__enum__"])
        if not issubclass(cls, Enum) or (annotation in _ALLOWED_CLASSES and cls is not annotation):
            _violation("enum_class_mismatch")
        try:
            return cls(term["value"])
        except ValueError:
            _violation("enum_value_mismatch")
    if set(term) != {"__dataclass__", "fields"}:
        _violation("dataclass_shape_mismatch")
    cls = _allowed_class(term["__dataclass__"])
    if not is_dataclass(cls) or (annotation in _ALLOWED_CLASSES and cls is not annotation):
        _violation("dataclass_class_mismatch")
    raw_items = term["fields"]
    if not isinstance(raw_items, list) or any(
        not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
        for item in raw_items
    ):
        _violation("dataclass_fields_shape_mismatch")
    raw_fields = dict(raw_items)
    declared = {field.name for field in fields(cls)}
    if len(raw_fields) != len(raw_items) or set(raw_fields) != declared:
        _violation("dataclass_fields_mismatch")
    hints = get_type_hints(cls)
    return cls(**{
        field.name: _closed_value(raw_fields[field.name], hints.get(field.name))
        for field in fields(cls)
    })


def _allowed_class(path: object) -> type:
    if not isinstance(path, str) or path not in _CLASS_BY_PATH:
        _violation("class_not_allowed")
    return _CLASS_BY_PATH[path]


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot closed term violation: {kind}",
    )


__all__ = ("writer_frontier_cursor_from_closed_terms",)
