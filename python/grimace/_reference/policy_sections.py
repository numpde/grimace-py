from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PUBLIC_OPTION_NAMES,
    coerce_required_public_options,
    public_option_values,
)
from grimace._reference.policy import ReferencePolicy


SAMPLING_KEYS = frozenset({"seed", "draw_budget", *MOL_TO_SMILES_PUBLIC_OPTION_NAMES})
IDENTITY_KEYS = frozenset({"parse_with_rdkit", *MOL_TO_SMILES_PUBLIC_OPTION_NAMES})


def _require_keys(section: Mapping[str, Any], expected: frozenset[str], section_name: str) -> None:
    actual = set(section)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{section_name} keys must match exactly; missing={missing}, extra={extra}"
        )


def _json_object_section(
    policy: ReferencePolicy,
    section_name: str,
    *,
    expected_keys: frozenset[str],
) -> Mapping[str, Any]:
    section = policy.data[section_name]
    if not isinstance(section, Mapping):
        raise TypeError(f"{section_name} policy must be a JSON object")
    _require_keys(section, expected_keys, section_name)
    return section


def _json_integer_field(section: Mapping[str, Any], field_name: str) -> int:
    value = section[field_name]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a JSON integer")
    return int(value)


def _json_boolean_field(section: Mapping[str, Any], field_name: str) -> bool:
    value = section[field_name]
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a JSON boolean")
    return value


def sampling_section(policy: ReferencePolicy) -> dict[str, object]:
    sampling = _json_object_section(
        policy,
        "sampling",
        expected_keys=SAMPLING_KEYS,
    )
    return {
        "seed": _json_integer_field(sampling, "seed"),
        "draw_budget": _json_integer_field(sampling, "draw_budget"),
        **coerce_required_public_options(
            MOL_TO_SMILES_OPTIONS,
            public_option_values(MOL_TO_SMILES_OPTIONS, sampling),
            context="sampling policy",
        ),
    }


def identity_section(policy: ReferencePolicy) -> dict[str, object]:
    identity = _json_object_section(
        policy,
        "identity_check",
        expected_keys=IDENTITY_KEYS,
    )
    return {
        "parse_with_rdkit": _json_boolean_field(identity, "parse_with_rdkit"),
        **coerce_required_public_options(
            MOL_TO_SMILES_OPTIONS,
            public_option_values(MOL_TO_SMILES_OPTIONS, identity),
            context="identity_check policy",
        ),
    }
