"""Internal MolToSmiles public option inventory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Literal


_ValueRule = Literal["bool_like", "root_atom"]
_Scope = Literal["prepared", "call"]


@dataclass(frozen=True, slots=True)
class _OptionSpec:
    public_name: str
    internal_name: str
    default: object
    value_rule: _ValueRule
    scope: _Scope


MOL_TO_SMILES_OPTIONS = (
    _OptionSpec(
        public_name="isomericSmiles",
        internal_name="isomeric_smiles",
        default=True,
        value_rule="bool_like",
        scope="prepared",
    ),
    _OptionSpec(
        public_name="kekuleSmiles",
        internal_name="kekule_smiles",
        default=False,
        value_rule="bool_like",
        scope="prepared",
    ),
    _OptionSpec(
        public_name="rootedAtAtom",
        internal_name="rooted_at_atom",
        default=-1,
        value_rule="root_atom",
        scope="call",
    ),
    _OptionSpec(
        public_name="canonical",
        internal_name="canonical",
        default=True,
        value_rule="bool_like",
        scope="call",
    ),
    _OptionSpec(
        public_name="allBondsExplicit",
        internal_name="all_bonds_explicit",
        default=False,
        value_rule="bool_like",
        scope="prepared",
    ),
    _OptionSpec(
        public_name="allHsExplicit",
        internal_name="all_hs_explicit",
        default=False,
        value_rule="bool_like",
        scope="prepared",
    ),
    _OptionSpec(
        public_name="doRandom",
        internal_name="do_random",
        default=False,
        value_rule="bool_like",
        scope="call",
    ),
    _OptionSpec(
        public_name="ignoreAtomMapNumbers",
        internal_name="ignore_atom_map_numbers",
        default=False,
        value_rule="bool_like",
        scope="prepared",
    ),
)

MOL_TO_SMILES_PREPARED_OPTIONS = tuple(
    spec for spec in MOL_TO_SMILES_OPTIONS if spec.scope == "prepared"
)
MOL_TO_SMILES_PUBLIC_OPTION_NAMES = frozenset(
    spec.public_name for spec in MOL_TO_SMILES_OPTIONS
)


def _reject_unknown_options(
    values: Mapping[str, object],
    *,
    allowed_names: set[str],
    context: str,
) -> None:
    unknown_names = sorted(repr(name) for name in set(values) - allowed_names)
    if unknown_names:
        raise TypeError(
            f"{context} got unknown option(s): {', '.join(unknown_names)}"
        )


def public_option_values(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
) -> dict[str, object]:
    return {
        spec.public_name: values[spec.public_name]
        for spec in specs
        if spec.public_name in values
    }


def internal_option_values(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
) -> dict[str, object]:
    return {
        spec.internal_name: values[spec.internal_name]
        for spec in specs
        if spec.internal_name in values
    }


def coerce_option(
    spec: _OptionSpec,
    value: object,
    *,
    context: str,
) -> object:
    if spec.value_rule == "bool_like":
        if value is None:
            return False
        if not isinstance(value, Integral):
            raise TypeError(
                f"{context} requires {spec.public_name} to follow RDKit's Python binding "
                "and be a bool, int, or None"
            )
        return bool(value)
    if spec.value_rule == "root_atom":
        if not isinstance(value, Integral):
            raise TypeError(
                f"{context} requires {spec.public_name} to follow RDKit's Python binding "
                "and be an integer"
            )
        return normalize_root_atom(int(value))
    raise RuntimeError(
        f"unsupported MolToSmiles option value_rule: {spec.value_rule!r}"
    )


def normalize_root_atom(value: int) -> int:
    return -1 if value < 0 else value


def coerce_public_options(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    _reject_unknown_options(
        values,
        allowed_names={spec.public_name for spec in specs},
        context=context,
    )
    return {
        spec.internal_name: coerce_option(
            spec,
            values.get(spec.public_name, spec.default),
            context=context,
        )
        for spec in specs
    }


def coerce_required_public_options(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    _reject_unknown_options(
        values,
        allowed_names={spec.public_name for spec in specs},
        context=context,
    )
    coerced: dict[str, object] = {}
    for spec in specs:
        if spec.public_name not in values:
            raise ValueError(f"{context} requires {spec.public_name}")
        coerced[spec.public_name] = coerce_option(
            spec,
            values[spec.public_name],
            context=context,
        )
    return coerced


def coerce_internal_options(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    _reject_unknown_options(
        values,
        allowed_names={spec.internal_name for spec in specs},
        context=context,
    )
    return {
        spec.internal_name: coerce_option(
            spec,
            values.get(spec.internal_name, spec.default),
            context=context,
        )
        for spec in specs
    }


def public_options_from_internal_options(
    specs: tuple[_OptionSpec, ...],
    values: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    _reject_unknown_options(
        values,
        allowed_names={spec.internal_name for spec in specs},
        context=context,
    )
    return {
        spec.public_name: coerce_option(
            spec,
            values.get(spec.internal_name, spec.default),
            context=context,
        )
        for spec in specs
    }
