from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
)
SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {"b", "c", "n", "o", "p", "s"}
)
SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {"se", "te"}
)
SOUTH_STAR_BRACKET_ONLY_MAIN_GROUP_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {"As", "Ge", "Sb", "Se", "Si", "Te"}
)
SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS: frozenset[str] = (
    SOUTH_STAR_BRACKET_ONLY_MAIN_GROUP_ATOM_TEXT_TOKENS
)
SOUTH_STAR_BRACKET_AROMATIC_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {
        "[nH]",
        "[15nH]",
        "[n:7]",
        "[nH+]",
    }
)
# Representative bracket-atom examples. The accepted bracket-token language is
# the predicate below, because isotope, charge, hydrogen, and map values are
# field-derived rather than a finite token list.
SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {
        "[H]",
        "[2H]",
        "[13CH3:7]",
        "[15NH3+]",
        "[H+]",
        "[Cl-]",
        "[NH4+]",
        "[C@H]",
        "[C@@H]",
        "[C@]",
        "[C@@]",
        "[CH3:1]",
        "[CH3]",
        "[O]",
        "[AsH2]",
        "[AsH3]",
        "[GeH3]",
        "[GeH4]",
        "[SeH]",
        "[SiH3]",
        "[SbH3]",
        "[TeH]",
        *SOUTH_STAR_BRACKET_AROMATIC_ATOM_TEXT_TOKENS,
        "[se]",
        "[te]",
    }
)
SOUTH_STAR_SUPPORTED_ATOM_SYMBOLS: frozenset[str] = frozenset(
    {
        "H",
        *SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS,
        *SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS,
    }
)


@dataclass(frozen=True, slots=True)
class SouthStarAtomTextFields:
    atom_idx: int
    atomic_number: int
    symbol: str
    isotope: int
    formal_charge: int
    radical_electron_count: int
    atom_map_number: int
    explicit_hydrogen_count: int
    chiral_tag: str
    is_aromatic: bool


@dataclass(frozen=True, slots=True)
class SouthStarAtomTextUnsupportedReason:
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class SouthStarAtomTextModifierObligation:
    fields: SouthStarAtomTextFields
    modifier_name: str
    field_name: str
    value: int
    unsupported_category: str | None
    renderer_requirement: str
    reason: str

    def __post_init__(self) -> None:
        if self.value == 0:
            raise ValueError("atom-text modifier obligations require nonzero input")
        if getattr(self.fields, self.field_name) != self.value:
            raise ValueError(
                "atom-text modifier obligation value must match source fields"
            )

    @property
    def atom_idx(self) -> int:
        return self.fields.atom_idx

    @property
    def unsupported_reason(self) -> SouthStarAtomTextUnsupportedReason:
        if self.unsupported_category is None:
            raise ValueError(
                "renderer-capable atom-text modifier has no unsupported reason"
            )
        return SouthStarAtomTextUnsupportedReason(
            category=self.unsupported_category,
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class SouthStarAtomTextObligation:
    atom_idx: int
    fields: SouthStarAtomTextFields
    emitted_text: str
    token_family: str
    bracket_obligations: tuple[str, ...]

    @property
    def uses_brackets(self) -> bool:
        return self.emitted_text.startswith("[") and self.emitted_text.endswith("]")


@dataclass(frozen=True, slots=True)
class _AtomTextModifierObligationSpec:
    modifier_name: str
    field_name: str
    unsupported_category: str | None
    renderer_requirement: str
    reason: str


_ATOM_TEXT_MODIFIER_OBLIGATION_SPECS = (
    _AtomTextModifierObligationSpec(
        modifier_name="isotope",
        field_name="isotope",
        unsupported_category=None,
        renderer_requirement="bracket_atom_isotope_prefix",
        reason=(
            "isotopic atom text is rendered as a bracket-atom isotope prefix"
        ),
    ),
    _AtomTextModifierObligationSpec(
        modifier_name="charge",
        field_name="formal_charge",
        unsupported_category=None,
        renderer_requirement="bracket_atom_charge_suffix",
        reason=(
            "charged atom text is rendered as a bracket-atom charge suffix"
        ),
    ),
    _AtomTextModifierObligationSpec(
        modifier_name="radical",
        field_name="radical_electron_count",
        unsupported_category=None,
        renderer_requirement="bracket_atom_radical_valence_semantics",
        reason=(
            "radical atom text is represented by bracket atom valence, explicit "
            "hydrogen, and bond context"
        ),
    ),
    _AtomTextModifierObligationSpec(
        modifier_name="atom_map",
        field_name="atom_map_number",
        unsupported_category=None,
        renderer_requirement="bracket_atom_map_suffix",
        reason=(
            "atom-map text is rendered as a bracket-atom map suffix"
        ),
    ),
)


def south_star_atom_text_fields(atom: Chem.Atom) -> SouthStarAtomTextFields:
    return SouthStarAtomTextFields(
        atom_idx=atom.GetIdx(),
        atomic_number=atom.GetAtomicNum(),
        symbol=atom.GetSymbol(),
        isotope=atom.GetIsotope(),
        formal_charge=atom.GetFormalCharge(),
        radical_electron_count=atom.GetNumRadicalElectrons(),
        atom_map_number=atom.GetAtomMapNum(),
        explicit_hydrogen_count=atom.GetNumExplicitHs(),
        chiral_tag=str(atom.GetChiralTag()),
        is_aromatic=atom.GetIsAromatic(),
    )


def unsupported_atom_text_reasons(
    fields: SouthStarAtomTextFields,
) -> tuple[SouthStarAtomTextUnsupportedReason, ...]:
    reasons = [
        obligation.unsupported_reason
        for obligation in atom_text_modifier_obligations(fields)
        if obligation.unsupported_category is not None
    ]
    if fields.symbol not in SOUTH_STAR_SUPPORTED_ATOM_SYMBOLS:
        reasons.append(
            SouthStarAtomTextUnsupportedReason(
                category="unsupported_atom_text",
                reason=(
                    f"atom symbol {fields.symbol!r} is outside first South Star "
                    "scope"
                ),
            )
        )
    return tuple(reasons)


def atom_text_modifier_obligations(
    fields: SouthStarAtomTextFields,
) -> tuple[SouthStarAtomTextModifierObligation, ...]:
    obligations: list[SouthStarAtomTextModifierObligation] = []
    for spec in _ATOM_TEXT_MODIFIER_OBLIGATION_SPECS:
        value = getattr(fields, spec.field_name)
        if value == 0:
            continue
        obligations.append(
            SouthStarAtomTextModifierObligation(
                fields=fields,
                modifier_name=spec.modifier_name,
                field_name=spec.field_name,
                value=value,
                unsupported_category=spec.unsupported_category,
                renderer_requirement=spec.renderer_requirement,
                reason=spec.reason,
            )
        )
    return tuple(obligations)


def atom_text_obligation_for_supported_atom(
    atom: Chem.Atom,
) -> SouthStarAtomTextObligation:
    fields = south_star_atom_text_fields(atom)
    return atom_text_obligation_for_supported_fields(fields)


def atom_text_obligation_for_supported_fields(
    fields: SouthStarAtomTextFields,
) -> SouthStarAtomTextObligation:
    unsupported = unsupported_atom_text_reasons(fields)
    if unsupported:
        categories = ", ".join(reason.category for reason in unsupported)
        raise NotImplementedError(
            f"South Star atom text unsupported for atom {fields.atom_idx}: "
            f"{categories}"
        )
    if fields.is_aromatic:
        return _aromatic_atom_text_obligation(fields)
    if _requires_bracket_atom_text(fields):
        return _bracket_atom_text_obligation(fields)
    if fields.symbol == "H":
        return SouthStarAtomTextObligation(
            atom_idx=fields.atom_idx,
            fields=fields,
            emitted_text="[H]",
            token_family="bracket_atom",
            bracket_obligations=("element_requires_bracket",),
        )
    if fields.symbol in SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS:
        return SouthStarAtomTextObligation(
            atom_idx=fields.atom_idx,
            fields=fields,
            emitted_text=fields.symbol,
            token_family="organic_subset",
            bracket_obligations=(),
        )
    raise AssertionError(f"unhandled South Star atom text symbol {fields.symbol!r}")


def _aromatic_atom_text_obligation(
    fields: SouthStarAtomTextFields,
) -> SouthStarAtomTextObligation:
    if _requires_bracket_atom_text(fields):
        return _bracket_aromatic_atom_text_obligation(fields)
    token = fields.symbol.lower()
    if token not in SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS:
        raise NotImplementedError(
            f"South Star aromatic atom text unsupported for symbol {fields.symbol!r}"
        )
    return SouthStarAtomTextObligation(
        atom_idx=fields.atom_idx,
        fields=fields,
        emitted_text=token,
        token_family="aromatic_subset",
        bracket_obligations=(),
    )


def is_south_star_bracket_atom_text_token(token: str) -> bool:
    if not (token.startswith("[") and token.endswith("]")):
        return False
    body = token[1:-1]
    if not body:
        return False
    rest = body
    while rest and rest[0].isdigit():
        rest = rest[1:]

    symbol = ""
    bracket_symbols = (
        SOUTH_STAR_SUPPORTED_ATOM_SYMBOLS
        | SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS
        | SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS
    )
    for candidate in sorted(bracket_symbols, key=len, reverse=True):
        if rest.startswith(candidate):
            symbol = candidate
            rest = rest[len(candidate) :]
            break
    if not symbol:
        return False

    is_aromatic_symbol = symbol in SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS
    if rest.startswith("@@") or rest.startswith("@"):
        if is_aromatic_symbol:
            return False
        if rest.startswith("@@"):
            rest = rest[2:]
        else:
            rest = rest[1:]

    if rest.startswith("H"):
        rest = rest[1:]
        while rest and rest[0].isdigit():
            rest = rest[1:]

    if rest.startswith("+") or rest.startswith("-"):
        rest = rest[1:]
        while rest and rest[0].isdigit():
            rest = rest[1:]

    if rest.startswith(":"):
        rest = rest[1:]
        if not rest or not rest.isdigit():
            return False
        rest = ""

    return rest == ""


def _requires_bracket_atom_text(fields: SouthStarAtomTextFields) -> bool:
    return (
        fields.symbol == "H"
        or fields.symbol in SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS
        or fields.isotope != 0
        or fields.formal_charge != 0
        or fields.atom_map_number != 0
        or fields.explicit_hydrogen_count != 0
        or fields.radical_electron_count != 0
    )


def _bracket_atom_text_obligation(
    fields: SouthStarAtomTextFields,
) -> SouthStarAtomTextObligation:
    obligations: list[str] = ["bracket_atom"]
    if fields.symbol == "H" and (
        fields.isotope == 0
        and fields.formal_charge == 0
        and fields.atom_map_number == 0
        and fields.explicit_hydrogen_count == 0
    ):
        obligations.append("element_requires_bracket")

    if fields.symbol in SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS:
        obligations.append("non_organic_symbol_requires_bracket")

    text = (
        "["
        f"{_isotope_text(fields)}"
        f"{fields.symbol}"
        f"{_hydrogen_text(fields)}"
        f"{_charge_text(fields.formal_charge)}"
        f"{_atom_map_text(fields.atom_map_number)}"
        "]"
    )
    if fields.isotope != 0:
        obligations.append("isotope_prefix")
    if fields.explicit_hydrogen_count != 0:
        obligations.append("explicit_hydrogen_count")
    if fields.formal_charge != 0:
        obligations.append("charge_suffix")
    if fields.atom_map_number != 0:
        obligations.append("atom_map_suffix")
    if fields.radical_electron_count != 0:
        obligations.append("radical_valence_semantics")
    if not is_south_star_bracket_atom_text_token(text):
        raise AssertionError(f"rendered unsupported South Star atom text {text!r}")
    return SouthStarAtomTextObligation(
        atom_idx=fields.atom_idx,
        fields=fields,
        emitted_text=text,
        token_family="bracket_atom",
        bracket_obligations=tuple(obligations),
    )


def _bracket_aromatic_atom_text_obligation(
    fields: SouthStarAtomTextFields,
) -> SouthStarAtomTextObligation:
    token = fields.symbol.lower()
    if token not in (
        SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS
        | SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS
    ):
        raise NotImplementedError(
            f"South Star aromatic atom text unsupported for symbol {fields.symbol!r}"
        )
    text = (
        "["
        f"{_isotope_text(fields)}"
        f"{token}"
        f"{_hydrogen_text(fields)}"
        f"{_charge_text(fields.formal_charge)}"
        f"{_atom_map_text(fields.atom_map_number)}"
        "]"
    )
    obligations = ["bracket_aromatic_atom"]
    if fields.isotope != 0:
        obligations.append("isotope_prefix")
    if fields.explicit_hydrogen_count != 0:
        obligations.append("explicit_hydrogen_count")
    if fields.formal_charge != 0:
        obligations.append("charge_suffix")
    if fields.atom_map_number != 0:
        obligations.append("atom_map_suffix")
    if fields.radical_electron_count != 0:
        obligations.append("radical_valence_semantics")
    if not is_south_star_bracket_atom_text_token(text):
        raise AssertionError(
            f"rendered unsupported South Star aromatic atom text {text!r}"
        )
    return SouthStarAtomTextObligation(
        atom_idx=fields.atom_idx,
        fields=fields,
        emitted_text=text,
        token_family="bracket_aromatic_atom",
        bracket_obligations=tuple(obligations),
    )


def _isotope_text(fields: SouthStarAtomTextFields) -> str:
    return "" if fields.isotope == 0 else str(fields.isotope)


def _hydrogen_text(fields: SouthStarAtomTextFields) -> str:
    if fields.explicit_hydrogen_count == 0:
        return ""
    if fields.explicit_hydrogen_count == 1:
        return "H"
    return f"H{fields.explicit_hydrogen_count}"


def _charge_text(charge: int) -> str:
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    return sign if magnitude == 1 else f"{sign}{magnitude}"


def _atom_map_text(atom_map_number: int) -> str:
    if atom_map_number == 0:
        return ""
    return f":{atom_map_number}"


def tetrahedral_atom_text_obligation(
    atom: Chem.Atom,
    *,
    stereo_token: str,
    implicit_hydrogen_count: int,
) -> SouthStarAtomTextObligation:
    fields = south_star_atom_text_fields(atom)
    _assert_supported_tetrahedral_atom_text_fields(
        fields,
        stereo_token=stereo_token,
        implicit_hydrogen_count=implicit_hydrogen_count,
    )
    hydrogen_text = "H" if implicit_hydrogen_count else ""
    hydrogen_obligation = (
        ("implicit_hydrogen_text",) if implicit_hydrogen_count else ()
    )
    return SouthStarAtomTextObligation(
        atom_idx=fields.atom_idx,
        fields=fields,
        emitted_text=f"[{fields.symbol}{stereo_token}{hydrogen_text}]",
        token_family="bracket_atom",
        bracket_obligations=(
            "tetrahedral_stereo_token",
            *hydrogen_obligation,
        ),
    )


def _assert_supported_tetrahedral_atom_text_fields(
    fields: SouthStarAtomTextFields,
    *,
    stereo_token: str,
    implicit_hydrogen_count: int,
) -> None:
    unsupported = unsupported_atom_text_reasons(fields)
    if unsupported:
        categories = ", ".join(reason.category for reason in unsupported)
        raise NotImplementedError(
            f"South Star tetrahedral atom text unsupported for atom "
            f"{fields.atom_idx}: {categories}"
        )
    if fields.is_aromatic:
        raise NotImplementedError(
            "South Star tetrahedral atom text rendering requires a non-aromatic atom"
        )
    if fields.symbol != "C":
        raise NotImplementedError(
            "South Star tetrahedral atom text currently supports carbon centers"
        )
    if stereo_token not in {"@", "@@"}:
        raise ValueError(f"unsupported tetrahedral stereo token {stereo_token!r}")
    if implicit_hydrogen_count not in {0, 1}:
        raise ValueError(
            "South Star tetrahedral atom text requires zero or one implicit hydrogen"
        )


def atom_text_for_supported_atom(atom: Chem.Atom) -> str:
    return atom_text_obligation_for_supported_atom(atom).emitted_text
