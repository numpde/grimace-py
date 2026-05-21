from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
)
SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS: frozenset[str] = frozenset(
    {
        "[H]",
        "[C@H]",
        "[C@@H]",
        "[C@]",
        "[C@@]",
    }
)
SOUTH_STAR_SUPPORTED_ATOM_SYMBOLS: frozenset[str] = frozenset(
    {"H", *SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS}
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
class SouthStarAtomTextObligation:
    atom_idx: int
    fields: SouthStarAtomTextFields
    emitted_text: str
    token_family: str
    bracket_obligations: tuple[str, ...]

    @property
    def uses_brackets(self) -> bool:
        return self.emitted_text.startswith("[") and self.emitted_text.endswith("]")


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
    reasons = []
    if fields.isotope != 0:
        reasons.append(
            SouthStarAtomTextUnsupportedReason(
                category="unsupported_atom_isotope",
                reason=(
                    "isotopic atom text is outside the current South Star "
                    "bracket-atom grammar contract"
                ),
            )
        )
    if fields.formal_charge != 0:
        reasons.append(
            SouthStarAtomTextUnsupportedReason(
                category="unsupported_atom_charge",
                reason=(
                    "charged atom text is outside the current South Star "
                    "bracket-atom grammar contract"
                ),
            )
        )
    if fields.radical_electron_count != 0:
        reasons.append(
            SouthStarAtomTextUnsupportedReason(
                category="unsupported_radical_atom",
                reason=(
                    "radical atom text is outside the current South Star "
                    "bracket-atom grammar contract"
                ),
            )
        )
    if fields.atom_map_number != 0:
        reasons.append(
            SouthStarAtomTextUnsupportedReason(
                category="unsupported_atom_map",
                reason=(
                    "atom-map text is outside the current South Star "
                    "bracket-atom grammar contract"
                ),
            )
        )
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


def atom_text_obligation_for_supported_atom(
    atom: Chem.Atom,
) -> SouthStarAtomTextObligation:
    fields = south_star_atom_text_fields(atom)
    unsupported = unsupported_atom_text_reasons(fields)
    if unsupported:
        categories = ", ".join(reason.category for reason in unsupported)
        raise NotImplementedError(
            f"South Star atom text unsupported for atom {fields.atom_idx}: "
            f"{categories}"
        )
    if fields.is_aromatic:
        raise NotImplementedError(
            "South Star atom text rendering requires a non-aromatic atom"
        )
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
