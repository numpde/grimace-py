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


def atom_text_for_supported_atom(atom: Chem.Atom) -> str:
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
        return "[H]"
    if fields.symbol in SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS:
        return fields.symbol
    raise AssertionError(f"unhandled South Star atom text symbol {fields.symbol!r}")
