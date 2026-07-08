"""Certified local atom-text relations for writer-shaped atom emission."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .ids import AtomId


@dataclass(frozen=True, slots=True)
class WriterBracketAtomTextEvidence:
    atom: AtomId
    element: str
    formal_charge: int
    hydrogen_count: int
    aromatic: bool
    isotope: object | None
    rendered_text: str
    bracket_required: bool


def writer_bracket_atom_text_evidence(
    atom: AtomFacts,
    *,
    rendered_text: str,
) -> WriterBracketAtomTextEvidence:
    evidence = WriterBracketAtomTextEvidence(
        atom=atom.id,
        element=atom.symbol,
        formal_charge=atom.formal_charge,
        hydrogen_count=atom.implicit_h_count,
        aromatic=atom.is_aromatic,
        isotope=atom.isotope,
        rendered_text=rendered_text,
        bracket_required=True,
    )
    validate_writer_bracket_atom_text_transition(
        atom=atom,
        rendered_text=rendered_text,
        evidence=evidence,
    )
    return evidence


def validate_writer_bracket_atom_text_transition(
    *,
    atom: AtomFacts,
    rendered_text: str,
    evidence: WriterBracketAtomTextEvidence,
) -> None:
    if evidence.atom != atom.id:
        _atom_text_violation("bracket_atom_id_mismatch")
    if evidence.element != atom.symbol:
        _atom_text_violation("bracket_atom_element_mismatch")
    if evidence.formal_charge != atom.formal_charge:
        _atom_text_violation("bracket_atom_charge_mismatch")
    if evidence.hydrogen_count != atom.implicit_h_count:
        _atom_text_violation("bracket_atom_hydrogen_count_mismatch")
    if evidence.aromatic != atom.is_aromatic:
        _atom_text_violation("bracket_atom_aromatic_mismatch")
    if evidence.isotope != atom.isotope:
        _atom_text_violation("bracket_atom_isotope_mismatch")
    if evidence.rendered_text != rendered_text:
        _atom_text_violation("bracket_atom_rendered_text_mismatch")
    if not evidence.bracket_required:
        _atom_text_violation("bracket_atom_not_required")
    if not rendered_text.startswith("[") or not rendered_text.endswith("]"):
        _atom_text_violation("bracket_atom_text_lacks_brackets")
    if atom.explicit_h_count != 0:
        _atom_text_violation("bracket_atom_explicit_h_unsupported")
    if atom.no_implicit:
        _atom_text_violation("bracket_atom_no_implicit_unsupported")
    if atom.is_aromatic:
        _atom_text_violation("bracket_atom_aromatic_unsupported")

    expected = bracket_atom_text(atom)
    if rendered_text != expected:
        _atom_text_violation("bracket_atom_text_mismatch")


def bracket_atom_text(atom: AtomFacts) -> str:
    if is_supported_charged_nitrogen_bracket_atom(atom):
        return _charged_nitrogen_bracket_text(atom.implicit_h_count)
    if is_supported_simple_isotope_carbon_bracket_atom(atom):
        isotope = atom.isotope
        if isotope is None:
            _atom_text_violation("bracket_atom_isotope_missing")
        return f"[{isotope}C{_hydrogen_suffix(atom.implicit_h_count)}]"
    if atom.isotope is not None:
        _atom_text_violation("bracket_atom_isotope_unsupported")
    if atom.symbol != "N":
        _atom_text_violation("bracket_atom_element_unsupported")
    if atom.formal_charge != 1:
        _atom_text_violation("bracket_atom_charge_unsupported")
    if atom.implicit_h_count not in {0, 1, 2, 3, 4}:
        _atom_text_violation("bracket_atom_hydrogen_count_unsupported")
    _atom_text_violation("bracket_atom_unsupported")


def charged_nitrogen_bracket_atom_text(atom: AtomFacts) -> str:
    if not is_supported_charged_nitrogen_bracket_atom(atom):
        _atom_text_violation("bracket_atom_not_supported_charged_nitrogen")
    return _charged_nitrogen_bracket_text(atom.implicit_h_count)


def is_supported_bracket_atom(atom: AtomFacts) -> bool:
    return (
        is_supported_charged_nitrogen_bracket_atom(atom)
        or is_supported_simple_isotope_carbon_bracket_atom(atom)
    )


def is_supported_charged_nitrogen_bracket_atom(atom: AtomFacts) -> bool:
    return (
        atom.symbol == "N"
        and atom.formal_charge == 1
        and atom.isotope is None
        and atom.explicit_h_count == 0
        and not atom.no_implicit
        and not atom.is_aromatic
        and atom.implicit_h_count in {0, 1, 2, 3, 4}
    )


def is_supported_simple_isotope_carbon_bracket_atom(atom: AtomFacts) -> bool:
    return (
        atom.symbol == "C"
        and atom.isotope is not None
        and isinstance(atom.isotope, int)
        and atom.isotope > 0
        and atom.formal_charge == 0
        and atom.explicit_h_count == 0
        and not atom.no_implicit
        and not atom.is_aromatic
        and atom.implicit_h_count in {0, 1, 2, 3, 4}
    )


def _charged_nitrogen_bracket_text(hydrogen_count: int) -> str:
    return f"[N{_hydrogen_suffix(hydrogen_count)}+]"


def _hydrogen_suffix(hydrogen_count: int) -> str:
    hydrogen = "H" if hydrogen_count == 1 else f"H{hydrogen_count}"
    if hydrogen_count == 0:
        hydrogen = ""
    return hydrogen


def _atom_text_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_ATOM,
        f"writer bracket atom text violation: {kind}",
    )


__all__ = (
    "WriterBracketAtomTextEvidence",
    "bracket_atom_text",
    "charged_nitrogen_bracket_atom_text",
    "is_supported_bracket_atom",
    "is_supported_charged_nitrogen_bracket_atom",
    "is_supported_simple_isotope_carbon_bracket_atom",
    "validate_writer_bracket_atom_text_transition",
    "writer_bracket_atom_text_evidence",
)
