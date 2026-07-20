"""Facts-derived unbracketed atom spelling for the ordinary dialect."""

from __future__ import annotations

from .facts import AtomFacts


_ORGANIC_SUBSET = frozenset({"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"})
_AROMATIC_SYMBOLS = {"C": "c", "N": "n", "O": "o", "S": "s"}


def ordinary_unbracketed_atom_text_for_facts(atom: AtomFacts) -> str | None:
    """Return the unique ordinary unbracketed token, if this atom has one."""

    if (
        atom.isotope is not None
        or atom.formal_charge != 0
        or atom.explicit_h_count != 0
        or atom.no_implicit
    ):
        return None
    if atom.is_aromatic:
        if atom.symbol == "N" and atom.implicit_h_count:
            return None
        return _AROMATIC_SYMBOLS.get(atom.symbol)
    if atom.symbol in _ORGANIC_SUBSET:
        return atom.symbol
    return None


__all__ = ("ordinary_unbracketed_atom_text_for_facts",)
