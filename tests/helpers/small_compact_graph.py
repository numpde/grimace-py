"""Interpret the selected C/O, branch, dot, and one-digit ring witness grammar."""

from rdkit import Chem


def small_compact_molecule(smiles: str) -> Chem.Mol:
    """Interpret selected C/O, branch, dot, and one-digit ring spellings.

    This temporary witness builds a graph independently of both the production
    decoder and RDKit's SMILES parser. It belongs in the shared fact/event/
    constraint model once that model can interpret emitted syntax.
    """
    graph = Chem.RWMol()
    rings: dict[str, int] = {}
    offset = 0

    def fail(reason: str) -> None:
        raise ValueError(f"small compact graph witness rejected {smiles!r}: {reason}")

    def chain(attachment: int | None, *, branch: bool) -> None:
        nonlocal offset
        if branch and offset < len(smiles) and smiles[offset] == ".":
            attachment = None
            offset += 1
        current = attachment
        needs_atom = True
        while offset < len(smiles) and smiles[offset] != ")":
            symbol = smiles[offset]
            if symbol not in "CO":
                fail(f"expected C or O atom at offset {offset}; unsupported token {symbol!r}")
            atom = graph.AddAtom(Chem.Atom(symbol))
            if current is not None:
                graph.AddBond(current, atom, Chem.BondType.SINGLE)
            current = atom
            offset += 1
            needs_atom = False

            while offset < len(smiles) and smiles[offset].isdigit():
                label = smiles[offset]
                if label not in "0123456789":
                    fail(f"unsupported ring label {label!r}")
                if label in rings:
                    start = rings.pop(label)
                    if start == atom:
                        fail(f"ring label {label!r} closes on the same atom")
                    if graph.GetBondBetweenAtoms(start, atom) is not None:
                        fail(f"ring label {label!r} duplicates an existing bond")
                    graph.AddBond(start, atom, Chem.BondType.SINGLE)
                else:
                    rings[label] = atom
                offset += 1

            while offset < len(smiles) and smiles[offset] == "(":
                offset += 1
                chain(current, branch=True)
                if offset >= len(smiles) or smiles[offset] != ")":
                    fail("unclosed branch")
                offset += 1

            if offset < len(smiles) and smiles[offset] == ".":
                offset += 1
                current = None
                needs_atom = True
        if needs_atom:
            fail("empty chain or trailing dot")

    chain(None, branch=False)
    if offset != len(smiles):
        fail(f"unexpected closing branch at offset {offset}")
    if rings:
        fail(f"unclosed ring labels {sorted(rings)!r}")
    molecule = graph.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def small_compact_graph(smiles: str) -> str:
    """Return a canonical comparison spelling for the interpreted graph."""
    return Chem.MolToSmiles(small_compact_molecule(smiles), canonical=True, isomericSmiles=True)

