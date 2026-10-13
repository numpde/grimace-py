"""Interpret one H-bearing C/F/Cl/Br tetrahedral event language.

OpenSMILES v1.0 §§3.7.1 and 3.8.2 permit a dot-started branch and order
tetrahedral ligands by bond occurrence; a bracket H occupies the first slot.
This temporary witness belongs in the shared fact/event/constraint/solver/
renderer model. It neither parses general SMILES nor asks RDKit to parse output.
"""

from __future__ import annotations


def tetra_signature(text: str) -> tuple[tuple[tuple[str, str], ...], int]:
    """Return graph edges and signed tetrahedral parity for the declared grammar."""
    position = 0
    atoms: list[str] = []
    edges: set[tuple[int, int]] = set()
    rings: dict[str, tuple[int, int | None]] = {}
    center: int | None = None
    marker: str | None = None
    ligands: list[int | str | None] = []

    def fail(reason: str) -> None:
        raise ValueError(f"leading-dot tetra witness rejected {text!r}: {reason}")

    def bond(a: int, b: int) -> None:
        if a == b or tuple(sorted((a, b))) in edges:
            fail("duplicate or self bond")
        edges.add(tuple(sorted((a, b))))

    def atom(parent: int | None) -> int:
        nonlocal position, center, marker
        if text.startswith("[C@@H]", position):
            symbol, new_marker, width = "C", "@@", 6
        elif text.startswith("[C@H]", position):
            symbol, new_marker, width = "C", "@", 5
        elif text.startswith(("Cl", "Br"), position):
            symbol, new_marker, width = text[position:position + 2], None, 2
        elif text.startswith("F", position):
            symbol, new_marker, width = "F", None, 1
        else:
            fail(f"unsupported atom at offset {position}")
        if symbol in atoms:
            fail(f"repeated atom {symbol}")
        index = len(atoms)
        atoms.append(symbol)
        position += width
        if new_marker is not None:
            if center is not None:
                fail("more than one center")
            center, marker = index, new_marker
            if parent is not None:
                ligands.append(parent)
            # The incoming atom is the viewing reference; H is first among
            # the remaining ligands (OpenSMILES §3.8.2, N[C@H](O)C).
            ligands.append("H")
        if parent is not None:
            bond(parent, index)
            if parent == center:
                ligands.append(index)
        return index

    def chain(parent: int | None, *, branch: bool = False) -> None:
        nonlocal position
        if branch and position < len(text) and text[position] == ".":
            parent = None
            position += 1
        start = position
        current = parent
        while position < len(text) and text[position] != ")":
            current = atom(current)
            while position < len(text) and text[position].isdigit():
                digit = text[position]
                position += 1
                if digit in rings:
                    opener, slot = rings.pop(digit)
                    bond(opener, current)
                    if slot is not None:
                        ligands[slot] = current
                    if current == center:
                        ligands.append(opener)
                else:
                    slot = len(ligands) if current == center else None
                    if slot is not None:
                        ligands.append(None)
                    rings[digit] = (current, slot)
            while position < len(text) and text[position] == "(":
                position += 1
                chain(current, branch=True)
                if position >= len(text) or text[position] != ")":
                    fail("unclosed branch")
                position += 1
            if position < len(text) and text[position] == ".":
                position += 1
                current = None
                if position == len(text) or text[position] == ")":
                    fail("trailing dot")
        if position == start:
            fail("empty chain")

    chain(None)
    if position != len(text) or rings:
        fail("unmatched closing branch or ring")
    if sorted(atoms) != ["Br", "C", "Cl", "F"] or len(edges) != 3:
        fail("expected the connected C/F/Cl/Br star")
    if center is None or marker is None:
        fail("expected one H-bearing tetrahedral center")
    if len(ligands) != 4 or any(ligand is None for ligand in ligands):
        fail("tetrahedral center lacks four resolved ligand slots")
    names = [ligand if ligand == "H" else atoms[ligand] for ligand in ligands]
    if sorted(names) != ["Br", "Cl", "F", "H"]:
        fail("center ligands do not match the declared star")
    graph = tuple(sorted(tuple(sorted((atoms[a], atoms[b]))) for a, b in edges))
    ranks = [sorted(names).index(name) for name in names]
    inversions = sum(ranks[a] > ranks[b] for a in range(4) for b in range(a + 1, 4))
    return graph, (inversions + (marker == "@@")) % 2
