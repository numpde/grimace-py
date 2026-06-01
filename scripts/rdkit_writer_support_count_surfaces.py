"""RDKit writer support-count flag surface helpers."""

from __future__ import annotations


CANDIDATE_MINING_SURFACE_FLAGS: dict[str, dict[str, bool]] = {
    "nonisomeric__random": {
        "isomericSmiles": False,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": False,
        "allBondsExplicit": False,
        "allHsExplicit": False,
        "ignoreAtomMapNumbers": False,
    },
    "isomeric__random": {
        "isomericSmiles": True,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": False,
        "allBondsExplicit": False,
        "allHsExplicit": False,
        "ignoreAtomMapNumbers": False,
    },
    "nonisomeric__random_kekule": {
        "isomericSmiles": False,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": True,
        "allBondsExplicit": False,
        "allHsExplicit": False,
        "ignoreAtomMapNumbers": False,
    },
    "nonisomeric__random_all_bonds_explicit": {
        "isomericSmiles": False,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": False,
        "allBondsExplicit": True,
        "allHsExplicit": False,
        "ignoreAtomMapNumbers": False,
    },
    "nonisomeric__random_all_hs_explicit": {
        "isomericSmiles": False,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": False,
        "allBondsExplicit": False,
        "allHsExplicit": True,
        "ignoreAtomMapNumbers": False,
    },
}


def surface_name(flags: dict[str, bool]) -> str:
    surface = "isomeric" if flags["isomericSmiles"] else "nonisomeric"
    modifiers = ["random"]
    if flags["kekuleSmiles"]:
        modifiers.append("kekule")
    if flags["allBondsExplicit"]:
        modifiers.append("all_bonds_explicit")
    if flags["allHsExplicit"]:
        modifiers.append("all_hs_explicit")
    if flags["ignoreAtomMapNumbers"]:
        modifiers.append("ignore_atom_maps")
    return f"{surface}__{'_'.join(modifiers)}"


def surface_flags(surface: str) -> dict[str, bool]:
    try:
        return dict(CANDIDATE_MINING_SURFACE_FLAGS[surface])
    except KeyError as exc:
        raise ValueError(f"unknown surface: {surface!r}") from exc
