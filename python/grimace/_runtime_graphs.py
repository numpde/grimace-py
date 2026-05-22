"""Prepared graph adapters for the public runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import cast

import grimace._prepared_mol as _prepared_mol
from grimace._prepared_mol import PreparedMol
from grimace._reference.prepared_graph import (
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)
from grimace._runtime_inputs import (
    MolToSmilesFlags,
    ensure_singly_connected_molecule,
    runtime_surface_kind,
    writer_flag_kwargs,
)

_core = importlib.import_module("grimace._core")


@dataclass(frozen=True, slots=True)
class _FragmentPlan:
    fragment: object
    rooted_at_atom: int | None


def as_disconnected_prepared_mol(mol_or_prepared: object) -> PreparedMol | None:
    if (
        isinstance(mol_or_prepared, PreparedMol)
        and _prepared_mol._fragment_count(mol_or_prepared) > 1
    ):
        return mol_or_prepared
    return None


def atom_count(mol_or_prepared: object) -> int:
    if isinstance(mol_or_prepared, PreparedMol):
        return _prepared_mol._atom_count(mol_or_prepared)
    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        return mol_or_prepared.atom_count
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        return cast(int, mol_or_prepared.atom_count)
    raise TypeError(f"Unsupported molecule/prepared type: {type(mol_or_prepared)!r}")


def prepared_mol_fragment_plans(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int | None,
) -> tuple[_FragmentPlan, ...]:
    rooted_fragments = _prepared_mol._rooted_fragments(
        prepared,
        rooted_at_atom=rooted_at_atom,
    )
    return tuple(
        _FragmentPlan(fragment, fragment_rooted_at_atom)
        for fragment, fragment_rooted_at_atom in rooted_fragments
    )


def connected_prepared_mol_fragment(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int,
) -> tuple[object, int]:
    rooted_at_atom_or_none = None if rooted_at_atom < 0 else rooted_at_atom
    plans = prepared_mol_fragment_plans(
        prepared,
        rooted_at_atom=rooted_at_atom_or_none,
    )
    if len(plans) != 1:
        raise NotImplementedError(
            "connected PreparedMol runtime requires one prepared fragment"
        )
    plan = plans[0]
    if plan.rooted_at_atom is None:
        return plan.fragment, -1
    return plan.fragment, plan.rooted_at_atom


def connected_prepared_mol_fragment_or_none(
    mol_or_prepared: object,
    *,
    rooted_at_atom: int,
) -> tuple[object, int] | None:
    if not isinstance(mol_or_prepared, PreparedMol):
        return None
    return connected_prepared_mol_fragment(
        mol_or_prepared,
        rooted_at_atom=rooted_at_atom,
    )


def prepare_smiles_graph(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    prepared = _coerce_prepared_graph(mol_or_prepared, flags=flags)
    if prepared is not None:
        return prepared

    surface_kind = runtime_surface_kind(mol_or_prepared, flags=flags)
    ensure_singly_connected_molecule(mol_or_prepared)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol_or_prepared,
        surface_kind=surface_kind,
        isomeric_smiles=flags.isomeric_smiles,
        kekule_smiles=flags.kekule_smiles,
        all_bonds_explicit=flags.all_bonds_explicit,
        all_hs_explicit=flags.all_hs_explicit,
        ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
    )
    return _core.PreparedSmilesGraph(reference_prepared)


def prepare_core_graph_for_static_inventory(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    prepared = _coerce_prepared_graph(mol_or_prepared, flags=flags)
    if prepared is None:
        raise TypeError(
            "Unsupported molecule/prepared type for static inventory: "
            f"{type(mol_or_prepared)!r}"
        )
    return prepared


def _coerce_prepared_graph(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object | None:
    surface_kind = runtime_surface_kind(mol_or_prepared, flags=flags)
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared

    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return _core.PreparedSmilesGraph(mol_or_prepared)

    return None


def _validate_surface_kind(
    prepared: object,
    *,
    surface_kind: str,
) -> None:
    if prepared.surface_kind != surface_kind:
        raise ValueError(
            f"PreparedSmilesGraph surface_kind={prepared.surface_kind!r} does not match "
            f"the requested surface_kind={surface_kind!r}"
        )


def _validate_writer_flags(
    prepared: object,
    flags: MolToSmilesFlags,
) -> None:
    expected = writer_flag_kwargs(flags)
    if isinstance(prepared, _core.PreparedSmilesGraph):
        matches = prepared.matches_writer_flags(**expected)
    else:
        matches = {
            "isomeric_smiles": bool(prepared.writer_do_isomeric_smiles),
            "kekule_smiles": bool(prepared.writer_kekule_smiles),
            "all_bonds_explicit": bool(prepared.writer_all_bonds_explicit),
            "all_hs_explicit": bool(prepared.writer_all_hs_explicit),
            "ignore_atom_map_numbers": bool(prepared.writer_ignore_atom_map_numbers),
        } == expected
    if not matches:
        raise ValueError(
            "PreparedSmilesGraph writer flags do not match the requested public runtime options"
        )
