"""Prepared molecule container and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from numbers import Integral
from typing import Any

from rdkit import Chem


_PREPARED_MOL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _PreparedMolWriterFlags:
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedMolFragment:
    atom_indices: tuple[int, ...]
    prepared_graph: object

    def __post_init__(self) -> None:
        atom_indices = _coerce_atom_indices(self.atom_indices)
        object.__setattr__(self, "atom_indices", atom_indices)
        _validate_fragment_graph_shape(atom_indices, self.prepared_graph)


class PreparedMol:
    """Opaque prepared molecule returned by PrepareMol."""

    __slots__ = ("_schema_version", "_writer_flags", "_fragments")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "PreparedMol cannot be constructed directly; use PrepareMol or "
            "PreparedMol.from_bytes"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PreparedMol is immutable")

    def to_bytes(self) -> bytes:
        return json.dumps(
            _prepared_mol_to_payload(self),
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "PreparedMol":
        if not isinstance(data, bytes):
            raise TypeError("PreparedMol.from_bytes requires bytes")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed PreparedMol payload") from exc
        return _prepared_mol_from_payload(payload)


def _prepared_mol_fragments(prepared: PreparedMol) -> tuple[_PreparedMolFragment, ...]:
    return prepared._fragments


def _prepared_mol_writer_flag_values(
    prepared: PreparedMol,
) -> tuple[bool, bool, bool, bool, bool]:
    return (
        prepared._writer_flags.isomeric_smiles,
        prepared._writer_flags.kekule_smiles,
        prepared._writer_flags.all_bonds_explicit,
        prepared._writer_flags.all_hs_explicit,
        prepared._writer_flags.ignore_atom_map_numbers,
    )


def _is_rdkit_mol(value: object) -> bool:
    return isinstance(value, Chem.Mol)


def _rdkit_mol_requires_stereo_surface(mol: Chem.Mol) -> bool:
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    )


def _rdkit_mol_atom_count(mol: Chem.Mol) -> int:
    return mol.GetNumAtoms()


def _rdkit_mol_fragment_count(mol: Chem.Mol) -> int:
    return len(Chem.GetMolFrags(mol))


def _rdkit_mol_fragments(mol: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(atom_idx) for atom_idx in fragment)
        for fragment in Chem.GetMolFrags(mol)
    )


def _rdkit_mol_fragment_mols(mol: Chem.Mol) -> tuple[Chem.Mol, ...]:
    return tuple(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def _make_prepared_mol(
    *,
    schema_version: int,
    writer_flags: _PreparedMolWriterFlags,
    fragments: tuple[_PreparedMolFragment, ...],
) -> PreparedMol:
    if schema_version != _PREPARED_MOL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported PreparedMol schema version: {schema_version}")
    if not isinstance(writer_flags, _PreparedMolWriterFlags):
        raise ValueError("PreparedMol writer_flags must be a prepared writer flag record")

    fragments = tuple(fragments)
    for fragment in fragments:
        if not isinstance(fragment, _PreparedMolFragment):
            raise ValueError("PreparedMol fragments must be prepared fragment records")
        _validate_fragment_writer_flags(fragment, writer_flags)
    _validate_fragment_atom_indices_are_unique(fragments)

    prepared = object.__new__(PreparedMol)
    object.__setattr__(prepared, "_schema_version", schema_version)
    object.__setattr__(prepared, "_writer_flags", writer_flags)
    object.__setattr__(prepared, "_fragments", fragments)
    return prepared


def _prepared_mol_to_payload(prepared: PreparedMol) -> dict[str, object]:
    return {
        "schema_version": prepared._schema_version,
        "writer_flags": {
            "isomeric_smiles": prepared._writer_flags.isomeric_smiles,
            "kekule_smiles": prepared._writer_flags.kekule_smiles,
            "all_bonds_explicit": prepared._writer_flags.all_bonds_explicit,
            "all_hs_explicit": prepared._writer_flags.all_hs_explicit,
            "ignore_atom_map_numbers": prepared._writer_flags.ignore_atom_map_numbers,
        },
        "fragments": [
            {
                "atom_indices": list(fragment.atom_indices),
                "prepared_graph": _prepared_graph_to_dict(fragment.prepared_graph),
            }
            for fragment in prepared._fragments
        ],
    }


def _prepared_mol_from_payload(payload: object) -> PreparedMol:
    if not isinstance(payload, dict):
        raise ValueError("PreparedMol payload must be an object")
    try:
        schema_version = payload["schema_version"]
        writer_flags_data = payload["writer_flags"]
        fragments_data = payload["fragments"]
    except KeyError as exc:
        raise ValueError("PreparedMol payload is missing required fields") from exc

    if not isinstance(fragments_data, list):
        raise ValueError("PreparedMol fragments must be an array")

    return _make_prepared_mol(
        schema_version=_require_int(schema_version, "schema_version"),
        writer_flags=_writer_flags_from_payload(writer_flags_data),
        fragments=tuple(_fragment_from_payload(fragment) for fragment in fragments_data),
    )


def _writer_flags_from_payload(payload: object) -> _PreparedMolWriterFlags:
    if not isinstance(payload, dict):
        raise ValueError("PreparedMol writer_flags must be an object")
    return _PreparedMolWriterFlags(
        isomeric_smiles=_require_bool(payload, "isomeric_smiles"),
        kekule_smiles=_require_bool(payload, "kekule_smiles"),
        all_bonds_explicit=_require_bool(payload, "all_bonds_explicit"),
        all_hs_explicit=_require_bool(payload, "all_hs_explicit"),
        ignore_atom_map_numbers=_require_bool(payload, "ignore_atom_map_numbers"),
    )


def _fragment_from_payload(payload: object) -> _PreparedMolFragment:
    if not isinstance(payload, dict):
        raise ValueError("PreparedMol fragment must be an object")
    try:
        atom_indices = payload["atom_indices"]
        prepared_graph = payload["prepared_graph"]
    except KeyError as exc:
        raise ValueError("PreparedMol fragment is missing required fields") from exc
    return _PreparedMolFragment(
        atom_indices=_coerce_atom_indices(atom_indices),
        prepared_graph=_prepared_graph_from_payload(prepared_graph),
    )


def PrepareMol(
    mol: Chem.Mol,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> PreparedMol:
    if not isinstance(mol, Chem.Mol):
        raise TypeError("PrepareMol requires an RDKit Chem.Mol")

    writer_flags = _PreparedMolWriterFlags(
        isomeric_smiles=_coerce_bool_flag("isomericSmiles", isomericSmiles),
        kekule_smiles=_coerce_bool_flag("kekuleSmiles", kekuleSmiles),
        all_bonds_explicit=_coerce_bool_flag("allBondsExplicit", allBondsExplicit),
        all_hs_explicit=_coerce_bool_flag("allHsExplicit", allHsExplicit),
        ignore_atom_map_numbers=_coerce_bool_flag(
            "ignoreAtomMapNumbers",
            ignoreAtomMapNumbers,
        ),
    )
    runtime = importlib.import_module("grimace._runtime")
    runtime_flags = runtime.MolToSmilesFlags(
        isomeric_smiles=writer_flags.isomeric_smiles,
        kekule_smiles=writer_flags.kekule_smiles,
        all_bonds_explicit=writer_flags.all_bonds_explicit,
        all_hs_explicit=writer_flags.all_hs_explicit,
        ignore_atom_map_numbers=writer_flags.ignore_atom_map_numbers,
    )

    if mol.GetNumAtoms() == 0:
        fragments = (
            _PreparedMolFragment(
                atom_indices=(),
                prepared_graph=runtime.prepare_smiles_graph(mol, flags=runtime_flags),
            ),
        )
    else:
        atom_indices_by_fragment = _rdkit_mol_fragments(mol)
        fragment_mols = _rdkit_mol_fragment_mols(mol)
        fragments = tuple(
            _PreparedMolFragment(
                atom_indices=atom_indices,
                prepared_graph=runtime.prepare_smiles_graph(fragment_mol, flags=runtime_flags),
            )
            for atom_indices, fragment_mol in zip(
                atom_indices_by_fragment,
                fragment_mols,
                strict=True,
            )
        )
    return _make_prepared_mol(
        schema_version=_PREPARED_MOL_SCHEMA_VERSION,
        writer_flags=writer_flags,
        fragments=fragments,
    )


def _coerce_bool_flag(name: str, value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Integral):
        raise TypeError(
            f"PrepareMol requires {name} to follow RDKit's Python binding "
            "and be a bool, int, or None"
        )
    return bool(value)


def _require_bool(data: dict[object, object], key: str) -> bool:
    try:
        value = data[key]
    except KeyError as exc:
        raise ValueError(f"PreparedMol writer_flags is missing {key!r}") from exc
    if not isinstance(value, bool):
        raise ValueError(f"PreparedMol writer flag {key!r} must be a bool")
    return value


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"PreparedMol {field_name} must be an integer")
    return value


def _coerce_atom_indices(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("PreparedMol fragment atom_indices must be an array")
    atom_indices: list[int] = []
    for atom_idx in value:
        if not isinstance(atom_idx, int) or isinstance(atom_idx, bool):
            raise ValueError("PreparedMol fragment atom indices must be integers")
        if atom_idx < 0:
            raise ValueError("PreparedMol fragment atom indices must be non-negative")
        atom_indices.append(atom_idx)
    if len(set(atom_indices)) != len(atom_indices):
        raise ValueError("PreparedMol fragment atom indices must be unique")
    return tuple(atom_indices)


def _validate_fragment_atom_indices_are_unique(
    fragments: tuple[_PreparedMolFragment, ...],
) -> None:
    seen: set[int] = set()
    for fragment in fragments:
        for atom_idx in fragment.atom_indices:
            if atom_idx in seen:
                raise ValueError("PreparedMol fragment atom indices overlap")
            seen.add(atom_idx)


def _prepared_graph_to_dict(prepared_graph: object) -> dict[str, Any]:
    to_dict = getattr(prepared_graph, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("PreparedMol fragment prepared_graph must support to_dict()")
    data = to_dict()
    if not isinstance(data, dict):
        raise ValueError("PreparedMol fragment prepared_graph.to_dict() must return a dict")
    return data


def _prepared_graph_from_payload(data: object) -> object:
    if not isinstance(data, dict):
        raise ValueError("PreparedMol fragment prepared_graph must be an object")
    _validate_prepared_graph_dict_writer_flag_fields(data)
    core = importlib.import_module("grimace._core")
    try:
        return core.PreparedSmilesGraph(data)
    except Exception as exc:
        raise ValueError("Malformed PreparedMol prepared_graph") from exc


def _validate_fragment_graph_shape(
    atom_indices: tuple[int, ...],
    prepared_graph: object,
) -> None:
    graph_data = _prepared_graph_to_dict(prepared_graph)
    try:
        atom_count = graph_data["atom_count"]
    except KeyError as exc:
        raise ValueError("PreparedMol prepared_graph is missing atom_count") from exc
    if atom_count != len(atom_indices):
        raise ValueError("PreparedMol fragment atom_indices length does not match graph atom_count")


def _validate_fragment_writer_flags(
    fragment: _PreparedMolFragment,
    writer_flags: _PreparedMolWriterFlags,
) -> None:
    graph_data = _prepared_graph_to_dict(fragment.prepared_graph)
    _validate_prepared_graph_dict_writer_flag_fields(graph_data)
    actual = (
        graph_data["writer_do_isomeric_smiles"],
        graph_data["writer_kekule_smiles"],
        graph_data["writer_all_bonds_explicit"],
        graph_data["writer_all_hs_explicit"],
        graph_data["writer_ignore_atom_map_numbers"],
    )
    expected = (
        writer_flags.isomeric_smiles,
        writer_flags.kekule_smiles,
        writer_flags.all_bonds_explicit,
        writer_flags.all_hs_explicit,
        writer_flags.ignore_atom_map_numbers,
    )
    if actual != expected:
        raise ValueError("PreparedMol writer flags do not match fragment prepared graph")


def _validate_prepared_graph_dict_writer_flag_fields(data: dict[str, object]) -> None:
    for key in (
        "writer_do_isomeric_smiles",
        "writer_kekule_smiles",
        "writer_all_bonds_explicit",
        "writer_all_hs_explicit",
        "writer_ignore_atom_map_numbers",
    ):
        try:
            value = data[key]
        except KeyError as exc:
            raise ValueError(f"PreparedMol prepared_graph is missing {key!r}") from exc
        if not isinstance(value, bool):
            raise ValueError(f"PreparedMol prepared_graph field {key!r} must be a bool")
