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


@dataclass(frozen=True, slots=True)
class PreparedMol:
    schema_version: int
    writer_flags: _PreparedMolWriterFlags
    fragments: tuple[_PreparedMolFragment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _PREPARED_MOL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported PreparedMol schema version: {self.schema_version}")
        if not isinstance(self.writer_flags, _PreparedMolWriterFlags):
            raise ValueError("PreparedMol writer_flags must be a prepared writer flag record")

        fragments = tuple(self.fragments)
        for fragment in fragments:
            if not isinstance(fragment, _PreparedMolFragment):
                raise ValueError("PreparedMol fragments must be prepared fragment records")
            _validate_fragment_writer_flags(fragment, self.writer_flags)
        _validate_fragment_atom_indices_are_unique(fragments)
        object.__setattr__(self, "fragments", fragments)

    def to_bytes(self) -> bytes:
        return json.dumps(
            _prepared_mol_to_payload(self),
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "PreparedMol":
        if not isinstance(data, bytes):
            raise TypeError("PreparedMol.from_bytes requires bytes")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed PreparedMol payload") from exc
        return _prepared_mol_from_payload(payload)


def _prepared_mol_to_payload(prepared: PreparedMol) -> dict[str, object]:
    return {
        "schema_version": prepared.schema_version,
        "writer_flags": {
            "isomeric_smiles": prepared.writer_flags.isomeric_smiles,
            "kekule_smiles": prepared.writer_flags.kekule_smiles,
            "all_bonds_explicit": prepared.writer_flags.all_bonds_explicit,
            "all_hs_explicit": prepared.writer_flags.all_hs_explicit,
            "ignore_atom_map_numbers": prepared.writer_flags.ignore_atom_map_numbers,
        },
        "fragments": [
            {
                "atom_indices": list(fragment.atom_indices),
                "prepared_graph": _prepared_graph_to_dict(fragment.prepared_graph),
            }
            for fragment in prepared.fragments
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

    return PreparedMol(
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
        atom_indices_by_fragment = Chem.GetMolFrags(mol)
        fragment_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        fragments = tuple(
            _PreparedMolFragment(
                atom_indices=tuple(int(atom_idx) for atom_idx in atom_indices),
                prepared_graph=runtime.prepare_smiles_graph(fragment_mol, flags=runtime_flags),
            )
            for atom_indices, fragment_mol in zip(
                atom_indices_by_fragment,
                fragment_mols,
                strict=True,
            )
        )
    return PreparedMol(
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
