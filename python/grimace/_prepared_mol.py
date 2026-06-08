"""Prepared molecule wrapper and RDKit preparation boundary."""

from __future__ import annotations

from importlib import resources
import json
from typing import Any

from rdkit import Chem

import grimace._core as _core
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_public_options,
    public_option_values,
)

_RAW_PREPARED_MOL_MAGIC = b"GPM\0"
_ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DICT_ID_SIZE_BY_FLAG = (0, 1, 2, 4)
_ZSTD_MANIFEST_FILE = "default_v1.json"
_ZSTD_DICTIONARY_FILE = "default_v1.zstdict"
_DEFAULT_ZSTD_LEVEL = 3
_DEFAULT_ZSTD_DICTIONARY_LEVEL = 3
_ZSTD_DICTIONARY_BY_ID: dict[int, object] = {}
_ZSTD_DICTIONARY_ID_BY_TRAINING_LEVEL: dict[int, int] = {}


class PreparedMol:
    """Opaque prepared molecule returned by PrepareMol."""

    __slots__ = ("_inner",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "PreparedMol cannot be constructed directly; use PrepareMol or "
            "PreparedMol.from_bytes"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PreparedMol is immutable")

    def to_bytes(
        self,
        *,
        compression: str | None = None,
        dictionary_level: int = _DEFAULT_ZSTD_DICTIONARY_LEVEL,
        level: int = _DEFAULT_ZSTD_LEVEL,
    ) -> bytes:
        raw_payload = self._inner.to_bytes()
        if compression is None:
            return raw_payload
        if compression != "zstd":
            raise ValueError("PreparedMol.to_bytes compression must be None or 'zstd'")
        _require_int(dictionary_level, "PreparedMol.to_bytes dictionary_level")
        _require_int(level, "PreparedMol.to_bytes level")
        if not 1 <= level <= 22:
            raise ValueError("PreparedMol.to_bytes level must be in zstd range 1..22")

        compression_dictionary = _zstd_dictionary_for_training_level(
            dictionary_level,
        )
        zstd = _load_zstd()
        return zstd.ZstdCompressor(
            level=level,
            dict_data=compression_dictionary,
            write_checksum=True,
            write_content_size=True,
        ).compress(raw_payload)

    @staticmethod
    def from_bytes(
        data: bytes,
    ) -> "PreparedMol":
        if not isinstance(data, bytes):
            raise TypeError("PreparedMol.from_bytes requires bytes")
        if data.startswith(_RAW_PREPARED_MOL_MAGIC):
            return _make_prepared_mol(_core.PreparedMol.from_bytes(data))
        if data.startswith(_ZSTD_FRAME_MAGIC):
            raw_payload = _decompress_prepared_mol_zstd(data)
            return _make_prepared_mol(_core.PreparedMol.from_bytes(raw_payload))
        raise ValueError("Malformed PreparedMol payload")


def _load_zstd() -> Any:
    import zstandard as zstd

    return zstd


def _require_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")


def _zstd_dictionary_for_training_level(
    training_level: int,
) -> object:
    _require_int(training_level, "PreparedMol zstd dictionary level")
    cached_id = _ZSTD_DICTIONARY_ID_BY_TRAINING_LEVEL.get(training_level)
    if cached_id is not None:
        cached = _ZSTD_DICTIONARY_BY_ID.get(cached_id)
        if cached is not None:
            return cached

    manifest = _zstd_dictionary_manifest_for_training_level(training_level)
    return _zstd_dictionary_from_manifest(manifest)


def _zstd_dictionary_for_id(dictionary_id: int) -> object:
    if dictionary_id == 0:
        raise ValueError("PreparedMol zstd frame does not name a dictionary")
    cached = _ZSTD_DICTIONARY_BY_ID.get(dictionary_id)
    if cached is not None:
        return cached
    manifest = _zstd_dictionary_manifest_for_id(dictionary_id)
    return _zstd_dictionary_from_manifest(manifest)


def _zstd_dictionary_from_manifest(manifest: dict[str, Any]) -> object:
    dictionary_id = _zstd_dictionary_manifest_id(manifest)
    cached = _ZSTD_DICTIONARY_BY_ID.get(dictionary_id)
    if cached is not None:
        return cached

    zstd = _load_zstd()
    root = _zstd_dictionary_root()
    artifact_dir = _zstd_dictionary_manifest_artifact_dir(manifest)
    dictionary_bytes = (
        root
        .joinpath(artifact_dir)
        .joinpath(_ZSTD_DICTIONARY_FILE)
        .read_bytes()
    )
    compression_dictionary = zstd.ZstdCompressionDict(dictionary_bytes)
    if compression_dictionary.dict_id() != dictionary_id:
        raise ValueError("PreparedMol zstd dictionary id does not match manifest")
    _ZSTD_DICTIONARY_ID_BY_TRAINING_LEVEL[
        _zstd_dictionary_manifest_training_level(manifest)
    ] = dictionary_id
    _ZSTD_DICTIONARY_BY_ID[dictionary_id] = compression_dictionary
    return compression_dictionary


def _zstd_dictionary_manifest_for_training_level(level: int) -> dict[str, Any]:
    matches = [
        manifest
        for manifest in _zstd_dictionary_manifests()
        if _zstd_dictionary_manifest_training_level(manifest) == level
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique PreparedMol zstd dictionary for level {level}")
    return matches[0]


def _zstd_dictionary_manifest_for_id(dictionary_id: int) -> dict[str, Any]:
    matches = [
        manifest
        for manifest in _zstd_dictionary_manifests()
        if _zstd_dictionary_manifest_id(manifest) == dictionary_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"No PreparedMol zstd dictionary for frame id {dictionary_id}"
        )
    return matches[0]


def _zstd_dictionary_manifests() -> tuple[dict[str, Any], ...]:
    manifests: list[dict[str, Any]] = []
    for artifact in _zstd_dictionary_root().iterdir():
        manifest_path = artifact.joinpath(_ZSTD_MANIFEST_FILE)
        if manifest_path.is_file():
            manifests.append(
                _read_zstd_dictionary_manifest(
                    manifest_path,
                    artifact_name=artifact.name,
                )
            )
    return tuple(manifests)


def _read_zstd_dictionary_manifest(
    manifest_path: resources.abc.Traversable,
    *,
    artifact_name: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("PreparedMol zstd dictionary manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("PreparedMol zstd dictionary manifest is invalid")
    if _zstd_dictionary_manifest_artifact_dir(manifest) != artifact_name:
        raise ValueError("PreparedMol zstd dictionary manifest has invalid artifact")
    if _zstd_dictionary_manifest_file(manifest) != _ZSTD_DICTIONARY_FILE:
        raise ValueError("PreparedMol zstd dictionary manifest has invalid files")
    _zstd_dictionary_manifest_id(manifest)
    _zstd_dictionary_manifest_training_level(manifest)
    return manifest


def _zstd_dictionary_manifest_artifact_dir(manifest: dict[str, Any]) -> str:
    artifact_dir = manifest.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("PreparedMol zstd dictionary manifest has invalid artifact")
    return artifact_dir


def _zstd_dictionary_manifest_file(manifest: dict[str, Any]) -> str:
    files = manifest.get("files")
    dictionary_file = files.get("dictionary") if isinstance(files, dict) else None
    if not isinstance(dictionary_file, str) or not dictionary_file:
        raise ValueError("PreparedMol zstd dictionary manifest has invalid files")
    return dictionary_file


def _zstd_dictionary_manifest_id(manifest: dict[str, Any]) -> int:
    dictionary_id = manifest.get("zstd_dictionary_id")
    if not isinstance(dictionary_id, int) or isinstance(dictionary_id, bool):
        raise ValueError("PreparedMol zstd dictionary manifest has invalid id")
    return dictionary_id


def _zstd_dictionary_manifest_training_level(manifest: dict[str, Any]) -> int:
    training_identity = manifest.get("training_identity")
    training_parameters = (
        training_identity.get("training_parameters")
        if isinstance(training_identity, dict)
        else None
    )
    training_level = (
        training_parameters.get("level")
        if isinstance(training_parameters, dict)
        else None
    )
    if not isinstance(training_level, int) or isinstance(training_level, bool):
        raise ValueError("PreparedMol zstd dictionary manifest has invalid level")
    return training_level


def _zstd_dictionary_root() -> resources.abc.Traversable:
    return resources.files("grimace").joinpath("data", "prepared_mol_zstd")


def _zstd_frame_dictionary_id(data: bytes) -> int:
    if not data.startswith(_ZSTD_FRAME_MAGIC):
        raise ValueError("PreparedMol payload is not a zstd frame")
    if len(data) < len(_ZSTD_FRAME_MAGIC) + 1:
        raise ValueError("PreparedMol zstd payload is truncated")

    descriptor = data[len(_ZSTD_FRAME_MAGIC)]
    checksum_flag = bool(descriptor & 0b100)
    single_segment_flag = bool(descriptor & 0b0010_0000)
    content_size_flag = descriptor >> 6
    if not checksum_flag:
        raise ValueError("PreparedMol zstd frame does not include checksum")
    if not (content_size_flag or single_segment_flag):
        raise ValueError("PreparedMol zstd frame does not include content size")

    dictionary_id_size = _ZSTD_DICT_ID_SIZE_BY_FLAG[descriptor & 0b11]
    offset = len(_ZSTD_FRAME_MAGIC) + 1
    if not single_segment_flag:
        offset += 1
    if dictionary_id_size == 0:
        raise ValueError("PreparedMol zstd frame does not name a dictionary")
    if len(data) < offset + dictionary_id_size:
        raise ValueError("PreparedMol zstd payload is truncated")
    return int.from_bytes(data[offset : offset + dictionary_id_size], "little")


def _decompress_prepared_mol_zstd(data: bytes) -> bytes:
    dictionary_id = _zstd_frame_dictionary_id(data)
    dictionary = _zstd_dictionary_for_id(dictionary_id)
    zstd = _load_zstd()
    decompressor = zstd.ZstdDecompressor(dict_data=dictionary)
    try:
        raw_payload = decompressor.decompress(data, allow_extra_data=False)
    except Exception as exc:
        raise ValueError("Malformed PreparedMol zstd payload") from exc
    if not isinstance(raw_payload, bytes) or not raw_payload.startswith(
        _RAW_PREPARED_MOL_MAGIC
    ):
        raise ValueError("Malformed PreparedMol zstd payload")
    return raw_payload


def _make_prepared_mol(inner: object) -> PreparedMol:
    prepared = object.__new__(PreparedMol)
    object.__setattr__(prepared, "_inner", inner)
    return prepared


def _matches_writer_flags(
    prepared: PreparedMol,
    *,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
) -> bool:
    return prepared._inner.matches_writer_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )


def _fragment_count(prepared: PreparedMol) -> int:
    return prepared._inner.fragment_count()


def _atom_count(prepared: PreparedMol) -> int:
    return prepared._inner.atom_count()


def _rooted_fragments(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int | None,
) -> tuple[tuple[object, int | None], ...]:
    return tuple(prepared._inner.rooted_fragments(rooted_at_atom))


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


def _rdkit_mol_fragment_mols_and_atom_indices(
    mol: Chem.Mol,
) -> tuple[tuple[Chem.Mol, tuple[int, ...]], ...]:
    atom_indices_by_fragment: list[tuple[int, ...]] = []
    fragment_mols = Chem.GetMolFrags(
        mol,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=atom_indices_by_fragment,
    )
    return tuple(zip(fragment_mols, atom_indices_by_fragment, strict=True))


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

    option_values = locals()
    writer_options = coerce_public_options(
        MOL_TO_SMILES_PREPARED_OPTIONS,
        public_option_values(MOL_TO_SMILES_PREPARED_OPTIONS, option_values),
        context="PrepareMol",
    )

    import grimace._runtime_graphs as runtime_graphs
    import grimace._runtime_inputs as runtime_inputs
    runtime_flags = runtime_inputs.MolToSmilesFlags(**writer_options)

    if mol.GetNumAtoms() == 0:
        fragments = [
            (
                [],
                runtime_graphs.prepare_smiles_graph(mol, flags=runtime_flags),
            )
        ]
    else:
        fragments = [
            (
                list(atom_indices),
                runtime_graphs.prepare_smiles_graph(
                    fragment_mol,
                    flags=runtime_flags,
                ),
            )
            for fragment_mol, atom_indices in _rdkit_mol_fragment_mols_and_atom_indices(
                mol
            )
        ]

    return _make_prepared_mol(
        _core.PreparedMol._from_parts(
            **writer_options,
            fragments=fragments,
        )
    )
