from __future__ import annotations

from dataclasses import dataclass
import base64
import importlib.resources as resources
import json
import subprocess
import sys
import textwrap
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import grimace
import grimace._prepared_mol as prepared_mol_module
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    choice_texts,
    make_decoder,
    make_determinized_decoder,
    prepared_writer_kwargs,
    public_enum_support,
    public_token_inventory,
    public_token_inventory_superset,
    supported_public_kwargs,
)


RAW_PREPARED_MOL_MAGIC = b"GPM\0"
ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"
ZSTD_SKIPPABLE_FRAME_MAGIC = b"\x50\x2a\x4d\x18"
ZSTD_DICT_ID_SIZE_BY_FLAG = (0, 1, 2, 4)
SHIPPED_DICTIONARY_LEVELS = (3, 10)


@dataclass(frozen=True, slots=True)
class ZstdFrameHeader:
    dictionary_id: int | None
    dictionary_id_offset: int | None
    dictionary_id_size: int
    content_size: int | None
    has_content_size: bool
    has_checksum: bool


def _read_zstd_frame_header(payload: bytes) -> ZstdFrameHeader:
    """Read only the stable zstd frame fields Grimace promises to set."""

    if not payload.startswith(ZSTD_FRAME_MAGIC):
        raise AssertionError("payload is not a zstd frame")
    if len(payload) < len(ZSTD_FRAME_MAGIC) + 1:
        raise AssertionError("payload is too short for a zstd frame header")

    descriptor = payload[len(ZSTD_FRAME_MAGIC)]
    dictionary_id_flag = descriptor & 0b11
    checksum_flag = bool(descriptor & 0b100)
    single_segment_flag = bool(descriptor & 0b0010_0000)
    content_size_flag = descriptor >> 6
    content_size_present = bool(content_size_flag or single_segment_flag)

    offset = len(ZSTD_FRAME_MAGIC) + 1
    if not single_segment_flag:
        # Window Descriptor is one byte in the zstd frame format.
        offset += 1

    dictionary_id_size = ZSTD_DICT_ID_SIZE_BY_FLAG[dictionary_id_flag]
    dictionary_id_offset: int | None = None
    dictionary_id: int | None = None
    if dictionary_id_size:
        dictionary_id_offset = offset
        if len(payload) < offset + dictionary_id_size:
            raise AssertionError("payload is too short for zstd dictionary ID")
        dictionary_id = int.from_bytes(
            payload[offset : offset + dictionary_id_size],
            "little",
        )
        offset += dictionary_id_size

    content_size: int | None = None
    if content_size_present:
        content_size_size = (0, 2, 4, 8)[content_size_flag]
        if single_segment_flag and content_size_flag == 0:
            content_size_size = 1
        if len(payload) < offset + content_size_size:
            raise AssertionError("payload is too short for zstd content size")
        content_size = int.from_bytes(
            payload[offset : offset + content_size_size],
            "little",
        )
        if content_size_size == 2:
            content_size += 256

    return ZstdFrameHeader(
        dictionary_id=dictionary_id,
        dictionary_id_offset=dictionary_id_offset,
        dictionary_id_size=dictionary_id_size,
        content_size=content_size,
        has_content_size=content_size_present,
        has_checksum=checksum_flag,
    )


def _zstd_frame_header_with_content_size(
    *,
    dictionary_id: int,
    content_size: int,
) -> bytes:
    if not 0 <= dictionary_id < (1 << 32):
        raise AssertionError("dictionary ID does not fit a 4-byte zstd header")
    if not 0 <= content_size < (1 << 32):
        raise AssertionError("content size does not fit a 4-byte zstd header")

    descriptor = 0b1000_0000 | 0b0010_0000 | 0b0000_0100 | 0b0000_0011
    return b"".join(
        (
            ZSTD_FRAME_MAGIC,
            bytes((descriptor,)),
            dictionary_id.to_bytes(4, "little"),
            content_size.to_bytes(4, "little"),
        ),
    )


def _with_zstd_dictionary_id(payload: bytes, dictionary_id: int) -> bytes:
    header = _read_zstd_frame_header(payload)
    if header.dictionary_id_offset is None:
        raise AssertionError("zstd payload does not carry a dictionary ID")
    if not 0 <= dictionary_id < (1 << (8 * header.dictionary_id_size)):
        raise AssertionError("replacement dictionary ID does not fit header")

    mutated = bytearray(payload)
    offset = header.dictionary_id_offset
    mutated[offset : offset + header.dictionary_id_size] = dictionary_id.to_bytes(
        header.dictionary_id_size,
        "little",
    )
    return bytes(mutated)


def _dictionary_manifests() -> tuple[dict[str, object], ...]:
    root = resources.files("grimace").joinpath("data", "prepared_mol_zstd")
    manifests: list[dict[str, object]] = []
    for artifact in root.iterdir():
        manifest_path = artifact.joinpath("default_v1.json")
        if manifest_path.is_file():
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not manifests:
        raise AssertionError("No shipped PreparedMol zstd dictionaries found")
    return tuple(manifests)


def _dictionary_id_for_training_level(level: int) -> int:
    manifest = _dictionary_manifest_for_training_level(level)
    dictionary_id = manifest["zstd_dictionary_id"]
    if not isinstance(dictionary_id, int):
        raise AssertionError("Dictionary manifest id is not an integer")
    return dictionary_id


def _dictionary_manifest_for_training_level(level: int) -> dict[str, object]:
    matches = tuple(
        manifest
        for manifest in _dictionary_manifests()
        if manifest["training_identity"]["training_parameters"]["level"] == level
    )
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one shipped dictionary for training level {level}, "
            f"got {len(matches)}"
        )
    return matches[0]


def _shipped_dictionary_ids() -> set[int]:
    dictionary_ids = {
        manifest["zstd_dictionary_id"] for manifest in _dictionary_manifests()
    }
    if not all(isinstance(dictionary_id, int) for dictionary_id in dictionary_ids):
        raise AssertionError("Dictionary manifest id is not an integer")
    return dictionary_ids


def _compress_with_shipped_dictionary(
    raw_payload: bytes,
    *,
    dictionary_level: int,
    write_checksum: bool = True,
    write_content_size: bool = True,
) -> bytes:
    import zstandard as zstd

    manifest = _dictionary_manifest_for_training_level(dictionary_level)
    artifact_dir = (
        resources.files("grimace")
        .joinpath("data", "prepared_mol_zstd")
        .joinpath(manifest["artifact_dir"])
    )
    dictionary = zstd.ZstdCompressionDict(
        artifact_dir.joinpath(manifest["files"]["dictionary"]).read_bytes(),
    )
    if dictionary.dict_id() != manifest["zstd_dictionary_id"]:
        raise AssertionError("Dictionary manifest id does not match dictionary bytes")
    return zstd.ZstdCompressor(
        level=3,
        dict_data=dictionary,
        write_checksum=write_checksum,
        write_content_size=write_content_size,
    ).compress(raw_payload)


def _compress_without_dictionary(raw_payload: bytes) -> bytes:
    import zstandard as zstd

    return zstd.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw_payload)


class PreparedMolZstdContractTests(unittest.TestCase):
    def _prepare(self, smiles: str, **kwargs: object) -> grimace.PreparedMol:
        return grimace.PrepareMol(parse_smiles(smiles), **kwargs)

    def _assert_public_runtime_matches(
        self,
        mol: object,
        restored: grimace.PreparedMol,
        *,
        kwargs: dict[str, object],
    ) -> None:
        self.assertEqual(
            public_enum_support(mol, **kwargs),
            public_enum_support(restored, **kwargs),
        )
        self.assertEqual(
            public_token_inventory(mol, **kwargs),
            public_token_inventory(restored, **kwargs),
        )
        self.assertEqual(
            public_token_inventory_superset(mol, **kwargs),
            public_token_inventory_superset(restored, **kwargs),
        )
        self.assertEqual(
            choice_texts(make_decoder(mol, **kwargs)),
            choice_texts(make_decoder(restored, **kwargs)),
        )
        self.assertEqual(
            choice_texts(make_determinized_decoder(mol, **kwargs)),
            choice_texts(make_determinized_decoder(restored, **kwargs)),
        )

        candidate = min(public_enum_support(mol, **kwargs))
        self.assertEqual(
            grimace.MolToSmilesDeviation(mol, candidate, **kwargs),
            grimace.MolToSmilesDeviation(restored, candidate, **kwargs),
        )

    def test_raw_bytes_remain_the_canonical_default(self) -> None:
        prepared = self._prepare("CCO.N", isomericSmiles=False)

        raw_payload = prepared.to_bytes()

        self.assertTrue(raw_payload.startswith(RAW_PREPARED_MOL_MAGIC))
        self.assertFalse(raw_payload.startswith(ZSTD_FRAME_MAGIC))
        self.assertEqual(raw_payload, prepared.to_bytes(compression=None))
        self.assertEqual(
            raw_payload,
            grimace.PreparedMol.from_bytes(raw_payload).to_bytes(),
        )

    def test_zstd_payloads_round_trip_through_from_bytes(self) -> None:
        cases = (
            (
                "connected_nonstereo",
                "CCO",
                supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
            ),
            (
                "all_roots",
                "CCO",
                supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1),
            ),
            (
                "disconnected_nonstereo",
                "CCO.N.Cl",
                supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3),
            ),
            (
                "stereo",
                "F[C@H](Cl)Br",
                supported_public_kwargs(isomericSmiles=True, rootedAtAtom=1),
            ),
        )

        for name, smiles, kwargs in cases:
            with self.subTest(name=name):
                mol = parse_smiles(smiles)
                prepared = grimace.PrepareMol(mol, **prepared_writer_kwargs(kwargs))
                raw_payload = prepared.to_bytes()

                zstd_payload = prepared.to_bytes(compression="zstd")
                restored = grimace.PreparedMol.from_bytes(zstd_payload)

                self.assertIsInstance(zstd_payload, bytes)
                self.assertTrue(zstd_payload.startswith(ZSTD_FRAME_MAGIC))
                self.assertFalse(zstd_payload.startswith(RAW_PREPARED_MOL_MAGIC))
                self.assertEqual(raw_payload, restored.to_bytes())
                self._assert_public_runtime_matches(
                    mol,
                    restored,
                    kwargs=kwargs,
                )

    def test_to_bytes_rejects_raw_payloads_above_size_limit(self) -> None:
        class FakePreparedMolInner:
            def to_bytes(self) -> bytes:
                return RAW_PREPARED_MOL_MAGIC + b"\0" * (
                    prepared_mol_module._MAX_PREPARED_MOL_RAW_BYTES
                    - len(RAW_PREPARED_MOL_MAGIC)
                    + 1
                )

        prepared = prepared_mol_module._make_prepared_mol(FakePreparedMolInner())

        with self.assertRaisesRegex(ValueError, "exceeds"):
            prepared.to_bytes()

    def test_from_bytes_rejects_raw_payloads_above_size_limit(self) -> None:
        raw_payload = RAW_PREPARED_MOL_MAGIC + b"\0" * (
            prepared_mol_module._MAX_PREPARED_MOL_RAW_BYTES
            - len(RAW_PREPARED_MOL_MAGIC)
            + 1
        )

        with self.assertRaisesRegex(ValueError, "exceeds"):
            grimace.PreparedMol.from_bytes(raw_payload)

    def test_from_bytes_rejects_declared_zstd_size_above_limit_before_lookup(
        self,
    ) -> None:
        dictionary_id = _dictionary_id_for_training_level(3)
        payload = _zstd_frame_header_with_content_size(
            dictionary_id=dictionary_id,
            content_size=prepared_mol_module._MAX_PREPARED_MOL_RAW_BYTES + 1,
        )

        with mock.patch.object(
            prepared_mol_module,
            "_zstd_dictionary_for_id",
            side_effect=AssertionError("dictionary lookup should not run"),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                grimace.PreparedMol.from_bytes(payload)

    def test_from_bytes_rejects_decompressed_zstd_payloads_above_size_limit(
        self,
    ) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")
        oversized_raw_payload = RAW_PREPARED_MOL_MAGIC + b"\0" * (
            prepared_mol_module._MAX_PREPARED_MOL_RAW_BYTES
            - len(RAW_PREPARED_MOL_MAGIC)
            + 1
        )

        class FakeZstdError(Exception):
            pass

        class FakeZstdDecompressor:
            def __init__(self, *, dict_data: object) -> None:
                pass

            def decompress(
                self,
                data: bytes,
                *,
                allow_extra_data: bool,
            ) -> bytes:
                if data != zstd_payload or allow_extra_data:
                    raise AssertionError("unexpected zstd decompression call")
                return oversized_raw_payload

        class FakeZstd:
            CONTENTSIZE_UNKNOWN = -1
            CONTENTSIZE_ERROR = -2
            ZstdError = FakeZstdError

            @staticmethod
            def frame_content_size(data: bytes) -> int:
                if data != zstd_payload:
                    raise AssertionError("unexpected zstd frame inspection")
                return len(prepared.to_bytes())

            ZstdDecompressor = FakeZstdDecompressor

        with (
            mock.patch.object(
                prepared_mol_module,
                "_zstd_dictionary_for_id",
                return_value=object(),
            ),
            mock.patch.object(prepared_mol_module, "_load_zstd", return_value=FakeZstd),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                grimace.PreparedMol.from_bytes(zstd_payload)

    def test_default_zstd_write_embeds_builtin_dictionary_selector(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        zstd_payload = prepared.to_bytes(compression="zstd")
        header = _read_zstd_frame_header(zstd_payload)

        self.assertEqual(_dictionary_id_for_training_level(3), header.dictionary_id)
        self.assertEqual(len(prepared.to_bytes()), header.content_size)
        self.assertTrue(header.has_content_size)
        self.assertTrue(header.has_checksum)

    def test_default_zstd_write_matches_explicit_level3_dictionary_level3(
        self,
    ) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        self.assertEqual(
            prepared.to_bytes(compression="zstd"),
            prepared.to_bytes(
                compression="zstd",
                dictionary_level=3,
                level=3,
            ),
        )

    def test_dictionary_level_selects_matching_builtin_dictionary(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        for dictionary_level in SHIPPED_DICTIONARY_LEVELS:
            with self.subTest(dictionary_level=dictionary_level):
                zstd_payload = prepared.to_bytes(
                    compression="zstd",
                    dictionary_level=dictionary_level,
                )
                header = _read_zstd_frame_header(zstd_payload)
                restored = grimace.PreparedMol.from_bytes(zstd_payload)

                self.assertEqual(
                    _dictionary_id_for_training_level(dictionary_level),
                    header.dictionary_id,
                )
                self.assertEqual(prepared.to_bytes(), restored.to_bytes())

    def test_runtime_level_does_not_change_dictionary_selector(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        expected_dictionary_id = _dictionary_id_for_training_level(3)

        for level in (1, 3, 10, 19):
            with self.subTest(level=level):
                zstd_payload = prepared.to_bytes(
                    compression="zstd",
                    dictionary_level=3,
                    level=level,
                )
                self.assertEqual(
                    expected_dictionary_id,
                    _read_zstd_frame_header(zstd_payload).dictionary_id,
                )

    def test_from_bytes_selects_builtin_dictionary_from_frame_id(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        for dictionary_level in SHIPPED_DICTIONARY_LEVELS:
            with self.subTest(dictionary_level=dictionary_level):
                zstd_payload = prepared.to_bytes(
                    compression="zstd",
                    dictionary_level=dictionary_level,
                )

                restored = grimace.PreparedMol.from_bytes(zstd_payload)

                self.assertEqual(prepared.to_bytes(), restored.to_bytes())

    def test_compression_rejects_invalid_dictionary_manifests(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        cases = (
            ("{", "manifest is invalid"),
            ("[]", "manifest is invalid"),
            (
                json.dumps(
                    {
                        "artifact_dir": "other",
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "training_identity": {"training_parameters": {"level": 3}},
                    },
                ),
                "invalid artifact",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "artifact",
                        "files": {"dictionary": "../default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "training_identity": {"training_parameters": {"level": 3}},
                    },
                ),
                "invalid files",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "artifact",
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": True,
                        "training_identity": {"training_parameters": {"level": 3}},
                    },
                ),
                "invalid id",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "artifact",
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "training_identity": {"training_parameters": {"level": "3"}},
                    },
                ),
                "invalid level",
            ),
        )
        for manifest_payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact = Path(tmpdir) / "artifact"
                    artifact.mkdir()
                    (artifact / "default_v1.json").write_text(
                        manifest_payload,
                        encoding="utf-8",
                    )

                    prepared_mol_module._ZSTD_DICTIONARY_BY_ID.clear()
                    prepared_mol_module._ZSTD_DICTIONARY_ID_BY_TRAINING_LEVEL.clear()
                    with mock.patch.object(
                        prepared_mol_module,
                        "_zstd_dictionary_root",
                        return_value=Path(tmpdir),
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            prepared.to_bytes(compression="zstd")

    def test_from_bytes_reuses_cached_builtin_dictionary(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        raw_payload = prepared.to_bytes()
        zstd_payload = _compress_with_shipped_dictionary(
            raw_payload,
            dictionary_level=3,
        )
        program = """
import base64
import importlib.resources
import sys
from unittest import mock

import grimace

raw_payload = base64.b64decode(sys.argv[1])
zstd_payload = base64.b64decode(sys.argv[2])
real_files = importlib.resources.files

with mock.patch("importlib.resources.files", wraps=real_files) as files:
    assert raw_payload == grimace.PreparedMol.from_bytes(zstd_payload).to_bytes()
    first_lookup_count = files.call_count
    assert first_lookup_count > 0, first_lookup_count
    assert raw_payload == grimace.PreparedMol.from_bytes(zstd_payload).to_bytes()
    assert first_lookup_count == files.call_count, files.call_count
"""

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(program),
                base64.b64encode(raw_payload).decode("ascii"),
                base64.b64encode(zstd_payload).decode("ascii"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            completed.returncode,
            completed.stderr or completed.stdout,
        )

    def test_zero_dictionary_id_is_rejected(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(_with_zstd_dictionary_id(zstd_payload, 0))

    def test_unknown_dictionary_id_is_rejected(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")
        unknown_dictionary_id = 1
        self.assertNotIn(unknown_dictionary_id, _shipped_dictionary_ids())

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(
                _with_zstd_dictionary_id(zstd_payload, unknown_dictionary_id),
            )

    def test_compressed_payload_without_dictionary_id_is_rejected(self) -> None:
        raw_payload = self._prepare("CCO", isomericSmiles=False).to_bytes()
        zstd_payload = _compress_without_dictionary(raw_payload)
        header = _read_zstd_frame_header(zstd_payload)
        self.assertIsNone(header.dictionary_id)
        self.assertTrue(header.has_content_size)
        self.assertTrue(header.has_checksum)

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(zstd_payload)

    def test_compressed_payload_without_checksum_is_rejected(self) -> None:
        raw_payload = self._prepare("CCO", isomericSmiles=False).to_bytes()
        zstd_payload = _compress_with_shipped_dictionary(
            raw_payload,
            dictionary_level=3,
            write_checksum=False,
        )
        self.assertFalse(_read_zstd_frame_header(zstd_payload).has_checksum)

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(zstd_payload)

    def test_compressed_payload_without_content_size_is_rejected(self) -> None:
        raw_payload = self._prepare("CCO", isomericSmiles=False).to_bytes()
        zstd_payload = _compress_with_shipped_dictionary(
            raw_payload,
            dictionary_level=3,
            write_content_size=False,
        )
        self.assertFalse(_read_zstd_frame_header(zstd_payload).has_content_size)

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(zstd_payload)

    def test_zstd_requested_for_tiny_payloads_never_silently_falls_back_to_raw(
        self,
    ) -> None:
        prepared = self._prepare("C", isomericSmiles=False)
        raw_payload = prepared.to_bytes()

        zstd_payload = prepared.to_bytes(compression="zstd")
        restored = grimace.PreparedMol.from_bytes(zstd_payload)

        self.assertTrue(zstd_payload.startswith(ZSTD_FRAME_MAGIC))
        self.assertNotEqual(raw_payload, zstd_payload)
        self.assertEqual(raw_payload, restored.to_bytes())

    def test_to_bytes_compression_option_surface_is_strict(self) -> None:
        prepared = self._prepare("C", isomericSmiles=False)

        with self.assertRaises(TypeError):
            prepared.to_bytes("zstd")  # type: ignore[call-arg]

        with self.assertRaises(ValueError):
            prepared.to_bytes(compression="gzip")

        with self.assertRaises(TypeError):
            prepared.to_bytes(dictionary=object())

        with self.assertRaises(TypeError):
            prepared.to_bytes(compression="zstd", dictionary=object())

        with self.assertRaises(TypeError):
            prepared.to_bytes(compression="zstd", dictionary_level="3")

        with self.assertRaises(ValueError):
            prepared.to_bytes(compression="zstd", dictionary_level=4)

        with self.assertRaises(TypeError):
            prepared.to_bytes(compression="zstd", level="3")

        for level in (0, 23):
            with self.subTest(level=level):
                with self.assertRaises(ValueError):
                    prepared.to_bytes(compression="zstd", level=level)

    def test_from_bytes_compression_option_surface_is_strict(self) -> None:
        raw_payload = self._prepare("C", isomericSmiles=False).to_bytes()

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(raw_payload, dictionary=object())

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(raw_payload, reload_dictionary=object())

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(bytearray(raw_payload))

    def test_malformed_zstd_like_payloads_are_rejected(self) -> None:
        skippable_empty_frame = ZSTD_SKIPPABLE_FRAME_MAGIC + b"\0\0\0\0"

        for payload in (
            ZSTD_FRAME_MAGIC,
            ZSTD_FRAME_MAGIC + b"not enough frame",
            skippable_empty_frame,
            skippable_empty_frame + ZSTD_FRAME_MAGIC,
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    grimace.PreparedMol.from_bytes(payload)

    def test_compressed_payloads_are_consumed_exactly(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        zstd_payload = prepared.to_bytes(compression="zstd")

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(zstd_payload + b"\0")

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(zstd_payload + zstd_payload)

    def test_corrupt_compressed_payloads_are_rejected(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = bytearray(prepared.to_bytes(compression="zstd"))
        zstd_payload[-1] ^= 0x01

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(bytes(zstd_payload))


if __name__ == "__main__":
    unittest.main()
