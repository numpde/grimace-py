from __future__ import annotations

from dataclasses import dataclass
import unittest

import grimace
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


@dataclass(frozen=True, slots=True)
class ZstdFrameHeader:
    dictionary_id: int | None
    dictionary_id_offset: int | None
    dictionary_id_size: int
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

    return ZstdFrameHeader(
        dictionary_id=dictionary_id,
        dictionary_id_offset=dictionary_id_offset,
        dictionary_id_size=dictionary_id_size,
        has_content_size=content_size_present,
        has_checksum=checksum_flag,
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

    def test_default_zstd_write_embeds_builtin_dictionary_selector(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)

        zstd_payload = prepared.to_bytes(compression="zstd")
        header = _read_zstd_frame_header(zstd_payload)

        self.assertIsNotNone(header.dictionary_id)
        self.assertNotEqual(0, header.dictionary_id)
        self.assertTrue(header.has_content_size)
        self.assertTrue(header.has_checksum)

    def test_from_bytes_selects_builtin_dictionary_from_frame_id(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")

        restored = grimace.PreparedMol.from_bytes(zstd_payload)

        self.assertEqual(prepared.to_bytes(), restored.to_bytes())

    def test_zero_dictionary_id_is_rejected(self) -> None:
        prepared = self._prepare("CCO", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(_with_zstd_dictionary_id(zstd_payload, 0))

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

    def test_from_bytes_compression_option_surface_is_strict(self) -> None:
        raw_payload = self._prepare("C", isomericSmiles=False).to_bytes()

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(raw_payload, dictionary=object())

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(bytearray(raw_payload))

    def test_from_bytes_rejects_invalid_dictionary_for_zstd_payload(self) -> None:
        prepared = self._prepare("C", isomericSmiles=False)
        zstd_payload = prepared.to_bytes(compression="zstd")

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(zstd_payload, dictionary=object())

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
