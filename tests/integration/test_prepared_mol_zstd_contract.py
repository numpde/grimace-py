from __future__ import annotations

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
