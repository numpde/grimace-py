from __future__ import annotations

import unittest

import grimace
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    prepared_writer_kwargs,
    public_enum_support,
    supported_public_kwargs,
)


RAW_PREPARED_MOL_MAGIC = b"GPM\0"
ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"


class PreparedMolZstdContractTests(unittest.TestCase):
    def _prepare(self, smiles: str, **kwargs: object) -> grimace.PreparedMol:
        return grimace.PrepareMol(parse_smiles(smiles), **kwargs)

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
                self.assertEqual(
                    public_enum_support(mol, **kwargs),
                    public_enum_support(restored, **kwargs),
                )

    def test_zstd_option_surface_is_strict(self) -> None:
        prepared = self._prepare("C", isomericSmiles=False)
        raw_payload = prepared.to_bytes()

        with self.assertRaises(TypeError):
            prepared.to_bytes("zstd")  # type: ignore[call-arg]

        with self.assertRaises(ValueError):
            prepared.to_bytes(compression="gzip")

        with self.assertRaises(TypeError):
            prepared.to_bytes(dictionary=object())

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(raw_payload, dictionary=object())

        with self.assertRaises(TypeError):
            grimace.PreparedMol.from_bytes(bytearray(raw_payload))

        with self.assertRaises(ValueError):
            grimace.PreparedMol.from_bytes(ZSTD_FRAME_MAGIC)


if __name__ == "__main__":
    unittest.main()
