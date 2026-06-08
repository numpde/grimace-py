from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rdkit import Chem, rdBase
import zstandard as zstd

import grimace
from scripts import prepared_mol_zstd_dictionary_generate as generator

TEST_CREATED_YYYYMMDD = "20000102"

generator.zstd = zstd


def _training_dictionary_bytes(dictionary_id: int) -> bytes:
    samples = [
        (b"GPM\0 sample %06d " % index) * 40 for index in range(256)
    ]
    return zstd.train_dictionary(4096, samples, dict_id=dictionary_id).as_bytes()


def _prepared_payload() -> bytes:
    mol = Chem.MolFromSmiles("CCO")
    if mol is None:
        raise AssertionError("Could not parse smoke molecule")
    return grimace.PrepareMol(mol, **generator.WRITER_OPTIONS).to_bytes()


def _artifact_identity(
    *,
    dictionary_bytes: bytes,
    dictionary_id: int,
) -> dict[str, object]:
    training_identity = {"test": "prepared-mol-zstd-postflight"}
    return generator.artifact_identity(
        training_identity=training_identity,
        training_identity_sha256=generator.sha256_hex(
            generator.canonical_json_bytes(training_identity),
        ),
        dictionary_bytes=dictionary_bytes,
        dictionary_id=dictionary_id,
    )


class PreparedMolZstdDictionaryFlightTests(unittest.TestCase):
    def test_generation_environment_matches_recipe(self) -> None:
        self.assertEqual(generator.EXPECTED_RDKIT_VERSION, rdBase.rdkitVersion)
        self.assertEqual(generator.EXPECTED_ZSTANDARD_VERSION, zstd.__version__)
        self.assertEqual(generator.EXPECTED_ZSTANDARD_BACKEND, zstd.backend)
        self.assertEqual(generator.EXPECTED_ZSTD_LIBRARY_VERSION, zstd.ZSTD_VERSION)

    def test_installed_prepared_mol_smoke(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        prepared = grimace.PrepareMol(mol, **generator.WRITER_OPTIONS)
        payload = prepared.to_bytes()
        restored = grimace.PreparedMol.from_bytes(payload)

        self.assertTrue(payload.startswith(generator.RAW_PREPARED_MOL_MAGIC))
        self.assertEqual(payload, restored.to_bytes())

    def test_artifact_postflight_accepts_usable_dictionary(self) -> None:
        dictionary_id = 123_456
        dictionary_bytes = _training_dictionary_bytes(dictionary_id)
        identity = _artifact_identity(
            dictionary_bytes=dictionary_bytes,
            dictionary_id=dictionary_id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = generator.write_artifact(
                output_root=Path(tmpdir),
                created_yyyymmdd=TEST_CREATED_YYYYMMDD,
                dictionary_bytes=dictionary_bytes,
                identity=identity,
                force=False,
                postflight_payload=_prepared_payload(),
            )

            generator.validate_artifact(
                artifact_dir,
                smoke_payload=_prepared_payload(),
            )

    def test_artifact_postflight_rejects_wrong_dictionary_id(self) -> None:
        dictionary_bytes = _training_dictionary_bytes(123_456)
        identity = _artifact_identity(
            dictionary_bytes=dictionary_bytes,
            dictionary_id=123_457,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "Dictionary ID mismatch"):
                generator.write_artifact(
                    output_root=output_root,
                    created_yyyymmdd=TEST_CREATED_YYYYMMDD,
                    dictionary_bytes=dictionary_bytes,
                    identity=identity,
                    force=False,
                    postflight_payload=_prepared_payload(),
                )

            self.assertEqual((), tuple(output_root.iterdir()))

    def test_artifact_postflight_rejects_dictionary_without_nonzero_id(self) -> None:
        dictionary_bytes = b"not a zstd dictionary"
        identity = _artifact_identity(
            dictionary_bytes=dictionary_bytes,
            dictionary_id=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "must be nonzero"):
                generator.write_artifact(
                    output_root=output_root,
                    created_yyyymmdd=TEST_CREATED_YYYYMMDD,
                    dictionary_bytes=dictionary_bytes,
                    identity=identity,
                    force=False,
                    postflight_payload=_prepared_payload(),
                )

            self.assertEqual((), tuple(output_root.iterdir()))

    def test_artifact_write_failure_removes_partial_output(self) -> None:
        dictionary_id = 123_456
        dictionary_bytes = _training_dictionary_bytes(dictionary_id)
        identity = _artifact_identity(
            dictionary_bytes=dictionary_bytes,
            dictionary_id=dictionary_id,
        )
        original_write_text = Path.write_text

        def fail_manifest_write(path: Path, *args: object, **kwargs: object) -> int:
            if path.name == f"{generator.ARTIFACT_STEM}.json":
                raise OSError("simulated manifest write failure")
            return original_write_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with mock.patch.object(Path, "write_text", fail_manifest_write):
                with self.assertRaisesRegex(OSError, "simulated manifest"):
                    generator.write_artifact(
                        output_root=output_root,
                        created_yyyymmdd=TEST_CREATED_YYYYMMDD,
                        dictionary_bytes=dictionary_bytes,
                        identity=identity,
                        force=False,
                        postflight_payload=_prepared_payload(),
                    )

            self.assertEqual((), tuple(output_root.iterdir()))

    def test_artifact_postflight_rejects_tampered_dictionary(self) -> None:
        dictionary_id = 123_456
        dictionary_bytes = _training_dictionary_bytes(dictionary_id)
        identity = _artifact_identity(
            dictionary_bytes=dictionary_bytes,
            dictionary_id=dictionary_id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = generator.write_artifact(
                output_root=Path(tmpdir),
                created_yyyymmdd=TEST_CREATED_YYYYMMDD,
                dictionary_bytes=dictionary_bytes,
                identity=identity,
                force=False,
                postflight_payload=_prepared_payload(),
            )
            dictionary_path = artifact_dir / f"{generator.ARTIFACT_STEM}.zstdict"
            tampered = bytearray(dictionary_path.read_bytes())
            tampered[-1] ^= 0x01
            dictionary_path.write_bytes(bytes(tampered))

            with self.assertRaisesRegex(RuntimeError, "Dictionary SHA-256 mismatch"):
                generator.validate_artifact(
                    artifact_dir,
                    smoke_payload=_prepared_payload(),
                )


if __name__ == "__main__":
    unittest.main()
