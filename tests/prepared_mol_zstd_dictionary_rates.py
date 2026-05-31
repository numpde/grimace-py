from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import unittest

from rdkit import Chem
import zstandard as zstd

import grimace
from scripts import generate_prepared_mol_zstd_dictionary as generator


SAMPLE_COUNT = 1024
EXPECTED_RAW_SIZE = 5_722_868
EXPECTED_PLAIN_ZSTD_SIZE = 685_686
EXPECTED_DICTIONARY_ZSTD_SIZE = 481_278
MAX_DICTIONARY_TO_RAW_RATIO = 0.10
MAX_DICTIONARY_TO_PLAIN_ZSTD_RATIO = 0.75


def _artifact_dir() -> Path:
    manifests = tuple(
        sorted(
            (generator.ROOT / generator.PACKAGE_DICTIONARY_ROOT).glob(
                f"*/{generator.ARTIFACT_STEM}.json",
            ),
        ),
    )
    if len(manifests) != 1:
        raise AssertionError(
            "Expected exactly one shipped default_v1 dictionary manifest, "
            f"got {len(manifests)}"
        )
    return manifests[0].parent


def _prepared_payloads(limit: int) -> list[bytes]:
    payloads: list[bytes] = []
    fixture_path = generator.ROOT / generator.FIXTURE_RELATIVE_PATH
    with gzip.open(fixture_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue
            payloads.append(
                grimace.PrepareMol(mol, **generator.WRITER_OPTIONS).to_bytes(),
            )
            if len(payloads) == limit:
                return payloads
    raise AssertionError(f"Fixture did not yield {limit} prepared payloads")


class PreparedMolZstdDictionaryRateTests(unittest.TestCase):
    def test_shipped_dictionary_improves_zstd_ratio_on_fixture_sample(self) -> None:
        artifact_dir = _artifact_dir()
        manifest = json.loads(
            (artifact_dir / f"{generator.ARTIFACT_STEM}.json").read_text(
                encoding="utf-8",
            ),
        )
        dictionary = zstd.ZstdCompressionDict(
            (artifact_dir / manifest["files"]["dictionary"]).read_bytes(),
        )
        self.assertEqual(manifest["zstd_dictionary_id"], dictionary.dict_id())

        payloads = _prepared_payloads(SAMPLE_COUNT)
        plain_compressor = zstd.ZstdCompressor(
            level=3,
            write_checksum=True,
            write_content_size=True,
        )
        dictionary_compressor = zstd.ZstdCompressor(
            level=3,
            dict_data=dictionary,
            write_checksum=True,
            write_content_size=True,
        )

        raw_size = sum(len(payload) for payload in payloads)
        plain_zstd_size = sum(
            len(plain_compressor.compress(payload)) for payload in payloads
        )
        dictionary_zstd_size = sum(
            len(dictionary_compressor.compress(payload)) for payload in payloads
        )
        self.assertEqual(EXPECTED_RAW_SIZE, raw_size)
        self.assertEqual(EXPECTED_PLAIN_ZSTD_SIZE, plain_zstd_size)
        self.assertEqual(EXPECTED_DICTIONARY_ZSTD_SIZE, dictionary_zstd_size)

        dictionary_to_raw = dictionary_zstd_size / raw_size
        dictionary_to_plain_zstd = dictionary_zstd_size / plain_zstd_size

        self.assertLessEqual(
            dictionary_to_raw,
            MAX_DICTIONARY_TO_RAW_RATIO,
            (
                f"dictionary/raw={dictionary_to_raw:.3f}, "
                f"raw={raw_size}, dictionary_zstd={dictionary_zstd_size}"
            ),
        )
        self.assertLessEqual(
            dictionary_to_plain_zstd,
            MAX_DICTIONARY_TO_PLAIN_ZSTD_RATIO,
            (
                f"dictionary/plain_zstd={dictionary_to_plain_zstd:.3f}, "
                f"plain_zstd={plain_zstd_size}, "
                f"dictionary_zstd={dictionary_zstd_size}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
