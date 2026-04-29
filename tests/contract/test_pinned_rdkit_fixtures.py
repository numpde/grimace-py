from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.rdkit_serializer_regressions import (
    load_pinned_serializer_regression_cases,
)


RDKIT_VERSION = "2099.01.1"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _base_payload(*cases: dict[str, object], rdkit_version: str = RDKIT_VERSION) -> dict[str, object]:
    return {
        "rdkit_version": rdkit_version,
        "cases": list(cases),
    }


def _base_case(case_id: str, *, source: str = "contract test") -> dict[str, object]:
    return {
        "id": case_id,
        "source": source,
    }


def _serializer_case(case_id: str, **overrides: object) -> dict[str, object]:
    case = {
        **_base_case(case_id),
        "smiles": "CCO",
        "rooted_at_atom": -1,
        "isomeric_smiles": True,
        "expected": ["C(C)O", "CCO", "OCC"],
        "expected_inventory": ["C", "O"],
    }
    case.update(overrides)
    return case


class PinnedRdkitFixtureLoaderTest(unittest.TestCase):
    def test_single_file_fixture_loads_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_path = root / f"{RDKIT_VERSION}.json"
            _write_json(fixture_path, _base_payload(_base_case("case_a")))

            cases = load_pinned_rdkit_fixture_cases(
                fixture_root=root,
                rdkit_version=RDKIT_VERSION,
                fixture_label="contract",
            )

        self.assertEqual(1, len(cases))
        self.assertEqual("case_a", cases[0].case_id)
        self.assertEqual("contract test", cases[0].source)
        self.assertEqual(fixture_path, cases[0].fixture_path)

    def test_directory_fixture_loads_shards_in_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_dir = root / RDKIT_VERSION
            _write_json(fixture_dir / "20_later.json", _base_payload(_base_case("case_b")))
            _write_json(fixture_dir / "10_earlier.json", _base_payload(_base_case("case_a")))

            cases = load_pinned_rdkit_fixture_cases(
                fixture_root=root,
                rdkit_version=RDKIT_VERSION,
                fixture_label="contract",
            )

        self.assertEqual(["case_a", "case_b"], [case.case_id for case in cases])

    def test_directory_fixture_rejects_duplicate_ids_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_dir = root / RDKIT_VERSION
            _write_json(fixture_dir / "10_first.json", _base_payload(_base_case("dup")))
            _write_json(fixture_dir / "20_second.json", _base_payload(_base_case("dup")))

            with self.assertRaisesRegex(ValueError, "duplicates case id 'dup'"):
                load_pinned_rdkit_fixture_cases(
                    fixture_root=root,
                    rdkit_version=RDKIT_VERSION,
                    fixture_label="contract",
                )

    def test_directory_fixture_rejects_empty_shard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / RDKIT_VERSION).mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "contains no JSON shards"):
                load_pinned_rdkit_fixture_cases(
                    fixture_root=root,
                    rdkit_version=RDKIT_VERSION,
                    fixture_label="contract",
                )

    def test_fixture_rejects_rdkit_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(
                root / f"{RDKIT_VERSION}.json",
                _base_payload(_base_case("case_a"), rdkit_version="2098.09.1"),
            )

            with self.assertRaisesRegex(ValueError, "expected '2099.01.1'"):
                load_pinned_rdkit_fixture_cases(
                    fixture_root=root,
                    rdkit_version=RDKIT_VERSION,
                    fixture_label="contract",
                )

    def test_fixture_rejects_empty_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(
                root / f"{RDKIT_VERSION}.json",
                _base_payload(_base_case("case_a", source="")),
            )

            with self.assertRaisesRegex(ValueError, "nonempty string source"):
                load_pinned_rdkit_fixture_cases(
                    fixture_root=root,
                    rdkit_version=RDKIT_VERSION,
                    fixture_label="contract",
                )


class SerializerRegressionFixtureLoaderTest(unittest.TestCase):
    def test_serializer_fixture_accepts_smiles_or_molblock_cases(self) -> None:
        molblock_case = _serializer_case("molblock_case", molblock="mol\nM  END\n")
        del molblock_case["smiles"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_dir = root / RDKIT_VERSION
            _write_json(
                fixture_dir / "00_cases.json",
                _base_payload(_serializer_case("smiles_case"), molblock_case),
            )

            cases = load_pinned_serializer_regression_cases(
                RDKIT_VERSION,
                fixture_root=root,
            )

        self.assertEqual(["smiles_case", "molblock_case"], [case.case_id for case in cases])
        self.assertEqual("CCO", cases[0].smiles)
        self.assertIsNone(cases[0].molblock)
        self.assertIsNone(cases[1].smiles)
        self.assertEqual("mol\nM  END\n", cases[1].molblock)

    def test_serializer_fixture_rejects_missing_and_duplicate_molecule_sources(self) -> None:
        duplicate_source_case = _serializer_case(
            "both_sources",
            molblock="mol\nM  END\n",
        )
        missing_source_case = _serializer_case("no_source")
        del missing_source_case["smiles"]
        for case in (duplicate_source_case, missing_source_case):
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaisesRegex(
                        ValueError,
                        "exactly one of 'smiles' or 'molblock'",
                    ):
                        load_pinned_serializer_regression_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )

    def test_serializer_fixture_rejects_coerced_boolean_and_integer_fields(self) -> None:
        invalid_cases = (
            _serializer_case("string_bool", isomeric_smiles="false"),
            _serializer_case("bool_root", rooted_at_atom=False),
            _serializer_case("string_budget", rdkit_sample_draw_budget="10"),
        )
        for case in invalid_cases:
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaises(ValueError):
                        load_pinned_serializer_regression_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )


if __name__ == "__main__":
    unittest.main()
