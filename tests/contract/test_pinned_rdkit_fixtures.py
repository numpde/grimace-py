from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_EXACT_SMALL_SUPPORT,
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_ROOTED_RANDOM,
    PINNED_RDKIT_SERIALIZER_REGRESSIONS,
    PINNED_RDKIT_WRITER_MEMBERSHIP,
    PINNED_STEREO_CONSTRAINT_MODEL,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_root,
    pinned_rdkit_fixture_versions,
)
from tests.helpers.rdkit_disconnected_sampling import load_disconnected_root_zero_smiles
from tests.helpers.rdkit_exact_small_support import (
    load_pinned_exact_small_support_cases,
)
from tests.helpers.rdkit_known_quirks import load_pinned_rdkit_known_quirk_cases
from tests.helpers.rdkit_rooted_random import load_pinned_rooted_random_cases
from tests.helpers.rdkit_serializer_regressions import (
    load_pinned_serializer_regression_cases,
)
from tests.helpers.rdkit_stereo_regressions import (
    load_stereo_expected_member_regressions,
    load_steroid_ring_coupled_component_regression,
)
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.helpers.stereo_constraint_model import (
    load_pinned_stereo_constraint_model_cases,
)


RDKIT_VERSION = "2099.01.1"
PINNED_FIXTURE_LOADERS = {
    PINNED_RDKIT_EXACT_SMALL_SUPPORT: load_pinned_exact_small_support_cases,
    PINNED_RDKIT_ROOTED_RANDOM: load_pinned_rooted_random_cases,
    PINNED_RDKIT_SERIALIZER_REGRESSIONS: load_pinned_serializer_regression_cases,
    PINNED_RDKIT_WRITER_MEMBERSHIP: load_pinned_writer_membership_cases,
}


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


def _stereo_constraint_model_case(case_id: str, **overrides: object) -> dict[str, object]:
    case = {
        **_base_case(case_id),
        "smiles": "C/C=C/C",
        "expected_component_side_domain_sizes": [[1, 1]],
        "expected_semantic_assignment_count": 1,
        "expected_rdkit_local_writer_assignment_count": 1,
        "expected_rdkit_traversal_writer_assignment_count": 1,
        "expected_grimace_runtime_support_count": 1,
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

    def test_serializer_fixture_rejects_non_string_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(
                root / f"{RDKIT_VERSION}.json",
                _base_payload(_serializer_case("bad_expected", expected=[123])),
            )

            with self.assertRaisesRegex(ValueError, "expected as strings"):
                load_pinned_serializer_regression_cases(
                    RDKIT_VERSION,
                    fixture_root=root,
                )


class StereoConstraintModelFixtureLoaderTest(unittest.TestCase):
    def test_stereo_constraint_model_fixture_derives_component_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(
                root / f"{RDKIT_VERSION}.json",
                _base_payload(
                    _stereo_constraint_model_case(
                        "case_a",
                        expected_component_side_domain_sizes=[[1, 2], [2, 1]],
                        expected_semantic_assignment_count=4,
                    )
                ),
            )

            cases = load_pinned_stereo_constraint_model_cases(
                RDKIT_VERSION,
                fixture_root=root,
            )

        self.assertEqual(1, len(cases))
        self.assertEqual(2, cases[0].expected_component_count)
        self.assertEqual(4, cases[0].expected_side_count)
        self.assertEqual((2, 2), cases[0].expected_component_domain_assignment_counts)

    def test_stereo_constraint_model_fixture_rejects_bad_domain_sizes(self) -> None:
        invalid_cases = (
            _stereo_constraint_model_case(
                "missing_domain_sizes",
                expected_component_side_domain_sizes=[],
            ),
            _stereo_constraint_model_case(
                "non_list_component",
                expected_component_side_domain_sizes=[1],
            ),
            _stereo_constraint_model_case(
                "non_int_domain_size",
                expected_component_side_domain_sizes=[["2"]],
            ),
            _stereo_constraint_model_case(
                "zero_domain_size",
                expected_component_side_domain_sizes=[[0]],
            ),
        )
        for case in invalid_cases:
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaises(ValueError):
                        load_pinned_stereo_constraint_model_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )


class CheckedInPinnedRdkitFixtureTest(unittest.TestCase):
    def test_all_checked_in_pinned_rdkit_fixtures_load(self) -> None:
        self.assertEqual(
            set(PINNED_RDKIT_PARITY_FIXTURE_FAMILIES),
            set(PINNED_FIXTURE_LOADERS),
        )
        for fixture_family in PINNED_RDKIT_PARITY_FIXTURE_FAMILIES:
            fixture_root = pinned_rdkit_fixture_root(fixture_family)
            versions = pinned_rdkit_fixture_versions(fixture_root)
            self.assertTrue(versions, fixture_family)

            for rdkit_version in versions:
                with self.subTest(
                    fixture_family=fixture_family,
                    rdkit_version=rdkit_version,
                ):
                    cases = PINNED_FIXTURE_LOADERS[fixture_family](
                        rdkit_version,
                        fixture_root=fixture_root,
                    )
                    self.assertTrue(cases)


class CheckedInRdkitCompatibilityFixtureTest(unittest.TestCase):
    def test_known_quirks_fixture_loads(self) -> None:
        fixture_root = pinned_rdkit_fixture_root(PINNED_RDKIT_KNOWN_QUIRKS)
        versions = pinned_rdkit_fixture_versions(fixture_root)

        self.assertTrue(versions)
        for rdkit_version in versions:
            with self.subTest(rdkit_version=rdkit_version):
                cases = load_pinned_rdkit_known_quirk_cases(
                    rdkit_version,
                    fixture_root=fixture_root,
                )

                self.assertTrue(cases)
                self.assertTrue(all(case.category for case in cases))

    def test_disconnected_root_zero_fixture_loads(self) -> None:
        cases = load_disconnected_root_zero_smiles()

        self.assertEqual(30, len(cases))
        self.assertEqual(30, len(set(cases)))
        self.assertIn("[Na+].[Cl-]", cases)

    def test_steroid_ring_coupled_component_fixture_loads(self) -> None:
        case = load_steroid_ring_coupled_component_regression()

        self.assertEqual(0, case.rooted_at_atom)
        self.assertIn("C[C@H]", case.input_smiles)
        self.assertIn("/C(=C/C=C1", case.expected_member)
        self.assertIn("\\C(=C\\C=C1", case.rejected_member)

    def test_stereo_expected_member_fixture_loads(self) -> None:
        cases = load_stereo_expected_member_regressions()

        self.assertEqual(2, len(cases))
        self.assertEqual(2, len({case.case_id for case in cases}))
        self.assertTrue(all(case.rooted_at_atom == 0 for case in cases))
        self.assertEqual([True, False], [case.validate_support for case in cases])

    def test_stereo_constraint_model_fixture_loads(self) -> None:
        fixture_root = pinned_rdkit_fixture_root(PINNED_STEREO_CONSTRAINT_MODEL)
        versions = pinned_rdkit_fixture_versions(fixture_root)

        self.assertTrue(versions)
        for rdkit_version in versions:
            with self.subTest(rdkit_version=rdkit_version):
                cases = load_pinned_stereo_constraint_model_cases(
                    rdkit_version,
                    fixture_root=fixture_root,
                )

                self.assertTrue(cases)
                self.assertTrue(
                    all(
                        case.expected_rdkit_traversal_writer_assignment_count
                        <= case.expected_rdkit_local_writer_assignment_count
                        <= case.expected_semantic_assignment_count
                        for case in cases
                    )
                )


class RdkitCompatibilityFixtureLoaderTest(unittest.TestCase):
    def test_disconnected_root_zero_fixture_rejects_bad_case_list(self) -> None:
        invalid_payloads = (
            {"cases": "CC.O"},
            {"cases": []},
            {"cases": ["CC.O", "CC.O"]},
            {"cases": ["CC.O", ""]},
            {"cases": ["CC.O", 123]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fixture_path = Path(tmpdir) / "root_zero_smiles.json"
                    _write_json(fixture_path, payload)

                    with self.assertRaises(ValueError):
                        load_disconnected_root_zero_smiles(fixture_path)

    def test_steroid_ring_coupled_component_fixture_rejects_bad_fields(self) -> None:
        valid_payload = {
            "input_smiles": "CC",
            "rooted_at_atom": 0,
            "expected_member": "CC",
            "rejected_member": "C.C",
        }
        invalid_overrides = (
            {"input_smiles": ""},
            {"rooted_at_atom": "0"},
            {"expected_member": ""},
            {"rejected_member": ""},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fixture_path = Path(tmpdir) / "steroid.json"
                    payload = {**valid_payload, **override}
                    _write_json(fixture_path, payload)

                    with self.assertRaises(ValueError):
                        load_steroid_ring_coupled_component_regression(fixture_path)

    def test_stereo_expected_member_fixture_rejects_bad_case_fields(self) -> None:
        valid_case = {
            "id": "case_a",
            "input_smiles": "CC",
            "rooted_at_atom": 0,
            "expected_member": "CC",
            "validate_support": True,
        }
        invalid_payloads = (
            {"cases": []},
            {"cases": "case_a"},
            {"cases": [{**valid_case, "id": ""}]},
            {"cases": [{**valid_case, "rooted_at_atom": "0"}]},
            {"cases": [{**valid_case, "validate_support": "true"}]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fixture_path = Path(tmpdir) / "rooted_membership.json"
                    _write_json(fixture_path, payload)

                    with self.assertRaises(ValueError):
                        load_stereo_expected_member_regressions(fixture_path)


if __name__ == "__main__":
    unittest.main()
