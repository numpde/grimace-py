from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_EXACT_SMALL_SUPPORT,
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
    PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_ROOTED_RANDOM,
    PINNED_RDKIT_SERIALIZER_REGRESSIONS,
    PINNED_RDKIT_WRITER_MEMBERSHIP,
    PINNED_RDKIT_WRITER_SUPPORT_COUNTS,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_root,
    pinned_rdkit_fixture_versions,
)
from tests.helpers.rdkit_disconnected_sampling import load_disconnected_root_zero_smiles
from tests.helpers.rdkit_exact_small_support import (
    load_pinned_exact_small_support_cases,
)
from tests.helpers.rdkit_known_quirks import load_pinned_rdkit_known_quirk_cases
from tests.helpers.rdkit_known_stereo_gaps import (
    load_pinned_known_stereo_gap_cases,
)
from tests.helpers.rdkit_rooted_random import load_pinned_rooted_random_cases
from tests.helpers.rdkit_serializer_regressions import (
    load_pinned_serializer_regression_cases,
)
from tests.helpers.rdkit_stereo_regressions import (
    load_stereo_expected_member_regressions,
    load_steroid_ring_coupled_component_regression,
)
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.helpers.rdkit_writer_support_counts import (
    load_pinned_writer_support_count_cases,
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


def _known_gap_case(case_id: str, **overrides: object) -> dict[str, object]:
    case = {
        **_base_case(case_id),
        "smiles": "CC/C=C\\C",
        "expected": "CC/C=C\\C",
        "rooted_at_atom": None,
        "isomeric_smiles": True,
        "rdkit_canonical": True,
    }
    case.update(overrides)
    return case


def _writer_support_count_flags(**overrides: object) -> dict[str, object]:
    flags = {
        "isomericSmiles": False,
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": False,
        "allBondsExplicit": False,
        "allHsExplicit": False,
        "ignoreAtomMapNumbers": False,
    }
    flags.update(overrides)
    return flags


def _writer_support_count_payload(
    *cases: dict[str, object],
    flags: dict[str, object] | None = None,
    rdkit_version: str = RDKIT_VERSION,
) -> dict[str, object]:
    return {
        "rdkit_version": rdkit_version,
        "flags": flags if flags is not None else _writer_support_count_flags(),
        "cases": list(cases),
    }


def _writer_support_count_case(case_id: str, **overrides: object) -> dict[str, object]:
    case = {
        **_base_case(case_id),
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "rooted_at_atom": -1,
        "support_count": 304,
        "evidence": {
            "method": "rdkit_random_adaptive_saturation",
            "criterion_version": 1,
            "min_draws": 10000,
            "unseen_mass_threshold": 0.001,
            "allowed_missing_variants": 1.0,
            "runs": [
                {
                    "seed": 12345,
                    "draw_count": 11000,
                    "support_count": 304,
                    "consecutive_draws_without_new_variant": 10000,
                    "singleton_count": 0,
                    "doubleton_count": 0,
                    "estimated_unseen_mass": 0.0,
                    "estimated_missing_variants": 0.0,
                },
                {
                    "seed": 54321,
                    "draw_count": 12000,
                    "support_count": 304,
                    "consecutive_draws_without_new_variant": 10000,
                    "singleton_count": 0,
                    "doubleton_count": 0,
                    "estimated_unseen_mass": 0.0,
                    "estimated_missing_variants": 0.0,
                },
            ],
        },
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

    def test_fixture_rejects_malformed_json_with_context(self) -> None:
        for relative_path in (
            Path(f"{RDKIT_VERSION}.json"),
            Path(RDKIT_VERSION) / "10_bad.json",
        ):
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    fixture_path = root / relative_path
                    fixture_path.parent.mkdir(parents=True, exist_ok=True)
                    fixture_path.write_text("{", encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "not readable JSON"):
                        load_pinned_rdkit_fixture_cases(
                            fixture_root=root,
                            rdkit_version=RDKIT_VERSION,
                            fixture_label="contract",
                        )

    def test_fixture_rejects_malformed_payloads(self) -> None:
        invalid_payloads = (
            [],
            {"rdkit_version": RDKIT_VERSION, "cases": []},
            {"rdkit_version": RDKIT_VERSION, "cases": "case_a"},
            {"rdkit_version": RDKIT_VERSION, "cases": ["case_a"]},
            {"rdkit_version": RDKIT_VERSION, "cases": [{"id": 123, "source": "x"}]},
            {"rdkit_version": RDKIT_VERSION, "cases": [{"id": "", "source": "x"}]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", payload)

                    with self.assertRaises(ValueError):
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


class KnownStereoGapFixtureLoaderTest(unittest.TestCase):
    def test_known_gap_fixture_accepts_each_molecule_source_kind(self) -> None:
        cases = [
            _known_gap_case("smiles_case"),
            _known_gap_case("molblock_case", molblock="mol\nM  END\n"),
            _known_gap_case(
                "writer_case",
                writer_membership_case_id="writer_membership_case",
            ),
        ]
        del cases[1]["smiles"]
        del cases[2]["smiles"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(*cases))

            loaded = load_pinned_known_stereo_gap_cases(
                RDKIT_VERSION,
                fixture_root=root,
            )

        self.assertEqual(
            ["smiles_case", "molblock_case", "writer_case"],
            [case.case_id for case in loaded],
        )
        self.assertEqual("CC/C=C\\C", loaded[0].smiles)
        self.assertEqual("mol\nM  END\n", loaded[1].molblock)
        self.assertEqual("writer_membership_case", loaded[2].writer_membership_case_id)

    def test_known_gap_fixture_rejects_bad_molecule_source_count(self) -> None:
        duplicate_source_case = _known_gap_case(
            "both_sources",
            molblock="mol\nM  END\n",
        )
        missing_source_case = _known_gap_case("no_source")
        del missing_source_case["smiles"]
        for case in (duplicate_source_case, missing_source_case):
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaisesRegex(
                        ValueError,
                        "exactly one of 'smiles', 'molblock', or "
                        "'writer_membership_case_id'",
                    ):
                        load_pinned_known_stereo_gap_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )


class WriterSupportCountFixtureLoaderTest(unittest.TestCase):
    def test_writer_support_count_fixture_loads_directory_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_dir = root / RDKIT_VERSION
            _write_json(
                fixture_dir / "nonisomeric__random.json",
                _writer_support_count_payload(
                    _writer_support_count_case("aspirin_count")
                ),
            )

            cases = load_pinned_writer_support_count_cases(
                RDKIT_VERSION,
                fixture_root=root,
            )

        self.assertEqual(["aspirin_count"], [case.case_id for case in cases])
        self.assertEqual(304, cases[0].support_count)
        self.assertFalse(cases[0].isomeric_smiles)
        self.assertEqual(-1, cases[0].rooted_at_atom)
        self.assertEqual(
            "rdkit_random_adaptive_saturation",
            cases[0].evidence.method,
        )
        self.assertEqual([12345, 54321], [run.seed for run in cases[0].evidence.runs])

    def test_writer_support_count_fixture_rejects_bad_flag_surface(self) -> None:
        invalid_payloads = (
            _writer_support_count_payload(
                _writer_support_count_case("missing_flag"),
                flags={
                    key: value
                    for key, value in _writer_support_count_flags().items()
                    if key != "canonical"
                },
            ),
            _writer_support_count_payload(
                _writer_support_count_case("canonical_true"),
                flags=_writer_support_count_flags(canonical=True),
            ),
            _writer_support_count_payload(
                _writer_support_count_case("random_false"),
                flags=_writer_support_count_flags(doRandom=False),
            ),
        )
        for payload in invalid_payloads:
            with self.subTest(flags=payload["flags"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / RDKIT_VERSION / "nonisomeric__random.json", payload)

                    with self.assertRaises(ValueError):
                        load_pinned_writer_support_count_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )

    def test_writer_support_count_fixture_rejects_filename_flag_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(
                root / RDKIT_VERSION / "isomeric__random.json",
                _writer_support_count_payload(
                    _writer_support_count_case("aspirin_count")
                ),
            )

            with self.assertRaisesRegex(ValueError, "filename must match"):
                load_pinned_writer_support_count_cases(
                    RDKIT_VERSION,
                    fixture_root=root,
                )

    def test_writer_support_count_fixture_rejects_bad_adaptive_evidence(self) -> None:
        bad_evidence_cases = (
            _writer_support_count_case("one_run"),
            _writer_support_count_case("mismatched_count"),
            _writer_support_count_case("same_seed"),
            _writer_support_count_case("high_unseen_mass"),
            _writer_support_count_case("high_missing"),
            _writer_support_count_case("short_run"),
            _writer_support_count_case("low_patience"),
            _writer_support_count_case("wrong_unseen_estimate"),
            _writer_support_count_case("wrong_missing_estimate"),
            _writer_support_count_case("unstable_missing_estimate"),
        )
        bad_evidence_cases[0]["evidence"]["runs"] = bad_evidence_cases[0]["evidence"][
            "runs"
        ][:1]
        bad_evidence_cases[1]["evidence"]["runs"][0]["support_count"] = 303
        bad_evidence_cases[2]["evidence"]["runs"][1]["seed"] = 12345
        bad_evidence_cases[3]["evidence"]["runs"][0]["estimated_unseen_mass"] = 0.01
        bad_evidence_cases[4]["evidence"]["runs"][0]["estimated_missing_variants"] = 2.0
        bad_evidence_cases[5]["evidence"]["runs"][0]["draw_count"] = 50
        bad_evidence_cases[6]["evidence"]["runs"][0][
            "consecutive_draws_without_new_variant"
        ] = 9999
        bad_evidence_cases[7]["evidence"]["runs"][0]["singleton_count"] = 1
        bad_evidence_cases[8]["evidence"]["runs"][0]["singleton_count"] = 2
        bad_evidence_cases[8]["evidence"]["runs"][0]["doubleton_count"] = 1
        bad_evidence_cases[8]["evidence"]["runs"][0][
            "estimated_unseen_mass"
        ] = 2 / 11000
        bad_evidence_cases[9]["evidence"]["runs"][0]["singleton_count"] = 1
        bad_evidence_cases[9]["evidence"]["runs"][0][
            "estimated_unseen_mass"
        ] = 1 / 11000

        for case in bad_evidence_cases:
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(
                        root / RDKIT_VERSION / "nonisomeric__random.json",
                        _writer_support_count_payload(case),
                    )

                    with self.assertRaises(ValueError):
                        load_pinned_writer_support_count_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )

    def test_known_gap_fixture_rejects_unpaired_random_vector_fields(self) -> None:
        invalid_cases = (
            _known_gap_case("seed_only", rdkit_random_vector_seed=1),
            _known_gap_case("index_only", rdkit_random_vector_index=0),
        )
        for case in invalid_cases:
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaisesRegex(ValueError, "must define both"):
                        load_pinned_known_stereo_gap_cases(
                            RDKIT_VERSION,
                            fixture_root=root,
                        )

    def test_known_gap_fixture_rejects_coerced_boolean_and_integer_fields(self) -> None:
        invalid_cases = (
            _known_gap_case("string_bool", isomeric_smiles="true"),
            _known_gap_case("bool_root", rooted_at_atom=False),
            _known_gap_case(
                "string_seed",
                rdkit_random_vector_seed="1",
                rdkit_random_vector_index=0,
            ),
        )
        for case in invalid_cases:
            with self.subTest(case_id=case["id"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    _write_json(root / f"{RDKIT_VERSION}.json", _base_payload(case))

                    with self.assertRaises(ValueError):
                        load_pinned_known_stereo_gap_cases(
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

    def test_known_stereo_gaps_fixture_loads(self) -> None:
        fixture_root = pinned_rdkit_fixture_root(PINNED_RDKIT_KNOWN_STEREO_GAPS)
        versions = pinned_rdkit_fixture_versions(fixture_root)

        self.assertTrue(versions)
        for rdkit_version in versions:
            with self.subTest(rdkit_version=rdkit_version):
                cases = load_pinned_known_stereo_gap_cases(
                    rdkit_version,
                    fixture_root=fixture_root,
                )

                self.assertTrue(cases)
                self.assertTrue(all(case.expected for case in cases))

    def test_writer_support_count_fixture_loads(self) -> None:
        fixture_root = pinned_rdkit_fixture_root(PINNED_RDKIT_WRITER_SUPPORT_COUNTS)
        versions = pinned_rdkit_fixture_versions(fixture_root)

        self.assertTrue(versions)
        for rdkit_version in versions:
            with self.subTest(rdkit_version=rdkit_version):
                cases = load_pinned_writer_support_count_cases(
                    rdkit_version,
                    fixture_root=fixture_root,
                )

                self.assertTrue(cases)
                self.assertTrue(all(case.support_count > 0 for case in cases))

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


class RdkitCompatibilityFixtureLoaderTest(unittest.TestCase):
    def test_disconnected_root_zero_fixture_rejects_bad_case_list(self) -> None:
        invalid_payloads = (
            [],
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
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "steroid.json"
            _write_json(fixture_path, [])

            with self.assertRaises(ValueError):
                load_steroid_ring_coupled_component_regression(fixture_path)

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
            [],
            {"cases": []},
            {"cases": "case_a"},
            {"cases": ["case_a"]},
            {"cases": [{**valid_case, "id": ""}]},
            {"cases": [valid_case, valid_case]},
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
