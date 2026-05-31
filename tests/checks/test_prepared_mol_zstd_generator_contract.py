from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "generate_prepared_mol_zstd_dictionary.py"


def generator_constants() -> dict[str, ast.AST]:
    module = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    constants: dict[str, ast.AST] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = statement.value
    return constants


def literal_constant(name: str) -> object:
    constants = generator_constants()

    def evaluate(node: ast.AST) -> object:
        if isinstance(node, ast.Name):
            try:
                return evaluate(constants[node.id])
            except KeyError as exc:
                raise AssertionError(f"Unknown generator constant: {node.id}") from exc
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(element) for element in node.elts)
        if isinstance(node, ast.List):
            return [evaluate(element) for element in node.elts]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
        ):
            return Path(*(evaluate(argument) for argument in node.args))
        return ast.literal_eval(node)

    try:
        return evaluate(constants[name])
    except KeyError as exc:
        raise AssertionError(f"Missing generator constant: {name}") from exc


def load_generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_prepared_mol_zstd_dictionary",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load generator module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PreparedMolZstdGeneratorContractTests(unittest.TestCase):
    def test_training_parameters_are_explicit_recipe_values(self) -> None:
        self.assertEqual(
            {
                "dict_size": 112_640,
                "dict_id": "derived-from-training-identity-sha256",
                "k": 0,
                "d": 8,
                "f": 20,
                "split_point": 0.75,
                "accel": 1,
                "notifications": 0,
                "level": 3,
                "steps": 4,
                "threads": 0,
            },
            literal_constant("ZSTD_TRAINING_PARAMETERS"),
        )

    def test_training_environment_is_pinned_beyond_python_package_name(
        self,
    ) -> None:
        self.assertEqual("2026.03.1", literal_constant("EXPECTED_RDKIT_VERSION"))
        self.assertEqual("0.25.0", literal_constant("EXPECTED_ZSTANDARD_VERSION"))
        self.assertEqual("cext", literal_constant("EXPECTED_ZSTANDARD_BACKEND"))
        self.assertEqual((1, 5, 7), literal_constant("EXPECTED_ZSTD_LIBRARY_VERSION"))

    def test_training_corpus_recipe_is_committed(self) -> None:
        self.assertEqual(
            Path("tests/fixtures/top_100000_CIDs.tsv.gz"),
            literal_constant("FIXTURE_RELATIVE_PATH"),
        )
        self.assertEqual(
            "67d1d31c3eb27da5ae5a8b8c1a3369531061113464c0dbfaf46e274493acc1ea",
            literal_constant("EXPECTED_FIXTURE_GZIP_SHA256"),
        )
        self.assertEqual(
            "605cbbd10e69225ccfb47e05594aa01b338df7238f2ba3da76d7f5060c08f1bf",
            literal_constant("EXPECTED_FIXTURE_UNCOMPRESSED_SHA256"),
        )
        self.assertEqual(
            {
                "isomericSmiles": True,
                "kekuleSmiles": False,
                "allBondsExplicit": False,
                "allHsExplicit": False,
                "ignoreAtomMapNumbers": False,
            },
            literal_constant("WRITER_OPTIONS"),
        )
        self.assertEqual(
            {
                "name": "all-parseable-preparable-v1",
                "included_rows": (
                    "all fixture rows that RDKit parses and grimace.PrepareMol "
                    "prepares successfully"
                ),
                "sample_order": "source row order",
                "deduplication": "none",
            },
            literal_constant("SELECTION_RULE"),
        )

    def test_dictionary_id_derivation_rule_is_the_committed_recipe(self) -> None:
        self.assertEqual(
            (
                "For each 4-byte little-endian word of "
                "training_identity_sha256, clear the top bit and choose the "
                "first value in 32768..2147483647 not already assigned to a "
                "shipped dictionary. Fail if no such value exists."
            ),
            literal_constant("DICT_ID_DERIVATION_RULE"),
        )

    def test_dictionary_id_derivation_skips_reserved_and_existing_ids(self) -> None:
        generator = load_generator_module()
        digest = b"".join(
            (
                (1).to_bytes(4, "little"),
                (123_456).to_bytes(4, "little"),
                (50_000).to_bytes(4, "little"),
                b"\0" * 20,
            )
        ).hex()

        self.assertEqual(
            123_456,
            generator.derive_dictionary_id(digest, existing_ids=set()),
        )
        self.assertEqual(
            50_000,
            generator.derive_dictionary_id(digest, existing_ids={123_456}),
        )

        reserved = ((1).to_bytes(4, "little") * 8).hex()
        with self.assertRaises(RuntimeError):
            generator.derive_dictionary_id(reserved, existing_ids=set())

    def test_existing_shipped_dictionary_ids_rejects_malformed_manifests(
        self,
    ) -> None:
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = (
                root
                / "python"
                / "grimace"
                / "data"
                / "prepared_mol_zstd"
                / "artifact"
            )
            artifact.mkdir(parents=True)
            (artifact / "default_v1.json").write_text(
                json.dumps({"zstd_dictionary_id": "not-an-int"}),
                encoding="utf-8",
            )

            generator.ROOT = root
            with self.assertRaises(RuntimeError):
                generator.existing_shipped_dictionary_ids()

    def test_existing_shipped_dictionary_ids_rejects_duplicate_ids(self) -> None:
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dictionary_root = (
                root / "python" / "grimace" / "data" / "prepared_mol_zstd"
            )
            for name in ("first", "second"):
                artifact = dictionary_root / name
                artifact.mkdir(parents=True)
                (artifact / "default_v1.json").write_text(
                    json.dumps({"zstd_dictionary_id": 123_456}),
                    encoding="utf-8",
                )

            generator.ROOT = root
            with self.assertRaises(RuntimeError):
                generator.existing_shipped_dictionary_ids()

    def test_existing_shipped_dictionary_ids_reads_valid_manifests(self) -> None:
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dictionary_root = (
                root / "python" / "grimace" / "data" / "prepared_mol_zstd"
            )
            for name, dictionary_id in (("first", 123_456), ("second", 50_000)):
                artifact = dictionary_root / name
                artifact.mkdir(parents=True)
                (artifact / "default_v1.json").write_text(
                    json.dumps({"zstd_dictionary_id": dictionary_id}),
                    encoding="utf-8",
                )

            generator.ROOT = root
            self.assertEqual(
                {50_000, 123_456},
                generator.existing_shipped_dictionary_ids(),
            )


if __name__ == "__main__":
    unittest.main()
