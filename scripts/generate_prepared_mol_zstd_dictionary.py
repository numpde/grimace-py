#!/usr/bin/env python3
"""Generate the built-in PreparedMol zstd dictionary artifact.

This script owns the production `default_v1` recipe. It intentionally has no
test-dictionary mode: all output is generated from the pinned checked-in
molecule fixture and the deterministic selection rule documented in
`notes/020_prepared_mol_zstd_dictionary.md`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ZSTANDARD_VERSION = "0.25.0"

SEMANTIC_ID = "prepared-mol-default-v1"
ARTIFACT_STEM = "default_v1"
# Bump for any output-affecting generator change. Refactors that preserve the
# training identity and dictionary bytes do not need a new artifact identity.
GENERATOR_VERSION = 1
EXPECTED_RDKIT_VERSION = "2026.03.1"
EXPECTED_ZSTANDARD_BACKEND = "cext"
EXPECTED_ZSTD_LIBRARY_VERSION = (1, 5, 7)
FIXTURE_RELATIVE_PATH = Path("tests/fixtures/top_100000_CIDs.tsv.gz")
PACKAGE_DICTIONARY_ROOT = Path("python/grimace/data/prepared_mol_zstd")
EXPECTED_FIXTURE_GZIP_SHA256 = (
    "67d1d31c3eb27da5ae5a8b8c1a3369531061113464c0dbfaf46e274493acc1ea"
)
EXPECTED_FIXTURE_UNCOMPRESSED_SHA256 = (
    "605cbbd10e69225ccfb47e05594aa01b338df7238f2ba3da76d7f5060c08f1bf"
)
RAW_PREPARED_MOL_MAGIC = b"GPM\0"
HEAD_SAMPLE_COUNT = 10_000
HASH_STRATIFIED_SAMPLE_COUNT = 10_000
ZSTD_DICTIONARY_SIZE = 112_640
ZSTD_TRAINING_PARAMETERS = {
    "dict_size": ZSTD_DICTIONARY_SIZE,
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
}
WRITER_OPTIONS = {
    "isomericSmiles": True,
    "kekuleSmiles": False,
    "allBondsExplicit": False,
    "allHsExplicit": False,
    "ignoreAtomMapNumbers": False,
}
SELECTION_RULE = {
    "name": "head-plus-cid-hash-stratified-v1",
    "head_parseable_prepared_rows": HEAD_SAMPLE_COUNT,
    "hash_stratified_parseable_prepared_rows": HASH_STRATIFIED_SAMPLE_COUNT,
    "hash_input": "ASCII decimal CID text",
    "hash": "sha256",
    "hash_order": "ascending digest bytes, then source row number",
    "final_sample_order": "source row order",
    "deduplication": "none",
}
DICT_ID_DERIVATION_RULE = (
    "For each 4-byte little-endian word of training_identity_sha256, clear the "
    "top bit and choose the first value in 32768..2147483647 not already "
    "assigned to a shipped dictionary. Fail if no such value exists."
)
MANIFEST_HASH_CANONICALIZATION = {
    "json": "utf-8, sort_keys=True, separators=(',', ':')",
    "excluded_fields": [
        "artifact_dir",
        "created_yyyymmdd",
        "manifest_sha256",
    ],
}

zstd: Any | None = None
Chem: Any | None = None
rdBase: Any | None = None
grimace: Any | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    row_number: int
    cid: str
    smiles: str
    raw_payload: bytes


@dataclass(frozen=True, slots=True)
class Corpus:
    candidate_success_count: int
    parse_failure_count: int
    preparation_failure_count: int
    selected: tuple[Candidate, ...]


def load_runtime_dependencies() -> None:
    global zstd, Chem, rdBase, grimace
    if zstd is not None:
        return

    try:
        import zstandard as loaded_zstd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Missing generator dependency: install "
            f"zstandard=={EXPECTED_ZSTANDARD_VERSION} in the development "
            "environment."
        ) from exc

    from rdkit import Chem as loaded_Chem
    from rdkit import rdBase as loaded_rdBase

    import grimace as loaded_grimace

    zstd = loaded_zstd
    Chem = loaded_Chem
    rdBase = loaded_rdBase
    grimace = loaded_grimace


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_uncompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_length_prefixed_bytes(values: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def digest_text_lines(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepared_mol_format_version(payload: bytes) -> int:
    if not payload.startswith(RAW_PREPARED_MOL_MAGIC):
        raise ValueError("PreparedMol payload does not start with GPM\\0")
    if len(payload) < len(RAW_PREPARED_MOL_MAGIC) + 4:
        raise ValueError("PreparedMol payload is too short for a format version")
    return int.from_bytes(
        payload[len(RAW_PREPARED_MOL_MAGIC) : len(RAW_PREPARED_MOL_MAGIC) + 4],
        "little",
    )


def require_expected_environment(fixture_path: Path) -> None:
    if rdBase.rdkitVersion != EXPECTED_RDKIT_VERSION:
        raise RuntimeError(
            f"RDKit version must be {EXPECTED_RDKIT_VERSION}, got {rdBase.rdkitVersion}"
        )
    if zstd.__version__ != EXPECTED_ZSTANDARD_VERSION:
        raise RuntimeError(
            "python-zstandard version must be "
            f"{EXPECTED_ZSTANDARD_VERSION}, got {zstd.__version__}"
        )
    if zstd.backend != EXPECTED_ZSTANDARD_BACKEND:
        raise RuntimeError(
            "python-zstandard backend must be "
            f"{EXPECTED_ZSTANDARD_BACKEND}, got {zstd.backend}"
        )
    if zstd.ZSTD_VERSION != EXPECTED_ZSTD_LIBRARY_VERSION:
        raise RuntimeError(
            "zstd library version must be "
            f"{EXPECTED_ZSTD_LIBRARY_VERSION}, got {zstd.ZSTD_VERSION}"
        )

    gzip_hash = file_sha256(fixture_path)
    if gzip_hash != EXPECTED_FIXTURE_GZIP_SHA256:
        raise RuntimeError(
            "Fixture gzip SHA-256 mismatch: "
            f"expected {EXPECTED_FIXTURE_GZIP_SHA256}, got {gzip_hash}"
        )

    uncompressed_hash = gzip_uncompressed_sha256(fixture_path)
    if uncompressed_hash != EXPECTED_FIXTURE_UNCOMPRESSED_SHA256:
        raise RuntimeError(
            "Fixture uncompressed SHA-256 mismatch: "
            f"expected {EXPECTED_FIXTURE_UNCOMPRESSED_SHA256}, got {uncompressed_hash}"
        )


def build_candidates(fixture_path: Path) -> Corpus:
    candidates: list[Candidate] = []
    parse_failure_count = 0
    preparation_failure_count = 0

    with gzip.open(fixture_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                parse_failure_count += 1
                continue

            try:
                prepared = grimace.PrepareMol(mol, **WRITER_OPTIONS)
                raw_payload = prepared.to_bytes()
            except Exception:
                preparation_failure_count += 1
                continue

            candidates.append(
                Candidate(
                    row_number=row_number,
                    cid=row["CID"],
                    smiles=smiles,
                    raw_payload=raw_payload,
                )
            )

    if len(candidates) < HEAD_SAMPLE_COUNT + HASH_STRATIFIED_SAMPLE_COUNT:
        raise RuntimeError(
            "Not enough parseable/preparable fixture rows for dictionary training: "
            f"need {HEAD_SAMPLE_COUNT + HASH_STRATIFIED_SAMPLE_COUNT}, "
            f"got {len(candidates)}"
        )

    head = candidates[:HEAD_SAMPLE_COUNT]
    remaining = candidates[HEAD_SAMPLE_COUNT:]
    stratified = sorted(
        remaining,
        key=lambda candidate: (
            hashlib.sha256(candidate.cid.encode("ascii")).digest(),
            candidate.row_number,
        ),
    )[:HASH_STRATIFIED_SAMPLE_COUNT]
    selected_rows = {candidate.row_number for candidate in (*head, *stratified)}
    selected = tuple(
        candidate for candidate in candidates if candidate.row_number in selected_rows
    )

    if len(selected) != HEAD_SAMPLE_COUNT + HASH_STRATIFIED_SAMPLE_COUNT:
        raise RuntimeError("Selection rule produced duplicate source rows")

    return Corpus(
        candidate_success_count=len(candidates),
        parse_failure_count=parse_failure_count,
        preparation_failure_count=preparation_failure_count,
        selected=selected,
    )


def zstandard_library_version() -> object:
    return getattr(zstd, "ZSTD_VERSION", None)


def existing_shipped_dictionary_ids() -> set[int]:
    dictionary_ids: dict[int, Path] = {}
    dictionary_root = ROOT / PACKAGE_DICTIONARY_ROOT
    if not dictionary_root.exists():
        return set()

    for manifest_path in dictionary_root.glob("*/*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dictionary_id = manifest.get("zstd_dictionary_id")
        if not isinstance(dictionary_id, int):
            raise RuntimeError(
                "Shipped dictionary manifest lacks an integer "
                f"zstd_dictionary_id: {manifest_path}"
            )
        if dictionary_id in dictionary_ids:
            raise RuntimeError(
                "Duplicate shipped zstd dictionary ID "
                f"{dictionary_id}: {dictionary_ids[dictionary_id]} and "
                f"{manifest_path}"
            )
        dictionary_ids[dictionary_id] = manifest_path
    return set(dictionary_ids)


def build_training_identity(corpus: Corpus, fixture_path: Path) -> dict[str, Any]:
    raw_payloads = tuple(candidate.raw_payload for candidate in corpus.selected)
    selected_cids = tuple(candidate.cid for candidate in corpus.selected)
    selected_source_rows = tuple(
        str(candidate.row_number) for candidate in corpus.selected
    )
    first_version = prepared_mol_format_version(raw_payloads[0])
    if any(
        prepared_mol_format_version(payload) != first_version for payload in raw_payloads
    ):
        raise RuntimeError(
            "Selected PreparedMol payloads do not share one raw format version"
        )

    return {
        "semantic_id": SEMANTIC_ID,
        "source_fixture": {
            "path": str(fixture_path.relative_to(ROOT)),
            "source_name": "PubChem-derived top 100000 CID fixture",
            "gzip_sha256": EXPECTED_FIXTURE_GZIP_SHA256,
            "uncompressed_sha256": EXPECTED_FIXTURE_UNCOMPRESSED_SHA256,
            "redistribution_terms": (
                "Dictionary training uses the checked-in repository fixture; "
                "the generated dictionary ships derived compression statistics, "
                "not raw training samples."
            ),
        },
        "rdkit_version": rdBase.rdkitVersion,
        "prepared_mol_raw_format": {
            "magic_hex": RAW_PREPARED_MOL_MAGIC.hex(),
            "version": first_version,
        },
        "writer_options": WRITER_OPTIONS,
        "selection_rule": SELECTION_RULE,
        "candidate_success_count": corpus.candidate_success_count,
        "parse_failure_count": corpus.parse_failure_count,
        "preparation_failure_count": corpus.preparation_failure_count,
        "sample_count": len(corpus.selected),
        "selected_cids_sha256": digest_text_lines(selected_cids),
        "selected_source_rows_sha256": digest_text_lines(selected_source_rows),
        "raw_sample_digest": {
            "algorithm": "sha256",
            "encoding": "8-byte little-endian length prefix followed by raw payload bytes",
            "value": digest_length_prefixed_bytes(raw_payloads),
        },
        "raw_sample_total_bytes": sum(len(payload) for payload in raw_payloads),
        "dictionary_size": ZSTD_DICTIONARY_SIZE,
        "dictionary_id_derivation_rule": DICT_ID_DERIVATION_RULE,
        "training_parameters": ZSTD_TRAINING_PARAMETERS,
        "generator": {
            "script": "scripts/generate_prepared_mol_zstd_dictionary.py",
            "version": GENERATOR_VERSION,
            "python_package": "zstandard",
            "python_package_version": zstd.__version__,
            "python_package_backend": zstd.backend,
            "zstd_library_version": zstandard_library_version(),
        },
    }


def derive_dictionary_id(
    training_identity_sha256: str,
    *,
    existing_ids: set[int],
) -> int:
    digest = bytes.fromhex(training_identity_sha256)
    for offset in range(0, len(digest), 4):
        value = int.from_bytes(digest[offset : offset + 4], "little") & 0x7FFF_FFFF
        if 32_768 <= value < 2_147_483_648 and value not in existing_ids:
            return value
    raise RuntimeError("Could not derive a non-reserved zstd dictionary ID")


def train_dictionary(corpus: Corpus, dict_id: int) -> bytes:
    samples = [candidate.raw_payload for candidate in corpus.selected]
    dictionary = zstd.train_dictionary(
        ZSTD_DICTIONARY_SIZE,
        samples,
        dict_id=dict_id,
        k=ZSTD_TRAINING_PARAMETERS["k"],
        d=ZSTD_TRAINING_PARAMETERS["d"],
        f=ZSTD_TRAINING_PARAMETERS["f"],
        split_point=ZSTD_TRAINING_PARAMETERS["split_point"],
        accel=ZSTD_TRAINING_PARAMETERS["accel"],
        notifications=ZSTD_TRAINING_PARAMETERS["notifications"],
        level=ZSTD_TRAINING_PARAMETERS["level"],
        steps=ZSTD_TRAINING_PARAMETERS["steps"],
        threads=ZSTD_TRAINING_PARAMETERS["threads"],
    )
    actual_dict_id = dictionary.dict_id()
    if actual_dict_id != dict_id:
        raise RuntimeError(
            f"Generated dictionary ID mismatch: expected {dict_id}, got {actual_dict_id}"
        )
    dictionary_bytes = dictionary.as_bytes()
    if len(dictionary_bytes) > ZSTD_DICTIONARY_SIZE:
        raise RuntimeError(
            f"Generated dictionary exceeds requested size: {len(dictionary_bytes)}"
        )
    return dictionary_bytes


def artifact_identity(
    *,
    training_identity: dict[str, Any],
    training_identity_sha256: str,
    dictionary_bytes: bytes,
    dictionary_id: int,
) -> dict[str, Any]:
    return {
        "semantic_id": SEMANTIC_ID,
        "training_identity_sha256": training_identity_sha256,
        "training_identity": training_identity,
        "manifest_hash_canonicalization": MANIFEST_HASH_CANONICALIZATION,
        "zstd_dictionary_id": dictionary_id,
        "zstd_dictionary_sha256": sha256_hex(dictionary_bytes),
        "zstd_dictionary_size_bytes": len(dictionary_bytes),
        "files": {
            "dictionary": f"{ARTIFACT_STEM}.zstdict",
            "manifest": f"{ARTIFACT_STEM}.json",
        },
    }


def build_manifest(
    *,
    created_yyyymmdd: str,
    artifact_dir: str,
    manifest_sha256: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        **identity,
        "artifact_dir": artifact_dir,
        "created_yyyymmdd": created_yyyymmdd,
        "manifest_sha256": manifest_sha256,
    }


def write_artifact(
    *,
    output_root: Path,
    created_yyyymmdd: str,
    dictionary_bytes: bytes,
    identity: dict[str, Any],
    force: bool,
) -> Path:
    manifest_sha = sha256_hex(canonical_json_bytes(identity))
    artifact_dir_name = f"{created_yyyymmdd}_{manifest_sha[:8]}"
    artifact_dir = output_root / artifact_dir_name
    existing_same_hash = tuple(sorted(output_root.glob(f"*_{manifest_sha[:8]}")))
    if existing_same_hash and artifact_dir not in existing_same_hash:
        raise FileExistsError(
            "Artifact with same manifest hash already exists under a different "
            f"date: {[str(path) for path in existing_same_hash]}"
        )
    if artifact_dir.exists():
        existing_manifest_path = artifact_dir / f"{ARTIFACT_STEM}.json"
        if existing_manifest_path.exists():
            existing_manifest = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            existing_manifest_sha = existing_manifest.get("manifest_sha256")
            if existing_manifest_sha != manifest_sha:
                raise FileExistsError(
                    "Artifact directory exists with a different manifest hash: "
                    f"{artifact_dir}"
                )
    manifest = build_manifest(
        created_yyyymmdd=created_yyyymmdd,
        artifact_dir=artifact_dir_name,
        manifest_sha256=manifest_sha,
        identity=identity,
    )

    if artifact_dir.exists():
        if not force:
            raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / f"{ARTIFACT_STEM}.zstdict").write_bytes(dictionary_bytes)
    (artifact_dir / f"{ARTIFACT_STEM}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the production PreparedMol zstd dictionary artifact."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "tmp/grimace-prepared-mol-zstd",
        help="Directory under which the versioned artifact directory is written.",
    )
    parser.add_argument(
        "--created-date",
        default=datetime.now(timezone.utc).strftime("%Y%m%d"),
        help="Artifact creation date used in the output directory name, YYYYMMDD.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the computed artifact directory if it already exists.",
    )
    return parser.parse_args(argv)


def validate_created_date(value: str) -> None:
    if re.fullmatch(r"[0-9]{8}", value) is None:
        raise ValueError(f"--created-date must be YYYYMMDD, got {value!r}")
    datetime.strptime(value, "%Y%m%d")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    validate_created_date(args.created_date)
    fixture_path = (ROOT / FIXTURE_RELATIVE_PATH).resolve()
    output_root = args.output_root.expanduser().resolve()

    load_runtime_dependencies()
    require_expected_environment(fixture_path)
    corpus = build_candidates(fixture_path)
    training_identity = build_training_identity(corpus, fixture_path)
    training_identity_sha = sha256_hex(canonical_json_bytes(training_identity))
    dict_id = derive_dictionary_id(
        training_identity_sha,
        existing_ids=existing_shipped_dictionary_ids(),
    )
    dictionary_bytes = train_dictionary(corpus, dict_id)
    identity = artifact_identity(
        training_identity=training_identity,
        training_identity_sha256=training_identity_sha,
        dictionary_bytes=dictionary_bytes,
        dictionary_id=dict_id,
    )
    artifact_dir = write_artifact(
        output_root=output_root,
        created_yyyymmdd=args.created_date,
        dictionary_bytes=dictionary_bytes,
        identity=identity,
        force=args.force,
    )

    print(f"Wrote {artifact_dir}")
    print(f"Dictionary ID: {dict_id}")
    print(f"Samples: {len(corpus.selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
