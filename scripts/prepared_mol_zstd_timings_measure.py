#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import prepared_mol_zstd_dictionary_generate as generator


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "prepared-mol-zstd-timings.tsv"
DEFAULT_LEVELS = (1, 3, 6, 9, 12, 15, 19)
DEFAULT_SAMPLE_SEED = 20260531

zstd: Any | None = None
Chem: Any | None = None
RDLogger: Any | None = None
grimace: Any | None = None


@dataclass(frozen=True, slots=True)
class TimingRow:
    sample_count: int
    sample_policy: str
    sample_seed: int
    sample_source_rows_sha256: str
    sample_cids_sha256: str
    raw_bytes: int
    level: int
    mode: str
    compressed_bytes: int
    compression_ratio: float
    compression_mean_s: float
    compression_std_s: float
    decompression_mean_s: float
    decompression_std_s: float

    @classmethod
    def fieldnames(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())


@dataclass(frozen=True, slots=True)
class Sample:
    source_row: int
    cid: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class ParseableRow:
    source_row: int
    cid: str
    smiles: str


@dataclass(frozen=True, slots=True)
class SampleBatch:
    samples: tuple[Sample, ...]

    @property
    def payloads(self) -> tuple[bytes, ...]:
        return tuple(sample.payload for sample in self.samples)

    @property
    def source_rows_sha256(self) -> str:
        return generator.digest_text_lines(
            tuple(str(sample.source_row) for sample in self.samples),
        )

    @property
    def cids_sha256(self) -> str:
        return generator.digest_text_lines(tuple(sample.cid for sample in self.samples))


def _runtime_trials(fn, *, repeats: int) -> list[float]:
    fn()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def load_runtime_dependencies() -> None:
    global zstd, Chem, RDLogger, grimace
    if zstd is not None:
        return

    import grimace as loaded_grimace
    import zstandard as loaded_zstd
    from rdkit import Chem as loaded_Chem
    from rdkit import RDLogger as loaded_RDLogger

    zstd = loaded_zstd
    Chem = loaded_Chem
    RDLogger = loaded_RDLogger
    grimace = loaded_grimace


def _artifact_dir() -> Path:
    manifests = tuple(
        sorted(
            (generator.ROOT / generator.PACKAGE_DICTIONARY_ROOT).glob(
                f"*/{generator.ARTIFACT_STEM}.json",
            ),
        ),
    )
    if len(manifests) != 1:
        raise RuntimeError(
            "Expected exactly one shipped default_v1 dictionary manifest, "
            f"got {len(manifests)}"
        )
    return manifests[0].parent


def _dictionary() -> zstd.ZstdCompressionDict:
    assert zstd is not None
    artifact_dir = _artifact_dir()
    manifest = json.loads(
        (artifact_dir / f"{generator.ARTIFACT_STEM}.json").read_text(
            encoding="utf-8",
        ),
    )
    dictionary = zstd.ZstdCompressionDict(
        (artifact_dir / manifest["files"]["dictionary"]).read_bytes(),
    )
    if dictionary.dict_id() != manifest["zstd_dictionary_id"]:
        raise RuntimeError("Dictionary ID does not match shipped manifest")
    return dictionary


def _prepared_sample(limit: int, *, seed: int) -> SampleBatch:
    assert Chem is not None
    assert RDLogger is not None
    assert grimace is not None

    candidates: list[ParseableRow] = []
    fixture_path = generator.ROOT / generator.FIXTURE_RELATIVE_PATH
    RDLogger.DisableLog("rdApp.*")
    try:
        with gzip.open(fixture_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=2):
                mol = Chem.MolFromSmiles(row["SMILES"])
                if mol is None:
                    continue
                candidates.append(
                    ParseableRow(
                        source_row=row_number,
                        cid=row["CID"],
                        smiles=row["SMILES"],
                    ),
                )
    finally:
        RDLogger.EnableLog("rdApp.*")
    if len(candidates) < limit:
        raise RuntimeError(
            f"Fixture yielded {len(candidates)} prepared payloads, "
            f"fewer than requested sample count {limit}"
        )
    rows = sorted(
        random.Random(seed).sample(candidates, limit),
        key=lambda row: row.source_row,
    )
    samples: list[Sample] = []
    RDLogger.DisableLog("rdApp.*")
    try:
        for row in rows:
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None:
                raise RuntimeError(f"Selected parseable row no longer parses: {row.cid}")
            samples.append(
                Sample(
                    source_row=row.source_row,
                    cid=row.cid,
                    payload=grimace.PrepareMol(
                        mol,
                        **generator.WRITER_OPTIONS,
                    ).to_bytes(),
                ),
            )
    finally:
        RDLogger.EnableLog("rdApp.*")
    return SampleBatch(tuple(samples))


def _compress_payloads(
    payloads: tuple[bytes, ...],
    *,
    compressor: zstd.ZstdCompressor,
) -> tuple[bytes, ...]:
    return tuple(compressor.compress(payload) for payload in payloads)


def _decompress_payloads(
    payloads: tuple[bytes, ...],
    *,
    decompressor: zstd.ZstdDecompressor,
) -> tuple[bytes, ...]:
    return tuple(decompressor.decompress(payload) for payload in payloads)


def _timing_row(
    payloads: tuple[bytes, ...],
    *,
    raw_bytes: int,
    sample_seed: int,
    sample_source_rows_sha256: str,
    sample_cids_sha256: str,
    level: int,
    mode: str,
    repeats: int,
    dictionary: zstd.ZstdCompressionDict | None,
) -> TimingRow:
    compressor_kwargs = {
        "level": level,
        "write_checksum": True,
        "write_content_size": True,
    }
    decompressor_kwargs = {}
    if dictionary is not None:
        compressor_kwargs["dict_data"] = dictionary
        decompressor_kwargs["dict_data"] = dictionary

    compressor = zstd.ZstdCompressor(**compressor_kwargs)
    compressed = _compress_payloads(payloads, compressor=compressor)
    compressed_bytes = sum(len(payload) for payload in compressed)
    decompressor = zstd.ZstdDecompressor(**decompressor_kwargs)
    if _decompress_payloads(compressed, decompressor=decompressor) != payloads:
        raise RuntimeError(f"{mode} level {level} did not round-trip")

    compression_times = _runtime_trials(
        lambda: _compress_payloads(payloads, compressor=compressor),
        repeats=repeats,
    )
    decompression_times = _runtime_trials(
        lambda: _decompress_payloads(compressed, decompressor=decompressor),
        repeats=repeats,
    )
    return TimingRow(
        sample_count=len(payloads),
        sample_policy="random-parseable-preparable-v1",
        sample_seed=sample_seed,
        sample_source_rows_sha256=sample_source_rows_sha256,
        sample_cids_sha256=sample_cids_sha256,
        raw_bytes=raw_bytes,
        level=level,
        mode=mode,
        compressed_bytes=compressed_bytes,
        compression_ratio=compressed_bytes / raw_bytes,
        compression_mean_s=statistics.mean(compression_times),
        compression_std_s=statistics.stdev(compression_times),
        decompression_mean_s=statistics.mean(decompression_times),
        decompression_std_s=statistics.stdev(decompression_times),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-molecule PreparedMol zstd compression timings.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=DEFAULT_LEVELS,
        help="zstd compression levels to measure.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    if args.sample_seed < 0:
        raise SystemExit("--sample-seed must be non-negative")
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    if any(level < 1 or level > 22 for level in args.levels):
        raise SystemExit("--levels must be in zstd range 1..22")

    load_runtime_dependencies()
    sample = _prepared_sample(args.sample_count, seed=args.sample_seed)
    payloads = sample.payloads
    raw_bytes = sum(len(payload) for payload in payloads)
    sample_source_rows_sha256 = sample.source_rows_sha256
    sample_cids_sha256 = sample.cids_sha256
    dictionary = _dictionary()

    rows: list[TimingRow] = []
    for level in args.levels:
        for mode, mode_dictionary in (
            ("no-dictionary", None),
            ("dictionary", dictionary),
        ):
            row = _timing_row(
                payloads,
                raw_bytes=raw_bytes,
                sample_seed=args.sample_seed,
                sample_source_rows_sha256=sample_source_rows_sha256,
                sample_cids_sha256=sample_cids_sha256,
                level=level,
                mode=mode,
                repeats=args.repeats,
                dictionary=mode_dictionary,
            )
            rows.append(row)
            print(
                f"level={level} mode={mode} "
                f"ratio={row.compression_ratio:.6f} "
                f"compress={row.compression_mean_s:.6f}s "
                f"decompress={row.decompression_mean_s:.6f}s"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TimingRow.fieldnames(),
            dialect="excel-tab",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
