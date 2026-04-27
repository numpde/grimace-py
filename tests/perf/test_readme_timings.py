from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import statistics
import time
import unittest
from dataclasses import asdict, dataclass

from rdkit import Chem, rdBase

import grimace
from tests.perf._history import append_history_record, current_run_metadata
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class TimingCase:
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool


@dataclass(frozen=True, slots=True)
class TimingRow:
    surface: str
    molecule: str
    atoms: int
    support: int
    enum_mean_s: float
    enum_std_s: float
    decoder_per_root_mean_s: float
    decoder_per_root_std_s: float
    determinized_decoder_per_root_mean_s: float
    determinized_decoder_per_root_std_s: float
    decoder_merged_mean_s: float
    decoder_merged_std_s: float
    determinized_decoder_merged_mean_s: float
    determinized_decoder_merged_std_s: float
    rdkit_half_mean_s: float
    rdkit_half_std_s: float
    rdkit_half_draw_mean: float
    rdkit_half_draw_std: float
    rdkit_full_mean_s: float
    rdkit_full_std_s: float
    rdkit_full_draw_mean: float
    rdkit_full_draw_std: float

    @classmethod
    def tsv_fieldnames(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())


def _runtime_trials(fn, *, repeats: int = 7) -> list[float]:
    fn()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def _format_draws(mean: float, stdev: float) -> str:
    return f"{mean:.1f} ± {stdev:.1f}"


def _format_bold_duration(mean: float, stdev: float) -> str:
    return f"**{mean * 1_000:.1f}** ± {stdev * 1_000:.1f} ms"


TIMING_MOLECULES = (
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1COCCC12CO2",
    "CN1CCC[C@H]1C2=CN=CC=C2",
    "C/C(=N\\\\OC(=O)NC)/SC",
    "C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O",
    "C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
    "C1[C@H]([C@@H]2[C@H](O1)[C@H](CO2)O[N+](=O)[O-])O[N+](=O)[O-]",
    "C[C@@H]([C@@H]1[C@@]2([C@@H](CC1)/C(=C/C=C1/C[C@@H](O)CCC1=C)CCC2)C)",
    r"CC\1=C(C2=C(/C1=C\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O",
)


@unittest.skipUnless(
    os.environ.get("RUN_PERF_TESTS") == "1",
    "set RUN_PERF_TESTS=1 to run performance checks",
)
class ReadmeTimingPerfTests(unittest.TestCase):
    OUTPUT_TSV_PATH = Path(__file__).resolve().parents[2] / "docs" / "timings.tsv"
    OUTPUT_MD_PATH = Path(__file__).resolve().parents[2] / "docs" / "timings.md"
    HISTORY_KIND = "timings_snapshot"
    CASES = tuple(
        TimingCase(smiles=smiles, rooted_at_atom=0, isomeric_smiles=isomeric_smiles)
        for isomeric_smiles in (False, True)
        for smiles in TIMING_MOLECULES
    )

    def test_generate_readme_timing_table(self) -> None:
        rows = self._measure_rows()
        self._write_tsv(rows)
        document = self._render_document_from_tsv()
        self.OUTPUT_MD_PATH.write_text(document, encoding="utf-8")
        self._append_history_snapshot(rows)

        print()
        print(f"Wrote timing data: {self.OUTPUT_TSV_PATH}")
        print(f"Wrote timing document: {self.OUTPUT_MD_PATH}")
        for row in document.splitlines():
            print(row)

    def _measure_rows(self) -> list[TimingRow]:
        rows: list[TimingRow] = []
        for case in self.CASES:
            mol = parse_smiles(case.smiles)
            canonical_smiles = Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=True,
                isomericSmiles=True,
            )
            with self.subTest(case=canonical_smiles):
                support = self._enumerate_all_roots(mol, case)
                support_size = len(support)
                self.assertGreater(support_size, 0)

                enum_times = _runtime_trials(
                    lambda: self._enumerate_all_roots(mol, case)
                )
                decoder_times = _runtime_trials(
                    lambda: self._enumerate_all_roots_with_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDecoder,
                    )
                )
                determinized_decoder_times = _runtime_trials(
                    lambda: self._enumerate_all_roots_with_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDeterminizedDecoder,
                    )
                )
                merged_decoder_times = _runtime_trials(
                    lambda: self._enumerate_all_roots_with_merged_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDecoder,
                    )
                )
                merged_determinized_decoder_times = _runtime_trials(
                    lambda: self._enumerate_all_roots_with_merged_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDeterminizedDecoder,
                    )
                )
                self.assertEqual(
                    support,
                    self._enumerate_all_roots_with_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDecoder,
                    ),
                )
                self.assertEqual(
                    support,
                    self._enumerate_all_roots_with_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDeterminizedDecoder,
                    ),
                )
                self.assertEqual(
                    support,
                    self._enumerate_all_roots_with_merged_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDecoder,
                    ),
                )
                self.assertEqual(
                    support,
                    self._enumerate_all_roots_with_merged_decoder(
                        mol,
                        case,
                        decoder_cls=grimace.MolToSmilesDeterminizedDecoder,
                    ),
                )

                half_target = math.ceil(support_size / 2)
                full_target = support_size
                max_draws = max(5_000, support_size * mol.GetNumAtoms() * 20)

                half_draws, half_times = self._sampling_trials(
                    mol,
                    case,
                    target_count=half_target,
                    max_draws=max_draws,
                )
                full_draws, full_times = self._sampling_trials(
                    mol,
                    case,
                    target_count=full_target,
                    max_draws=max_draws,
                )

                rows.append(
                    TimingRow(
                        surface=(
                            "stereo" if case.isomeric_smiles else "non-stereo"
                        ),
                        molecule=canonical_smiles,
                        atoms=mol.GetNumAtoms(),
                        support=support_size,
                        enum_mean_s=statistics.mean(enum_times),
                        enum_std_s=statistics.stdev(enum_times),
                        decoder_per_root_mean_s=statistics.mean(decoder_times),
                        decoder_per_root_std_s=statistics.stdev(decoder_times),
                        determinized_decoder_per_root_mean_s=statistics.mean(
                            determinized_decoder_times
                        ),
                        determinized_decoder_per_root_std_s=statistics.stdev(
                            determinized_decoder_times
                        ),
                        decoder_merged_mean_s=statistics.mean(merged_decoder_times),
                        decoder_merged_std_s=statistics.stdev(merged_decoder_times),
                        determinized_decoder_merged_mean_s=statistics.mean(
                            merged_determinized_decoder_times
                        ),
                        determinized_decoder_merged_std_s=statistics.stdev(
                            merged_determinized_decoder_times
                        ),
                        rdkit_half_mean_s=statistics.mean(half_times),
                        rdkit_half_std_s=statistics.stdev(half_times),
                        rdkit_half_draw_mean=statistics.mean(half_draws),
                        rdkit_half_draw_std=statistics.stdev(half_draws),
                        rdkit_full_mean_s=statistics.mean(full_times),
                        rdkit_full_std_s=statistics.stdev(full_times),
                        rdkit_full_draw_mean=statistics.mean(full_draws),
                        rdkit_full_draw_std=statistics.stdev(full_draws),
                    )
                )
        return rows

    def _write_tsv(self, rows: list[TimingRow]) -> None:
        with self.OUTPUT_TSV_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=TimingRow.tsv_fieldnames(),
                dialect="excel-tab",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    def _append_history_snapshot(self, rows: list[TimingRow]) -> None:
        append_history_record(
            {
                "kind": self.HISTORY_KIND,
                **current_run_metadata(),
                "benchmark": "tests.perf.test_readme_timings",
                "timings_tsv": str(self.OUTPUT_TSV_PATH.relative_to(self.OUTPUT_TSV_PATH.parents[1])),
                "timings_md": str(self.OUTPUT_MD_PATH.relative_to(self.OUTPUT_MD_PATH.parents[1])),
                "rows": [asdict(row) for row in rows],
            }
        )

    def _render_document_from_tsv(self) -> str:
        header = (
            "| Molecule | Atoms | Support | Grimace enum (per-root union) | "
            "Decoder enum (branch-preserving, per-root) | "
            "Decoder enum (determinized, per-root) | "
            "Decoder enum (branch-preserving, merged) | "
            "Decoder enum (determinized, merged) | RDKit to 1/2 support | "
            "RDKit to full support |"
        )
        separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"

        rendered_rows = {"non-stereo": [], "stereo": []}

        with self.OUTPUT_TSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, dialect="excel-tab")
            for row in reader:
                rendered_rows[row["surface"]].append(
                    "| "
                    f"`{row['molecule']}` | {row['atoms']} | {row['support']} | "
                    f"{_format_bold_duration(float(row['enum_mean_s']), float(row['enum_std_s']))} | "
                    f"{_format_bold_duration(float(row['decoder_per_root_mean_s']), float(row['decoder_per_root_std_s']))} | "
                    f"{_format_bold_duration(float(row['determinized_decoder_per_root_mean_s']), float(row['determinized_decoder_per_root_std_s']))} | "
                    f"{_format_bold_duration(float(row['decoder_merged_mean_s']), float(row['decoder_merged_std_s']))} | "
                    f"{_format_bold_duration(float(row['determinized_decoder_merged_mean_s']), float(row['determinized_decoder_merged_std_s']))} | "
                    f"{_format_bold_duration(float(row['rdkit_half_mean_s']), float(row['rdkit_half_std_s']))} "
                    f"({_format_draws(float(row['rdkit_half_draw_mean']), float(row['rdkit_half_draw_std']))} draws) | "
                    f"{_format_bold_duration(float(row['rdkit_full_mean_s']), float(row['rdkit_full_std_s']))} "
                    f"({_format_draws(float(row['rdkit_full_draw_mean']), float(row['rdkit_full_draw_std']))} draws) |"
                )

        lines = [
            "# Timings",
            "",
            "This file is generated by `tests/perf/test_readme_timings.py`.",
            "The benchmark first writes `docs/timings.tsv`, then renders this",
            "Markdown table from that TSV.",
            "",
            "Example timings from the opt-in performance benchmark, measured in release mode",
            "on one development machine. Treat them as indicative, not as a portability,",
            "stability, or universality guarantee.",
            "",
            "- This is a small curated benchmark: 9 molecules, 2 writer modes, and",
            "  7 timing repeats per row.",
            "- This is not a workload study and not an exact-versus-exact comparison.",
            "- `Support`: the size of the exact rooted SMILES support across all root atoms.",
            "- `Grimace enum (per-root union)`: union of",
            "  `MolToSmilesEnum(..., rootedAtAtom=root_idx, canonical=False, doRandom=True, isomericSmiles=<table mode>)`",
            "  over every root atom.",
            "- The direct public `MolToSmilesEnum(..., rootedAtAtom=-1, ...)` path is",
            "  not timed in this column and can differ materially from the explicit",
            "  per-root union shown here.",
            "- `Decoder enum (branch-preserving, per-root)`: exhaustive traversal of",
            "  `MolToSmilesDecoder(..., rootedAtAtom=root_idx, canonical=False, doRandom=True, isomericSmiles=<table mode>).next_choices`",
            "  over every root atom, then unioned.",
            "- `Decoder enum (determinized, per-root)`: the same per-root traversal,",
            "  using `MolToSmilesDeterminizedDecoder(...)`.",
            "- `Decoder enum (branch-preserving, merged)`: exhaustive traversal of",
            "  `MolToSmilesDecoder(..., rootedAtAtom=-1, canonical=False, doRandom=True, isomericSmiles=<table mode>).next_choices`.",
            "- `Decoder enum (determinized, merged)`: the same merged traversal,",
            "  using `MolToSmilesDeterminizedDecoder(...)`.",
            "- `RDKit to 1/2 support`: repeated RDKit `MolToSmiles(..., canonical=False,",
            "  doRandom=True, rootedAtAtom=root_idx, isomericSmiles=<table mode>)` draws",
            "  across all roots until half of the exact support has been seen.",
            "- `RDKit to full support`: the same sampling process until the full exact",
            "  support has been seen.",
            "- `Non-stereo` means `isomericSmiles=False`.",
            "- `Stereo` means `isomericSmiles=True`.",
            "- All timing columns are shown as `time mean ± std`.",
            "- The two RDKit columns also show `(draw mean ± std)` over repeated seeded",
            "  trials.",
            "- The published table does not directly rank every public exact path:",
            "  it times `Grimace enum (per-root union)` rather than the direct",
            "  public `MolToSmilesEnum(..., rootedAtAtom=-1)` path, and some",
            "  merged decoder rows are numerically lower than that per-root",
            "  union column.",
            "- The merged decoder rows expose the public all-roots decoder path directly,",
            "  so they can diverge substantially from the explicit per-root rows.",
            "- Read the RDKit comparison as 'faster on this benchmark against this",
            "  sampling baseline', not as a general claim about every molecule or",
            "  every SMILES-writing workload.",
            "- The determinized decoder can reduce exhaustive decoder cost on some",
            "  molecules, but direct exact enumeration is still faster on these cases.",
            "",
            "## Non-stereo (`isomericSmiles=False`)",
            "",
            header,
            separator,
            *rendered_rows["non-stereo"],
            "",
            "## Stereo (`isomericSmiles=True`)",
            "",
            header,
            separator,
            *rendered_rows["stereo"],
            "",
        ]
        return "\n".join(lines)

    def _sampling_trials(
        self,
        mol: Chem.Mol,
        case: TimingCase,
        *,
        target_count: int,
        max_draws: int,
        trials: int = 7,
    ) -> tuple[list[int], list[float]]:
        draws_taken: list[int] = []
        elapsed_times: list[float] = []

        for seed in range(trials):
            draws, elapsed = self._sample_until_support(
                mol,
                case,
                target_count=target_count,
                max_draws=max_draws,
                seed=seed,
            )
            self.assertGreaterEqual(draws, target_count)
            self.assertLessEqual(draws, max_draws)
            draws_taken.append(draws)
            elapsed_times.append(elapsed)

        return draws_taken, elapsed_times

    def _sample_until_support(
        self,
        mol: Chem.Mol,
        case: TimingCase,
        *,
        target_count: int,
        max_draws: int,
        seed: int,
    ) -> tuple[int, float]:
        rdBase.SeedRandomNumberGenerator(seed)
        seen: set[str] = set()
        roots = list(range(mol.GetNumAtoms()))
        start = time.perf_counter()

        for draw_idx in range(1, max_draws + 1):
            sampled = Chem.MolToSmiles(
                Chem.Mol(mol),
                rootedAtAtom=roots[(draw_idx - 1) % len(roots)],
                canonical=False,
                doRandom=True,
                isomericSmiles=case.isomeric_smiles,
            )
            seen.add(sampled)
            if len(seen) >= target_count:
                return draw_idx, time.perf_counter() - start

        self.fail(
            f"RDKit sampling did not reach target_count={target_count} for {case.smiles} "
            f"within max_draws={max_draws}"
        )

    def _enumerate_all_roots(self, mol: Chem.Mol, case: TimingCase) -> set[str]:
        return {
            smiles
            for root_idx in range(mol.GetNumAtoms())
            for smiles in grimace.MolToSmilesEnum(
                mol,
                rootedAtAtom=root_idx,
                isomericSmiles=case.isomeric_smiles,
                canonical=False,
                doRandom=True,
            )
        }

    def _enumerate_all_roots_with_decoder(
        self,
        mol: Chem.Mol,
        case: TimingCase,
        *,
        decoder_cls,
    ) -> set[str]:
        outputs: set[str] = set()

        for root_idx in range(mol.GetNumAtoms()):
            root_decoder = decoder_cls(
                mol,
                rootedAtAtom=root_idx,
                isomericSmiles=case.isomeric_smiles,
                canonical=False,
                doRandom=True,
            )
            stack = [root_decoder]

            while stack:
                decoder = stack.pop()
                if decoder.is_terminal:
                    outputs.add(decoder.prefix)
                    continue
                for choice in reversed(decoder.next_choices):
                    stack.append(choice.next_state)

        return outputs

    def _enumerate_all_roots_with_merged_decoder(
        self,
        mol: Chem.Mol,
        case: TimingCase,
        *,
        decoder_cls,
    ) -> set[str]:
        outputs: set[str] = set()
        stack = [
            decoder_cls(
                mol,
                rootedAtAtom=-1,
                isomericSmiles=case.isomeric_smiles,
                canonical=False,
                doRandom=True,
            )
        ]

        while stack:
            decoder = stack.pop()
            if decoder.is_terminal:
                outputs.add(decoder.prefix)
                continue
            for choice in reversed(decoder.next_choices):
                stack.append(choice.next_state)

        return outputs


if __name__ == "__main__":
    unittest.main()
