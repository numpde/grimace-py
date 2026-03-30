from __future__ import annotations

import math
import os
import statistics
import time
import unittest
from dataclasses import dataclass

from rdkit import Chem, rdBase

import grimace
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class TimingCase:
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool


def _runtime_trials(fn, *, repeats: int = 7) -> list[float]:
    fn()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1_000:.1f} ms"


def _format_bold_ms(seconds: float) -> str:
    return f"**{seconds * 1_000:.1f}** ms"


def _format_draws(mean: float, stdev: float) -> str:
    return f"{mean:.1f} ± {stdev:.1f}"


def _format_duration(mean: float, stdev: float) -> str:
    return f"{mean * 1_000:.1f} ± {stdev * 1_000:.1f} ms"


def _format_bold_duration(mean: float, stdev: float) -> str:
    return f"**{mean * 1_000:.1f}** ± {stdev * 1_000:.1f} ms"


@unittest.skipUnless(
    os.environ.get("RUN_PERF_TESTS") == "1",
    "set RUN_PERF_TESTS=1 to run performance checks",
)
class ReadmeTimingPerfTests(unittest.TestCase):
    CASES = (
        TimingCase(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        TimingCase(
            smiles="C1COCCC12CO2",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        TimingCase(
            smiles="CN1CCC[C@H]1C2=CN=CC=C2",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        TimingCase(
            smiles="C/C(=N\\\\OC(=O)NC)/SC",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        TimingCase(
            smiles="C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        TimingCase(
            smiles="C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
    )

    def test_generate_readme_timing_table(self) -> None:
        rows: list[str] = []
        rows.append(
            "| Molecule | Atoms | Support | Grimace enum (all roots) | "
            "Decoder enum (all roots) | RDKit to 1/2 support | "
            "RDKit to full support |"
        )
        rows.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")

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
                    lambda: self._enumerate_all_roots_with_decoder(mol, case)
                )
                self.assertEqual(support, self._enumerate_all_roots_with_decoder(mol, case))

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
                    "| "
                    f"`{canonical_smiles}` | {mol.GetNumAtoms()} | {support_size} | "
                    f"{_format_bold_duration(statistics.mean(enum_times), statistics.stdev(enum_times))} | "
                    f"{_format_bold_duration(statistics.mean(decoder_times), statistics.stdev(decoder_times))} | "
                    f"{_format_bold_duration(statistics.mean(half_times), statistics.stdev(half_times))} "
                    f"({_format_draws(statistics.mean(half_draws), statistics.stdev(half_draws))} draws) | "
                    f"{_format_bold_duration(statistics.mean(full_times), statistics.stdev(full_times))} "
                    f"({_format_draws(statistics.mean(full_draws), statistics.stdev(full_draws))} draws) |"
                )

        print()
        print("README timing table")
        for row in rows:
            print(row)

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

    def _enumerate_all_roots_with_decoder(self, mol: Chem.Mol, case: TimingCase) -> set[str]:
        outputs: set[str] = set()

        for root_idx in range(mol.GetNumAtoms()):
            root_decoder = grimace.MolToSmilesDecoder(
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
                for token in reversed(decoder.next_tokens):
                    next_state = decoder.copy()
                    next_state.advance(token)
                    stack.append(next_state)

        return outputs


if __name__ == "__main__":
    unittest.main()
