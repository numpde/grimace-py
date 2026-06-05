from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import unittest
from unittest import mock

import grimace
import grimace._sampling as _sampling
import grimace._runtime_walks as _runtime_walks
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    SAMPLING_MODE_PAIRS,
    prepared_writer_kwargs,
    public_enum_support,
    supported_public_kwargs,
)


def _step_choice_pairs(step: object) -> tuple[tuple[str, int], ...]:
    return tuple(zip(step.choice_tokens, step.choice_branch_counts, strict=True))


def _valid_sample_step() -> grimace.SmilesSampleStep:
    return grimace.SmilesSampleStep(("C",), (1,), 0, "C")


def _valid_sample() -> grimace.SmilesSample:
    return grimace.SmilesSample(
        ("C",),
        "C",
        "determinized",
        "uniform_token",
        (_valid_sample_step(),),
    )


class PublicSamplingTests(unittest.TestCase):
    def _assert_all_modes_return_legal_sample(
        self,
        mol: object,
        *,
        kwargs: Mapping[str, object],
        seed: int,
    ) -> None:
        support = public_enum_support(mol, **kwargs)

        for decoder_view, sampling_mode in SAMPLING_MODE_PAIRS:
            with self.subTest(
                decoder_view=decoder_view,
                sampling_mode=sampling_mode,
            ):
                sample = grimace.MolToSmilesSample(
                    mol,
                    seed=seed,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                    **kwargs,
                )

                self._assert_sample_invariants(
                    sample,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                )
                self.assertIn(sample.smiles, support)

    def _assert_sample_invariants(
        self,
        sample: object,
        *,
        decoder_view: str,
        sampling_mode: str,
    ) -> None:
        self.assertIsInstance(sample, grimace.SmilesSample)
        self.assertIsInstance(sample.tokens, tuple)
        self.assertTrue(all(isinstance(token, str) for token in sample.tokens))
        self.assertEqual("".join(sample.tokens), sample.smiles)
        self.assertEqual(decoder_view, sample.decoder_view)
        self.assertEqual(sampling_mode, sample.sampling_mode)
        self.assertIsInstance(sample.steps, tuple)
        self.assertEqual(len(sample.tokens), len(sample.steps))

        for selected_token, step in zip(sample.tokens, sample.steps, strict=True):
            with self.subTest(prefix_token=selected_token):
                self.assertIsInstance(step, grimace.SmilesSampleStep)
                self.assertIsInstance(step.choice_tokens, tuple)
                self.assertIsInstance(step.choice_branch_counts, tuple)
                self.assertEqual(
                    len(step.choice_tokens),
                    len(step.choice_branch_counts),
                )
                self.assertGreater(len(step.choice_tokens), 0)
                self.assertEqual(len(step.choice_tokens), len(set(step.choice_tokens)))
                self.assertTrue(
                    all(isinstance(token, str) for token in step.choice_tokens)
                )
                self.assertTrue(
                    all(
                        type(branch_count) is int and branch_count > 0
                        for branch_count in step.choice_branch_counts
                    )
                )
                self.assertIs(type(step.selected_index), int)
                self.assertGreaterEqual(step.selected_index, 0)
                self.assertLess(step.selected_index, len(step.choice_tokens))
                self.assertEqual(
                    step.choice_tokens[step.selected_index],
                    step.selected_token,
                )
                self.assertEqual(selected_token, step.selected_token)

    def test_public_sampling_exports_result_surface(self) -> None:
        self.assertIn("MolToSmilesSample", grimace.__all__)
        self.assertIn("SmilesSample", grimace.__all__)
        self.assertIn("SmilesSampleStep", grimace.__all__)
        self.assertTrue(callable(grimace.MolToSmilesSample))
        self.assertTrue(callable(grimace.SmilesSample))
        self.assertTrue(callable(grimace.SmilesSampleStep))

    def test_sampling_mode_pairs_match_public_contract(self) -> None:
        self.assertEqual(
            frozenset(SAMPLING_MODE_PAIRS),
            frozenset(_sampling._SAMPLING_WALKERS),
        )

    def test_sample_records_reject_mutable_payload_containers(self) -> None:
        step = _valid_sample_step()

        with self.assertRaisesRegex(TypeError, "choice_tokens must be a tuple"):
            grimace.SmilesSampleStep(["C"], (1,), 0, "C")
        with self.assertRaisesRegex(TypeError, "choice_branch_counts must be a tuple"):
            grimace.SmilesSampleStep(("C",), [1], 0, "C")
        with self.assertRaisesRegex(TypeError, "sample tokens must be a tuple"):
            grimace.SmilesSample(
                ["C"],
                "C",
                "determinized",
                "uniform_token",
                (step,),
            )
        with self.assertRaisesRegex(TypeError, "sample steps must be a tuple"):
            grimace.SmilesSample(
                ("C",),
                "C",
                "determinized",
                "uniform_token",
                [step],
            )

    def test_sample_step_rejects_nonstring_choice_tokens_first(self) -> None:
        with self.assertRaisesRegex(TypeError, "choice tokens must be strings"):
            grimace.SmilesSampleStep(([],), (1,), 0, "C")

    def test_sample_records_reject_invalid_scalars(self) -> None:
        step = _valid_sample_step()
        sample = _valid_sample()
        cases = (
            (
                lambda: replace(step, choice_branch_counts=(True,)),
                ValueError,
                "branch counts",
            ),
            (
                lambda: replace(step, selected_index=True),
                TypeError,
                "selected_index",
            ),
            (
                lambda: replace(step, selected_token=[]),
                TypeError,
                "selected_token",
            ),
            (
                lambda: replace(sample, tokens=([],)),
                TypeError,
                "sample tokens",
            ),
            (
                lambda: replace(sample, smiles=[]),
                TypeError,
                "sample smiles",
            ),
            (
                lambda: replace(sample, steps=(object(),)),
                TypeError,
                "SmilesSampleStep",
            ),
        )

        for make_record, exception_type, expected_regex in cases:
            with self.subTest(expected_regex=expected_regex):
                with self.assertRaisesRegex(exception_type, expected_regex):
                    make_record()

    def test_sample_records_reject_invalid_relationships(self) -> None:
        step = _valid_sample_step()
        sample = _valid_sample()
        cases = (
            (
                lambda: replace(step, choice_branch_counts=(1, 1)),
                "lengths differ",
            ),
            (
                lambda: replace(step, choice_tokens=(), choice_branch_counts=()),
                "at least one choice",
            ),
            (
                lambda: replace(
                    step,
                    choice_tokens=("C", "C"),
                    choice_branch_counts=(1, 1),
                ),
                "must be unique",
            ),
            (
                lambda: replace(step, selected_index=1),
                "out of range",
            ),
            (
                lambda: replace(step, selected_token="N"),
                "does not match selected_index",
            ),
            (
                lambda: replace(sample, smiles="N"),
                "must equal joined tokens",
            ),
            (
                lambda: replace(sample, steps=()),
                "step count",
            ),
            (
                lambda: replace(sample, tokens=("N",), smiles="N"),
                "does not match selected step token",
            ),
            (
                lambda: replace(
                    sample,
                    decoder_view="branch_preserving",
                    sampling_mode="uniform_token",
                ),
                "decoder_view/sampling_mode",
            ),
        )

        for make_record, expected_regex in cases:
            with self.subTest(expected_regex=expected_regex):
                with self.assertRaisesRegex(ValueError, expected_regex):
                    make_record()

    def test_sampling_modes_return_legal_smiles_with_step_context(self) -> None:
        mol = parse_smiles("CCO")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)

        self._assert_all_modes_return_legal_sample(mol, kwargs=kwargs, seed=17)

    def test_sampling_accepts_rooted_writer_surface_flags(self) -> None:
        mol = parse_smiles("C1CC1O")
        kwargs = supported_public_kwargs(
            rootedAtAtom=0,
            isomericSmiles=False,
            kekuleSmiles=True,
            allBondsExplicit=True,
            allHsExplicit=True,
        )

        self._assert_all_modes_return_legal_sample(mol, kwargs=kwargs, seed=5)

    def test_sampling_is_reproducible_for_same_seed_and_mode(self) -> None:
        mol = parse_smiles("F[C@H](Cl)Br")
        kwargs = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1)

        for decoder_view, sampling_mode in SAMPLING_MODE_PAIRS:
            with self.subTest(
                decoder_view=decoder_view,
                sampling_mode=sampling_mode,
            ):
                first = grimace.MolToSmilesSample(
                    mol,
                    seed=123,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                    **kwargs,
                )
                second = grimace.MolToSmilesSample(
                    mol,
                    seed=123,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                    **kwargs,
                )

                self.assertEqual(first, second)

    def test_sampling_accepts_rooted_stereo(self) -> None:
        mol = parse_smiles("F[C@H](Cl)Br")
        kwargs = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=1)

        self._assert_all_modes_return_legal_sample(mol, kwargs=kwargs, seed=31)

    def test_sampling_accepts_prepared_mol_bytes_round_trip(self) -> None:
        mol = parse_smiles("CCO.N")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        prepared = grimace.PrepareMol(mol, **prepared_writer_kwargs(kwargs))
        restored = grimace.PreparedMol.from_bytes(prepared.to_bytes())

        for decoder_view, sampling_mode in SAMPLING_MODE_PAIRS:
            with self.subTest(
                decoder_view=decoder_view,
                sampling_mode=sampling_mode,
            ):
                mol_sample = grimace.MolToSmilesSample(
                    mol,
                    seed=9,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                    **kwargs,
                )
                prepared_sample = grimace.MolToSmilesSample(
                    restored,
                    seed=9,
                    decoder_view=decoder_view,
                    sampling_mode=sampling_mode,
                    **kwargs,
                )

                self.assertEqual(mol_sample, prepared_sample)
                self.assertIn(".", mol_sample.tokens)
                separator_step = mol_sample.steps[mol_sample.tokens.index(".")]
                self.assertEqual(((".", 1),), _step_choice_pairs(separator_step))

    def test_sampling_rejects_invalid_mode_pairs(self) -> None:
        mol = parse_smiles("CCO")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0)
        valid_views = sorted({view for view, _mode in SAMPLING_MODE_PAIRS})
        valid_modes = sorted({mode for _view, mode in SAMPLING_MODE_PAIRS})

        invalid_cases = (
            (None, "uniform_token"),
            ("missing", "uniform_token"),
            ("determinized", None),
            ("determinized", "missing"),
            *(
                (decoder_view, sampling_mode)
                for decoder_view in valid_views
                for sampling_mode in valid_modes
                if (decoder_view, sampling_mode) not in SAMPLING_MODE_PAIRS
            ),
        )

        for decoder_view, sampling_mode in invalid_cases:
            with self.subTest(
                decoder_view=decoder_view,
                sampling_mode=sampling_mode,
            ):
                with self.assertRaisesRegex(ValueError, "decoder_view/sampling_mode"):
                    grimace.MolToSmilesSample(
                        mol,
                        seed=0,
                        decoder_view=decoder_view,
                        sampling_mode=sampling_mode,
                        **kwargs,
                    )

    def test_sampling_reuses_supported_runtime_option_rejection(self) -> None:
        mol = parse_smiles("CCO")
        with self.assertRaisesRegex(NotImplementedError, "canonical=False"):
            grimace.MolToSmilesSample(
                mol,
                seed=0,
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        with self.assertRaisesRegex(NotImplementedError, "doRandom=True"):
            grimace.MolToSmilesSample(
                mol,
                seed=0,
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
            )

    def test_sampling_rejects_unsupported_options_before_core_sampler_lookup(
        self,
    ) -> None:
        mol = parse_smiles("CCO")
        # Unsupported writer options are cheap public-input errors; they must
        # not depend on constructing the extension-backed RNG first.
        with mock.patch.object(_runtime_walks, "_core", object()):
            with self.assertRaisesRegex(NotImplementedError, "canonical=False"):
                grimace.MolToSmilesSample(
                    mol,
                    seed=0,
                    rootedAtAtom=0,
                    isomericSmiles=False,
                )

    def test_sampling_rejects_invalid_seed_values(self) -> None:
        mol = parse_smiles("CCO")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0)

        for seed in (-1, True, 1 << 64):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
                    grimace.MolToSmilesSample(
                        mol,
                        seed=seed,
                        decoder_view="determinized",
                        sampling_mode="uniform_token",
                        **kwargs,
                    )

    def test_sampling_rejects_invalid_seed_before_preparing_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
            grimace.MolToSmilesSample(
                object(),
                seed=-1,
                decoder_view="determinized",
                sampling_mode="uniform_token",
                **supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
            )

    def test_sampling_requires_explicit_seed(self) -> None:
        with self.assertRaises(TypeError):
            grimace.MolToSmilesSample(
                parse_smiles("CCO"),
                decoder_view="determinized",
                sampling_mode="uniform_token",
                **supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
            )


if __name__ == "__main__":
    unittest.main()
