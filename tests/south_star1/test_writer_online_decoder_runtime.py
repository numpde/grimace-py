"""Online decoder coverage for the writer-runtime route."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.writer_online_decoder_certificates import (
    writer_online_choice_result_certificate,
)
from grimace._south_star1.writer_online_decoder_certificates import (
    writer_online_eos_choice_certificate,
)
from grimace._south_star1.writer_online_decoder_certificates import (
    writer_online_text_choice_certificate,
)

from grimace._south_star1.online_decoder_api import make_branch_preserving_online_decoder
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_writer_shaped_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_online_decoder import WriterRuntimeOnlineStats
from grimace._south_star1.writer_online_decoder import WriterShapedOnlineDecoderState
from grimace._south_star1.writer_online_decoder import make_writer_shaped_online_decoder
from grimace._south_star1.writer_runtime import WriterRuntimeState
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from tests.helpers.module_boundaries import scan_module_boundaries
from tests.south_star1.helpers import cco_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER_ONLINE_DECODER_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "writer_online_decoder.py"
)


class WriterOnlineDecoderRuntimeTest(unittest.TestCase):
    def test_named_writer_shaped_decoder_factory_uses_live_runtime(self) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            include_eos=True,
        )
        support = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        initial = decoder.initial_state()
        result = initial.choices_with_stats()
        self.assertIsInstance(initial.raw_state, WriterRuntimeState)
        self.assertIsInstance(result.stats, WriterRuntimeOnlineStats)
        self.assertEqual(_reachable_eos_prefixes(initial), set(support.strings))

    def test_named_writer_shaped_decoder_preserves_runtime_options(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            runtime_options=options,
            include_eos=True,
        )
        support = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=options,
        )

        self.assertEqual(decoder.runtime_options, options)
        self.assertEqual(
            _reachable_eos_prefixes(decoder.initial_state()),
            set(support.strings),
        )

    def test_generic_online_factories_reject_writer_shaped_runtime(self) -> None:
        prepared = _prepare(cco_facts())
        for factory in (
            make_branch_preserving_online_decoder,
            make_determinized_online_decoder,
        ):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "make_writer_shaped_online_decoder",
                ):
                    factory(
                        prepared=prepared,
                        runtime_options=_writer_options(),
                    )

    def test_writer_shaped_route_never_falls_through_to_legacy_raw_state(self) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(prepared=prepared)
        invalid_state = WriterShapedOnlineDecoderState(
            prefix="",
            raw_state=object(),
            decoder=decoder,
        )

        with self.assertRaisesRegex(ValueError, "non-writer state"):
            decoder.choices_with_stats(invalid_state)

    def test_writer_online_decoder_boundary_has_no_legacy_online_imports(self) -> None:
        scan = scan_module_boundaries(
            WRITER_ONLINE_DECODER_PATH,
            banned_modules={
                "online_continuation",
                "online_decoder_api",
                "online_decoder_state",
                "online_decisions",
                "online_residual_continuation",
                "online_search_vm",
            },
            banned_imported_names={
                "_advance_writer_runtime_state_by_choice",
            },
            banned_calls={
                "_advance_writer_runtime_state_by_choice",
                "make_branch_preserving_online_decoder",
                "make_determinized_online_decoder",
                "online_branch_preserving_choice_result",
                "online_determinized_choice_result",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_writer_shaped_choices_carry_text_certificates(self) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            include_eos=False,
        )
        result = decoder.initial_state().choices_with_stats()
        self.assertGreater(len(result.choices), 0)
        for choice in result.choices:
            self.assertFalse(choice.is_eos)
            self.assertIsNotNone(choice.choice_certificate)
            assert choice.choice_certificate is not None
            self.assertEqual(choice.choice_certificate.text, choice.text)
            self.assertIsNotNone(
                choice.choice_certificate.text_projection_certificate
            )
            self.assertIsNotNone(
                choice.choice_certificate.snapshot_step_certificate
            )
            self.assertIsNotNone(choice.choice_certificate.checked_frontier_certificate)
            self.assertIsNotNone(choice.choice_certificate.count_certificate)
            self.assertEqual(
                choice.choice_certificate.snapshot_step_certificate.advanced_snapshot,
                choice.next_state.raw_state.snapshot,
            )

    def test_writer_shaped_eos_choice_carries_terminal_certificates(self) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            include_eos=True,
        )
        result = decoder.initial_state().choices_with_stats()
        terminal_choices = tuple(choice for choice in result.choices if choice.is_eos)
        if not terminal_choices:
            self.skipTest("fixture has no immediate EOS")
        choice = terminal_choices[0]
        self.assertIsNotNone(choice.choice_certificate)
        assert choice.choice_certificate is not None
        self.assertTrue(choice.choice_certificate.kind.name.startswith("EOS"))
        self.assertIsNone(choice.choice_certificate.prefix_after)
        self.assertIsNotNone(choice.choice_certificate.terminal_projection_certificate)
        self.assertTrue(choice.choice_certificate.terminal_certificates)

    def test_writer_shaped_result_certificate_covers_choices(self) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            include_eos=True,
        )
        result = decoder.initial_state().choices_with_stats()
        self.assertIsNotNone(result.result_certificate)
        assert result.result_certificate is not None
        self.assertEqual(
            tuple(choice.choice_certificate for choice in result.choices),
            result.result_certificate.choice_certificates,
        )
        self.assertEqual(
            result.checked_frontier_certificate,
            result.result_certificate.checked_frontier_certificate,
        )
        self.assertEqual(
            result.count_certificate,
            result.result_certificate.count_certificate,
        )

    def test_writer_shaped_online_preserves_runtime_frontier_certificates(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(
            prepared=prepared,
            include_eos=True,
        )
        runtime_state = decoder.initial_state().raw_state
        runtime_transitions = writer_runtime_choice_transitions(
            prepared=prepared,
            state=runtime_state,
        )
        result = WriterShapedOnlineDecoderState(
            prefix="",
            raw_state=runtime_state,
            decoder=decoder,
        ).choices_with_stats()
        self.assertEqual(
            result.checked_frontier_certificate,
            runtime_transitions.checked_frontier_certificate,
        )
        self.assertEqual(
            result.count_certificate,
            runtime_transitions.count_certificate,
        )

    def test_writer_shaped_online_choice_certificate_rejects_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        decoder = make_writer_shaped_online_decoder(prepared=prepared)
        state = decoder.initial_state()
        transitions = writer_runtime_choice_transitions(
            prepared=prepared,
            state=state.raw_state,
        )
        transition = transitions.transitions[0]
        choice = transition.choice
        projection = transitions.text_choice_projection_certificates[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "snapshot_step_projection_mismatch",
        ):
            writer_online_text_choice_certificate(
                prefix=state.prefix,
                choice=choice,
                next_state=WriterShapedOnlineDecoderState(
                    prefix=state.prefix + choice.emitted_text,
                    raw_state=transition.next_state,
                    decoder=decoder,
                ),
                snapshot_step_certificate=transition.snapshot_step_certificate,
                text_projection_certificate=SimpleNamespace(
                    emitted_text="DIFFERENT",
                    branch_certificates=projection.branch_certificates,
                ),
                checked_frontier_certificate=(
                    transitions.checked_frontier_certificate
                ),
                count_certificate=transitions.count_certificate,
            )

        if transitions.terminal is not None:
            with self.assertRaisesRegex(
                SouthStarError,
                "eos_choice_text_mismatch",
            ):
                writer_online_eos_choice_certificate(
                    prefix=state.prefix,
                    eos_text="END",
                    terminal=transitions.terminal,
                    terminal_projection_certificate=(
                        transitions.terminal_projection_certificate
                    ),
                    checked_frontier_certificate=(
                        transitions.checked_frontier_certificate
                    ),
                    count_certificate=transitions.count_certificate,
                )

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_count_mismatch",
        ):
            writer_online_choice_result_certificate(
                prefix=state.prefix,
                choices=(choice,),
                choice_certificates=(),
                checked_frontier_certificate=transitions.checked_frontier_certificate,
                count_certificate=transitions.count_certificate,
            )


def _reachable_eos_prefixes(state) -> set[str]:
    pending = [state]
    seen_prefixes: set[str] = set()
    out: set[str] = set()
    while pending:
        current = pending.pop()
        if current.prefix in seen_prefixes:
            continue
        seen_prefixes.add(current.prefix)
        for choice in current.choices():
            if choice.is_eos:
                # EOS is an observation on the current writer frontier; it is
                # not an emitted SMILES token and must not extend the prefix.
                out.add(current.prefix)
                continue
            if choice.next_state is None:
                raise AssertionError("non-EOS writer online choice lacks next_state")
            pending.append(choice.next_state)
    return out


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
