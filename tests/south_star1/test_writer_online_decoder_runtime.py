"""Online decoder coverage for the writer-runtime route."""

from __future__ import annotations

import unittest

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
from tests.south_star1.helpers import cco_facts


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
        self.assertEqual(decoder.rooted_at_atom, 1)
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
