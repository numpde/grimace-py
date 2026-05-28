"""Writer-owned residual stereo tests."""

from __future__ import annotations

import unittest

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import add_factor_checked
from grimace._south_star1.residual_constraints import tetra_var
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class WriterStereoResidualTest(unittest.TestCase):
    def test_tetrahedral_stereo_prunes_invalid_atom_tokens(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=1),
        )

        self.assertEqual(
            support.strings,
            ("F[C@@H](Br)Cl", "F[C@H](Cl)Br"),
        )
        self.assertEqual(support.distinct_count, 2)
        self.assertEqual(support.witness_count, 2)

    def test_tetra_frontier_counts_are_pruned_per_token(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        choices = writer_frontier_choices(prepared, after_f)

        self.assertEqual(
            tuple(
                (choice.emitted_text, choice.support_count, choice.completion_count)
                for choice in choices.choices
            ),
            (("[C@@H]", 1, 1), ("[C@H]", 1, 1)),
        )

    def test_directional_stereo_prunes_invalid_carrier_marks(self) -> None:
        prepared = _prepare(directional_facts())
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=2),
        )

        self.assertEqual(support.strings, ("F/C=C/Cl", "F\\C=C\\Cl"))
        self.assertEqual(support.distinct_count, 2)
        self.assertEqual(support.witness_count, 2)

    def test_directional_frontier_drops_zero_completion_mark_choice(self) -> None:
        prepared = _prepare(directional_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=2),
        )
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        choices = writer_frontier_choices(prepared, after_f)

        self.assertEqual(
            tuple(choice.emitted_text for choice in choices.choices),
            ("/", "\\"),
        )
        self.assertEqual(
            tuple(choice.completion_count for choice in choices.choices),
            (1, 1),
        )

    def test_add_factor_checked_rolls_back_rejected_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(store.assign(var, TetraToken.ATAT))
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=(0, 1, 2, 3),
            local_order=(0, 1, 2, 3),
        )

        self.assertFalse(add_factor_checked(store, factor))
        self.assertEqual(store.value_snapshot().factors, ())
        self.assertEqual(
            ResidualStore.from_value_snapshot(store.value_snapshot()).value_snapshot(),
            store.value_snapshot(),
        )


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
