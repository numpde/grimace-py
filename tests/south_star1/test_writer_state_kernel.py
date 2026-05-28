"""Tests for the writer-shaped state/frontier MVP."""

from __future__ import annotations

import ast
import contextlib
import unittest
from pathlib import Path
from unittest.mock import patch

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import count_writer_witness_completions
from grimace._south_star1.writer_frontier import initial_writer_frontier
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import writer_state_key
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
SOUTH_STAR1_ROOT = REPO_ROOT / "python" / "grimace" / "_south_star1"


class WriterStateKernelTest(unittest.TestCase):
    def test_writer_shaped_acyclic_support_uses_writer_frontier(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(),
            )

        self.assertEqual(
            support.strings,
            ("C(C)O", "C(O)C", "CCO", "OCC"),
        )
        self.assertEqual(support.distinct_count, 4)
        self.assertEqual(support.witness_count, 4)

    def test_writer_frontier_groups_same_emitted_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        frontier = initial_writer_frontier(prepared, _writer_options())

        choices = writer_frontier_choices(prepared, frontier)

        self.assertFalse(choices.eos_available)
        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertEqual(choices.choices[0].immediate_multiplicity, 2)
        self.assertEqual(len(choices.choices[0].successor.states), 2)

    def test_writer_support_image_keeps_witness_count_separate(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("CC",))
        self.assertEqual(support.distinct_count, 1)
        self.assertEqual(support.witness_count, 2)

    def test_writer_witness_completions_can_exceed_support_count(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))
        frontier = initial_writer_frontier(prepared, _writer_options())

        self.assertEqual(count_writer_frontier_support(prepared, frontier), 2)
        self.assertEqual(count_writer_witness_completions(prepared, frontier), 4)

    def test_writer_support_count_does_not_call_streaming_support(self) -> None:
        prepared = _prepare(cco_facts())
        frontier = initial_writer_frontier(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier.iter_writer_frontier_support",
            side_effect=AssertionError("count-only path streamed support strings"),
        ):
            self.assertEqual(count_writer_frontier_support(prepared, frontier), 4)

    def test_unique_child_is_inline_for_rooted_chain(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(support.strings, ("CCC",))
        self.assertNotIn("(", support.strings[0])

    def test_true_side_branches_remain_expressible(self) -> None:
        prepared = _prepare(cco_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=1),
        )

        self.assertEqual(support.strings, ("C(C)O", "C(O)C"))

    def test_double_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "O", BondOrder.DOUBLE))
        frontier = initial_writer_frontier(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, frontier).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "=")
        self.assertEqual(third.emitted_text, "O")
        self.assertEqual(support.strings, ("C=O",))

    def test_triple_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "C", BondOrder.TRIPLE))
        frontier = initial_writer_frontier(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, frontier).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "#")
        self.assertEqual(third.emitted_text, "C")

    def test_writer_shaped_disconnected_components_emit_dot(self) -> None:
        prepared = _prepare(disconnected_co_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("C.O",))

    def test_writer_root_restricts_initial_frontier_without_plan_route(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(rooted_at_atom=2),
            )

        self.assertEqual(support.strings, ("OCC",))

    def test_writer_shaped_cyclic_fails_closed_before_forbidden_routes(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_shaped_cycle_plus_isolate_component_fails_closed(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_shaped_stereo_fails_closed_before_forbidden_routes(self) -> None:
        for facts in (tetrahedral_facts(), directional_facts()):
            with self.subTest(facts=facts):
                prepared = _prepare(facts)
                with _forbidden_exhaustive_routes():
                    with self.assertRaises(SouthStarError) as caught:
                        enumerate_prepared_stereo_support(
                            prepared=prepared,
                            runtime_options=_writer_options(),
                        )

                self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_writer_state_key_excludes_rendered_payloads(self) -> None:
        fields = set(WriterState.__dataclass_fields__)

        self.assertNotIn("rendered", fields)
        self.assertNotIn("prefix", fields)
        self.assertNotIn("suffix", fields)
        state = next(iter(initial_writer_frontier(_prepare(cco_facts()), _writer_options()).states))
        self.assertIs(writer_state_key(state), state)

    def test_writer_modules_do_not_import_exhaustive_routes(self) -> None:
        forbidden = {
            "skeleton",
            "exhaustive_online_traversal",
            "online_stereo_witness",
            "online_search_vm",
        }

        for module_name in (
            "writer_state.py",
            "writer_transitions.py",
            "writer_frontier.py",
            "writer_support.py",
        ):
            with self.subTest(module=module_name):
                tree = ast.parse((SOUTH_STAR1_ROOT / module_name).read_text(encoding="utf-8"))
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.extend(
                            alias.name.rsplit(".", 1)[-1]
                            for alias in node.names
                            if alias.name.rsplit(".", 1)[-1] in forbidden
                        )
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        imported.extend(
                            part for part in forbidden if module.endswith(part)
                        )
                self.assertEqual(imported, [])


def _prepare(facts: MoleculeFacts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


@contextlib.contextmanager
def _forbidden_exhaustive_routes():
    paths = (
        "grimace._south_star1.skeleton.enumerate_traversal_skeletons",
        "grimace._south_star1.exhaustive_online_traversal.iter_exhaustive_online_traversal_traces",
        "grimace._south_star1.exhaustive_online_traversal.iter_prepared_exhaustive_online_traversal_traces",
        "grimace._south_star1.online_stereo_witness.iter_exhaustive_online_stereo_witnesses",
        "grimace._south_star1.online_search_vm.ExhaustiveOnlineSearchVM",
        "grimace._south_star1.skeleton.TraversalSkeleton",
        "grimace._south_star1.exhaustive_online_traversal.ExhaustiveTraversalTrace",
        "grimace._south_star1.skeleton._component_spanning_trees",
        "grimace._south_star1.exhaustive_online_traversal._iter_spanning_forest_choices_lazy",
    )
    with contextlib.ExitStack() as stack:
        for path in paths:
            stack.enter_context(
                patch(
                    path,
                    side_effect=AssertionError(f"writer-shaped called {path}"),
                )
            )
        yield


def chain_facts(symbols: tuple[str, ...]) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, symbol) for index, symbol in enumerate(symbols)),
        bonds=tuple(
            single_bond(index, index, index + 1)
            for index in range(len(symbols) - 1)
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(len(symbols))),
                bonds=tuple(BondId(index) for index in range(len(symbols) - 1)),
            ),
        ),
    )


def disconnected_co_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


def two_atom_facts(
    left: str,
    right: str,
    order: BondOrder,
) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, left), atom(1, right)),
        bonds=(bond(0, 0, 1, order),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def cycle_plus_isolate_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "C")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
