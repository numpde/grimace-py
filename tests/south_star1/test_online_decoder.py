"""Tests for online prefix feasibility and next-character decoding."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_decoder import OnlineDecodeToken
from grimace._south_star1.online_decoder import online_allowed_next_characters
from grimace._south_star1.online_decoder import online_allowed_next_tokens
from grimace._south_star1.online_decoder import online_prefix_has_completion
from grimace._south_star1.online_render_sink import PrefixConstrainedSink
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_DECODER_PATH = REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_decoder.py"


class OnlineDecoderTest(unittest.TestCase):
    def test_online_prefix_has_completion_for_empty_prefix(self) -> None:
        facts = tetrahedral_facts()

        self.assertTrue(
            online_prefix_has_completion(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                prefix="",
            )
        )

    def test_online_prefix_rejects_impossible_prefix(self) -> None:
        facts = tetrahedral_facts()

        self.assertFalse(
            online_prefix_has_completion(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                prefix="not-smiles",
            )
        )

    def test_online_allowed_next_characters_matches_online_witness_multiset_tetra(
        self,
    ) -> None:
        self._assert_next_chars_match_witnesses(tetrahedral_facts(), "[")

    def test_online_allowed_next_characters_matches_online_witness_multiset_directional(
        self,
    ) -> None:
        self._assert_next_chars_match_witnesses(directional_facts(), "C(")

    def test_online_allowed_next_characters_matches_online_witness_multiset_ring_tetra(
        self,
    ) -> None:
        self._assert_next_chars_match_witnesses(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
            "[C",
        )

    def test_online_decoder_does_not_deduplicate_support(self) -> None:
        facts = ethane_facts()
        witnesses = _witnesses(facts)
        prefix = "C"

        self.assertGreater(sum(Counter(witnesses).values()), len(set(witnesses)))
        self.assertEqual(
            tuple(
                sorted(
                    {
                        witness[len(prefix)]
                        for witness in witnesses
                        if witness.startswith(prefix) and len(witness) > len(prefix)
                    }
                )
            ),
            tuple(
                sorted(
                    online_allowed_next_characters(
                        facts=facts,
                        policy=ordinary_policy_for_facts(facts),
                        semantics=OrdinarySmilesSemantics(),
                        prefix=prefix,
                    )
                )
            ),
        )

    def test_prefix_sink_prunes_early(self) -> None:
        sink = PrefixConstrainedSink(required_prefix="C(")

        self.assertTrue(sink.append("C"))
        self.assertFalse(sink.append("C"))
        self.assertEqual(sink.value(), "C")

    def test_prefix_sink_rollback_restores_state(self) -> None:
        sink = PrefixConstrainedSink(required_prefix="C(")
        self.assertTrue(sink.append("C"))
        checkpoint = sink.checkpoint()
        self.assertTrue(sink.append("("))
        self.assertEqual(sink.value(), "C(")

        sink.rollback(checkpoint)

        self.assertEqual(sink.value(), "C")
        self.assertTrue(sink.append("("))
        self.assertTrue(sink.complete())

    def test_online_allowed_next_characters_excludes_eos_by_default(self) -> None:
        facts = tetrahedral_facts()
        witness = _witnesses(facts)[0]

        self.assertEqual(
            online_allowed_next_characters(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                prefix=witness,
            ),
            (),
        )

    def test_online_allowed_next_tokens_uses_supplied_token_vocabulary(self) -> None:
        facts = tetrahedral_facts()
        tokens = (
            OnlineDecodeToken("[", "atom_text"),
            OnlineDecodeToken("Z", "atom_text"),
        )

        self.assertEqual(
            online_allowed_next_tokens(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                prefix="",
                tokens=tokens,
            ),
            (tokens[0],),
        )

    def test_online_decoder_boundary_no_support_image_or_artifact_imports(self) -> None:
        tree = ast.parse(ONLINE_DECODER_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "compile_support_artifact",
            "enumerate_stereo_support",
            "render_image_from_witnesses",
        }
        imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])

    def _assert_next_chars_match_witnesses(self, facts, prefix: str) -> None:
        witnesses = _witnesses(facts)
        expected = tuple(
            sorted(
                {
                    witness[len(prefix)]
                    for witness in witnesses
                    if witness.startswith(prefix) and len(witness) > len(prefix)
                }
            )
        )

        actual = tuple(
            sorted(
                online_allowed_next_characters(
                    facts=facts,
                    policy=ordinary_policy_for_facts(facts),
                    semantics=OrdinarySmilesSemantics(),
                    prefix=prefix,
                )
            )
        )

        self.assertEqual(actual, expected)


def _witnesses(facts) -> tuple[str, ...]:
    return tuple(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def ethane_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
