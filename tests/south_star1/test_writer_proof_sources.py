from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.south_star1.helpers import cco_facts
from tests.south_star1.writer_proof_sources import (
    SHARED_RING_BRANCH_SOURCE_ADDRESSES,
    SharedRingBranchSourceAddress,
    first_terminal_proof_source,
    shared_ring_branch_source,
    shared_ring_branch_sources,
    validate_shared_ring_branch_source_addresses,
)
import tests.south_star1.writer_proof_sources as proof_sources
from tests.south_star1.writer_test_context import writer_runtime_options
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_branch_transition_artifact import (
    writer_branch_transition_artifact_for_support,
)


class WriterProofSourcesTest(unittest.TestCase):
    def test_shared_ring_address_registry_is_valid(self) -> None:
        validate_shared_ring_branch_source_addresses()
        self.assertEqual(len(SHARED_RING_BRANCH_SOURCE_ADDRESSES), 6)

    def test_shared_ring_address_registry_rejects_malformed_values(self) -> None:
        address = SHARED_RING_BRANCH_SOURCE_ADDRESSES[0]
        with self.assertRaises(ValueError):
            validate_shared_ring_branch_source_addresses(
                (address, *SHARED_RING_BRANCH_SOURCE_ADDRESSES[2:])
            )
        with self.assertRaises(ValueError):
            validate_shared_ring_branch_source_addresses(
                (
                    SharedRingBranchSourceAddress(
                        address.phase,
                        address.direction_mark,
                        address.predecessor_branch_certificate_digests,
                        "not-a-digest",
                        address.target_branch_certificate_digest,
                        address.target_emitted_text,
                        address.target_successor_cursor_digest,
                        address.expected_branch_artifact_digest,
                    ),
                    *SHARED_RING_BRANCH_SOURCE_ADDRESSES[1:],
                )
            )

    def test_shared_ring_sources_are_immutable_and_keyed(self) -> None:
        sources = shared_ring_branch_sources()
        self.assertEqual(len(sources), 6)
        keys = [(source.phase, source.direction_mark) for source in sources]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            keys,
            sorted(keys, key=lambda item: (item[0], item[1].value)),
        )
        self.assertIs(shared_ring_branch_source(*keys[0]), sources[0])

    def test_replay_matches_every_pinned_address(self) -> None:
        sources = shared_ring_branch_sources()
        for source, address in zip(sources, SHARED_RING_BRANCH_SOURCE_ADDRESSES):
            self.assertEqual((source.phase, source.direction_mark), (address.phase, address.direction_mark))
            self.assertEqual(source.support.emitted_text, address.target_emitted_text)
            self.assertEqual(
                _identity_digest(source.support.checked_branch_certificate),
                address.target_branch_certificate_digest,
            )
            self.assertEqual(
                _identity_digest(source.support.successor_cursor),
                address.target_successor_cursor_digest,
            )
            artifact = writer_branch_transition_artifact_for_support(
                prepared=source.prepared,
                snapshot=source.snapshot,
                support=source.support,
            )
            self.assertEqual(artifact["digest"], address.expected_branch_artifact_digest)

    def test_replay_captures_one_snapshot_per_selected_source(self) -> None:
        proof_sources.shared_ring_branch_sources.cache_clear()
        with patch.object(
            proof_sources,
            "capture_writer_frontier_snapshot",
            wraps=proof_sources.capture_writer_frontier_snapshot,
        ) as capture:
            proof_sources.shared_ring_branch_sources()
        self.assertEqual(capture.call_count, len(SHARED_RING_BRANCH_SOURCE_ADDRESSES))

    def test_replay_uses_only_unique_prefix_and_source_batches(self) -> None:
        prefixes = {
            address.predecessor_branch_certificate_digests[:index]
            for address in SHARED_RING_BRANCH_SOURCE_ADDRESSES
            for index in range(1, len(address.predecessor_branch_certificate_digests) + 1)
        }
        expected_batch_count = len(prefixes) + 1
        proof_sources.shared_ring_branch_sources.cache_clear()
        with patch.object(
            proof_sources,
            "_checked_writer_frontier_branch_supports",
            wraps=proof_sources._checked_writer_frontier_branch_supports,
        ) as checked:
            proof_sources.shared_ring_branch_sources()
        self.assertEqual(checked.call_count, expected_batch_count)

    def test_replay_selector_has_no_exhaustive_search_constructs(self) -> None:
        tree = ast.parse(Path(proof_sources.__file__).read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "shared_ring_branch_sources"
        )
        self.assertFalse(any(isinstance(node, ast.While) for node in ast.walk(function)))
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"repr", "setdefault"}
                for node in ast.walk(function)
            )
        )

    def test_terminal_selector_preserves_supplied_context(self) -> None:
        options = writer_runtime_options(rooted_at_atom=0)
        facts = cco_facts()
        policy = None
        source = first_terminal_proof_source(facts, options, policy=policy)
        self.assertEqual(source.facts, facts)
        self.assertIs(source.runtime_options, options)
        self.assertEqual(source.policy, policy)
        self.assertEqual(source.snapshot.decoder_boundary.consumed_token_count, 3)
        self.assertIsNotNone(source.support)

    def test_shared_support_modules_import_no_test_modules(self) -> None:
        root = Path(__file__).parent
        for name in ("writer_test_context.py", "writer_test_fixtures.py", "writer_proof_sources.py"):
            tree = ast.parse((root / name).read_text(), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.south_star1.test_"):
                    self.fail(f"test-module import remains in {name}: {node.module}")


if __name__ == "__main__":
    unittest.main()
