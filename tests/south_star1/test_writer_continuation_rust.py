"""Rust continuation-core and asset-backed decoder contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from grimace import MolToSmilesContinuationDecoder
from grimace import SouthStarError
from grimace._south_star1.writer_continuation_asset import (
    open_writer_continuation_core,
)
from grimace._south_star1.writer_continuation_asset import (
    write_writer_continuation_asset,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_choices,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_completion_count,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_is_terminal,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_probabilities,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_support_count,
)
from grimace._south_star1.writer_continuation_automaton import (
    WriterContinuationCursor,
)
from grimace._south_star1.writer_frontier import iter_writer_frontier_support
from grimace._south_star1.writer_continuation_rust import _core_terms
from grimace._south_star1.writer_continuation_rust import _rust_core_from_terms
from grimace._south_star1.writer_continuation_rust import (
    open_writer_continuation_rust_core,
)
from grimace._south_star1.writer_envelope_terms import _identity_digest
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_writer_stereo_residual import (
    _directional_non_single_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    _directional_ring_carrier_facts,
)
from tests.south_star1.test_writer_support_artifact_fact_verifier import (
    _initial_snapshot,
)
from tests.south_star1.test_writer_support_artifact_fact_verifier import _prepare
from tests.south_star1.test_writer_support_artifact_fact_verifier import (
    _writer_options,
)


class WriterContinuationRustTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.path = Path(cls._temporary.name) / "asset"
        cls.prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(cls.prepared, _writer_options())
        write_writer_continuation_asset(
            path=cls.path,
            prepared=cls.prepared,
            snapshot=snapshot,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_rust_matches_python_core_at_every_node(self) -> None:
        asset = open_writer_continuation_core(self.path)
        rust = open_writer_continuation_rust_core(self.path)
        self.assertEqual(rust.manifest_digest, asset.manifest_digest)
        self.assertEqual(rust.node_count, len(asset.core.nodes))
        self.assertEqual(
            rust.edge_count,
            sum(len(node.choices) for node in asset.core.nodes),
        )
        for node in asset.core.nodes:
            python_cursor = WriterContinuationCursor(node.node_id, 3)
            rust_cursor = rust.cursor(node.node_id, 3)
            self.assertEqual(
                rust.is_terminal(rust_cursor),
                writer_continuation_is_terminal(asset.core, python_cursor),
            )
            self.assertEqual(
                int(rust.support_count(rust_cursor)),
                writer_continuation_support_count(asset.core, python_cursor),
            )
            self.assertEqual(
                int(rust.completion_count(rust_cursor)),
                writer_continuation_completion_count(
                    asset.core, python_cursor
                ),
            )
            python_choices = writer_continuation_choices(
                asset.core, python_cursor
            )
            rust_choices = rust.choices(rust_cursor)
            self.assertEqual(
                tuple(
                    (
                        item.text,
                        int(item.immediate_multiplicity),
                        item.next_cursor.node_id,
                        int(item.next_cursor.completion_scale),
                        int(item.support_count),
                        int(item.completion_count),
                    )
                    for item in rust_choices
                ),
                tuple(
                    (
                        item.emitted_text,
                        item.immediate_multiplicity,
                        item.successor_node_id,
                        item.successor_scale,
                        item.support_count,
                        item.completion_count,
                    )
                    for item in python_choices
                ),
            )
            self.assertEqual(
                tuple(
                    (item.text, int(item.numerator), int(item.denominator))
                    for item in rust.probabilities(rust_cursor)
                ),
                tuple(
                    (item.emitted_text, item.numerator, item.denominator)
                    for item in writer_continuation_probabilities(
                        asset.core, python_cursor
                    )
                ),
            )

    def test_decoder_enumerates_exact_support_and_resumes(self) -> None:
        decoder = MolToSmilesContinuationDecoder.from_asset(self.path)
        self.assertEqual((decoder.support_count, decoder.completion_count), (4, 4))
        self.assertEqual(
            sum(item.numerator for item in decoder.exact_probabilities()),
            decoder.completion_count,
        )
        support = _enumerate_decoder(decoder)
        self.assertEqual(support, ("C(C)O", "C(O)C", "CCO", "OCC"))
        advanced = decoder.next_choices[0].next_state
        resumed = MolToSmilesContinuationDecoder.from_snapshot(
            self.path, advanced.snapshot()
        )
        self.assertEqual(resumed.cache_key(), advanced.cache_key())
        self.assertEqual(_enumerate_decoder(resumed), _enumerate_decoder(advanced))
        copied = advanced.copy()
        self.assertEqual(copied.cache_key(), advanced.cache_key())
        self.assertIs(copied._state.core, advanced._state.core)

    def test_core_only_loader_reads_no_provenance_or_source_snapshot(self) -> None:
        from grimace._south_star1 import writer_continuation_asset as asset_module

        read_chunk = asset_module._read_chunk

        def core_chunk_only(path, descriptor):
            if descriptor["kind"] != "automaton_core":
                raise AssertionError("non-core chunk read")
            return read_chunk(path, descriptor)

        with (
            patch(
                "grimace._south_star1.writer_continuation_asset._read_chunk",
                side_effect=core_chunk_only,
            ),
            patch(
                "grimace._south_star1.writer_continuation_asset.WriterContinuationAsset.records",
                side_effect=AssertionError("provenance read"),
            ),
            patch(
                "grimace._south_star1.writer_continuation_asset._source_snapshot_from_asset",
                side_effect=AssertionError("source snapshot read"),
            ),
            patch(
                "grimace._south_star1.writer_continuation_asset._frontier_batch",
                side_effect=AssertionError("live frontier read"),
            ),
            patch(
                "grimace._south_star1.writer_continuation_asset.reconstruct_writer_cursor_from_asset",
                side_effect=AssertionError("cursor replay"),
            ),
            patch(
                "grimace._south_star1.prepared_runtime.prepare_south_star_mol_from_facts",
                side_effect=AssertionError("facts preparation"),
            ),
        ):
            decoder = MolToSmilesContinuationDecoder.from_asset(self.path)
            self.assertTrue(decoder.next_choices)
            self.assertEqual(decoder.completion_count, 4)

    def test_small_fixture_support_images_match(self) -> None:
        cases = (
            (tetrahedral_facts(), _writer_options()),
            (directional_facts(), _writer_options(rooted_at_atom=2)),
            (
                shared_acyclic_directional_facts(),
                _writer_options(rooted_at_atom=0),
            ),
            (
                _directional_ring_carrier_facts(),
                _writer_options(rooted_at_atom=0),
            ),
        )
        for facts, options in cases:
            with self.subTest(facts=facts), TemporaryDirectory() as directory:
                prepared = _prepare(facts)
                snapshot = _initial_snapshot(prepared, options)
                path = Path(directory) / "asset"
                write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
                decoder = MolToSmilesContinuationDecoder.from_asset(path)
                self.assertEqual(
                    _enumerate_decoder(decoder),
                    tuple(
                        iter_writer_frontier_support(
                            prepared, snapshot.cursor
                        )
                    ),
                )

    def test_non_single_ring_asset_publishes_after_complete_local_proof(self) -> None:
        facts = _directional_non_single_ring_carrier_facts()
        options = _writer_options(rooted_at_atom=0)
        prepared = _prepare(facts)
        snapshot = _initial_snapshot(prepared, options)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            self.assertTrue(path.is_dir())
            asset = open_writer_continuation_core(path)
            edges = asset.records("edge_records")
            terminals = asset.records("terminal_records")
            self.assertEqual(len(asset.records("raw_cursor_records")), 456)
            self.assertEqual(len(edges), 455)
            self.assertEqual(
                sum(len(edge.branch_certificate_digests) for edge in edges),
                491,
            )
            self.assertEqual(len(terminals), 72)
            self.assertEqual(
                sum(
                    len(terminal.terminal_support_identity_digests)
                    for terminal in terminals
                ),
                72,
            )
            decoder = MolToSmilesContinuationDecoder.from_asset(path)
            self.assertEqual(
                _enumerate_decoder(decoder),
                tuple(iter_writer_frontier_support(prepared, snapshot.cursor)),
            )

    def test_proof_mode_reconstructs_branch_and_terminal_lazily(self) -> None:
        decoder = MolToSmilesContinuationDecoder.from_asset(
            self.path,
            proof_capable=True,
            prepared=self.prepared,
        )
        asset = decoder._state.proof_asset
        edge = asset.edges_from(
            decoder._state.proof_cursor.raw_cursor_digest
        )[0]
        branch = decoder.branch_artifact(edge.branch_certificate_digests[0])
        self.assertEqual(branch["schema_name"], "writer_branch_transition_artifact")

        while not decoder.is_terminal:
            decoder = decoder.next_choices[0].next_state
        terminal = asset.terminal_record(
            decoder._state.proof_cursor.raw_cursor_digest
        )
        proof = decoder.terminalization_artifact(
            terminal.terminal_support_identity_digests[0]
        )
        self.assertEqual(proof["schema_name"], "writer_terminalization_artifact")

    def test_big_integers_terminal_with_choices_and_owned_copy(self) -> None:
        huge = 2**160
        terms = [
            (
                0,
                _identity_digest((True, huge, huge, ())),
                True,
                huge,
                huge,
                (),
                1,
                huge,
            ),
            (
                1,
                _identity_digest(
                    (True, 1, 1, (("x", huge, huge, 0),))
                ),
                True,
                1,
                1,
                (("x", huge, 0, huge, 1, huge * huge),),
                2,
                1 + huge * huge,
            ),
        ]
        core = _rust_core_from_terms(
            manifest_digest="synthetic",
            root_node_id=1,
            root_scale=huge,
            nodes=terms,
        )
        terms[1] = terms[0]
        cursor = core.root_cursor()
        self.assertTrue(core.is_terminal(cursor))
        self.assertEqual(len(core.choices(cursor)), 1)
        self.assertGreater(int(core.completion_count(cursor)), 2**128)

    def test_rust_rejects_core_mutations_and_foreign_cursors(self) -> None:
        asset = open_writer_continuation_core(self.path)
        original = list(_core_terms(asset.core))
        cases = []

        gap = deepcopy(original)
        gap[0] = (9, *gap[0][1:])
        cases.append(gap)

        wrong_total = deepcopy(original)
        wrong_total[-1] = (*wrong_total[-1][:-1], wrong_total[-1][-1] + 1)
        cases.append(wrong_total)

        wrong_signature = deepcopy(original)
        wrong_signature[0] = (
            wrong_signature[0][0],
            "0" * 64,
            *wrong_signature[0][2:],
        )
        cases.append(wrong_signature)

        unsorted = deepcopy(original)
        parent = next(index for index, node in enumerate(unsorted) if len(node[5]) > 1)
        node = unsorted[parent]
        unsorted[parent] = (*node[:5], tuple(reversed(node[5])), *node[6:])
        cases.append(unsorted)

        bad_scale = deepcopy(original)
        parent = next(index for index, node in enumerate(bad_scale) if node[5])
        node = bad_scale[parent]
        choice = node[5][0]
        choices = ((choice[0], choice[1], choice[2], 0, choice[4], choice[5]), *node[5][1:])
        bad_scale[parent] = (*node[:5], choices, *node[6:])
        cases.append(bad_scale)

        for terms in cases:
            with self.subTest(case=cases.index(terms)):
                with self.assertRaises((OverflowError, ValueError)):
                    _rust_core_from_terms(
                        manifest_digest=asset.manifest_digest,
                        root_node_id=asset.core.root.node_id,
                        root_scale=asset.core.root.completion_scale,
                        nodes=terms,
                    )

        first = open_writer_continuation_rust_core(self.path)
        second = open_writer_continuation_rust_core(self.path)
        with self.assertRaisesRegex(ValueError, "cursor_core_mismatch"):
            first.support_count(second.root_cursor())

    def test_snapshot_rejects_tampering_and_wrong_proof_mode(self) -> None:
        decoder = MolToSmilesContinuationDecoder.from_asset(self.path)
        snapshot = decoder.next_choices[0].next_state.snapshot()
        for field, value in (
            ("token_count", 99),
            ("emitted_texts", ["not-a-choice"]),
            ("asset_manifest_digest", "0" * 64),
        ):
            forged = deepcopy(snapshot)
            forged[field] = value
            forged["digest"] = _snapshot_digest_for_test(forged)
            with self.subTest(field=field):
                with self.assertRaises((ValueError, SouthStarError)):
                    MolToSmilesContinuationDecoder.from_snapshot(
                        self.path, forged
                    )
        forged = deepcopy(snapshot)
        forged["cursor"]["raw_cursor_digest"] = "1" * 64
        forged["digest"] = _snapshot_digest_for_test(forged)
        with self.assertRaisesRegex(ValueError, "proof_cursor_mismatch"):
            MolToSmilesContinuationDecoder.from_snapshot(self.path, forged)


def _enumerate_decoder(decoder) -> tuple[str, ...]:
    values = []
    pending = [decoder]
    while pending:
        current = pending.pop()
        if current.is_terminal:
            values.append(current.prefix)
        pending.extend(choice.next_state for choice in current.next_choices)
    return tuple(sorted(values))


def _snapshot_digest_for_test(snapshot) -> str:
    import hashlib
    import json

    unsigned = dict(snapshot)
    unsigned.pop("digest", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
