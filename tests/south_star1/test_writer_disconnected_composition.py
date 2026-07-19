"""Fixed-order disconnected composition across facts, proofs, and runtime."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rdkit import Chem
from rdkit import rdBase

from grimace import MolToSmilesContinuationDecoder
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import component_root_domains_for_prepared
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_branch_transition_artifact import (
    branch_transition_artifact_manifest,
)
from grimace._south_star1.writer_branch_transition_artifact import (
    verify_writer_branch_transition_artifact_envelope,
)
from grimace._south_star1.writer_branch_transition_artifact import (
    writer_branch_transition_artifact_for_support,
)
from grimace._south_star1.writer_branch_transition_artifact_checker import (
    verify_writer_branch_transition_artifact_consistency,
)
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import (
    verify_writer_branch_transition_artifact_for_facts,
)
from grimace._south_star1.writer_frontier import (
    _checked_writer_frontier_branch_supports,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_continuation_asset import (
    write_writer_continuation_asset,
)
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support import enumerate_prepared_writer_shaped_support
from grimace._south_star1.writer_support_artifact_checker import artifact_metrics
from grimace._south_star1.writer_support_artifact_checker import (
    support_artifact_object_identity_term,
)
from tests.helpers.rdkit_south_star_disconnected_audit import (
    load_pinned_south_star_disconnected_audit_cases,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_WRITER_CAPABILITY_CASES,
)


class WriterDisconnectedCompositionTest(unittest.TestCase):
    def test_fixed_order_support_is_the_exact_component_product(self) -> None:
        ledger = {item.name: item for item in DEFAULT_WRITER_CAPABILITY_CASES}
        fixtures = load_pinned_south_star_disconnected_audit_cases(
            rdBase.rdkitVersion
        )
        for fixture in fixtures:
            with self.subTest(case=fixture.name):
                case = ledger[fixture.name]
                facts = ordinary_molecule_facts_from_smiles(
                    case.smiles, case.extraction_options
                )
                self.assertEqual(
                    tuple(tuple(map(int, component.atoms)) for component in facts.components),
                    fixture.component_order,
                )
                prepared = _prepare(facts)
                self.assertEqual(
                    tuple(
                        tuple(map(int, domain))
                        for _component, domain in component_root_domains_for_prepared(
                            prepared=prepared,
                            rooted_at_atom=case.rooted_at_atom,
                        )
                    ),
                    fixture.component_root_domains,
                )
                image = enumerate_prepared_writer_shaped_support(
                    prepared=prepared,
                    runtime_options=_options(case.rooted_at_atom),
                )
                component_supports, component_completions = _component_products(
                    case.smiles,
                    rooted_at_atom=case.rooted_at_atom,
                    extraction_options=case.extraction_options,
                )
                expected = tuple(
                    sorted(
                        ".".join(parts)
                        for parts in product(*component_supports)
                    )
                )
                completion_count = 1
                for count in component_completions:
                    completion_count *= count
                self.assertEqual(expected, fixture.expected_support)
                self.assertEqual(tuple(sorted(image.strings)), expected)
                self.assertEqual(image.distinct_count, fixture.support_count)
                self.assertEqual(image.witness_count, completion_count)
                self.assertEqual(completion_count, fixture.completion_count)

    def test_every_dot_branch_has_semantic_relation_credit(self) -> None:
        ledger = {item.name: item for item in DEFAULT_WRITER_CAPABILITY_CASES}
        for fixture in load_pinned_south_star_disconnected_audit_cases(
            rdBase.rdkitVersion
        ):
            with self.subTest(case=fixture.name):
                case = ledger[fixture.name]
                facts = ordinary_molecule_facts_from_smiles(
                    case.smiles, case.extraction_options
                )
                prepared = _prepare(facts)
                options = _options(case.rooted_at_atom)
                dot_supports = _dot_supports(prepared, options)
                self.assertTrue(dot_supports)
                for cursor, support in dot_supports:
                    snapshot = capture_writer_frontier_snapshot(
                        prepared=prepared,
                        runtime_options=options,
                        cursor=cursor,
                    )
                    artifact = writer_branch_transition_artifact_for_support(
                        prepared=prepared,
                        snapshot=snapshot,
                        support=support,
                    )
                    structural = verify_writer_branch_transition_artifact_consistency(
                        artifact
                    )
                    live = verify_writer_branch_transition_artifact_envelope(
                        prepared=prepared,
                        artifact=artifact,
                    )
                    replay = verify_writer_branch_transition_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )
                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(replay.accepted, replay.reason)
                    self.assertEqual(
                        replay.checked_relation_families,
                        ("component_boundary_transition",),
                    )
                    self.assertEqual(replay.unchecked_obligation_families, ())

    def test_coherent_wrong_next_root_is_rejected_semantically(self) -> None:
        case = next(
            item
            for item in DEFAULT_WRITER_CAPABILITY_CASES
            if item.name == "disconnected_cc_oxygen"
        )
        facts = ordinary_molecule_facts_from_smiles(
            case.smiles, case.extraction_options
        )
        prepared = _prepare(facts)
        options = _options(case.rooted_at_atom)
        cursor, support = _dot_supports(prepared, options)[0]
        artifact = writer_branch_transition_artifact_for_support(
            prepared=prepared,
            snapshot=capture_writer_frontier_snapshot(
                prepared=prepared,
                runtime_options=options,
                cursor=cursor,
            ),
            support=support,
        )
        forged = deepcopy(artifact)
        branch = next(
            item for item in forged["objects"] if item["kind"] == "branch_support"
        )
        event = next(
            item
            for item in branch["payload"]["graph_ring_delta"]["manifest"][
                "event_manifests"
            ]
            if item["kind"] == "component_boundary_emitted"
        )
        event["next_root"] = 0
        delta = branch["payload"]["graph_ring_delta"]
        delta["digest"] = _identity_digest(
            {"kind": delta["kind"], "manifest": delta["manifest"]}
        )
        _redigest_branch_artifact(forged, branch)

        structural = verify_writer_branch_transition_artifact_consistency(forged)
        replay = verify_writer_branch_transition_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=forged,
        )
        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(replay.accepted)
        self.assertIn("component_boundary_next_root_mismatch", replay.reason)

    def test_unreplayed_lifecycle_cannot_credit_component_boundary(self) -> None:
        case = next(
            item
            for item in DEFAULT_WRITER_CAPABILITY_CASES
            if item.name == "disconnected_cc_oxygen"
        )
        facts = ordinary_molecule_facts_from_smiles(
            case.smiles, case.extraction_options
        )
        prepared = _prepare(facts)
        options = _options(case.rooted_at_atom)
        cursor, support = _dot_supports(prepared, options)[0]
        artifact = writer_branch_transition_artifact_for_support(
            prepared=prepared,
            snapshot=capture_writer_frontier_snapshot(
                prepared=prepared, runtime_options=options, cursor=cursor
            ),
            support=support,
        )
        forged = deepcopy(artifact)
        branch = next(
            item for item in forged["objects"] if item["kind"] == "branch_support"
        )
        lifecycle = branch["payload"]["obligation_manifests"][
            "stereo_lifecycle"
        ][0]
        for field in ("is_noop", "is_empty", "is_discharged", "terminal_clean"):
            lifecycle[field] = False
        _redigest_branch_artifact(forged, branch)

        structural = verify_writer_branch_transition_artifact_consistency(forged)
        replay = verify_writer_branch_transition_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=forged,
        )
        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(replay.accepted, replay.reason)
        self.assertEqual(replay.checked_relation_families, ())
        self.assertIn("stereo_lifecycle", replay.unchecked_obligation_families)

    def test_rust_dot_choice_and_snapshot_resume_are_exact(self) -> None:
        ledger = {item.name: item for item in DEFAULT_WRITER_CAPABILITY_CASES}
        for fixture in load_pinned_south_star_disconnected_audit_cases(
            rdBase.rdkitVersion
        ):
            with self.subTest(case=fixture.name), TemporaryDirectory() as directory:
                case = ledger[fixture.name]
                facts = ordinary_molecule_facts_from_smiles(
                    case.smiles, case.extraction_options
                )
                prepared = _prepare(facts)
                options = _options(case.rooted_at_atom)
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=initial_writer_frontier_cursor(prepared, options),
                )
                path = Path(directory) / "asset"
                write_writer_continuation_asset(
                    path=path, prepared=prepared, snapshot=snapshot
                )
                decoder = MolToSmilesContinuationDecoder.from_asset(path)
                pending = [decoder]
                seen = set()
                dot_states = []
                while pending:
                    state = pending.pop()
                    if state.cache_key() in seen:
                        continue
                    seen.add(state.cache_key())
                    choices = state.next_choices
                    dot_states.extend(
                        (state, item.next_state)
                        for item in choices
                        if item.text == "."
                    )
                    pending.extend(item.next_state for item in choices)
                self.assertTrue(dot_states)
                for before, after in dot_states:
                    self.assertEqual(after.prefix, before.prefix + ".")
                    self.assertEqual(
                        MolToSmilesContinuationDecoder.from_snapshot(
                            path, before.snapshot()
                        ).cache_key(),
                        before.cache_key(),
                    )
                    self.assertEqual(
                        MolToSmilesContinuationDecoder.from_snapshot(
                            path, after.snapshot()
                        ).cache_key(),
                        after.cache_key(),
                    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts, writer_surface=SouthStarWriterSurface()
    )


def _options(rooted_at_atom: int) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _component_products(source, *, rooted_at_atom, extraction_options):
    molecule = Chem.MolFromSmiles(source)
    mappings: list[tuple[int, ...]] = []
    fragments = Chem.GetMolFrags(
        molecule,
        asMols=True,
        sanitizeFrags=True,
        fragsMolAtomMapping=mappings,
    )
    supports = []
    completions = []
    for fragment, mapping in zip(fragments, mappings, strict=True):
        facts = ordinary_molecule_facts_from_rdkit(fragment, extraction_options)
        local_root = mapping.index(rooted_at_atom) if rooted_at_atom in mapping else -1
        image = enumerate_prepared_writer_shaped_support(
            prepared=_prepare(facts),
            runtime_options=_options(local_root),
        )
        supports.append(tuple(sorted(image.strings)))
        completions.append(image.witness_count)
    return tuple(supports), tuple(completions)


def _dot_supports(prepared, options):
    pending = [initial_writer_frontier_cursor(prepared, options)]
    seen = set()
    found = []
    while pending:
        cursor = pending.pop(0)
        if cursor in seen:
            continue
        seen.add(cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        matches = tuple(item for item in batch.supports if item.emitted_text == ".")
        if matches:
            found.extend((cursor, item) for item in matches)
        pending.extend(
            item.successor_cursor for item in batch.text_choice_projection_certificates
        )
    return tuple(found)


def _redigest_branch_artifact(artifact, branch) -> None:
    old_branch_ref = branch["object_id"]
    branch_digest = _identity_digest(
        support_artifact_object_identity_term(branch["kind"], branch["payload"])
    )
    branch["digest"] = branch_digest
    branch["object_id"] = f"obj:{branch_digest}"
    projection = next(
        item for item in artifact["objects"] if item["kind"] == "text_projection"
    )
    projection["payload"]["branch_support_refs"] = [
        branch["object_id"] if ref == old_branch_ref else ref
        for ref in projection["payload"]["branch_support_refs"]
    ]
    old_projection_ref = projection["object_id"]
    projection_digest = _identity_digest(
        support_artifact_object_identity_term(
            projection["kind"], projection["payload"]
        )
    )
    projection["digest"] = projection_digest
    projection["object_id"] = f"obj:{projection_digest}"
    artifact["roots"]["branch_support_ref"] = branch["object_id"]
    if artifact["roots"]["text_projection_ref"] == old_projection_ref:
        artifact["roots"]["text_projection_ref"] = projection["object_id"]
    artifact["metrics"] = {
        **artifact_metrics(artifact["objects"]),
        "reachable_object_count": 3,
        "unreferenced_object_count": 0,
    }
    artifact["digest"] = _digest_terms_bounded(
        branch_transition_artifact_manifest(artifact),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.disconnected_branch_manifest.digest",
    )


if __name__ == "__main__":
    unittest.main()
