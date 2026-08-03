"""Architecture checks for the shared writer-test context boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import unittest


@dataclass(frozen=True, slots=True)
class DirectWriterContextConstructionException:
    module: str
    qualified_function: str
    operation: str
    reason: str


# These are tests of a constructor, a deliberately non-default fixture, or a
# snapshot contract at a production boundary.  Ordinary writer setup is not
# exempted here; it uses writer_test_context and its three primitives.
DIRECT_CONSTRUCTION_EXCEPTIONS = (
    DirectWriterContextConstructionException(
        "test_writer_continuation_automaton.py",
        "WriterContinuationAutomatonTest.test_weight_normalization_scales_completion_not_support",
        "prepare", "tests explicit preparation weighting behavior",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py", "_certified_rdkit_support",
        "prepare", "cached corpus certification owns its prepared source",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py", "_certified_rdkit_support",
        "snapshot", "cached corpus certification owns its initial snapshot",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py",
        "WriterDefaultContinuationCorpusTest._cross_all_continuation_tiers",
        "prepare", "fast corpus traversal owns its prepared source",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py",
        "WriterDefaultContinuationCorpusTest._cross_all_continuation_tiers",
        "snapshot", "fast corpus traversal owns its initial snapshot",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py",
        "WriterDefaultContinuationCorpusTest._cross_cached_continuation_tiers",
        "prepare", "cached continuation traversal owns its prepared source",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_continuation_corpus.py",
        "WriterDefaultContinuationCorpusTest._cross_cached_continuation_tiers",
        "snapshot", "cached continuation traversal owns its initial snapshot",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_stereo_audit_fixture.py",
        "_support_for_replaced_directional_site", "prepare",
        "fixture replacement deliberately changes the prepared site",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_stereo_audit_fixture.py",
        "WriterDefaultStereoAuditFixtureTest.setUpClass", "options",
        "fixture setup binds its explicit audit root",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_stereo_audit_fixture.py",
        "WriterDefaultStereoAuditFixtureTest.setUpClass", "prepare",
        "fixture setup binds its selected audit facts",
    ),
    DirectWriterContextConstructionException(
        "test_writer_default_stereo_audit_fixture.py",
        "WriterDefaultStereoAuditFixtureTest.setUpClass", "snapshot",
        "fixture setup binds its selected audit snapshot",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_shared_directional_ring_carrier_with_bond_text_choices",
        "prepare", "domain fixture owns its bond-text choice policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_directional_ring_carrier_with_bond_text_choices",
        "prepare", "domain fixture owns its bond-text choice policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_non_single_closure_triangle_with_ring_endpoint_choices",
        "prepare", "domain fixture owns its ring-endpoint choice policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_single_closure_triangle_with_ring_endpoint_choices",
        "prepare", "domain fixture owns its ring-endpoint choice policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_aromatic_triangle_with_bond_text_choices",
        "prepare", "domain fixture owns its aromatic bond-text policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_bridge_separated_two_cycle_with_policy_slots",
        "prepare", "domain fixture owns its policy-slot layout",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "_prepare_with_ordinary_policy_options_and_slots",
        "prepare", "domain fixture owns its explicit policy-slot layout",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "WriterStateKernelTest.test_writer_frontier_counts_duplicate_token_paths_to_same_state",
        "prepare", "test directly exercises frontier count construction",
    ),
    DirectWriterContextConstructionException(
        "test_writer_state_kernel.py", "WriterStateKernelTest.test_missing_writer_bond_domain_fails_closed",
        "prepare", "test directly constructs the invalid preparation boundary",
    ),
    DirectWriterContextConstructionException(
        "test_writer_stereo_residual.py", "_prepare_directional_ring_carrier_facts",
        "prepare", "domain fixture owns its joint closure policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_stereo_residual.py", "_prepare_directional_non_single_ring_carrier_facts",
        "prepare", "domain fixture owns its joint closure policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_stereo_residual.py", "_prepare_shared_directional_ring_carrier_facts",
        "prepare", "domain fixture owns its joint closure policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_stereo_residual.py", "_prepare_three_site_shared_directional_ring_carrier_facts",
        "prepare", "domain fixture owns its joint closure policy",
    ),
    DirectWriterContextConstructionException(
        "test_writer_stereo_residual.py",
        "WriterStereoResidualTest.test_joint_directional_non_single_ring_carrier_support_is_certified",
        "snapshot", "test directly checks the initial snapshot certificate",
    ),
    DirectWriterContextConstructionException(
        "test_writer_snapshot.py", "_terminal_tetra_key", "prepare",
        "snapshot key fixture deliberately prepares a terminal source",
    ),
    DirectWriterContextConstructionException(
        "test_writer_snapshot.py", "_prepare_aromatic_triangle", "prepare",
        "aromatic snapshot fixture owns its non-default policy",
    ),
)

# Snapshot tests below intentionally capture the initial cursor directly to
# exercise the production snapshot API and its certificates.
for _name in (
    "test_snapshot_advance_emits_step_certificate",
    "test_snapshot_advance_outcome_carries_product_projection_identity",
    "test_snapshot_advance_invalid_text_has_no_projection_match",
    "test_snapshot_replay_sequence_invalid_text_carries_certificate",
    "test_snapshot_advance_invalid_text_certificate_rejects_match",
    "test_snapshot_advance_returns_blocked_product_for_unsupported_capability",
    "test_snapshot_advance_blocked_error_does_not_delegate_to_choice_snapshot_blockers",
    "test_snapshot_blocked_advance_certificate_rejects_cursor_mismatch",
    "test_snapshot_advance_successor_cursor_comes_from_text_projection_certificate",
    "test_snapshot_advance_outcome_rejects_stale_step_projection",
    "test_snapshot_replay_sequence_rejects_projection_chain_mismatch",
    "test_snapshot_replay_certificate_tracks_prefix_steps",
    "test_empty_snapshot_replay_has_empty_certificate",
    "test_prefix_read_exposes_replay_certificate",
    "test_prefix_read_certificate_binds_final_frontier_counts",
    "test_prefix_read_certificate_binds_replay_final_snapshot",
    "test_prefix_read_certificate_rejects_replay_final_snapshot_mismatch",
    "test_invalid_snapshot_advance_has_no_step_certificate",
    "test_snapshot_step_certificate_rejects_malformed_inputs",
    "test_snapshot_replay_certificate_rejects_malformed_inputs",
):
    DIRECT_CONSTRUCTION_EXCEPTIONS += (
        DirectWriterContextConstructionException(
            "test_writer_snapshot.py", f"WriterSnapshotTest.{_name}", "snapshot",
            "test directly exercises the initial snapshot production boundary",
        ),
    )


def _qualified_function(tree: ast.AST, node: ast.AST) -> str:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    names: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(id(current))
    return ".".join(reversed(names)) or "<module>"


def _observed_constructions() -> set[tuple[str, str, str]]:
    observed: set[tuple[str, str, str]] = set()
    root = Path(__file__).parent
    for path in sorted(root.glob("test_writer*.py")):
        if path.name in {"test_writer_test_context.py", Path(__file__).name}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            operation = None
            if node.func.id == "SouthStarRuntimeOptions":
                keywords = {item.arg for item in node.keywords if item.arg}
                serial = next(
                    (item.value for item in node.keywords
                     if item.arg == "serialization_language"), None,
                )
                canonical = next(
                    (item.value for item in node.keywords if item.arg == "canonical"), None,
                )
                random = next(
                    (item.value for item in node.keywords if item.arg == "do_random"), None,
                )
                if (
                    isinstance(serial, ast.Attribute)
                    and serial.attr == "WRITER_SHAPED"
                    and keywords <= {"rooted_at_atom", "canonical", "do_random", "serialization_language"}
                    and (canonical is None or isinstance(canonical, ast.Constant) and canonical.value is False)
                    and (random is None or isinstance(random, ast.Constant) and random.value is True)
                ):
                    operation = "options"
            elif node.func.id == "prepare_south_star_mol_from_facts":
                surface = next(
                    (item.value for item in node.keywords if item.arg == "writer_surface"), None,
                )
                if (
                    isinstance(surface, ast.Call)
                    and isinstance(surface.func, ast.Name)
                    and surface.func.id == "SouthStarWriterSurface"
                    and not surface.args and not surface.keywords
                ):
                    operation = "prepare"
            elif node.func.id == "capture_writer_frontier_snapshot":
                cursor = next(
                    (item.value for item in node.keywords if item.arg == "cursor"), None,
                )
                if (
                    isinstance(cursor, ast.Call)
                    and isinstance(cursor.func, ast.Name)
                    and cursor.func.id == "initial_writer_frontier_cursor"
                ):
                    operation = "snapshot"
            if operation:
                observed.add((path.name, _qualified_function(tree, node), operation))
    return observed


class WriterTestContextAuthorityTest(unittest.TestCase):
    def test_all_generic_direct_constructions_are_explicitly_accounted_for(self):
        observed = _observed_constructions()
        allowed = {
            (item.module, item.qualified_function, item.operation)
            for item in DIRECT_CONSTRUCTION_EXCEPTIONS
        }
        self.assertEqual(observed - allowed, set())
        self.assertEqual(allowed - observed, set())
        self.assertTrue(all(item.reason.strip() for item in DIRECT_CONSTRUCTION_EXCEPTIONS))

    def test_no_top_level_generic_context_helper_clones_remain(self):
        names = {"_prepare", "_prepared", "_prepare_default", "_writer_options", "_options", "_initial_snapshot"}
        for path in Path(__file__).parent.glob("test_writer*.py"):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertNotIn(node.name, names, path.name)
