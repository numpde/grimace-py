"""Product-level regressions for ordered coupled tetrahedral transitions."""

from __future__ import annotations

from collections import deque
import unittest

from rdkit import Chem

from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.writer_branch_transition_artifact import (
    writer_branch_transition_artifact_for_support,
)
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import (
    verify_writer_branch_transition_artifact_for_facts,
)
from grimace._south_star1.writer_frontier import (
    _checked_writer_frontier_branch_supports,
    initial_writer_frontier_cursor,
)
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_envelope_terms import _identity_digest
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_test_context import writer_test_context


class CoupledTetrahedralWriterTest(unittest.TestCase):
    def test_remote_identity_matrix_quotients_only_symmetry_equivalents(self) -> None:
        sources = {
            "A": "[C@H](F)([C@](F)(Cl)Br)[C@@](F)(Cl)Br",
            "B": "[C@@H](F)([C@](F)(Cl)Br)[C@@](F)(Cl)Br",
            "C": "[C@H](F)([C@@](F)(Cl)Br)[C@](F)(Cl)Br",
            "D": "[C@@H](F)([C@@](F)(Cl)Br)[C@](F)(Cl)Br",
        }
        facts = {
            name: ordinary_molecule_facts_from_rdkit(
                Chem.MolFromSmiles(smiles),
                options=RdkitOrdinaryExtractionOptions(
                    stereo_site_options=OrdinaryStereoSiteOptions(
                        ligand_equivalence="exact_stereochemical_graph_automorphism"
                    ),
                    stereo_site_discovery_mode="specified_closure",
                ),
            )
            for name, smiles in sources.items()
        }
        self.assertTrue(facts_are_isomorphic(facts["A"], facts["D"]).isomorphic)
        self.assertTrue(facts_are_isomorphic(facts["B"], facts["C"]).isomorphic)
        self.assertFalse(facts_are_isomorphic(facts["A"], facts["B"]).isomorphic)
        self.assertFalse(facts_are_isomorphic(facts["A"], facts["C"]).isomorphic)

    def test_all_multi_transition_branches_replay_as_ordered_chains(self) -> None:
        smiles = "[C@H](F)([C@](F)(Cl)Br)[C@@](F)(Cl)Br"
        facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles(smiles),
            options=RdkitOrdinaryExtractionOptions(
                stereo_site_options=OrdinaryStereoSiteOptions(
                    ligand_equivalence="exact_stereochemical_graph_automorphism"
                ),
                stereo_site_discovery_mode="specified_closure",
            ),
        )
        context = writer_test_context(facts, rooted_at_atom=0)
        prepared = context.prepared
        runtime_options = context.runtime_options
        initial = context.initial_snapshot
        pending = deque([initial])
        seen: set[str] = set()
        checked = 0
        while pending:
            snapshot = pending.popleft()
            cursor_digest = _identity_digest(snapshot.cursor)
            if cursor_digest in seen:
                continue
            seen.add(cursor_digest)
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                snapshot.cursor,
                include_counts=False,
                include_frontier_certificate=True,
                include_count_certificate=False,
            )
            for support in batch.supports:
                if len(support.residual_work_evidence) < 2:
                    continue
                artifact = writer_branch_transition_artifact_for_support(
                    prepared=prepared,
                    snapshot=snapshot,
                    support=support,
                )
                replay = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=runtime_options,
                    artifact=artifact,
                )
                self.assertTrue(replay.accepted, replay.reason)
                self.assertEqual(
                    replay.semantically_replayed_operations,
                    (
                        "tetrahedral atom-token restriction",
                        "tetrahedral local-order factor closure",
                    ),
                )
                self.assertEqual(replay.unchecked_obligation_families, ())
                checked += 1
            pending.extend(
                capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=runtime_options,
                    cursor=projection.successor_cursor,
                )
                for projection in batch.text_choice_projection_certificates
            )
        self.assertEqual(checked, 48)
