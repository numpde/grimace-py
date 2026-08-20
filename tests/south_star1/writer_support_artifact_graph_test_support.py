"""Non-test support for rich support-artifact relationships."""

from dataclasses import replace
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from tests.south_star1.writer_support_artifact_queries import first_graph_ring_delta_event
from tests.south_star1.writer_support_artifact_queries import first_local_evidence





def first_directional_bond_delta_branch(artifact):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        delta = item["payload"]["graph_ring_delta"]
        if delta["kind"] != "bond_advance":
            continue
        event = first_graph_ring_delta_event(item, "bond_emitted")
        direction_mark = event["direction_mark"]
        if direction_mark["value"] != 0:
            return item
    raise AssertionError("missing directional bond delta branch")


def first_closure_evidence_item(artifact):
    evidence = first_local_evidence(artifact, "closure_bond_text")
    return evidence["manifest"]["items"][0]


def tetra_facts_with_implicit_h_only_outside_specified_site(facts):
    site = facts.stereo.tetrahedral[0]
    outside_occurrence = LigandOccurrence(
        id=OccurrenceId(99),
        site=SiteId(99),
        kind=LigandKind.IMPLICIT_H,
        atom=site.center,
        bond=None,
    )
    return replace(
        facts,
        stereo=replace(
            facts.stereo,
            tetrahedral=(
                replace(
                    site,
                    ligand_occurrences=site.ligand_occurrences[:-1],
                    reference_order=site.reference_order[:-1],
                ),
            ),
        ),
        ligand_occurrences=facts.ligand_occurrences[:-1] + (outside_occurrence,),
    )

