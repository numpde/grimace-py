"""Directional and ring support-artifact replay contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    if name in {
        "test_directional_ring_carrier_root_zero_artifact_builds_with_default_budget",
        "test_non_single_directional_ring_root_zero_artifact_replays_completely",
    }:
        return False
    if name.endswith("coherent_forgeries_are_rejected") or name.endswith(
        "coherent_forgeries_reject_semantically"
    ):
        return False
    return name.startswith((
        "test_reduced_directional_ring_",
        "test_directional_ring_pair_",
        "test_directional_ring_opening_",
    ))


class WriterSupportArtifactDirectionalReplayTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactDirectionalReplayTest, _name, _method)
