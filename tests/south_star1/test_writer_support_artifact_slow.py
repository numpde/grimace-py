"""Explicit slow rich support-artifact probes."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


class WriterSupportArtifactSlowTest(unittest.TestCase):
    test_directional_ring_carrier_root_zero_artifact_builds_with_default_budget = getattr(
        WriterSupportArtifactDomainMethods,
        "test_directional_ring_carrier_root_zero_artifact_builds_with_default_budget",
    )
    test_non_single_directional_ring_root_zero_artifact_replays_completely = getattr(
        WriterSupportArtifactDomainMethods,
        "test_non_single_directional_ring_root_zero_artifact_replays_completely",
    )
