"""Acyclic and shared directional support-artifact replay contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    if name.endswith("coherent_forgeries_reject_semantically"):
        return False
    return name.startswith((
        "test_directional_rooted_",
        "test_shared_acyclic_directional_",
        "test_shared_ring_carrier_",
        "test_directional_carrier_",
    ))


class WriterSupportArtifactDirectionalAcyclicTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactDirectionalAcyclicTest, _name, _method)
