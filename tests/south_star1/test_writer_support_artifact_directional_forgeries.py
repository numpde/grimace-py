"""Directional and ring coherent-forgery contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name.startswith((
        "test_shared_acyclic_directional_coherent_forgeries_",
        "test_directional_carrier_coherent_forgeries_",
    ))


class WriterSupportArtifactDirectionalForgeryTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactDirectionalForgeryTest, _name, _method)
