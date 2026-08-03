"""Graph/ring and local-evidence support-artifact contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name.startswith((
        "test_offline_bracket_atom_replay_",
        "test_offline_tetra_bracket_atom_replay_",
        "test_offline_joint_closure_replay_",
        "test_graph_ring_",
        "test_branch_projection_",
        "test_local_",
    ))


class WriterSupportArtifactGraphRelationTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactGraphRelationTest, _name, _method)
