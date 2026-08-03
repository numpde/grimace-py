"""Specified-tetrahedral lifecycle-link contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    if name == "test_specified_tetra_residual_manifest_digest_mismatch_is_rejected":
        return False
    return name.startswith((
        "test_specified_tetra_residual_",
        "test_specified_tetra_no_second_authority_",
    ))


class WriterSupportArtifactTetraLifecycleTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactTetraLifecycleTest, _name, _method)
