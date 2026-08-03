"""Specified-tetrahedral transition and local-order contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name in {
        "test_specified_tetra_raw_smiles_blocks_without_potential_sites",
        "test_supported_specified_tetra_artifact_is_offline_complete",
        "test_specified_tetra_residual_manifest_digest_mismatch_is_rejected",
    } or name.startswith((
        "test_specified_tetra_transition_",
        "test_specified_tetra_atom_token_",
        "test_specified_tetra_local_order_",
    ))


class WriterSupportArtifactTetraTransitionTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactTetraTransitionTest, _name, _method)
