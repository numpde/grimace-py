"""Support-string replay and terminal identity contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name.startswith((
        "test_support_string_replay_paths_",
        "test_replay_path_",
        "test_terminal_support_identities_",
        "test_terminal_projection_",
        "test_terminal_support_ordinal_",
        "test_terminal_support_ref_",
    )) or name == "test_terminal_support_identity_forgeries_reject_after_redigest"


class WriterSupportArtifactPathIdentityTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactPathIdentityTest, _name, _method)
