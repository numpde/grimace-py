"""Obligation classification and replay-credit contracts."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import WriterSupportArtifactDomainMethods


def _selected(name: str) -> bool:
    return name in {
        "test_stereo_lifecycle_requires_exact_replay_credit",
        "test_default_corpus_obligations_are_classified",
    } or name.startswith((
        "test_terminal_clean_",
        "test_terminal_manifest_",
        "test_terminal_obligation_",
        "test_ring_finite_relation_",
        "test_ring_summary_",
        "test_ring_obligation_",
        "test_stereo_lifecycle_flags_",
        "test_descriptive_flags_",
        "test_branch_local_ledger_",
        "test_obligation_summary_",
    ))


class WriterSupportArtifactObligationReplayTest(unittest.TestCase):
    pass


for _name, _method in vars(WriterSupportArtifactDomainMethods).items():
    if _selected(_name):
        setattr(WriterSupportArtifactObligationReplayTest, _name, _method)
