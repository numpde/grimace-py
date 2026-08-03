"""Top-level rich support-artifact facts-bound contract tests."""

import unittest

from tests.south_star1.writer_support_artifact_domain_methods import (
    WriterSupportArtifactDomainMethods,
)


_NAMES = (
    "test_snapshot_artifact_verifies_against_matching_facts",
    "test_prefix_artifact_verifies_against_matching_facts",
    "test_wrong_facts_are_rejected",
    "test_wrong_runtime_options_are_rejected",
    "test_wrong_explicit_policy_is_rejected",
    "test_mutated_prepared_identity_is_rejected",
    "test_mutated_source_prepared_identity_is_rejected",
    "test_structurally_invalid_artifact_is_rejected",
    "test_unknown_object_kind_is_rejected_by_structural_checker",
    "test_facts_bound_verifier_reports_bracket_atom_offline_check",
    "test_facts_bound_verifier_reports_isotope_atom_offline_check",
    "test_facts_bound_verifier_reports_joint_double_closure_offline_check",
    "test_facts_bound_verifier_reports_joint_triple_closure_offline_check",
    "test_offline_coverage_ledger_classifies_partial_replay",
    "test_offline_coverage_ledger_covers_artifact_object_kinds",
)


class WriterSupportArtifactFactVerifierTest(unittest.TestCase):
    pass


for _name in _NAMES:
    setattr(WriterSupportArtifactFactVerifierTest, _name, getattr(WriterSupportArtifactDomainMethods, _name))
