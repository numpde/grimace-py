from __future__ import annotations

from dataclasses import dataclass
import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
    SOUTH_STAR_PRIVATE_DOMAIN,
    SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS,
    SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES,
    SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_spec_oracle import south_star_spec_oracle_report
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


SOUTH_STAR_SHARED_PIPELINE_GENERATION_BASIS = (
    "south_star_graph_native_equation_solved_tree_traversal"
)
SOUTH_STAR_PIPELINE_PROVENANCE_STAGES: tuple[str, ...] = (
    "molecule_facts",
    "traversal_plan",
    "observation",
    "constraint_family",
    "solver_assignment",
    "renderer",
    "semantic_evidence",
)


@dataclass(frozen=True, slots=True)
class SouthStarReadinessMatrix:
    unified_reference_backed_case_ids: tuple[str, ...]
    unified_reference_promotion_candidate_case_ids: tuple[str, ...]
    temporary_witness_case_ids: tuple[str, ...]
    regression_backed_case_ids: tuple[str, ...]
    public_api_blocker_case_ids: tuple[str, ...]
    supported_feature_areas: tuple[str, ...]
    unsupported_categories: tuple[str, ...]
    policy_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarPublicApiPromotionGate:
    gate_id: str
    evidence: str
    verification: str


@dataclass(frozen=True, slots=True)
class SouthStarUnifiedReferencePromotionCheck:
    case_id: str
    current_authority: str
    shared_pipeline_generated: bool
    pipeline_coverage: tuple[SouthStarPipelineCoverageRecord, ...]
    spine_bypass_count: int
    promoted: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarAuthorityPromotionCandidateInventoryItem:
    case_id: str
    fixture_family: str
    feature_area: str
    current_authority: str
    authority_class: str
    shared_spine_coverage: str
    fixture_string_generation_risk: str
    temporary_witness_dependency: str
    output_count: int
    estimated_product_size: int
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarAuthorityPromotionChecklistRecord:
    case_id: str
    support_authority: str
    shared_spine_coverage: str
    fixture_string_generation_prohibited: bool
    temporary_witness_as_cross_check_only: bool
    semantic_parseback: str
    output_count: int
    estimated_product_size: int
    unresolved_blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarPipelineCoverageRecord:
    stage_id: str
    status: str
    evidence: str


SOUTH_STAR_PUBLIC_API_PROMOTION_GATES: tuple[SouthStarPublicApiPromotionGate, ...] = (
    SouthStarPublicApiPromotionGate(
        gate_id="private_boundary",
        evidence="MolToSmilesEnumS is not exported until all gates pass.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_private_api_boundary -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="supported_domain_manifest",
        evidence="Supported domains, authorities, policies, and blockers are "
        "declared by the South Star domain manifest.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_domain_manifest -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="grammar_conformance",
        evidence="Outputs satisfy the declared South Star grammar subset.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_grammar_conformance -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="semantic_identity",
        evidence="Outputs parse back to the intended graph and stereo identity.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_semantic_identity "
        "tests.south_star.test_output_correctness_harness -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="support_evidence_classification",
        evidence="Every promoted supported domain classifies support evidence as "
        "unified-reference-backed, temporary witness, or regression witness.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_first_domain_completeness "
        "tests.south_star.test_expanded_support_fixtures -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="unified_reference_promotion",
        evidence="A domain may be treated as unified-reference-backed only when "
        "it is generated by the shared fact/event/constraint/renderer pipeline "
        "and its manifest authority is no longer temporary or regression-only.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_package_readiness -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="pipeline_provenance",
        evidence="Package-ready cases must have explicit MoleculeFacts, "
        "TraversalPlan, Observation, ConstraintFamily, SolverAssignment, "
        "Renderer, and semantic-evidence coverage; emitted constrained syntax "
        "must not bypass typed assignments.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_package_readiness -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="unsupported_category_completeness",
        evidence="Unsupported molecule classes fail before enumeration with "
        "manifested categories.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_support_gates -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="complexity_guardrails",
        evidence="Generation diagnostics expose product-size and assignment "
        "guardrails for representative promoted domains.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_complexity_guardrails -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="documentation_contract",
        evidence="Docs name the contract, policy set, supported and unsupported "
        "domains, parser dependency, and RDKit-parity distinction.",
        verification="explicit review: docs/enum-s.md and public API docs match "
        "the exported surface.",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="performance_evidence_boundary",
        evidence="EnumS complexity diagnostics are guardrails, not release "
        "performance claims; user-facing speed claims require a separate "
        "semantic-enumerator benchmark artifact.",
        verification="explicit review: docs/enum-s.md performance evidence "
        "boundary and release notes avoid unsupported speed claims.",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="release_notes_scope",
        evidence="Release notes state the semantic contract and distinguish it "
        "from MolToSmilesEnum RDKit writer parity.",
        verification="explicit review: release notes match the exact exported "
        "surface and readiness matrix.",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="full_readiness_runner",
        evidence="The named package-readiness runner passes as the promotion "
        "entry point.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.run_south_star_package_readiness -q",
    ),
)


class SouthStarPackageReadinessTests(unittest.TestCase):
    def test_public_api_promotion_gate_is_explicit(self) -> None:
        gate_ids = tuple(gate.gate_id for gate in SOUTH_STAR_PUBLIC_API_PROMOTION_GATES)

        self.assertEqual(
            (
                "private_boundary",
                "supported_domain_manifest",
                "grammar_conformance",
                "semantic_identity",
                "support_evidence_classification",
                "unified_reference_promotion",
                "pipeline_provenance",
                "unsupported_category_completeness",
                "complexity_guardrails",
                "documentation_contract",
                "performance_evidence_boundary",
                "release_notes_scope",
                "full_readiness_runner",
            ),
            gate_ids,
        )
        for gate in SOUTH_STAR_PUBLIC_API_PROMOTION_GATES:
            with self.subTest(gate_id=gate.gate_id):
                self.assertTrue(gate.evidence)
                self.assertTrue(
                    gate.verification.startswith("PYTHONPATH=python:.")
                    or gate.verification.startswith("explicit review:")
                )

    def test_readiness_matrix_reports_evidence_classes(self) -> None:
        matrix = south_star_package_readiness_matrix()

        self.assertEqual(
            (
                "explicit_bracket_hydrogen_h2",
                "radical_atom_text_hydrogen",
                "radical_atom_text_methyl",
                "radical_atom_text_oxygen",
                "charged_atom_text_chloride",
                "charged_atom_text_ammonium",
                "charged_atom_text_methylammonium",
            ),
            matrix.unified_reference_backed_case_ids,
        )
        self.assertIn(
            "isolated_alkene_z",
            matrix.unified_reference_promotion_candidate_case_ids,
        )
        self.assertIn(
            "explicit_bracket_hydrogen_h2",
            matrix.unified_reference_promotion_candidate_case_ids,
        )
        self.assertIn(
            "simple_saturated_monocycle_cyclohexane",
            matrix.unified_reference_promotion_candidate_case_ids,
        )
        self.assertIn("isolated_alkene_z", matrix.temporary_witness_case_ids)
        self.assertNotIn(
            "explicit_bracket_hydrogen_h2",
            matrix.regression_backed_case_ids,
        )
        self.assertIn("isolated_alkene_z", matrix.public_api_blocker_case_ids)
        self.assertNotIn(
            "radical_atom_text_methyl",
            matrix.public_api_blocker_case_ids,
        )
        self.assertNotIn(
            "charged_atom_text_methylammonium",
            matrix.public_api_blocker_case_ids,
        )
        self.assertIn("explicit_bracket_hydrogen", matrix.supported_feature_areas)
        self.assertIn("charged_atom_text", matrix.supported_feature_areas)
        self.assertIn("radical_atom_text", matrix.supported_feature_areas)
        self.assertIn("polycyclic_ring_stereo", matrix.supported_feature_areas)
        self.assertIn("ring_tetrahedral_monocycle", matrix.supported_feature_areas)
        self.assertNotIn("unsupported_radical_atom", matrix.unsupported_categories)
        self.assertEqual(
            (
                "maximal_eligible_carrier",
                "all_fragment_orders",
                "first_occurrence_deduplication",
            ),
            matrix.policy_names,
        )

    def test_every_expanded_case_has_package_readiness_diagnostics(self) -> None:
        for case in load_south_star_expanded_support_cases():
            with self.subTest(case_id=case.case_id):
                report = south_star_support_gate_report(parse_smiles(case.source_smiles))
                self.assertTrue(report.supported, report.unsupported_features)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(case.expected_support, result.outputs)
                self.assertIsNotNone(result.generation_diagnostics)
                self.assertIn(
                    case.support_authority,
                    SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
                )

    def test_unified_reference_promotion_checks_keep_temporary_witnesses_blocking(
        self,
    ) -> None:
        checks = south_star_unified_reference_promotion_checks()
        checks_by_id = {check.case_id: check for check in checks}

        self.assertIn("isolated_alkene_z", checks_by_id)
        self.assertIn("simple_saturated_monocycle_cyclohexane", checks_by_id)
        self.assertIn("radical_atom_text_methyl", checks_by_id)
        for case_id in (
            "isolated_alkene_z",
            "simple_saturated_monocycle_cyclohexane",
        ):
            check = checks_by_id[case_id]
            with self.subTest(case_id=case_id):
                self.assertTrue(check.shared_pipeline_generated)
                self.assertFalse(check.promoted)
                self.assertIn(
                    "support_authority_is_not_unified_reference",
                    check.blockers,
                )
                self.assertEqual(0, check.spine_bypass_count)
                self.assertEqual(
                    SOUTH_STAR_PIPELINE_PROVENANCE_STAGES,
                    tuple(
                        record.stage_id for record in check.pipeline_coverage
                    ),
                )

        promoted = checks_by_id["radical_atom_text_methyl"]
        self.assertTrue(promoted.shared_pipeline_generated)
        self.assertTrue(promoted.promoted)
        self.assertEqual((), promoted.blockers)
        self.assertEqual(0, promoted.spine_bypass_count)

        self.assertTrue(
            all(
                check.promoted
                == (check.current_authority in SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES)
                for check in checks
            )
        )

    def test_pipeline_provenance_classifies_spine_coverage(self) -> None:
        checks = south_star_unified_reference_promotion_checks()
        checks_by_id = {check.case_id: check for check in checks}

        alkene = checks_by_id["isolated_alkene_z"]
        tetrahedral = checks_by_id["implicit_h_tetrahedral_center"]
        ring_tetrahedral = checks_by_id["ring_tetrahedral_monocycle_center"]
        hydrogen = checks_by_id["explicit_bracket_hydrogen_h2"]

        self.assertEqual(
            "covered",
            _coverage_status(alkene, "constraint_family"),
        )
        self.assertEqual(
            "covered",
            _coverage_status(tetrahedral, "constraint_family"),
        )
        self.assertEqual(
            "covered",
            _coverage_status(ring_tetrahedral, "constraint_family"),
        )
        self.assertEqual(
            "not_required",
            _coverage_status(hydrogen, "constraint_family"),
        )
        for check in checks:
            with self.subTest(case_id=check.case_id):
                self.assertEqual(0, check.spine_bypass_count)
                if check.promoted:
                    self.assertFalse(
                        {
                            record.status
                            for record in check.pipeline_coverage
                            if record.status == "missing"
                        }
                    )


    def test_authority_promotion_inventory_is_reviewable(self) -> None:
        inventory = south_star_authority_promotion_candidate_inventory()
        inventory_by_id = {item.case_id: item for item in inventory}

        expected_case_ids = {
            case.case_id
            for case in (
                *load_south_star_exact_first_domain_cases(),
                *load_south_star_expanded_support_cases(),
            )
        }
        self.assertEqual(expected_case_ids, set(inventory_by_id))
        self.assertEqual(len(expected_case_ids), len(inventory))

        alkene = inventory_by_id["isolated_alkene_z"]
        self.assertEqual("exact_first_domain", alkene.fixture_family)
        self.assertEqual("first_domain_directional_bond_stereo", alkene.feature_area)
        self.assertEqual("temporary_witness", alkene.authority_class)
        self.assertEqual(
            "none_detected_shared_spine_matches_expected_support",
            alkene.fixture_string_generation_risk,
        )
        self.assertEqual(
            "temporary_witness_first_domain_shared_spine",
            alkene.temporary_witness_dependency,
        )
        self.assertEqual(alkene.output_count, alkene.estimated_product_size)
        self.assertEqual(
            ("support_authority_is_not_unified_reference",),
            alkene.blockers,
        )

        methyl = inventory_by_id["radical_atom_text_methyl"]
        self.assertEqual("expanded_support", methyl.fixture_family)
        self.assertEqual("radical_atom_text", methyl.feature_area)
        self.assertEqual("unified_reference", methyl.authority_class)
        self.assertEqual("none", methyl.temporary_witness_dependency)
        self.assertEqual((), methyl.blockers)

        for item in inventory:
            with self.subTest(case_id=item.case_id):
                self.assertIn(
                    item.shared_spine_coverage,
                    {"complete", "incomplete"},
                )
                self.assertGreater(item.output_count, 0)
                self.assertGreater(item.estimated_product_size, 0)
                if item.authority_class == "temporary_witness":
                    self.assertNotEqual("none", item.temporary_witness_dependency)
                if item.authority_class == "unified_reference":
                    self.assertEqual("none", item.temporary_witness_dependency)


    def test_authority_promotion_checklist_records_are_explicit(self) -> None:
        records = south_star_authority_promotion_checklist_records()
        records_by_id = {record.case_id: record for record in records}
        inventory_by_id = {
            item.case_id: item
            for item in south_star_authority_promotion_candidate_inventory()
        }

        self.assertEqual(set(inventory_by_id), set(records_by_id))

        alkene = records_by_id["isolated_alkene_z"]
        self.assertEqual(
            "temporary_witness_first_domain_shared_spine",
            alkene.support_authority,
        )
        self.assertEqual("complete", alkene.shared_spine_coverage)
        self.assertTrue(alkene.fixture_string_generation_prohibited)
        self.assertTrue(alkene.temporary_witness_as_cross_check_only)
        self.assertEqual("covered", alkene.semantic_parseback)
        self.assertEqual(
            ("support_authority_is_not_unified_reference",),
            alkene.unresolved_blockers,
        )

        methyl = records_by_id["radical_atom_text_methyl"]
        self.assertTrue(methyl.fixture_string_generation_prohibited)
        self.assertFalse(methyl.temporary_witness_as_cross_check_only)
        self.assertEqual("covered", methyl.semantic_parseback)
        self.assertEqual((), methyl.unresolved_blockers)

        for record in records:
            with self.subTest(case_id=record.case_id):
                inventory_item = inventory_by_id[record.case_id]
                self.assertEqual(
                    inventory_item.current_authority,
                    record.support_authority,
                )
                self.assertEqual(
                    inventory_item.shared_spine_coverage,
                    record.shared_spine_coverage,
                )
                self.assertEqual(
                    inventory_item.output_count,
                    record.output_count,
                )
                self.assertEqual(
                    inventory_item.estimated_product_size,
                    record.estimated_product_size,
                )
                self.assertEqual(
                    inventory_item.blockers,
                    record.unresolved_blockers,
                )
                self.assertIn(record.semantic_parseback, {"covered", "missing"})


def south_star_package_readiness_matrix() -> SouthStarReadinessMatrix:
    first_domain_cases = load_south_star_exact_first_domain_cases()
    expanded_cases = load_south_star_expanded_support_cases()
    promotion_checks = south_star_unified_reference_promotion_checks()
    unified_reference_backed_case_ids = tuple(
        check.case_id for check in promotion_checks if check.promoted
    )
    unified_reference_promotion_candidate_case_ids = tuple(
        check.case_id
        for check in promotion_checks
        if check.shared_pipeline_generated
    )
    temporary_witness_case_ids = tuple(
        case.case_id
        for case in first_domain_cases
        if case.support_authority in SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES
    ) + tuple(
        case.case_id
        for case in expanded_cases
        if case.support_authority in SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES
    )
    regression_backed_case_ids = tuple(
        case.case_id
        for case in expanded_cases
        if case.support_authority == SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
    )
    public_api_blocker_case_ids = tuple(
        check.case_id for check in promotion_checks if check.blockers
    )
    return SouthStarReadinessMatrix(
        unified_reference_backed_case_ids=unified_reference_backed_case_ids,
        unified_reference_promotion_candidate_case_ids=(
            unified_reference_promotion_candidate_case_ids
        ),
        temporary_witness_case_ids=temporary_witness_case_ids,
        regression_backed_case_ids=regression_backed_case_ids,
        public_api_blocker_case_ids=public_api_blocker_case_ids,
        supported_feature_areas=tuple(
            sorted(SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas)
        ),
        unsupported_categories=tuple(
            sorted(SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories)
        ),
        policy_names=(
            "maximal_eligible_carrier",
            "all_fragment_orders",
            "first_occurrence_deduplication",
        ),
    )


def south_star_authority_promotion_candidate_inventory(
) -> tuple[SouthStarAuthorityPromotionCandidateInventoryItem, ...]:
    checks_by_id = {
        check.case_id: check for check in south_star_unified_reference_promotion_checks()
    }
    items: list[SouthStarAuthorityPromotionCandidateInventoryItem] = []

    for case in load_south_star_exact_first_domain_cases():
        items.append(
            _authority_inventory_item(
                case_id=case.case_id,
                source_smiles=case.source_smiles,
                fixture_family="exact_first_domain",
                feature_area="first_domain_directional_bond_stereo",
                current_authority=case.support_authority,
                expected_support=case.expected_support,
                check=checks_by_id[case.case_id],
            )
        )

    for case in load_south_star_expanded_support_cases():
        items.append(
            _authority_inventory_item(
                case_id=case.case_id,
                source_smiles=case.source_smiles,
                fixture_family="expanded_support",
                feature_area=case.feature_area,
                current_authority=case.support_authority,
                expected_support=case.expected_support,
                check=checks_by_id[case.case_id],
            )
        )

    return tuple(items)


def south_star_authority_promotion_checklist_records(
) -> tuple[SouthStarAuthorityPromotionChecklistRecord, ...]:
    checks_by_id = {
        check.case_id: check for check in south_star_unified_reference_promotion_checks()
    }
    return tuple(
        _authority_promotion_checklist_record(
            inventory_item,
            check=checks_by_id[inventory_item.case_id],
        )
        for inventory_item in south_star_authority_promotion_candidate_inventory()
    )


def south_star_unified_reference_promotion_checks(
) -> tuple[SouthStarUnifiedReferencePromotionCheck, ...]:
    semantic_cases_by_id = {
        case.case_id: case for case in load_south_star_semantic_cases()
    }
    checks: list[SouthStarUnifiedReferencePromotionCheck] = []
    for exact_case in load_south_star_exact_first_domain_cases():
        semantic_case = semantic_cases_by_id[exact_case.case_id]
        result = mol_to_smiles_enum_s_graph_native(
            semantic_case.source_smiles,
            case_id=semantic_case.case_id,
        )
        checks.append(
            _promotion_check(
                case_id=exact_case.case_id,
                current_authority=exact_case.support_authority,
                shared_pipeline_generated=(
                    result.generation_basis == SOUTH_STAR_SHARED_PIPELINE_GENERATION_BASIS
                    and result.outputs == exact_case.expected_support
                ),
                pipeline_coverage=_pipeline_coverage_for_case(
                    semantic_case,
                    expected_support=exact_case.expected_support,
                ),
            )
        )

    for case in load_south_star_expanded_support_cases():
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )
        checks.append(
            _promotion_check(
                case_id=case.case_id,
                current_authority=case.support_authority,
                shared_pipeline_generated=(
                    case.feature_area
                    in SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS
                    and result.generation_basis
                    == SOUTH_STAR_SHARED_PIPELINE_GENERATION_BASIS
                    and result.outputs == case.expected_support
                ),
                pipeline_coverage=_pipeline_coverage_for_case(
                    case,
                    expected_support=case.expected_support,
                ),
            )
        )
    return tuple(checks)


def _authority_inventory_item(
    *,
    case_id: str,
    source_smiles: str,
    fixture_family: str,
    feature_area: str,
    current_authority: str,
    expected_support: tuple[str, ...],
    check: SouthStarUnifiedReferencePromotionCheck,
) -> SouthStarAuthorityPromotionCandidateInventoryItem:
    result = mol_to_smiles_enum_s_graph_native(source_smiles, case_id=case_id)
    diagnostics = result.generation_diagnostics
    if diagnostics is None:
        raise AssertionError("authority inventory requires generation diagnostics")
    if result.outputs != expected_support:
        raise AssertionError(
            f"authority inventory case {case_id!r} no longer matches expected support"
        )
    return SouthStarAuthorityPromotionCandidateInventoryItem(
        case_id=case_id,
        fixture_family=fixture_family,
        feature_area=feature_area,
        current_authority=current_authority,
        authority_class=_authority_class(current_authority),
        shared_spine_coverage=(
            "complete" if check.shared_pipeline_generated else "incomplete"
        ),
        fixture_string_generation_risk=(
            "none_detected_shared_spine_matches_expected_support"
            if check.shared_pipeline_generated
            else "risk_expected_support_remains_generation_oracle"
        ),
        temporary_witness_dependency=_temporary_witness_dependency(current_authority),
        output_count=diagnostics.output_count,
        estimated_product_size=diagnostics.estimated_product_size,
        blockers=check.blockers,
    )


def _authority_promotion_checklist_record(
    inventory_item: SouthStarAuthorityPromotionCandidateInventoryItem,
    *,
    check: SouthStarUnifiedReferencePromotionCheck,
) -> SouthStarAuthorityPromotionChecklistRecord:
    return SouthStarAuthorityPromotionChecklistRecord(
        case_id=inventory_item.case_id,
        support_authority=inventory_item.current_authority,
        shared_spine_coverage=inventory_item.shared_spine_coverage,
        fixture_string_generation_prohibited=(
            inventory_item.fixture_string_generation_risk
            == "none_detected_shared_spine_matches_expected_support"
        ),
        temporary_witness_as_cross_check_only=(
            inventory_item.authority_class == "temporary_witness"
            and inventory_item.shared_spine_coverage == "complete"
        ),
        semantic_parseback=_coverage_status(check, "semantic_evidence"),
        output_count=inventory_item.output_count,
        estimated_product_size=inventory_item.estimated_product_size,
        unresolved_blockers=inventory_item.blockers,
    )


def _promotion_check(
    *,
    case_id: str,
    current_authority: str,
    shared_pipeline_generated: bool,
    pipeline_coverage: tuple[SouthStarPipelineCoverageRecord, ...],
) -> SouthStarUnifiedReferencePromotionCheck:
    blockers = []
    if not shared_pipeline_generated:
        blockers.append("not_generated_by_shared_pipeline")
    if _coverage_has_missing_stage(pipeline_coverage):
        blockers.append("pipeline_provenance_incomplete")
    spine_bypass_count = _spine_bypass_count(case_id)
    if spine_bypass_count:
        blockers.append("emitted_syntax_bypasses_spine")
    if current_authority not in SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES:
        blockers.append("support_authority_is_not_unified_reference")
    return SouthStarUnifiedReferencePromotionCheck(
        case_id=case_id,
        current_authority=current_authority,
        shared_pipeline_generated=shared_pipeline_generated,
        pipeline_coverage=pipeline_coverage,
        spine_bypass_count=spine_bypass_count,
        promoted=not blockers,
        blockers=tuple(blockers),
    )


def _authority_class(current_authority: str) -> str:
    if current_authority in SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES:
        return "unified_reference"
    if current_authority in SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES:
        return "temporary_witness"
    if current_authority == SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY:
        return "regression_witness"
    raise AssertionError(f"unclassified South Star authority {current_authority!r}")


def _temporary_witness_dependency(current_authority: str) -> str:
    if current_authority in SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES:
        return current_authority
    return "none"


def _pipeline_coverage_for_case(
    case: object,
    *,
    expected_support: tuple[str, ...],
) -> tuple[SouthStarPipelineCoverageRecord, ...]:
    mol = parse_smiles(case.source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    result = mol_to_smiles_enum_s_graph_native(
        case.source_smiles,
        case_id=case.case_id,
    )
    spec_oracle_report = south_star_spec_oracle_report(
        source_smiles=case.source_smiles,
        candidate_smiles=result.outputs,
    )
    diagnostics = result.generation_diagnostics
    if diagnostics is None:
        raise AssertionError("graph-native generation must return diagnostics")
    traversals = _tree_traversals_or_empty(case)
    has_marker_slots = diagnostics.marker_slot_count > 0
    has_renderer_inputs = any(
        event.renderer_input is not None
        for traversal in traversals
        for event in traversal.events
    )
    has_constraints = has_marker_slots or has_renderer_inputs
    has_stereo = bool(
        molecule_facts.components or molecule_facts.tetrahedral_center_facts
    )
    return (
        SouthStarPipelineCoverageRecord(
            stage_id="molecule_facts",
            status="covered",
            evidence=f"atom_count={molecule_facts.graph_topology.atom_count}",
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="traversal_plan",
            status="covered" if diagnostics.spanning_tree_count else "missing",
            evidence=f"spanning_tree_count={diagnostics.spanning_tree_count}",
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="observation",
            status=(
                "covered"
                if has_constraints
                else "not_required"
                if not has_stereo
                else "missing"
            ),
            evidence=(
                f"marker_slot_count={diagnostics.marker_slot_count}; "
                f"renderer_input_count={_renderer_input_count(traversals)}"
            ),
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="constraint_family",
            status=(
                "covered"
                if has_constraints
                else "not_required"
                if not has_stereo
                else "missing"
            ),
            evidence=(
                f"stereo_component_count={diagnostics.stereo_component_count}; "
                f"marker_slot_count={diagnostics.marker_slot_count}; "
                f"renderer_input_count={_renderer_input_count(traversals)}"
            ),
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="solver_assignment",
            status="covered" if diagnostics.solved_assignment_count else "missing",
            evidence=f"solved_assignment_count={diagnostics.solved_assignment_count}",
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="renderer",
            status="covered" if result.outputs else "missing",
            evidence=f"output_count={len(result.outputs)}",
        ),
        SouthStarPipelineCoverageRecord(
            stage_id="semantic_evidence",
            status=(
                "covered"
                if result.outputs == expected_support
                and spec_oracle_report.all_accepted
                else "missing"
            ),
            evidence=(
                "spec_oracle_accepted="
                f"{spec_oracle_report.accepted_count}/"
                f"{spec_oracle_report.candidate_count}; "
                "generation_authority="
                f"{spec_oracle_report.generation_authority}"
            ),
        ),
    )


def _tree_traversals_or_empty(case: object) -> tuple[object, ...]:
    try:
        return mol_to_smiles_enum_s_tree_traversals_for_case(case)
    except NotImplementedError:
        return ()


def _renderer_input_count(traversals: tuple[object, ...]) -> int:
    return sum(
        1
        for traversal in traversals
        for event in traversal.events
        if event.renderer_input is not None
    )


def _spine_bypass_count(case_id: str) -> int:
    all_cases = {
        case.case_id: case
        for case in (
            *load_south_star_semantic_cases(),
            *load_south_star_expanded_support_cases(),
        )
    }
    traversals = _tree_traversals_or_empty(all_cases[case_id])
    return sum(
        _event_spine_bypass_count(event, traversal.marker_assignments)
        for traversal in traversals
        for event in traversal.events
    )


def _event_spine_bypass_count(event: object, marker_assignments: tuple[object, ...]) -> int:
    bypass_count = 0
    if event.marker_slot is not None and not any(
        assignment.slot_id == event.marker_slot.slot_id
        for assignment in marker_assignments
    ):
        bypass_count += 1
    if "@" in event.text and event.renderer_input is None:
        bypass_count += 1
    return bypass_count


def _coverage_has_missing_stage(
    pipeline_coverage: tuple[SouthStarPipelineCoverageRecord, ...],
) -> bool:
    return any(record.status == "missing" for record in pipeline_coverage)


def _coverage_status(
    check: SouthStarUnifiedReferencePromotionCheck,
    stage_id: str,
) -> str:
    return next(
        record.status for record in check.pipeline_coverage if record.stage_id == stage_id
    )


if __name__ == "__main__":
    unittest.main()
