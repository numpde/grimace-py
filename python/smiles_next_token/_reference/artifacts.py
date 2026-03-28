from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from smiles_next_token._reference.dataset import (
    MoleculeCase,
    iter_molecule_cases_from_input_source,
)
from smiles_next_token._reference.policy import ReferencePolicy
from smiles_next_token._reference.rdkit_random import sample_and_validate_rdkit_random


DEFAULT_CORE_SELECTION_LIMIT = 100


def _artifact_header(
    *,
    policy: ReferencePolicy,
    selection: dict[str, Any],
) -> dict[str, Any]:
    input_source = policy.data["input_source"]
    if policy.source_path is None:
        policy_path = None
    elif policy.source_path.is_absolute():
        try:
            policy_path = str(policy.source_path.relative_to(Path.cwd()))
        except ValueError:
            policy_path = str(policy.source_path)
    else:
        policy_path = str(policy.source_path)
    return {
        "policy_name": policy.policy_name,
        "policy_kind": policy.policy_kind,
        "branch_family": policy.branch_family,
        "policy_digest": policy.digest(),
        "policy_path": policy_path,
        "input_source": input_source,
        "source_path": str(input_source["path"]),
        "selection": selection,
    }


def _selection(limit: int | None, max_smiles_length: int | None) -> tuple[dict[str, Any], str]:
    if limit is None and max_smiles_length is None:
        return {"kind": "all"}, "full"
    if limit is not None and max_smiles_length is None:
        return {"kind": "first_n", "count": limit}, f"first_{limit}"
    if limit is None and max_smiles_length is not None:
        return {"kind": "max_smiles_length", "max_smiles_length": max_smiles_length}, f"len_le_{max_smiles_length}"
    return (
        {
            "kind": "first_n_with_max_smiles_length",
            "count": limit,
            "max_smiles_length": max_smiles_length,
        },
        f"first_{limit}_len_le_{max_smiles_length}",
    )


def _evaluate_case(
    case: MoleculeCase,
    *,
    policy: ReferencePolicy,
    include_sampled_set: bool,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(case.smiles)
    if mol is None:
        artifact_case = {
            "cid": case.cid,
            "name": case.name,
            "input_smiles": case.smiles,
            "parsed": False,
            "roundtrip_ok": False,
            "distinct_count": 0,
            "validation_issue_count": 0,
            "first_validation_issue": "failed to parse input smiles",
        }
        if include_sampled_set:
            artifact_case["sampled_set"] = []
        return artifact_case

    result = sample_and_validate_rdkit_random(mol, policy)
    artifact_case = {
        "cid": case.cid,
        "name": case.name,
        "input_smiles": case.smiles,
        "parsed": True,
        "roundtrip_ok": result.is_valid,
        "distinct_count": result.distinct_count,
        "validation_issue_count": len(result.validation_issues),
        "first_validation_issue": (
            None
            if not result.validation_issues
            else {
                "sampled_smiles": result.validation_issues[0].sampled_smiles,
                "reason": result.validation_issues[0].reason,
            }
        ),
    }
    if include_sampled_set:
        artifact_case["sampled_set"] = list(result.sampled_smiles)
    return artifact_case


def build_core_exact_sets_artifact(
    policy: ReferencePolicy,
    *,
    limit: int = DEFAULT_CORE_SELECTION_LIMIT,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")

    cases = [
        _evaluate_case(case, policy=policy, include_sampled_set=True)
        for case in iter_molecule_cases_from_input_source(policy.data["input_source"], limit=limit)
    ]
    return {
        **_artifact_header(
            policy=policy,
            selection={"kind": "first_n", "count": limit},
        ),
        "case_count": len(cases),
        "cases": cases,
    }


def build_full_metrics_artifact(
    policy: ReferencePolicy,
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> dict[str, Any]:
    selection, _ = _selection(limit, max_smiles_length)
    cases = [
        _evaluate_case(case, policy=policy, include_sampled_set=False)
        for case in iter_molecule_cases_from_input_source(
            policy.data["input_source"],
            limit=limit,
            max_smiles_length=max_smiles_length,
        )
    ]
    return {
        **_artifact_header(policy=policy, selection=selection),
        "case_count": len(cases),
        "cases": cases,
    }


def write_core_exact_sets_artifact(
    policy: ReferencePolicy,
    *,
    limit: int = DEFAULT_CORE_SELECTION_LIMIT,
    path: str | Path | None = None,
) -> Path:
    output_path = policy.core_exact_sets_path() if path is None else Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_core_exact_sets_artifact(policy, limit=limit)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_full_metrics_artifact(
    policy: ReferencePolicy,
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
    path: str | Path | None = None,
) -> Path:
    _, selection_tag = _selection(limit, max_smiles_length)
    output_path = policy.metrics_path(selection_tag) if path is None else Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_full_metrics_artifact(policy, limit=limit, max_smiles_length=max_smiles_length)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(artifact, handle, sort_keys=True)
    return output_path
