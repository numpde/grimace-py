"""Table-backed durable artifacts for complete writer support images."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_terms import _term
from .writer_atom_text_lifecycle import bracket_atom_text
from .writer_atom_text_lifecycle import is_supported_bracket_atom
from .writer_support_artifact_checker import SCHEMA_NAME
from .writer_support_artifact_checker import SCHEMA_VERSION
from .writer_support_artifact_checker import artifact_manifest
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_checker import verify_writer_support_artifact_consistency as _check_writer_support_artifact_consistency
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_frontier import _checked_writer_frontier_product
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_prefix_read
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_snapshot
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import _terminal_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_support_identity_envelope_from_certificate
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
from .writer_support_image_envelope import _support_image_certificate_for_source
from .writer_support_image_envelope import _text_projection_bucket_key
from .writer_support_string_envelope import _support_string_replay_certificate_digest

_PLAIN_ATOM_TEXT_ELEMENTS = frozenset(("C", "N", "O"))


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactEnvelopeVerification:
    accepted: bool
    source_kind: str
    support_count: int | None = None
    witness_count: int | None = None
    reason: str | None = None


def writer_support_artifact_envelope_for_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    product = _checked_product(prepared=prepared, snapshot=snapshot)
    count_envelope = writer_frontier_count_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
        budget=budget,
    )
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=snapshot,
        product=product,
    )
    return _artifact_from_image(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        count_envelope=count_envelope,
        product=product,
        image=image,
        budget=budget,
    )


def writer_support_artifact_envelope_for_prefix_read(
    *,
    prepared: SouthStarPreparedMol,
    prefix_read_envelope: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=prefix_read_envelope,
        budget=budget,
    )
    if not prefix.accepted:
        _artifact_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable" or prefix.final_snapshot is None:
        _artifact_violation("prefix_read_envelope_not_readable")
    product = _checked_product(prepared=prepared, snapshot=prefix.final_snapshot)
    count_envelope = writer_frontier_count_envelope_for_prefix_read(
        prepared=prepared,
        prefix_read_envelope=prefix_read_envelope,
        budget=budget,
    )
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=prefix.final_snapshot,
        product=product,
    )
    return _artifact_from_image(
        prepared=prepared,
        source_kind="prefix_read",
        source_snapshot=prefix.final_snapshot,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        product=product,
        image=image,
        budget=budget,
    )


def verify_writer_support_artifact_consistency(
    envelope: object,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactEnvelopeVerification:
    result = _check_writer_support_artifact_consistency(envelope, budget=budget)
    return WriterSupportArtifactEnvelopeVerification(
        accepted=result.accepted,
        source_kind=(
            str(envelope.get("source_kind", "unknown"))
            if isinstance(envelope, Mapping)
            else "unknown"
        ),
        support_count=result.support_count,
        witness_count=result.witness_count,
        reason=result.reason,
    )


def verify_writer_support_artifact_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_support_artifact_consistency(
            envelope,
            budget=budget,
        )
        if not structural.accepted:
            return structural
        assert isinstance(envelope, Mapping)
        source_kind = str(envelope["source_kind"])
        source_snapshot = _source_snapshot_for_artifact(
            prepared=prepared,
            envelope=envelope,
            budget=budget,
        )
        if source_kind == "snapshot":
            expected = writer_support_artifact_envelope_for_snapshot(
                prepared=prepared,
                snapshot=source_snapshot,
                budget=budget,
            )
        else:
            expected = writer_support_artifact_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=envelope["prefix_read_envelope"],
                budget=budget,
            )
        if expected != envelope:
            return WriterSupportArtifactEnvelopeVerification(
                accepted=False,
                source_kind=source_kind,
                reason="artifact_terms_mismatch",
            )
        root = next(
            (
                item
                for item in envelope["objects"]
                if item["object_id"] == envelope["roots"]["support_image_root"]
            ),
            None,
        )
        if root is None:
            _artifact_violation("support_image_root_missing")
        return WriterSupportArtifactEnvelopeVerification(
            accepted=True,
            source_kind=source_kind,
            support_count=int(root["payload"]["distinct_count"]),
            witness_count=int(root["payload"]["witness_count"]),
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _artifact_from_image(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    product,
    image,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, object]:
    del product
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_envelope",
        metric="support_string_count",
        actual=len(image.string_certificates),
        limit=budget.max_support_strings,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_envelope",
        metric="total_emitted_text_bytes",
        actual=sum(
            len(text.encode("utf-8"))
            for certificate in image.string_certificates
            for text in certificate.emitted_texts
        ),
        limit=budget.max_total_emitted_text_bytes,
    )
    table = _ObjectTable(budget)
    source_identity = _snapshot_identity_envelope(
        source_snapshot,
        budget=budget,
        operation="support_artifact.source_snapshot.digest",
    )
    source_ref = table.add(
        "source_snapshot",
        source_identity,
        operation="support_artifact.source_snapshot.object",
    )
    count_ref = table.add(
        "count_envelope",
        _count_payload(
            count_envelope,
            count_dag_ref=table.add(
                "count_dag",
                count_envelope["count_dag"],
                operation="support_artifact.count_dag.object",
            ),
        ),
        operation="support_artifact.count.object",
    )
    frontier_ref = table.add(
        "frontier_product",
        count_envelope["frontier_product"],
        operation="support_artifact.frontier_product.object",
    )
    support_string_refs = []
    for index, certificate in enumerate(image.string_certificates):
        support_string_refs.append(
            _add_support_string(
                table,
                index=index,
                certificate=certificate,
                source_ref=source_ref,
                count_ref=count_ref,
                facts=prepared.facts,
                budget=budget,
            )
        )
    coverage_ref = _add_coverage(
        table,
        coverage=image.enumeration_coverage_certificate,
        support_string_refs=support_string_refs,
        budget=budget,
    )
    support_image_ref = table.add(
        "support_image",
        {
            "source_ref": source_ref,
            "count_ref": count_ref,
            "frontier_product_ref": frontier_ref,
            "support_string_refs": support_string_refs,
            "coverage_ref": coverage_ref,
            "support_strings": [certificate.string for certificate in image.string_certificates],
            "distinct_count": image.distinct_count,
            "witness_count": image.witness_count,
            "support_count_certificate_digest": count_envelope["support_count_certificate"]["digest"],
            "witness_count_certificate_digest": count_envelope["completion_count_certificate"]["digest"],
        },
        operation="support_artifact.support_image.object",
    )
    objects = table.objects()
    roots = {
        "source_ref": source_ref,
        "count_ref": count_ref,
        "frontier_product_ref": frontier_ref,
        "support_image_root": support_image_ref,
    }
    metrics = artifact_metrics(objects, roots=roots)
    envelope = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(
            source_snapshot.prepared_identity,
            budget=budget,
            operation="support_artifact.prepared_identity.digest",
        ),
        "source_kind": source_kind,
        "source_snapshot": source_identity if source_kind == "snapshot" else None,
        "prefix_read_envelope": prefix_read_envelope,
        "objects": objects,
        "roots": roots,
        "metrics": metrics,
    }
    envelope["digest"] = _digest_terms_bounded(
        artifact_manifest(envelope),
        budget=budget,
        operation="support_artifact.manifest.digest",
    )
    checked = _check_writer_support_artifact_consistency(
        envelope,
        budget=budget,
    )
    if not checked.accepted:
        _artifact_violation(checked.reason or "artifact_checker_rejected")
    return envelope


def _add_support_string(
    table,
    *,
    index: int,
    certificate,
    source_ref: str,
    count_ref: str,
    facts,
    budget: WriterEnvelopeWorkBudget,
) -> str:
    text_projection_refs = [
        _add_text_projection(
            table,
            projection=projection,
            facts=facts,
            budget=budget,
        )
        for projection in certificate.text_projection_certificates
    ]
    terminal_projection = _terminal_projection_certificate_identity_envelope(
        certificate.terminal_projection_certificate,
        budget=budget,
    )
    replay_ref = table.add(
        "replay_path",
        {
            "source_ref": source_ref,
            "emitted_texts": list(certificate.emitted_texts),
            "text_projection_refs": text_projection_refs,
            "replay_certificate_digest": _support_string_replay_certificate_digest(
                certificate.replay_certificate,
                budget=budget,
            ),
            "final_cursor_digest": terminal_projection["source_cursor"]["digest"],
            "final_snapshot_digest": _snapshot_identity_envelope(
                certificate.final_snapshot,
                budget=budget,
                operation="support_artifact.replay.final_snapshot.digest",
            )["digest"],
        },
        operation="support_artifact.replay_path.object",
    )
    terminal_projection_ref = table.add(
        "terminal_projection",
        terminal_projection,
        operation="support_artifact.terminal_projection.object",
    )
    terminal_support_refs = [
        table.add(
            "terminal_support",
            {
                **_terminal_support_identity_envelope_from_certificate(
                    terminal,
                    budget=budget,
                ),
                "obligation_summary": _terminal_obligation_summary(terminal),
                "obligation_manifests": _terminal_obligation_manifests(
                    terminal,
                    budget=budget,
                ),
            },
            operation="support_artifact.terminal_support.object",
        )
        for terminal in certificate.terminal_projection_certificate.terminal_certificates
    ]
    return table.add(
        "support_string",
        {
            "index": index,
            "string": certificate.string,
            "emitted_texts": list(certificate.emitted_texts),
            "source_ref": source_ref,
            "count_ref": count_ref,
            "replay_path_ref": replay_ref,
            "text_projection_refs": text_projection_refs,
            "terminal_projection_ref": terminal_projection_ref,
            "terminal_support_refs": terminal_support_refs,
        },
        operation="support_artifact.support_string.object",
    )


def _add_text_projection(
    table,
    *,
    projection,
    facts,
    budget: WriterEnvelopeWorkBudget,
) -> str:
    envelope = _text_projection_certificate_identity_envelope(
        projection,
        budget=budget,
    )
    branch_support_refs = [
        _add_branch_support(
            table,
            branch=branch,
            text_projection=envelope,
            facts=facts,
            budget=budget,
        )
        for branch in projection.branch_certificates
    ]
    return table.add(
        "text_projection",
        {
            **envelope,
            "branch_support_refs": branch_support_refs,
        },
        operation="support_artifact.text_projection.object",
    )


def _add_branch_support(
    table,
    *,
    branch,
    text_projection: Mapping[str, object],
    facts,
    budget: WriterEnvelopeWorkBudget,
) -> str:
    envelope = _branch_certificate_identity_envelope(branch, budget=budget)
    local_evidence = _branch_local_evidence_envelope(
        branch,
        facts=facts,
        budget=budget,
    )
    graph_ring_delta = _branch_graph_ring_delta_envelope(
        branch,
        branch_identity=envelope,
        text_projection=text_projection,
        local_evidence=local_evidence,
        budget=budget,
    )
    return table.add(
        "branch_support",
        {
            "emitted_text": envelope["emitted_text"],
            "source_state_digest": envelope["source_state_digest"],
            "successor_state_digest": envelope["successor_state_digest"],
            "source_cursor_digest": text_projection["source_cursor"]["digest"],
            "successor_cursor_digest": text_projection["successor_cursor"]["digest"],
            "parent_weight": envelope["parent_weight"],
            "branch_ordinal": envelope["branch_ordinal"],
            "transition_kind": envelope["transition_kind"],
            "graph_action_surface_digest": envelope["graph_action_surface_digest"],
            "successor_state_certificate_digest": (
                envelope["successor_state_certificate_digest"]
            ),
            "checked_branch_certificate_digest": envelope["digest"],
            "local_evidence": local_evidence,
            "graph_ring_delta": graph_ring_delta,
            "obligation_summary": _branch_obligation_summary(branch),
            "obligation_manifests": _branch_obligation_manifests(
                branch,
                branch_identity=envelope,
                budget=budget,
            ),
            "digest": envelope["digest"],
        },
        operation="support_artifact.branch_support.object",
    )


def _branch_local_evidence_envelope(
    branch,
    *,
    facts,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, object]:
    successor = branch.successor_state_certificate
    directional = tuple(
        getattr(successor, "directional_ring_closure_bond_text_lifecycle_evidence", ())
    )
    ring_replay = getattr(successor, "ring_replay_certificate", None)
    closure = (
        ()
        if ring_replay is None
        else tuple(getattr(ring_replay, "closure_bond_text_lifecycle_evidence", ()))
    )
    if directional:
        manifest = {
            "closure_bond_text": [
                _closure_bond_text_evidence_manifest(item, budget=budget)
                for item in closure
            ],
            "directional_coupled_digests": [
                _identity_digest(
                    item,
                    budget=budget,
                    operation="support_artifact.local_directional_evidence.digest",
                )
                for item in directional
            ],
            "directional_coupled_count": len(directional),
        }
        return _local_evidence("directional_ring_closure_bond_text", manifest, budget)
    if closure:
        manifest = {
            "items": [
                _closure_bond_text_evidence_manifest(item, budget=budget)
                for item in closure
            ],
        }
        return _local_evidence("closure_bond_text", manifest, budget)
    atom_id = _branch_atom_text_atom_id(branch)
    atom_by_id = {atom.id: atom for atom in facts.atoms}
    atom = atom_by_id.get(atom_id)
    if atom is not None and is_supported_bracket_atom(atom):
        rendered = bracket_atom_text(atom)
        if rendered == branch.emitted_text:
            manifest = {
                "atom_id": _term(atom.id),
                "element": atom.symbol,
                "isotope": atom.isotope,
                "formal_charge": atom.formal_charge,
                "hydrogen_count": atom.implicit_h_count,
                "aromatic": atom.is_aromatic,
                "rendered_text": rendered,
                "bracket_required": True,
            }
            return _local_evidence("bracket_atom_text", manifest, budget)
    if atom is not None:
        rendered = _plain_atom_text(atom)
        if rendered is not None and rendered == branch.emitted_text:
            manifest = {
                "atom_id": _term(atom.id),
                "element": atom.symbol,
                "aromatic": atom.is_aromatic,
                "rendered_text": rendered,
                "bracket_required": False,
            }
            return _local_evidence("plain_atom_text", manifest, budget)
    return _local_evidence("other_structural", {}, budget)


def _branch_obligation_summary(branch) -> dict[str, int]:
    successor = branch.successor_state_certificate
    return {
        "residual_work_count": len(branch.residual_work_evidence),
        "finite_relation_work_count": len(branch.finite_relation_work_evidence),
        "graph_obligation_work_count": len(branch.graph_obligation_work_evidence),
        "stereo_lifecycle_count": (
            len(branch.stereo_lifecycle_evidence)
            + len(branch.stereo_branch_certificates)
        ),
        "residual_attachment_lifecycle_count": (
            len(branch.residual_attachment_lifecycle_evidence)
            + len(branch.residual_attachment_branch_certificates)
        ),
        "closure_candidate_lifecycle_count": (
            len(branch.closure_candidate_lifecycle_evidence)
            + len(branch.closure_candidate_branch_certificates)
        ),
        "directional_ring_closure_lifecycle_count": len(
            getattr(
                successor,
                "directional_ring_closure_bond_text_lifecycle_evidence",
                (),
            )
        ),
    }


def _branch_obligation_manifests(
    branch,
    *,
    branch_identity: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, list[dict[str, object]]]:
    successor = branch.successor_state_certificate
    graph_replay = getattr(successor, "graph_replay_certificate", None)
    stereo_replay = getattr(successor, "stereo_replay_certificate", None)
    residual_attachment_replay = getattr(
        successor,
        "residual_attachment_lifecycle_replay_certificate",
        None,
    )
    closure_candidate_replay = getattr(
        successor,
        "closure_candidate_lifecycle_replay_certificate",
        None,
    )
    return {
        "residual_work": _obligation_family_manifests(
            family="residual_work",
            records=branch.residual_work_evidence,
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=False,
            budget=budget,
        ),
        "finite_relation_work": _obligation_family_manifests(
            family="finite_relation_work",
            records=branch.finite_relation_work_evidence,
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=(
                _replay_complete(residual_attachment_replay)
                or _replay_complete(closure_candidate_replay)
            ),
            budget=budget,
        ),
        "graph_obligation_work": _obligation_family_manifests(
            family="graph_obligation_work",
            records=branch.graph_obligation_work_evidence,
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=_replay_complete(graph_replay),
            budget=budget,
        ),
        "stereo_lifecycle": _obligation_family_manifests(
            family="stereo_lifecycle",
            records=(
                *branch.stereo_lifecycle_evidence,
                *branch.stereo_branch_certificates,
            ),
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=_replay_complete(stereo_replay),
            budget=budget,
        ),
        "residual_attachment_lifecycle": _obligation_family_manifests(
            family="residual_attachment_lifecycle",
            records=(
                *branch.residual_attachment_lifecycle_evidence,
                *branch.residual_attachment_branch_certificates,
            ),
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=_replay_complete(residual_attachment_replay),
            budget=budget,
        ),
        "closure_candidate_lifecycle": _obligation_family_manifests(
            family="closure_candidate_lifecycle",
            records=(
                *branch.closure_candidate_lifecycle_evidence,
                *branch.closure_candidate_branch_certificates,
            ),
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=_replay_complete(closure_candidate_replay),
            budget=budget,
        ),
        "directional_ring_closure_lifecycle": _obligation_family_manifests(
            family="directional_ring_closure_lifecycle",
            records=getattr(
                successor,
                "directional_ring_closure_bond_text_lifecycle_evidence",
                (),
            ),
            source_digest=branch_identity["source_state_digest"],
            successor_digest=branch_identity["successor_state_digest"],
            replay_complete=False,
            budget=budget,
        ),
    }


def _terminal_obligation_summary(terminal) -> dict[str, int]:
    return {
        "terminal_residual_work_count": len(terminal.terminal_residual_work_evidence),
        "terminal_stereo_lifecycle_count": len(
            terminal.terminal_stereo_lifecycle_evidence
        ),
        "graph_obligation_work_count": len(terminal.graph_obligation_work_evidence),
    }


def _terminal_obligation_manifests(
    terminal,
    *,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, list[dict[str, object]]]:
    source_digest = _identity_digest(
        terminal.source_state,
        budget=budget,
        operation="support_artifact.terminal_obligation.source.digest",
    )
    finalized_digest = _identity_digest(
        terminal.finalized_state,
        budget=budget,
        operation="support_artifact.terminal_obligation.finalized.digest",
    )
    terminal_noop = terminal.source_state == terminal.finalized_state
    terminal_graph_clean = _terminal_graph_clean(terminal)
    terminal_stereo_clean = _terminal_stereo_clean(terminal)
    return {
        "terminal_residual_work": _obligation_family_manifests(
            family="terminal_residual_work",
            records=terminal.terminal_residual_work_evidence,
            source_digest=source_digest,
            successor_digest=finalized_digest,
            replay_complete=False,
            budget=budget,
        ),
        "terminal_stereo_lifecycle": _obligation_family_manifests(
            family="terminal_stereo_lifecycle",
            records=terminal.terminal_stereo_lifecycle_evidence,
            source_digest=source_digest,
            successor_digest=finalized_digest,
            replay_complete=terminal_noop,
            terminal_clean=terminal_stereo_clean,
            budget=budget,
        ),
        "terminal_graph_obligation_work": _obligation_family_manifests(
            family="terminal_graph_obligation_work",
            records=terminal.graph_obligation_work_evidence,
            source_digest=source_digest,
            successor_digest=finalized_digest,
            replay_complete=terminal_noop,
            terminal_clean=terminal_graph_clean,
            budget=budget,
        ),
    }


def _obligation_family_manifests(
    *,
    family: str,
    records: tuple[object, ...],
    source_digest: str,
    successor_digest: str,
    replay_complete: bool,
    budget: WriterEnvelopeWorkBudget,
    terminal_clean: bool = False,
) -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "operation": getattr(record, "operation", record.__class__.__name__),
            "source_digest": source_digest,
            "successor_digest": successor_digest,
            "is_noop": source_digest == successor_digest,
            "is_empty": False,
            "is_discharged": bool(replay_complete),
            "terminal_clean": bool(terminal_clean),
            "evidence_digest": _identity_digest(
                record,
                budget=budget,
                operation=f"support_artifact.obligation.{family}.evidence.digest",
            ),
        }
        for record in records
    ]


def _terminal_graph_clean(terminal) -> bool:
    for certificate in terminal.terminal_certificates:
        if getattr(getattr(certificate, "kind", None), "value", None) != "graph_complete":
            continue
        status = getattr(certificate, "graph_completion_status", None)
        if status is None:
            continue
        if (
            getattr(status, "complete", False)
            and not tuple(getattr(status, "unresolved_kinds", ()))
            and not tuple(getattr(status, "unresolved_bonds", ()))
        ):
            return True
    return False


def _terminal_stereo_clean(terminal) -> bool:
    for certificate in terminal.terminal_certificates:
        if (
            getattr(getattr(certificate, "kind", None), "value", None)
            == "stereo_terminalized"
        ):
            return True
    return False


def _replay_complete(certificate) -> bool:
    if certificate is None:
        return False
    if getattr(certificate, "replay_complete", False):
        return True
    nested = getattr(certificate, "obligation_replay_certificate", None)
    return bool(nested is not None and getattr(nested, "replay_complete", False))


def _plain_atom_text(atom) -> str | None:
    if atom.symbol not in _PLAIN_ATOM_TEXT_ELEMENTS:
        return None
    if atom.is_aromatic:
        return None
    if atom.isotope is not None:
        return None
    if atom.formal_charge != 0:
        return None
    return atom.symbol


def _branch_atom_text_atom_id(branch) -> object | None:
    atom_id = getattr(branch.transition_evidence, "atom", None)
    if atom_id is not None:
        return atom_id
    for event in branch.events:
        if hasattr(event, "atom") and hasattr(event, "text"):
            return event.atom
    source_atoms = dict(branch.source_state.policy_state.atom_text)
    successor_atoms = dict(branch.successor_state.policy_state.atom_text)
    added = [
        atom
        for atom, text in successor_atoms.items()
        if source_atoms.get(atom) != text and text == branch.emitted_text
    ]
    if len(added) == 1:
        return added[0]
    return None


def _local_evidence(
    kind: str,
    manifest: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, object]:
    envelope = {
        "kind": kind,
        "manifest": dict(manifest),
    }
    envelope["digest"] = _identity_digest(
        envelope,
        budget=budget,
        operation=f"support_artifact.local_evidence.{kind}.digest",
    )
    return envelope


def _branch_graph_ring_delta_envelope(
    branch,
    *,
    branch_identity: Mapping[str, object],
    text_projection: Mapping[str, object],
    local_evidence: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, object]:
    event_manifests = [
        _writer_event_manifest(event, budget=budget)
        for event in branch.events
    ]
    kind = _graph_ring_delta_kind(
        transition_kind=branch.transition_kind.value,
        event_manifests=event_manifests,
        local_evidence=local_evidence,
    )
    manifest = {
        "source_state_digest": branch_identity["source_state_digest"],
        "successor_state_digest": branch_identity["successor_state_digest"],
        "source_cursor_digest": text_projection["source_cursor"]["digest"],
        "successor_cursor_digest": text_projection["successor_cursor"]["digest"],
        "transition_kind": branch_identity["transition_kind"],
        "emitted_text": branch_identity["emitted_text"],
        "graph_action_surface_digest": branch_identity["graph_action_surface_digest"],
        "successor_state_certificate_digest": (
            branch_identity["successor_state_certificate_digest"]
        ),
        "checked_branch_certificate_digest": branch_identity["digest"],
        "local_evidence_digest": local_evidence["digest"],
        "event_manifests": event_manifests,
    }
    envelope = {
        "kind": kind,
        "manifest": manifest,
    }
    envelope["digest"] = _identity_digest(
        envelope,
        budget=budget,
        operation=f"support_artifact.graph_ring_delta.{kind}.digest",
    )
    return envelope


def _graph_ring_delta_kind(
    *,
    transition_kind: str,
    event_manifests: list[dict[str, object]],
    local_evidence: Mapping[str, object],
) -> str:
    event_kinds = {str(event["kind"]) for event in event_manifests}
    if "ring_endpoint_paired" in event_kinds:
        if local_evidence["kind"] in (
            "closure_bond_text",
            "directional_ring_closure_bond_text",
        ):
            return "ring_endpoint_pair_non_single"
        return "ring_endpoint_pair"
    if "ring_endpoint_emitted" in event_kinds:
        return "ring_endpoint_open"
    if "branch_opened" in event_kinds:
        return "branch_open"
    if "branch_closed" in event_kinds:
        return "branch_return"
    if "bond_emitted" in event_kinds:
        return "bond_advance"
    if "atom_emitted" in event_kinds:
        if transition_kind == "atom":
            return "atom_start"
        return "atom_advance"
    return "other_structural"


def _writer_event_manifest(event, *, budget: WriterEnvelopeWorkBudget) -> dict[str, object]:
    if event.__class__.__name__ == "WriterAtomEmitted":
        return {
            "kind": "atom_emitted",
            "atom": _term(event.atom),
            "text": event.text,
            "tetra_token": _term(event.tetra_token),
            "parent": _term(event.parent),
            "incoming_bond": _term(event.incoming_bond),
        }
    if event.__class__.__name__ == "WriterBondEmitted":
        return {
            "kind": "bond_emitted",
            "bond": _term(event.bond),
            "parent": _term(event.parent),
            "child": _term(event.child),
            "text": event.text,
            "direction_mark": _term(event.direction_mark),
        }
    if event.__class__.__name__ == "WriterBranchOpened":
        return {
            "kind": "branch_opened",
            "parent": _term(event.parent),
            "child": _term(event.child),
            "bond": _term(event.bond),
        }
    if event.__class__.__name__ == "WriterBranchClosed":
        return {
            "kind": "branch_closed",
            "atom": _term(event.atom),
        }
    if event.__class__.__name__ == "WriterComponentBoundaryEmitted":
        return {
            "kind": "component_boundary_emitted",
            "next_root": _term(event.next_root),
        }
    if event.__class__.__name__ == "WriterLocalOrderClosed":
        return {
            "kind": "local_order_closed",
            "atom": _term(event.atom),
        }
    if event.__class__.__name__ == "WriterRingLabelAllocated":
        return {
            "kind": "ring_label_allocated",
            "label": _term(event.label),
            "source": event.source,
        }
    if event.__class__.__name__ == "WriterRingLabelReleased":
        return {
            "kind": "ring_label_released",
            "label": _term(event.label),
            "destination": event.destination,
        }
    if event.__class__.__name__ == "WriterRingEndpointEmitted":
        return {
            "kind": "ring_endpoint_emitted",
            "bond": _term(event.bond),
            "endpoint_atom": _term(event.endpoint_atom),
            "partner_atom": _term(event.partner_atom),
            "label": _term(event.label),
            "endpoint_text": event.endpoint_text,
            "bond_text": event.bond_text,
            "direction_mark": _term(event.direction_mark),
            "side": event.side,
        }
    if event.__class__.__name__ == "WriterRingEndpointPaired":
        return {
            "kind": "ring_endpoint_paired",
            "bond": _term(event.bond),
            "endpoint_atom": _term(event.endpoint_atom),
            "partner_atom": _term(event.partner_atom),
            "label": _term(event.label),
            "endpoint_text": event.endpoint_text,
            "bond_text": event.bond_text,
            "direction_mark": _term(event.direction_mark),
            "first_endpoint_bond_text": event.first_endpoint_bond_text,
            "first_endpoint_direction_mark": _term(event.first_endpoint_direction_mark),
            "side": event.side,
        }
    return {
        "kind": "unknown",
        "class_name": event.__class__.__name__,
        "digest": _identity_digest(
            event,
            budget=budget,
            operation="support_artifact.unknown_event.digest",
        ),
    }


def _closure_bond_text_evidence_manifest(evidence, *, budget) -> dict[str, object]:
    return {
        "bond": _term(evidence.bond),
        "bond_order": evidence.bond_order,
        "label": _term(evidence.label),
        "opening_atom": _term(evidence.opening_atom),
        "closing_atom": _term(evidence.closing_atom),
        "opening_marker": evidence.opening_marker,
        "closing_marker": evidence.closing_marker,
        "marker_side": evidence.marker_side,
        "event_kind": evidence.event_kind,
        "closed_closure_record_digest": (
            None
            if evidence.closed_closure_record is None
            else _identity_digest(
                evidence.closed_closure_record,
                budget=budget,
                operation="support_artifact.closed_closure_record.digest",
            )
        ),
    }


def _add_coverage(
    table,
    *,
    coverage,
    support_string_refs: list[str],
    budget: WriterEnvelopeWorkBudget,
) -> str:
    text_buckets = []
    for bucket in coverage.text_buckets:
        indices = [
            coverage.string_certificates.index(certificate)
            for certificate in bucket.string_certificates
        ]
        text_buckets.append(
            {
                "text_projection": _text_projection_certificate_identity_envelope(
                    bucket.support_count_term.text_projection_certificate,
                    budget=budget,
                ),
                "support_count": bucket.support_count,
                "string_refs": [support_string_refs[index] for index in indices],
            }
        )
    terminal = coverage.terminal_bucket
    terminal_bucket = None
    if terminal is not None:
        terminal_bucket = {
            "terminal_projection": None
            if terminal.terminal_support_term is None
            else _terminal_projection_certificate_identity_envelope(
                terminal.terminal_support_term.terminal_projection_certificate,
                budget=budget,
            ),
            "support_count": terminal.support_count,
            "string_ref": None
            if terminal.string_certificate is None
            else support_string_refs[
                coverage.string_certificates.index(terminal.string_certificate)
            ],
        }
    return table.add(
        "support_image_coverage",
        {
            "text_buckets": text_buckets,
            "terminal_bucket": terminal_bucket,
            "distinct_count": coverage.distinct_count,
            "support_count": coverage.support_count,
        },
        operation="support_artifact.coverage.object",
    )


def _count_payload(
    count_envelope: Mapping[str, object],
    *,
    count_dag_ref: str,
) -> dict[str, object]:
    dag_metrics = count_envelope["count_dag"]["metrics"]
    return {
        "schema_name": count_envelope["schema_name"],
        "schema_version": count_envelope["schema_version"],
        "source_kind": count_envelope["source_kind"],
        "count_dag_ref": count_dag_ref,
        "frontier_snapshot_digest": count_envelope["frontier_snapshot"]["digest"],
        "frontier_product_digest": count_envelope["frontier_product"]["digest"],
        "count_dag_digest": count_envelope["count_dag"]["digest"],
        "support_count": count_envelope["support_count"],
        "completion_count": count_envelope["completion_count"],
        "support_count_certificate_digest": count_envelope["support_count_certificate"]["digest"],
        "completion_count_certificate_digest": count_envelope["completion_count_certificate"]["digest"],
        "count_dag_node_count": dag_metrics["node_count"],
        "count_dag_edge_count": dag_metrics["edge_count"],
    }


class _ObjectTable:
    def __init__(self, budget: WriterEnvelopeWorkBudget):
        self._budget = budget
        self._objects_by_id: dict[str, dict[str, object]] = {}

    def add(self, kind: str, payload, *, operation: str) -> str:
        digest = _identity_digest(
            {"kind": kind, "payload": payload},
            budget=self._budget,
            operation=operation,
        )
        object_id = f"obj:{digest}"
        if object_id not in self._objects_by_id:
            self._objects_by_id[object_id] = {
                "object_id": object_id,
                "kind": kind,
                "payload": payload,
                "digest": digest,
            }
        return object_id

    def objects(self) -> list[dict[str, object]]:
        return sorted(self._objects_by_id.values(), key=lambda item: item["object_id"])


def _source_snapshot_for_artifact(*, prepared, envelope, budget):
    if envelope["source_kind"] == "snapshot":
        return _source_snapshot_from_envelope(
            prepared=prepared,
            envelope=envelope,
            budget=budget,
        )
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=envelope["prefix_read_envelope"],
        budget=budget,
    )
    if not prefix.accepted:
        _artifact_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _artifact_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _artifact_violation("prefix_read_envelope_lacks_final_snapshot")
    return prefix.final_snapshot


def _checked_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )


def _artifact_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support artifact envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSupportArtifactEnvelopeVerification",
    "verify_writer_support_artifact_consistency",
    "verify_writer_support_artifact_envelope",
    "writer_support_artifact_envelope_for_prefix_read",
    "writer_support_artifact_envelope_for_snapshot",
)
