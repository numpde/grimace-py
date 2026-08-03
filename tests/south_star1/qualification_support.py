"""Shared, non-test vocabulary for South Star qualification tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import grimace

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_frontier import _snapshot_advance_writer_frontier_product
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_runtime import writer_runtime_state_from_snapshot
from grimace._south_star1.writer_snapshot import _writer_snapshot_advance_outcome_by_emitted_text
from grimace._south_star1.writer_snapshot_envelope import writer_snapshot_advance_envelope_for_emitted_text
from grimace._south_star1.writer_snapshot_envelope import verify_writer_snapshot_advance_envelope
from grimace._south_star1.writer_snapshot_prefix_envelope import writer_snapshot_prefix_read_envelope_for_emitted_texts
from grimace._south_star1.writer_support import enumerate_prepared_writer_shaped_support
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_snapshot
from grimace._south_star1.writer_frontier_count_envelope import verify_writer_frontier_count_envelope
from grimace._south_star1.writer_frontier_count_envelope import writer_frontier_count_envelope_for_snapshot
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_envelope import verify_writer_support_artifact_envelope
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.default_writer_capability_ledger import DefaultWriterCapabilityCase
from tests.south_star1.helpers import two_atom_facts


def facts_for_case(case: DefaultWriterCapabilityCase):
    return ordinary_molecule_facts_from_smiles(case.smiles, case.extraction_options)


def prepare_default_case(facts):
    return prepare_south_star_mol_from_facts(facts, writer_surface=SouthStarWriterSurface())


def runtime_options_for_case(case: DefaultWriterCapabilityCase) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=case.rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def runtime_options_for_root(rooted_at_atom: int = 0) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def count_writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=-1,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def count_prepare(facts):
    return prepare_default_case(facts)


def count_initial_snapshot(prepared):
    options = count_writer_options()
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def legal_count_prefix(prepared, snapshot, *, length: int) -> tuple[str, ...]:
    from grimace._south_star1.writer_frontier import writer_frontier_choices

    emitted: list[str] = []
    current = snapshot
    for _ in range(length):
        choice = writer_frontier_choices(prepared, current.cursor).choices[0].emitted_text
        envelope = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared, snapshot=current, emitted_text=choice
        )
        verification = verify_writer_snapshot_advance_envelope(prepared=prepared, envelope=envelope)
        if not verification.accepted or verification.advanced_snapshot is None:
            raise AssertionError("legal prefix helper failed to advance")
        emitted.append(envelope["emitted_text"])
        current = verification.advanced_snapshot
    return tuple(emitted)


def terminal_count_prefix_read_envelope():
    prepared = count_prepare(two_atom_facts())
    snapshot = count_initial_snapshot(prepared)
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=legal_count_prefix(prepared, snapshot, length=2),
    )
    return prepared, prefix


def initial_snapshot_for_case(prepared, case: DefaultWriterCapabilityCase):
    options = runtime_options_for_case(case)
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def initial_snapshot_for_prepared(prepared, rooted_at_atom: int = 0):
    options = runtime_options_for_root(rooted_at_atom)
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def support_image_for_case(case: DefaultWriterCapabilityCase):
    return enumerate_prepared_writer_shaped_support(
        prepared=prepare_default_case(facts_for_case(case)),
        runtime_options=runtime_options_for_case(case),
    )


def support_artifact_for_case(case: DefaultWriterCapabilityCase):
    prepared = prepare_default_case(facts_for_case(case))
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=initial_snapshot_for_case(prepared, case),
    )


def support_artifact_for_prepared(prepared, rooted_at_atom: int = 0):
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=initial_snapshot_for_prepared(prepared, rooted_at_atom),
    )


def decoder_support_strings(decoder) -> tuple[str, ...]:
    pending = [decoder]
    strings: list[str] = []
    while pending:
        state = pending.pop()
        if state.is_terminal:
            strings.append(state.prefix)
        pending.extend(choice.next_state for choice in state.next_choices)
    return tuple(sorted(strings))


def support_strings_digest(strings: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(strings, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def bundle_bytes(path: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (str(item.relative_to(path)), item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def blocked_case_result(case: DefaultWriterCapabilityCase) -> dict[str, object]:
    try:
        prepared = prepare_default_case(facts_for_case(case))
    except SouthStarError as error:
        return {"stage": "prepare", "error_kind": error.kind, "message": str(error)}

    pending = [
        initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=runtime_options_for_root(),
        ).snapshot
    ]
    seen = set()
    blockers = []
    while pending:
        snapshot = pending.pop(0)
        if snapshot.cursor in seen:
            continue
        seen.add(snapshot.cursor)
        product = _snapshot_advance_writer_frontier_product(prepared, snapshot.cursor)
        if product.blocked:
            blockers.extend(
                item.blocker
                for item in product.blocked_frontier_certificate.stereo_policy_blocker_certificates
            )
            continue
        projection = product.projection_certificate
        if projection.terminal_projection_certificate is not None:
            continue
        for text_projection in projection.text_choice_projection_certificates:
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text=text_projection.emitted_text,
            )
            if outcome.advanced_snapshot is not None:
                pending.append(outcome.advanced_snapshot)
    return {"stage": "frontier", "blockers": tuple(blockers)}


@dataclass(frozen=True, slots=True)
class PublicProofCursorTargets:
    source_raw_cursor_digest: str
    state: grimace.MolToSmilesContinuationDecoder
    branch_locators: tuple[grimace.MolToSmilesBranchProofLocator, ...]
    terminal_locators: tuple[grimace.MolToSmilesTerminalProofLocator, ...]


def _branch_locator_key(locator):
    return (locator.source_raw_cursor_digest, locator.emitted_text, locator.branch_certificate_digest)


def _terminal_locator_key(locator):
    return (locator.source_raw_cursor_digest, locator.terminal_support_identity_digest)


def public_proof_cursor_targets(decoder, *, asset=None):
    pending = [decoder]
    visited_states = set()
    groups = []
    seen_branches = set()
    seen_terminals = set()
    while pending:
        state = pending.pop()
        if state.cache_key() in visited_states:
            continue
        visited_states.add(state.cache_key())
        branches = tuple(state.branch_proof_locators)
        terminals = tuple(state.terminal_proof_locators)
        locators = (*branches, *terminals)
        if locators:
            source_digests = {locator.source_raw_cursor_digest for locator in locators}
            if len(source_digests) != 1:
                raise AssertionError("proof locators split source cursor")
            source = next(iter(source_digests))
            branch_keys = {_branch_locator_key(item) for item in branches}
            terminal_keys = {_terminal_locator_key(item) for item in terminals}
            if len(branch_keys) != len(branches) or branch_keys & seen_branches:
                raise AssertionError("duplicate public branch locator")
            if len(terminal_keys) != len(terminals) or terminal_keys & seen_terminals:
                raise AssertionError("duplicate public terminal locator")
            seen_branches.update(branch_keys)
            seen_terminals.update(terminal_keys)
            if any(group.source_raw_cursor_digest == source for group in groups):
                raise AssertionError("duplicate public source cursor group")
            groups.append(PublicProofCursorTargets(
                source_raw_cursor_digest=source,
                state=state,
                branch_locators=tuple(sorted(branches, key=_branch_locator_key)),
                terminal_locators=tuple(sorted(terminals, key=_terminal_locator_key)),
            ))
        pending.extend(choice.next_state for choice in state.next_choices)
    if asset is not None:
        asset_branches = {
            (edge.source_raw_cursor_digest, edge.emitted_text, digest)
            for edge in asset.records("edge_records")
            for digest in edge.branch_certificate_digests
        }
        asset_terminals = {
            (record.source_raw_cursor_digest, digest)
            for record in asset.records("terminal_records")
            for digest in record.terminal_support_identity_digests
        }
        if seen_branches != asset_branches or seen_terminals != asset_terminals:
            raise AssertionError("public proof inventory mismatch")
    return tuple(sorted(groups, key=lambda item: item.source_raw_cursor_digest))


def partition_public_proof_targets(groups, *, shard_count=None):
    from tests.south_star1.qualification_plan import PUBLIC_PROOF_SHARD_COUNT

    if shard_count is None:
        shard_count = PUBLIC_PROOF_SHARD_COUNT
    if shard_count != PUBLIC_PROOF_SHARD_COUNT:
        raise ValueError("South Star public proof qualification requires the declared shard count")
    if any(
        len({_branch_locator_key(item) for item in group.branch_locators}) != len(group.branch_locators)
        or len({_terminal_locator_key(item) for item in group.terminal_locators}) != len(group.terminal_locators)
        for group in groups
    ):
        raise AssertionError("duplicate locator in source cursor group")
    shards = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    for group in sorted(groups, key=lambda item: (-(len(item.branch_locators) + len(item.terminal_locators)), item.source_raw_cursor_digest)):
        index = min(range(shard_count), key=lambda item: (weights[item], item))
        shards[index].append(group)
        weights[index] += len(group.branch_locators) + len(group.terminal_locators)
    return tuple(tuple(sorted(shard, key=lambda item: item.source_raw_cursor_digest)) for shard in shards)


@dataclass(frozen=True, slots=True)
class AcceptedCaseResult:
    support_count: int
    completion_count: int
    materialized_support_count: int
    materialized_witness_count: int
    artifact_support_count: int
    artifact_witness_count: int
    artifact_metrics: dict[str, object]
    structural_accepted: bool
    live_accepted: bool
    facts_bound_accepted: bool
    facts_bound_offline_complete: bool
    live_frontier_agreement_complete: bool
    live_count_agreement_complete: bool
    snapshot_resume_agreement_complete: bool
    facts_bound_object_kinds: tuple[str, ...]
    facts_bound_unchecked_object_kinds: tuple[str, ...]
    facts_bound_unchecked_obligation_families: tuple[str, ...]
    facts_bound_relation_families: tuple[str, ...]

def accepted_case_result(case: DefaultWriterCapabilityCase) -> AcceptedCaseResult:
    facts = facts_for_case(case)
    prepared = prepare_default_case(facts)
    options = runtime_options_for_case(case)
    state = initial_writer_runtime_state(prepared=prepared, runtime_options=options)
    image = enumerate_prepared_writer_shaped_support(prepared=prepared, runtime_options=options)
    snapshot = initial_snapshot_for_case(prepared, case)
    count_envelope = writer_frontier_count_envelope_for_snapshot(prepared=prepared, snapshot=snapshot)
    count_verification = verify_writer_frontier_count_envelope(prepared=prepared, envelope=count_envelope)
    artifact = writer_support_artifact_envelope_for_snapshot(prepared=prepared, snapshot=snapshot)
    structural = verify_writer_support_artifact_consistency(artifact)
    live = verify_writer_support_artifact_envelope(prepared=prepared, envelope=artifact)
    fact_bound = verify_writer_support_artifact_for_facts(facts=facts, runtime_options=options, artifact=artifact)
    assert count_verification.accepted, count_verification.reason
    assert structural.accepted, structural.reason
    assert live.accepted, live.reason
    assert fact_bound.accepted, fact_bound.reason
    snapshot_resume = _snapshot_resume_agreement(prepared, snapshot)
    live_count_agreement_complete = (
        count_writer_runtime_support(prepared=prepared, state=state) == image.distinct_count == structural.support_count == artifact["metrics"]["support_string_count"]
        and count_writer_runtime_completions(prepared=prepared, state=state) == image.witness_count == structural.witness_count
    )
    return AcceptedCaseResult(
        support_count=count_writer_runtime_support(prepared=prepared, state=state),
        completion_count=count_writer_runtime_completions(prepared=prepared, state=state),
        materialized_support_count=image.distinct_count,
        materialized_witness_count=image.witness_count,
        artifact_support_count=structural.support_count,
        artifact_witness_count=structural.witness_count,
        artifact_metrics=artifact["metrics"],
        structural_accepted=structural.accepted,
        live_accepted=live.accepted,
        facts_bound_accepted=fact_bound.accepted,
        facts_bound_offline_complete=fact_bound.offline_replay_complete,
        live_frontier_agreement_complete=snapshot_resume["frontier_traversal_complete"] and count_verification.accepted and live.accepted,
        live_count_agreement_complete=live_count_agreement_complete,
        snapshot_resume_agreement_complete=snapshot_resume["frontier_traversal_complete"] and snapshot_resume["strings"] == set(image.strings),
        facts_bound_object_kinds=fact_bound.offline_checked_object_kinds,
        facts_bound_unchecked_object_kinds=fact_bound.offline_unchecked_object_kinds,
        facts_bound_unchecked_obligation_families=fact_bound.offline_unchecked_obligation_families,
        facts_bound_relation_families=fact_bound.offline_checked_relation_families,
    )


def _snapshot_resume_agreement(prepared, snapshot) -> dict[str, object]:
    pending = [(snapshot, "")]
    seen = set()
    strings = set()
    complete = True
    while pending:
        current, emitted = pending.pop(0)
        seen_key = (current.cursor, emitted)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        resumed_state = writer_runtime_state_from_snapshot(current, prepared=prepared)
        resumed_choices = resume_writer_frontier_choices_from_snapshot(current, prepared=prepared)
        runtime_choices = writer_runtime_choices(prepared=prepared, state=resumed_state)
        if resumed_choices != runtime_choices:
            complete = False
            continue
        product = _snapshot_advance_writer_frontier_product(prepared, current.cursor)
        if product.blocked:
            complete = False
            continue
        projection = product.projection_certificate
        if projection.terminal_projection_certificate is not None:
            strings.add(emitted)
            continue
        for choice in resumed_choices.choices:
            advanced = advance_writer_frontier_snapshot(current, prepared=prepared, emitted_text=choice.emitted_text)
            pending.append((advanced, emitted + choice.emitted_text))
    return {"frontier_traversal_complete": complete, "strings": strings}
