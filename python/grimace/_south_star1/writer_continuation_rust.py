"""Rust-backed online execution for verified continuation assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from grimace import _core

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarWriterSurface
from .public_continuation_asset import prepare_public_continuation_molecule
from .writer_continuation_asset import advance_writer_continuation_proof
from .writer_continuation_asset import _continuation_asset_proof_batch
from .writer_continuation_asset import open_writer_continuation_core
from .writer_continuation_asset import _source_snapshot_from_asset
from .writer_continuation_asset import (
    verified_branch_artifact_from_continuation_asset,
)
from .writer_continuation_asset import (
    verified_terminal_artifact_from_continuation_asset,
)
from .writer_continuation_asset import verify_writer_continuation_asset_consistency
from .writer_continuation_asset import (
    writer_continuation_asset_runtime_options,
)
from .writer_envelope_terms import _identity_digest
from .writer_facts_replay_context import _writer_facts_replay_context

if TYPE_CHECKING:
    from .writer_continuation_asset import WriterContinuationAsset


SNAPSHOT_SCHEMA_NAME = "writer_continuation_decoder_snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
_SNAPSHOT_FIELDS = frozenset(
    (
        "schema_name",
        "schema_version",
        "asset_manifest_digest",
        "cursor",
        "emitted_texts",
        "token_count",
        "digest",
    )
)
_CURSOR_FIELDS = frozenset(
    ("node_id", "completion_scale", "raw_cursor_digest")
)


def _core_terms(core):
    return tuple(
        (
            node.node_id,
            node.signature_digest,
            node.terminal_available,
            node.terminal_multiplicity,
            node.terminal_completion_count,
            tuple(
                (
                    choice.emitted_text,
                    choice.immediate_multiplicity,
                    choice.successor_node_id,
                    choice.successor_scale,
                    choice.support_count,
                    choice.completion_count,
                )
                for choice in node.choices
            ),
            node.support_count,
            node.completion_count,
        )
        for node in core.nodes
    )


def _rust_core_from_terms(
    *, manifest_digest, root_node_id, root_scale, nodes
):
    """Private mutation-test boundary for Rust-owned core validation."""

    return _core._writer_continuation_rust_core_from_verified_terms(
        manifest_digest,
        root_node_id,
        root_scale,
        nodes,
    )


def open_writer_continuation_rust_core(path):
    """Load the verified core chunk and copy it into immutable Rust storage."""

    asset = open_writer_continuation_core(path)
    return _rust_core_for_asset(asset)


def _rust_core_for_asset(asset):
    return _rust_core_from_terms(
        manifest_digest=asset.manifest_digest,
        root_node_id=asset.core.root.node_id,
        root_scale=asset.core.root.completion_scale,
        nodes=_core_terms(asset.core),
    )


@dataclass(frozen=True, slots=True)
class MolToSmilesContinuationProbability:
    text: str | None
    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class MolToSmilesWeightedChoice:
    text: str
    immediate_multiplicity: int
    support_count: int
    completion_count: int
    probability_numerator: int
    probability_denominator: int
    next_state: "MolToSmilesContinuationDecoder"


@dataclass(frozen=True, slots=True)
class MolToSmilesBranchProofLocator:
    asset_manifest_digest: str
    source_raw_cursor_digest: str
    emitted_text: str
    branch_certificate_digest: str


@dataclass(frozen=True, slots=True)
class MolToSmilesTerminalProofLocator:
    asset_manifest_digest: str
    source_raw_cursor_digest: str
    terminal_support_identity_digest: str


class _ContinuationProofSession:
    __slots__ = ("asset", "prepared", "facts_context", "_batches")

    def __init__(self, *, asset, prepared, facts_context) -> None:
        self.asset = asset
        self.prepared = prepared
        self.facts_context = facts_context
        self._batches = {}

    def batch(self, raw_cursor_digest: str):
        value = self._batches.get(raw_cursor_digest)
        if value is None:
            value = _continuation_asset_proof_batch(
                prepared=self.prepared,
                asset=self.asset,
                source_raw_cursor_digest=raw_cursor_digest,
            )
            self._batches[raw_cursor_digest] = value
        return value


@dataclass(frozen=True, slots=True)
class _ContinuationAssetStateAdapter:
    core: object
    cursor: object
    emitted_texts: tuple[str, ...]
    asset_manifest_digest: str
    proof_asset: "WriterContinuationAsset | None" = None
    proof_cursor: object | None = None
    proof_session: object | None = None

    def prefix(self) -> str:
        return "".join(self.emitted_texts)

    def is_terminal(self) -> bool:
        return bool(self.core.is_terminal(self.cursor))

    def copy(self) -> "_ContinuationAssetStateAdapter":
        return _ContinuationAssetStateAdapter(
            core=self.core,
            cursor=self.cursor.copy(),
            emitted_texts=self.emitted_texts,
            asset_manifest_digest=self.asset_manifest_digest,
            proof_asset=self.proof_asset,
            proof_cursor=self.proof_cursor,
            proof_session=self.proof_session,
        )

    def cache_key(self) -> tuple[object, ...]:
        return (
            "continuation_asset",
            self.asset_manifest_digest,
            self.cursor.node_id,
            int(self.cursor.completion_scale),
            self.prefix(),
            None
            if self.proof_cursor is None
            else self.proof_cursor.raw_cursor_digest,
        )

    def _advance(self, text: str) -> "_ContinuationAssetStateAdapter":
        cursor = self.core.advance(self.cursor, text)
        return self._successor(text, cursor)

    def _successor(self, text, cursor) -> "_ContinuationAssetStateAdapter":
        proof_cursor = None
        if self.proof_cursor is not None:
            proof_cursor = advance_writer_continuation_proof(
                self.proof_asset,
                self.proof_cursor,
                text,
            )
            if (
                proof_cursor.node_id != cursor.node_id
                or proof_cursor.completion_scale
                != int(cursor.completion_scale)
            ):
                raise ValueError("continuation_asset_rust_proof_advance_mismatch")
        return _ContinuationAssetStateAdapter(
            core=self.core,
            cursor=cursor,
            emitted_texts=(*self.emitted_texts, text),
            asset_manifest_digest=self.asset_manifest_digest,
            proof_asset=self.proof_asset,
            proof_cursor=proof_cursor,
            proof_session=self.proof_session,
        )

    def choice_successor_states(
        self,
    ) -> tuple[tuple[str, "_ContinuationAssetStateAdapter"], ...]:
        return tuple(
            (choice.text, self._successor(choice.text, choice.next_cursor))
            for choice in self.core.choices(self.cursor)
        )

    def grouped_successor_states(
        self,
    ) -> tuple[tuple[str, "_ContinuationAssetStateAdapter"], ...]:
        return self.choice_successor_states()

    @property
    def branch_proof_locators(self) -> tuple[MolToSmilesBranchProofLocator, ...]:
        self._require_proof_context()
        raw = self.proof_cursor.raw_cursor_digest
        return tuple(
            MolToSmilesBranchProofLocator(
                asset_manifest_digest=self.asset_manifest_digest,
                source_raw_cursor_digest=raw,
                emitted_text=edge.emitted_text,
                branch_certificate_digest=digest,
            )
            for edge in self.proof_asset.edges_from(raw)
            for digest in edge.branch_certificate_digests
        )

    @property
    def terminal_proof_locators(
        self,
    ) -> tuple[MolToSmilesTerminalProofLocator, ...]:
        self._require_proof_context()
        raw = self.proof_cursor.raw_cursor_digest
        record = self.proof_asset.terminal_record(raw)
        if record is None:
            return ()
        return tuple(
            MolToSmilesTerminalProofLocator(
                asset_manifest_digest=self.asset_manifest_digest,
                source_raw_cursor_digest=raw,
                terminal_support_identity_digest=digest,
            )
            for digest in record.terminal_support_identity_digests
        )

    def branch_artifact(self, locator):
        self._require_proof_context()
        locator = self._resolve_branch_locator(locator)
        return verified_branch_artifact_from_continuation_asset(
            context=self.proof_session.facts_context,
            prepared=self.proof_session.prepared,
            asset=self.proof_asset,
            source_raw_cursor_digest=locator.source_raw_cursor_digest,
            emitted_text=locator.emitted_text,
            branch_certificate_digest=locator.branch_certificate_digest,
            proof_batch=self.proof_session.batch(
                locator.source_raw_cursor_digest
            ),
        )

    def terminalization_artifact(self, locator):
        self._require_proof_context()
        locator = self._resolve_terminal_locator(locator)
        return verified_terminal_artifact_from_continuation_asset(
            context=self.proof_session.facts_context,
            prepared=self.proof_session.prepared,
            asset=self.proof_asset,
            source_raw_cursor_digest=locator.source_raw_cursor_digest,
            terminal_support_identity_digest=(
                locator.terminal_support_identity_digest
            ),
            proof_batch=self.proof_session.batch(
                locator.source_raw_cursor_digest
            ),
        )

    def _resolve_branch_locator(self, locator) -> MolToSmilesBranchProofLocator:
        if isinstance(locator, str):
            matches = tuple(
                item
                for item in self.branch_proof_locators
                if item.branch_certificate_digest == locator
            )
            if len(matches) != 1:
                _proof_error("continuation_asset_branch_proof_identity_not_unique")
            return matches[0]
        if not isinstance(locator, MolToSmilesBranchProofLocator):
            _proof_error("continuation_asset_branch_proof_locator_type_mismatch")
        if locator not in self.branch_proof_locators:
            _proof_error("continuation_asset_branch_proof_locator_mismatch")
        return locator

    def _resolve_terminal_locator(
        self, locator
    ) -> MolToSmilesTerminalProofLocator:
        if isinstance(locator, str):
            matches = tuple(
                item
                for item in self.terminal_proof_locators
                if item.terminal_support_identity_digest == locator
            )
            if len(matches) != 1:
                _proof_error("continuation_asset_terminal_proof_identity_not_unique")
            return matches[0]
        if not isinstance(locator, MolToSmilesTerminalProofLocator):
            _proof_error("continuation_asset_terminal_proof_locator_type_mismatch")
        if locator not in self.terminal_proof_locators:
            _proof_error("continuation_asset_terminal_proof_locator_mismatch")
        return locator

    def _require_proof_context(self) -> None:
        if self.proof_asset is None or self.proof_cursor is None:
            _proof_error(
                "continuation_asset_proof_mode_required",
                kind=SouthStarErrorKind.UNSUPPORTED_POLICY,
            )
        if self.proof_session is None:
            _proof_error("continuation_asset_proof_session_required")


def _proof_error(reason: str, *, kind=SouthStarErrorKind.SEMANTIC_MISMATCH):
    raise SouthStarError(kind, reason)


def _verified_root_proof_cursor(*, asset, prepared):
    source = _source_snapshot_from_asset(prepared=prepared, asset=asset)
    proof_cursor = asset.root_proof_cursor
    if _identity_digest(source.cursor) != proof_cursor.raw_cursor_digest:
        _proof_error("continuation_asset_proof_root_cursor_mismatch")
    record = asset.raw_cursor_record(proof_cursor.raw_cursor_digest)
    if record is None:
        _proof_error("continuation_asset_proof_root_record_missing")
    if (
        record.compiled_node_id != proof_cursor.node_id
        or record.normalization_scale != proof_cursor.completion_scale
    ):
        _proof_error("continuation_asset_proof_root_mapping_mismatch")
    return proof_cursor


class MolToSmilesContinuationDecoder:
    """Explicit decoder over one structurally verified continuation asset."""

    __slots__ = ("_state", "_choices_cache")

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "MolToSmilesContinuationDecoder must be constructed with from_asset()"
        )

    @classmethod
    def _from_state(
        cls, state: _ContinuationAssetStateAdapter
    ) -> "MolToSmilesContinuationDecoder":
        instance = cls.__new__(cls)
        instance._state = state
        instance._choices_cache = None
        return instance

    @classmethod
    def from_asset(
        cls,
        path,
        *,
        expected_manifest_digest: str | None = None,
        proof_capable: bool = False,
        prepared=None,
        mol=None,
    ) -> "MolToSmilesContinuationDecoder":
        if prepared is not None and mol is not None:
            _proof_error(
                "continuation_asset_proof_multiple_molecule_bindings",
                kind=SouthStarErrorKind.UNSUPPORTED_POLICY,
            )
        if (prepared is not None or mol is not None) and not proof_capable:
            _proof_error(
                "continuation_asset_proof_binding_without_mode",
                kind=SouthStarErrorKind.UNSUPPORTED_POLICY,
            )
        if proof_capable and prepared is None and mol is None:
            _proof_error(
                "continuation_asset_proof_molecule_required",
                kind=SouthStarErrorKind.UNSUPPORTED_POLICY,
            )
        asset = open_writer_continuation_core(path)
        if (
            expected_manifest_digest is not None
            and asset.manifest_digest != expected_manifest_digest
        ):
            _proof_error("continuation_asset_manifest_digest_mismatch")
        core = _rust_core_for_asset(asset)
        proof_cursor = None
        facts_context = None
        proof_session = None
        if proof_capable:
            structural = verify_writer_continuation_asset_consistency(asset.path)
            if not structural.accepted:
                _proof_error(
                    structural.reason
                    or "continuation_asset_proof_structural_rejection"
                )
            runtime_options = writer_continuation_asset_runtime_options(asset)
            if mol is not None:
                prepared = prepare_public_continuation_molecule(
                    mol,
                    writer_surface=SouthStarWriterSurface(),
                    runtime_options=runtime_options,
                )
            facts_context = _writer_facts_replay_context(
                facts=prepared.facts,
                runtime_options=runtime_options,
                policy=prepared.policy,
            )
            if facts_context.expected_identity != asset.manifest["prepared_identity"]:
                _proof_error("continuation_asset_proof_prepared_identity_mismatch")
            proof_cursor = _verified_root_proof_cursor(
                asset=asset,
                prepared=facts_context.prepared,
            )
            prepared = facts_context.prepared
            proof_session = _ContinuationProofSession(
                asset=asset,
                prepared=prepared,
                facts_context=facts_context,
            )
        return cls._from_state(
            _ContinuationAssetStateAdapter(
                core=core,
                cursor=core.root_cursor(),
                emitted_texts=(),
                asset_manifest_digest=asset.manifest_digest,
                proof_asset=asset if proof_capable else None,
                proof_cursor=proof_cursor,
                proof_session=proof_session,
            )
        )

    @classmethod
    def from_snapshot(
        cls,
        path,
        snapshot,
        *,
        proof_capable: bool = False,
        prepared=None,
        mol=None,
    ) -> "MolToSmilesContinuationDecoder":
        _validate_snapshot_shape(snapshot)
        if snapshot["digest"] != _snapshot_digest(snapshot):
            raise ValueError("continuation_decoder_snapshot_digest_mismatch")
        decoder = cls.from_asset(
            path,
            expected_manifest_digest=snapshot["asset_manifest_digest"],
            proof_capable=proof_capable,
            prepared=prepared,
            mol=mol,
        )
        emitted_texts = tuple(snapshot["emitted_texts"])
        state = decoder._state
        for text in emitted_texts:
            state = state._advance(text)
        cursor = snapshot["cursor"]
        if (
            state.cursor.node_id != cursor["node_id"]
            or int(state.cursor.completion_scale)
            != cursor["completion_scale"]
        ):
            raise ValueError("continuation_decoder_snapshot_cursor_mismatch")
        actual_raw = (
            None
            if state.proof_cursor is None
            else state.proof_cursor.raw_cursor_digest
        )
        if actual_raw != cursor["raw_cursor_digest"]:
            raise ValueError("continuation_decoder_snapshot_proof_cursor_mismatch")
        return cls._from_state(state)

    @property
    def prefix(self) -> str:
        return self._state.prefix()

    @property
    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    @property
    def support_count(self) -> int:
        return int(self._state.core.support_count(self._state.cursor))

    @property
    def completion_count(self) -> int:
        return int(self._state.core.completion_count(self._state.cursor))

    @property
    def terminal_completion_count(self) -> int:
        return int(
            self._state.core.terminal_completion_count(self._state.cursor)
        )

    @property
    def next_choices(self) -> tuple[MolToSmilesWeightedChoice, ...]:
        if self._choices_cache is None:
            self._choices_cache = self.choices()
        return self._choices_cache

    def choices(self) -> tuple[MolToSmilesWeightedChoice, ...]:
        denominator = self.completion_count
        return tuple(
            MolToSmilesWeightedChoice(
                text=choice.text,
                immediate_multiplicity=int(choice.immediate_multiplicity),
                support_count=int(choice.support_count),
                completion_count=int(choice.completion_count),
                probability_numerator=int(choice.completion_count),
                probability_denominator=denominator,
                next_state=type(self)._from_state(
                    self._state._successor(
                        choice.text, choice.next_cursor
                    )
                ),
            )
            for choice in self._state.core.choices(self._state.cursor)
        )

    def exact_probabilities(
        self,
    ) -> tuple[MolToSmilesContinuationProbability, ...]:
        probabilities = tuple(
            MolToSmilesContinuationProbability(
                text=item.text,
                numerator=int(item.numerator),
                denominator=int(item.denominator),
            )
            for item in self._state.core.probabilities(self._state.cursor)
        )
        if sum(item.numerator for item in probabilities) != self.completion_count:
            raise ValueError("continuation_decoder_probability_normalization_mismatch")
        return probabilities

    def advance(self, emitted_text: str) -> "MolToSmilesContinuationDecoder":
        return type(self)._from_state(self._state._advance(emitted_text))

    def copy(self) -> "MolToSmilesContinuationDecoder":
        return type(self)._from_state(self._state.copy())

    def cache_key(self) -> tuple[object, ...]:
        return self._state.cache_key()

    def snapshot(self) -> dict[str, object]:
        raw_cursor_digest = (
            None
            if self._state.proof_cursor is None
            else self._state.proof_cursor.raw_cursor_digest
        )
        snapshot = {
            "schema_name": SNAPSHOT_SCHEMA_NAME,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "asset_manifest_digest": self._state.asset_manifest_digest,
            "cursor": {
                "node_id": self._state.cursor.node_id,
                "completion_scale": int(
                    self._state.cursor.completion_scale
                ),
                "raw_cursor_digest": raw_cursor_digest,
            },
            "emitted_texts": list(self._state.emitted_texts),
            "token_count": len(self._state.emitted_texts),
        }
        snapshot["digest"] = _snapshot_digest(snapshot)
        return snapshot

    @property
    def branch_proof_locators(self) -> tuple[MolToSmilesBranchProofLocator, ...]:
        return self._state.branch_proof_locators

    @property
    def terminal_proof_locators(
        self,
    ) -> tuple[MolToSmilesTerminalProofLocator, ...]:
        return self._state.terminal_proof_locators

    def branch_artifact(self, locator):
        return self._state.branch_artifact(locator)

    def terminalization_artifact(self, locator):
        return self._state.terminalization_artifact(locator)

    @property
    def rust_resident_bytes(self) -> int:
        return int(self._state.core.resident_bytes)


def _validate_snapshot_shape(snapshot) -> None:
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("continuation_decoder_snapshot_shape_mismatch")
    if (
        snapshot["schema_name"] != SNAPSHOT_SCHEMA_NAME
        or snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("continuation_decoder_snapshot_unknown_schema")
    if (
        not isinstance(snapshot["cursor"], dict)
        or set(snapshot["cursor"]) != _CURSOR_FIELDS
    ):
        raise ValueError("continuation_decoder_snapshot_cursor_shape_mismatch")
    if (
        not isinstance(snapshot["emitted_texts"], list)
        or not all(isinstance(item, str) for item in snapshot["emitted_texts"])
        or not isinstance(snapshot["token_count"], int)
        or isinstance(snapshot["token_count"], bool)
        or snapshot["token_count"] != len(snapshot["emitted_texts"])
    ):
        raise ValueError("continuation_decoder_snapshot_token_count_mismatch")
    cursor = snapshot["cursor"]
    if (
        not isinstance(cursor["node_id"], int)
        or isinstance(cursor["node_id"], bool)
        or cursor["node_id"] < 0
    ):
        raise ValueError("continuation_decoder_snapshot_node_invalid")
    if (
        not isinstance(cursor["completion_scale"], int)
        or isinstance(cursor["completion_scale"], bool)
        or cursor["completion_scale"] <= 0
    ):
        raise ValueError("continuation_decoder_snapshot_scale_nonpositive")
    if cursor["raw_cursor_digest"] is not None and not isinstance(
        cursor["raw_cursor_digest"], str
    ):
        raise ValueError("continuation_decoder_snapshot_raw_cursor_invalid")
    if not isinstance(snapshot["asset_manifest_digest"], str):
        raise ValueError("continuation_decoder_snapshot_asset_digest_invalid")


def _snapshot_digest(snapshot) -> str:
    unsigned = dict(snapshot)
    unsigned.pop("digest", None)
    payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
