"""Rust-backed online execution for verified continuation assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from grimace import _core

from .writer_continuation_asset import advance_writer_continuation_proof
from .writer_continuation_asset import (
    branch_transition_artifact_from_continuation_asset,
)
from .writer_continuation_asset import open_writer_continuation_core
from .writer_continuation_asset import _source_snapshot_from_asset
from .writer_continuation_asset import (
    terminalization_artifact_from_continuation_asset,
)
from .writer_envelope_terms import _identity_digest

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
class _ContinuationAssetStateAdapter:
    core: object
    cursor: object
    emitted_texts: tuple[str, ...]
    asset_manifest_digest: str
    proof_asset: "WriterContinuationAsset | None" = None
    proof_cursor: object | None = None
    proof_prepared: object | None = None

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
            proof_prepared=self.proof_prepared,
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
            proof_prepared=self.proof_prepared,
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

    def branch_artifact(self, branch_certificate_digest: str):
        self._require_proof_context()
        return branch_transition_artifact_from_continuation_asset(
            prepared=self.proof_prepared,
            asset=self.proof_asset,
            source_raw_cursor_digest=self.proof_cursor.raw_cursor_digest,
            emitted_text=self._branch_text(branch_certificate_digest),
            branch_certificate_digest=branch_certificate_digest,
        )

    def terminalization_artifact(self, terminal_support_identity_digest: str):
        self._require_proof_context()
        return terminalization_artifact_from_continuation_asset(
            prepared=self.proof_prepared,
            asset=self.proof_asset,
            source_raw_cursor_digest=self.proof_cursor.raw_cursor_digest,
            terminal_support_identity_digest=terminal_support_identity_digest,
        )

    def _branch_text(self, branch_certificate_digest: str) -> str:
        matches = tuple(
            edge.emitted_text
            for edge in self.proof_asset.edges_from(
                self.proof_cursor.raw_cursor_digest
            )
            if branch_certificate_digest in edge.branch_certificate_digests
        )
        if len(matches) != 1:
            raise ValueError("continuation_asset_branch_proof_identity_not_unique")
        return matches[0]

    def _require_proof_context(self) -> None:
        if self.proof_asset is None or self.proof_cursor is None:
            raise ValueError("continuation_asset_proof_mode_required")
        if self.proof_prepared is None:
            raise ValueError("continuation_asset_proof_prepared_required")


def _verified_root_proof_cursor(*, asset, prepared):
    source = _source_snapshot_from_asset(prepared=prepared, asset=asset)
    proof_cursor = asset.root_proof_cursor
    if _identity_digest(source.cursor) != proof_cursor.raw_cursor_digest:
        raise ValueError("continuation_asset_proof_root_cursor_mismatch")
    record = asset.raw_cursor_record(proof_cursor.raw_cursor_digest)
    if record is None:
        raise ValueError("continuation_asset_proof_root_record_missing")
    if (
        record.compiled_node_id != proof_cursor.node_id
        or record.normalization_scale != proof_cursor.completion_scale
    ):
        raise ValueError("continuation_asset_proof_root_mapping_mismatch")
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
    ) -> "MolToSmilesContinuationDecoder":
        if prepared is not None and not proof_capable:
            raise ValueError("continuation_asset_proof_prepared_without_mode")
        if proof_capable and prepared is None:
            raise ValueError("continuation_asset_proof_prepared_required")
        asset = open_writer_continuation_core(path)
        if (
            expected_manifest_digest is not None
            and asset.manifest_digest != expected_manifest_digest
        ):
            raise ValueError("continuation_asset_manifest_digest_mismatch")
        core = _rust_core_for_asset(asset)
        proof_cursor = None
        if proof_capable:
            proof_cursor = _verified_root_proof_cursor(
                asset=asset,
                prepared=prepared,
            )
        return cls._from_state(
            _ContinuationAssetStateAdapter(
                core=core,
                cursor=core.root_cursor(),
                emitted_texts=(),
                asset_manifest_digest=asset.manifest_digest,
                proof_asset=asset if proof_capable else None,
                proof_cursor=proof_cursor,
                proof_prepared=prepared,
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
    ) -> "MolToSmilesContinuationDecoder":
        _validate_snapshot_shape(snapshot)
        if snapshot["digest"] != _snapshot_digest(snapshot):
            raise ValueError("continuation_decoder_snapshot_digest_mismatch")
        decoder = cls.from_asset(
            path,
            expected_manifest_digest=snapshot["asset_manifest_digest"],
            proof_capable=proof_capable,
            prepared=prepared,
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

    def branch_artifact(self, branch_certificate_digest: str):
        return self._state.branch_artifact(branch_certificate_digest)

    def terminalization_artifact(self, terminal_support_identity_digest: str):
        return self._state.terminalization_artifact(
            terminal_support_identity_digest
        )

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
