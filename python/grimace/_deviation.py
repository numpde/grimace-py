"""SMILES candidate diagnostics built on the public runtime decoder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from grimace import _runtime
from grimace._mol_to_smiles_options import MOL_TO_SMILES_OPTIONS
from grimace._runtime_states import DecoderCacheKey

SmilesDeviationReason: TypeAlias = Literal["unexpected_text", "unexpected_token", "incomplete"]


@dataclass(frozen=True, slots=True)
class SmilesDeviation:
    reason: SmilesDeviationReason
    char_index: int
    token_index: int | None
    offset_in_token: int | None
    accepted_text: str
    rejected_text: str
    legal_next_tokens: tuple[str, ...]


DecoderOptions: TypeAlias = dict[str, object]


def _sorted_tokens(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _decoder_key(decoder: _runtime.MolToSmilesDeterminizedDecoder) -> DecoderCacheKey:
    return decoder._cache_key()


def _dedupe_decoders(
    decoders: Iterable[_runtime.MolToSmilesDeterminizedDecoder],
) -> tuple[_runtime.MolToSmilesDeterminizedDecoder, ...]:
    by_key: dict[DecoderCacheKey, _runtime.MolToSmilesDeterminizedDecoder] = {}
    for decoder in decoders:
        by_key.setdefault(_decoder_key(decoder), decoder)
    return tuple(by_key.values())


def _next_token_texts(
    decoders: Iterable[_runtime.MolToSmilesDeterminizedDecoder],
) -> tuple[str, ...]:
    return _sorted_tokens(
        choice.text
        for decoder in decoders
        for choice in decoder.next_choices
    )


def _string_deviation(
    mol_or_prepared: object,
    candidate: str,
    options: DecoderOptions,
) -> SmilesDeviation | None:
    initial = _runtime.MolToSmilesDeterminizedDecoder(mol_or_prepared, **options)
    active_by_offset: dict[int, dict[DecoderCacheKey, _runtime.MolToSmilesDeterminizedDecoder]] = {
        0: {_decoder_key(initial): initial}
    }
    pending_offsets = {0}
    best_char_index = 0
    best_legal_next_tokens: tuple[str, ...] = ()

    while pending_offsets:
        offset = min(pending_offsets)
        pending_offsets.remove(offset)
        if offset > len(candidate):
            continue

        decoders = tuple(active_by_offset.get(offset, {}).values())
        choices_by_decoder = tuple(decoder.next_choices for decoder in decoders)
        legal_next_tokens = _sorted_tokens(
            choice.text
            for choices in choices_by_decoder
            for choice in choices
        )

        if offset >= best_char_index:
            best_char_index = offset
            best_legal_next_tokens = legal_next_tokens

        suffix = candidate[offset:]
        for choices in choices_by_decoder:
            for choice in choices:
                common_len = _common_prefix_len(choice.text, suffix)
                if common_len and offset + common_len > best_char_index:
                    best_char_index = offset + common_len
                    best_legal_next_tokens = legal_next_tokens
                if not suffix.startswith(choice.text):
                    continue
                next_offset = offset + len(choice.text)
                next_bucket = active_by_offset.setdefault(next_offset, {})
                key = _decoder_key(choice.next_state)
                if key not in next_bucket:
                    next_bucket[key] = choice.next_state
                    pending_offsets.add(next_offset)

    final_decoders = tuple(active_by_offset.get(len(candidate), {}).values())
    if any(decoder.is_terminal for decoder in final_decoders):
        return None

    if final_decoders:
        return SmilesDeviation(
            reason="incomplete",
            char_index=len(candidate),
            token_index=None,
            offset_in_token=None,
            accepted_text=candidate,
            rejected_text="",
            legal_next_tokens=_next_token_texts(final_decoders),
        )

    reason: SmilesDeviationReason = (
        "incomplete" if best_char_index == len(candidate) else "unexpected_text"
    )
    return SmilesDeviation(
        reason=reason,
        char_index=best_char_index,
        token_index=None,
        offset_in_token=None,
        accepted_text=candidate[:best_char_index],
        rejected_text=candidate[best_char_index:],
        legal_next_tokens=best_legal_next_tokens,
    )


def _candidate_token_text_and_starts(
    candidate: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    tokens = tuple(candidate)
    if not all(type(token) is str for token in tokens):
        raise TypeError("candidate token sequence must contain strings")
    if any(token == "" for token in tokens):
        raise ValueError("candidate token sequence must not contain empty strings")

    starts: list[int] = []
    offset = 0
    for token in tokens:
        starts.append(offset)
        offset += len(token)
    return "".join(tokens), tokens, tuple(starts)


def _sequence_deviation(
    mol_or_prepared: object,
    candidate: Sequence[str],
    options: DecoderOptions,
) -> SmilesDeviation | None:
    candidate_text, tokens, token_starts = _candidate_token_text_and_starts(candidate)
    active_decoders = (
        _runtime.MolToSmilesDeterminizedDecoder(mol_or_prepared, **options),
    )

    for token_index, token in enumerate(tokens):
        choices_by_text: dict[str, list[_runtime.MolToSmilesDeterminizedDecoder]] = {}
        for decoder in active_decoders:
            for choice in decoder.next_choices:
                choices_by_text.setdefault(choice.text, []).append(choice.next_state)

        token_start = token_starts[token_index]
        if token not in choices_by_text:
            return SmilesDeviation(
                reason="unexpected_token",
                char_index=token_start,
                token_index=token_index,
                offset_in_token=0,
                accepted_text=candidate_text[:token_start],
                rejected_text=candidate_text[token_start:],
                legal_next_tokens=_sorted_tokens(choices_by_text),
            )

        active_decoders = _dedupe_decoders(choices_by_text[token])

    if any(decoder.is_terminal for decoder in active_decoders):
        return None

    return SmilesDeviation(
        reason="incomplete",
        char_index=len(candidate_text),
        token_index=len(tokens) - 1 if tokens else None,
        offset_in_token=len(tokens[-1]) if tokens else None,
        accepted_text=candidate_text,
        rejected_text="",
        legal_next_tokens=_next_token_texts(active_decoders),
    )


def mol_to_smiles_deviation(
    mol_or_prepared: object,
    candidate: str | Sequence[str],
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> SmilesDeviation | None:
    """Return the first candidate location outside the molecule's SMILES language."""

    option_values = locals()
    options: DecoderOptions = {
        spec.internal_name: option_values[spec.internal_name]
        for spec in MOL_TO_SMILES_OPTIONS
    }

    if isinstance(candidate, str):
        return _string_deviation(mol_or_prepared, candidate, options)

    if not isinstance(candidate, Sequence):
        raise TypeError("candidate must be a string or a sequence of strings")
    return _sequence_deviation(mol_or_prepared, candidate, options)


__all__ = [
    "SmilesDeviation",
    "SmilesDeviationReason",
    "mol_to_smiles_deviation",
]
