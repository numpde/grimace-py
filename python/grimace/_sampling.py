"""Public SMILES sampling records and wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import grimace._runtime as _runtime
from grimace._runtime_walks import (
    _TokenWalkResult,
    _seeded_branch_multiplicity_chooser,
    _seeded_branch_preserving_chooser,
    _seeded_uniform_token_chooser,
    _validate_walk_seed,
    _walk_branch_transitions,
    _walk_token_transitions,
)


DecoderView = Literal["determinized", "branch_preserving"]
SamplingMode = Literal[
    "uniform_token",
    "branch_multiplicity",
    "branch_preserving",
]
_SamplingPair = tuple[DecoderView, SamplingMode]


_SAMPLING_WALKERS = {
    ("determinized", "uniform_token"): (
        _walk_token_transitions,
        _seeded_uniform_token_chooser,
    ),
    ("determinized", "branch_multiplicity"): (
        _walk_token_transitions,
        _seeded_branch_multiplicity_chooser,
    ),
    ("branch_preserving", "branch_preserving"): (
        _walk_branch_transitions,
        _seeded_branch_preserving_chooser,
    ),
}
_VALID_MODE_PAIRS = frozenset(_SAMPLING_WALKERS)


@dataclass(frozen=True, slots=True)
class SmilesSampleStep:
    choice_tokens: tuple[str, ...]
    choice_branch_counts: tuple[int, ...]
    selected_index: int
    selected_token: str

    def __post_init__(self) -> None:
        if type(self.choice_tokens) is not tuple:
            raise TypeError("sample step choice_tokens must be a tuple")
        if type(self.choice_branch_counts) is not tuple:
            raise TypeError("sample step choice_branch_counts must be a tuple")
        if len(self.choice_tokens) != len(self.choice_branch_counts):
            raise ValueError("choice token and branch-count lengths differ")
        if not self.choice_tokens:
            raise ValueError("sample step requires at least one choice")
        if len(set(self.choice_tokens)) != len(self.choice_tokens):
            raise ValueError("sample step choice tokens must be unique")
        if not all(isinstance(token, str) for token in self.choice_tokens):
            raise TypeError("sample step choice tokens must be strings")
        if not all(
            type(branch_count) is int and branch_count > 0
            for branch_count in self.choice_branch_counts
        ):
            raise ValueError("sample step branch counts must be positive ints")
        if type(self.selected_index) is not int:
            raise TypeError("sample step selected_index must be an int")
        if not 0 <= self.selected_index < len(self.choice_tokens):
            raise ValueError("sample step selected_index is out of range")
        if not isinstance(self.selected_token, str):
            raise TypeError("sample step selected_token must be a string")
        if self.selected_token != self.choice_tokens[self.selected_index]:
            raise ValueError("sample step selected_token does not match selected_index")


@dataclass(frozen=True, slots=True)
class SmilesSample:
    tokens: tuple[str, ...]
    smiles: str
    decoder_view: str
    sampling_mode: str
    steps: tuple[SmilesSampleStep, ...]

    def __post_init__(self) -> None:
        if type(self.tokens) is not tuple:
            raise TypeError("sample tokens must be a tuple")
        if not all(isinstance(token, str) for token in self.tokens):
            raise TypeError("sample tokens must be strings")
        if not isinstance(self.smiles, str):
            raise TypeError("sample smiles must be a string")
        if self.smiles != "".join(self.tokens):
            raise ValueError("sample smiles must equal joined tokens")
        _validate_mode_pair(self.decoder_view, self.sampling_mode)
        if type(self.steps) is not tuple:
            raise TypeError("sample steps must be a tuple")
        if not all(isinstance(step, SmilesSampleStep) for step in self.steps):
            raise TypeError("sample steps must be SmilesSampleStep instances")
        if len(self.steps) != len(self.tokens):
            raise ValueError("sample step count must match token count")
        for token, step in zip(self.tokens, self.steps):
            if token != step.selected_token:
                raise ValueError("sample token does not match selected step token")


def mol_to_smiles_sample(
    mol_or_prepared: object,
    *,
    seed: int,
    decoder_view: str = "determinized",
    sampling_mode: str = "uniform_token",
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> SmilesSample:
    decoder_view, sampling_mode = _validate_mode_pair(
        decoder_view,
        sampling_mode,
    )
    _validate_walk_seed(seed)
    initial_state = _runtime._make_decoder_state(
        mol_or_prepared,
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )

    walk, seeded_chooser = _SAMPLING_WALKERS[(decoder_view, sampling_mode)]
    result = walk(initial_state, seeded_chooser(seed))

    return _public_sample_from_walk_result(
        result,
        decoder_view=decoder_view,
        sampling_mode=sampling_mode,
    )


def _validate_mode_pair(
    decoder_view: object,
    sampling_mode: object,
) -> tuple[DecoderView, SamplingMode]:
    if not isinstance(decoder_view, str):
        raise ValueError("invalid decoder_view/sampling_mode pair")
    if not isinstance(sampling_mode, str):
        raise ValueError("invalid decoder_view/sampling_mode pair")
    pair = (decoder_view, sampling_mode)
    if pair not in _VALID_MODE_PAIRS:
        raise ValueError(
            f"invalid decoder_view/sampling_mode pair: {pair!r}"
        )
    return decoder_view, sampling_mode


def _public_sample_from_walk_result(
    result: _TokenWalkResult,
    *,
    decoder_view: str,
    sampling_mode: str,
) -> SmilesSample:
    if len(result.tokens) != len(result.selected_indices):
        raise ValueError("walk token count does not match selected-index count")
    if len(result.tokens) != len(result.choice_counts):
        raise ValueError("walk token count does not match choice-count count")
    if len(result.choice_tokens) != len(result.choice_branch_counts):
        raise ValueError("walk choice token and branch-count lengths differ")

    steps: list[SmilesSampleStep] = []
    offset = 0
    for token, selected_index, choice_count in zip(
        result.tokens,
        result.selected_indices,
        result.choice_counts,
        strict=True,
    ):
        stop = offset + choice_count
        steps.append(
            SmilesSampleStep(
                choice_tokens=result.choice_tokens[offset:stop],
                choice_branch_counts=result.choice_branch_counts[offset:stop],
                selected_index=selected_index,
                selected_token=token,
            )
        )
        offset = stop
    if offset != len(result.choice_tokens):
        raise ValueError("walk choice counts do not span choice payload")

    return SmilesSample(
        tokens=result.tokens,
        smiles="".join(result.tokens),
        decoder_view=decoder_view,
        sampling_mode=sampling_mode,
        steps=tuple(steps),
    )
