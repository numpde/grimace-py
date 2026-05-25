"""Online prefix feasibility and next-character decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .facts import MoleculeFacts
from .online_render_sink import PrefixConstrainedSink
from .online_stereo_witness import iter_online_stereo_witnesses_with_sink
from .policy import SmilesPolicy
from .semantics import ParserSemantics


@dataclass(frozen=True, slots=True)
class OnlineDecodeToken:
    text: str
    kind: Literal[
        "atom_text",
        "bond_text",
        "direction_mark",
        "ring_label",
        "branch_open",
        "branch_close",
        "dot",
    ]


def online_prefix_has_completion(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
) -> bool:
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: PrefixConstrainedSink(required_prefix=prefix),
    ):
        return True
    return False


def online_allowed_next_characters(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
    alphabet: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if alphabet is None:
        alphabet = online_decoder_alphabet(policy)
    return tuple(
        ch
        for ch in alphabet
        if online_prefix_has_completion(
            facts=facts,
            policy=policy,
            semantics=semantics,
            prefix=prefix + ch,
        )
    )


def online_allowed_next_tokens(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
    tokens: tuple[OnlineDecodeToken, ...] | None = None,
) -> tuple[OnlineDecodeToken, ...]:
    if tokens is None:
        tokens = online_decode_tokens(policy)
    return tuple(
        token
        for token in tokens
        if token.text
        and online_prefix_has_completion(
            facts=facts,
            policy=policy,
            semantics=semantics,
            prefix=prefix + token.text,
        )
    )


def online_decoder_alphabet(policy: SmilesPolicy) -> tuple[str, ...]:
    chars: set[str] = set("()/\\.")
    for label in policy.ring_labels:
        chars.update(label.text())
    for domain in policy.atom_text_domains:
        for choice in domain.choices:
            for _, text in choice.text_by_tetra:
                chars.update(text)
    for domain in policy.bond_text_domains:
        for choice in domain.choices:
            chars.update(choice.base_text)
            if choice.permits_direction:
                chars.update("/\\")
    return tuple(sorted(chars))


def online_decode_tokens(policy: SmilesPolicy) -> tuple[OnlineDecodeToken, ...]:
    tokens: set[OnlineDecodeToken] = {
        OnlineDecodeToken("(", "branch_open"),
        OnlineDecodeToken(")", "branch_close"),
        OnlineDecodeToken(".", "dot"),
        OnlineDecodeToken("/", "direction_mark"),
        OnlineDecodeToken("\\", "direction_mark"),
    }
    for label in policy.ring_labels:
        tokens.add(OnlineDecodeToken(label.text(), "ring_label"))
    for domain in policy.atom_text_domains:
        for choice in domain.choices:
            for _, text in choice.text_by_tetra:
                tokens.add(OnlineDecodeToken(text, "atom_text"))
    for domain in policy.bond_text_domains:
        for choice in domain.choices:
            if choice.base_text:
                tokens.add(OnlineDecodeToken(choice.base_text, "bond_text"))
    return tuple(sorted(tokens, key=lambda token: (token.text, token.kind)))


__all__ = (
    "OnlineDecodeToken",
    "online_allowed_next_characters",
    "online_allowed_next_tokens",
    "online_decode_tokens",
    "online_decoder_alphabet",
    "online_prefix_has_completion",
)
