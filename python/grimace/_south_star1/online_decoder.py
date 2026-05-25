"""Online prefix feasibility and next-character decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .facts import MoleculeFacts
from .online_render_sink import PrefixFrontierSink
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


@dataclass(slots=True)
class OnlineFrontierStats:
    dfs_runs: int = 0
    sink_rejections: int = 0
    completions_seen: int = 0
    committed_frontier_chars: int = 0
    committed_frontier_tokens: int = 0


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


def online_allowed_next_characters_bruteforce(
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


def online_allowed_next_characters_one_pass(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
) -> tuple[str, ...]:
    frontier, _ = online_allowed_next_characters_one_pass_with_stats(
        facts=facts,
        policy=policy,
        semantics=semantics,
        prefix=prefix,
    )
    return frontier


def online_allowed_next_characters_one_pass_with_stats(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
) -> tuple[tuple[str, ...], OnlineFrontierStats]:
    sink = PrefixFrontierSink(required_prefix=prefix)
    stats = OnlineFrontierStats(dfs_runs=1)
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: sink,
    ):
        pass
    stats.sink_rejections = sink.sink_rejections
    stats.completions_seen = sink.completions_seen
    stats.committed_frontier_chars = len(sink.frontier_chars)
    stats.committed_frontier_tokens = len(sink.frontier_token_texts)
    return tuple(sorted(sink.frontier_chars)), stats


def online_allowed_next_token_texts_one_pass(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
) -> tuple[str, ...]:
    sink = PrefixFrontierSink(required_prefix=prefix)
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: sink,
    ):
        pass
    return tuple(sorted(sink.frontier_token_texts))


def online_allowed_next_tokens_bruteforce(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
    tokens: tuple[OnlineDecodeToken, ...] | None = None,
) -> tuple[OnlineDecodeToken, ...]:
    if tokens is None:
        tokens = online_decode_tokens(policy)
    out: list[OnlineDecodeToken] = []
    for token in tokens:
        if not token.text:
            continue
        sink = PrefixFrontierSink(required_prefix=prefix)
        for _ in iter_online_stereo_witnesses_with_sink(
            facts=facts,
            policy=policy,
            semantics=semantics,
            sink_factory=lambda: sink,
        ):
            pass
        if token.text in sink.frontier_token_texts:
            out.append(token)
    return tuple(out)


def online_allowed_next_characters(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
    alphabet: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if alphabet is not None:
        frontier = set(
            online_allowed_next_characters_one_pass(
                facts=facts,
                policy=policy,
                semantics=semantics,
                prefix=prefix,
            )
        )
        return tuple(ch for ch in alphabet if ch in frontier)
    return online_allowed_next_characters_one_pass(
        facts=facts,
        policy=policy,
        semantics=semantics,
        prefix=prefix,
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
    token_texts = set(
        online_allowed_next_token_texts_one_pass(
            facts=facts,
            policy=policy,
            semantics=semantics,
            prefix=prefix,
        )
    )
    return tuple(token for token in tokens if token.text in token_texts)


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
    "OnlineFrontierStats",
    "online_allowed_next_characters",
    "online_allowed_next_characters_bruteforce",
    "online_allowed_next_characters_one_pass",
    "online_allowed_next_characters_one_pass_with_stats",
    "online_allowed_next_tokens",
    "online_allowed_next_tokens_bruteforce",
    "online_allowed_next_token_texts_one_pass",
    "online_decode_tokens",
    "online_decoder_alphabet",
    "online_prefix_has_completion",
)
