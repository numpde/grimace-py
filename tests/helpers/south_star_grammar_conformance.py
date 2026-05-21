from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.atom_text import SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import is_south_star_bracket_atom_text_token


SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS = "south_star_declared_subset_grammar_v1"
_ATOM_TOKENS: frozenset[str] = SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS
_TWO_CHAR_ATOM_TOKENS: frozenset[str] = frozenset(
    token for token in _ATOM_TOKENS if len(token) == 2
)
_BOND_TOKENS: frozenset[str] = frozenset({"=", "#", "/", "\\"})
_RING_LABEL_TOKENS: frozenset[str] = frozenset("123456789")

@dataclass(frozen=True, slots=True)
class SouthStarGrammarConformance:
    passed: bool
    basis: str
    detail: str
    rejection_code: str = ""
    tokens: tuple[str, ...] = ()


def south_star_grammar_conformance(smiles: str) -> SouthStarGrammarConformance:
    tokenization = _tokenize_declared_subset(smiles)
    if tokenization is None:
        return _rejected("unsupported_token", "candidate contains unsupported syntax")
    tokens = tokenization
    if not tokens:
        return _rejected("empty_string", "candidate is empty", tokens=tokens)

    structural_rejection = _first_structural_rejection(tokens)
    if structural_rejection is not None:
        code, detail = structural_rejection
        return _rejected(code, detail, tokens=tokens)

    return SouthStarGrammarConformance(
        passed=True,
        basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
        detail="candidate belongs to the declared South Star grammar subset",
        tokens=tokens,
    )


def _tokenize_declared_subset(smiles: str) -> tuple[str, ...] | None:
    tokens = []
    index = 0
    while index < len(smiles):
        if smiles[index] == "[":
            end_index = smiles.find("]", index + 1)
            if end_index == -1:
                return None
            token = smiles[index : end_index + 1]
            if not is_south_star_bracket_atom_text_token(token):
                return None
            tokens.append(token)
            index = end_index + 1
            continue

        two_char = smiles[index : index + 2]
        if two_char in _TWO_CHAR_ATOM_TOKENS:
            tokens.append(two_char)
            index += 2
            continue

        one_char = smiles[index]
        if (
            one_char in _ATOM_TOKENS
            or one_char in _BOND_TOKENS
            or one_char in _RING_LABEL_TOKENS
            or one_char in {"(", ")", "."}
        ):
            tokens.append(one_char)
            index += 1
            continue

        return None
    return tuple(tokens)


def _first_structural_rejection(tokens: tuple[str, ...]) -> tuple[str, str] | None:
    previous_kind = "start"
    branch_depth = 0
    ring_label_counts: dict[str, int] = {}
    pending_bond = False

    for token in tokens:
        kind = _token_kind(token)
        if kind == "atom":
            if previous_kind not in {
                "start",
                "atom",
                "ring_label",
                "branch_open",
                "branch_close",
                "bond",
                "dot",
            }:
                return ("atom_context", "atom token appears in an invalid context")
            pending_bond = False
        elif kind == "bond":
            if pending_bond:
                return ("consecutive_bonds", "bond or marker tokens cannot repeat")
            if previous_kind not in {"atom", "ring_label", "branch_open", "branch_close"}:
                return ("bond_context", "bond or marker token has invalid left context")
            pending_bond = True
        elif kind == "ring_label":
            if previous_kind not in {"atom", "ring_label", "branch_close", "bond"}:
                return ("ring_label_context", "ring label has invalid left context")
            ring_label_counts[token] = ring_label_counts.get(token, 0) + 1
            if ring_label_counts[token] > 2:
                return ("ring_label_reuse", "ring label occurs more than twice")
            pending_bond = False
        elif kind == "branch_open":
            if previous_kind not in {"atom", "ring_label", "branch_close"}:
                return ("branch_open_context", "branch open has invalid left context")
            branch_depth += 1
            pending_bond = False
        elif kind == "branch_close":
            if branch_depth == 0:
                return ("branch_underflow", "branch close has no matching open")
            if previous_kind not in {"atom", "ring_label", "branch_close"}:
                return ("branch_close_context", "branch close has invalid left context")
            branch_depth -= 1
            pending_bond = False
        elif kind == "dot":
            if previous_kind not in {"atom", "ring_label", "branch_close"}:
                return ("dot_context", "fragment separator has invalid left context")
            pending_bond = False
        else:
            raise AssertionError(f"unhandled South Star grammar token kind {kind!r}")

        previous_kind = kind

    if pending_bond:
        return ("trailing_bond", "bond or marker token cannot terminate a string")
    if branch_depth != 0:
        return ("unbalanced_branch", "branch parentheses are not balanced")
    unpaired_ring_labels = tuple(
        label for label, count in sorted(ring_label_counts.items()) if count != 2
    )
    if unpaired_ring_labels:
        return ("unpaired_ring_label", "ring labels must occur exactly twice")
    if previous_kind == "dot":
        return ("trailing_dot", "fragment separator cannot terminate a string")
    return None


def _token_kind(token: str) -> str:
    if token in _ATOM_TOKENS or is_south_star_bracket_atom_text_token(token):
        return "atom"
    if token in _BOND_TOKENS:
        return "bond"
    if token in _RING_LABEL_TOKENS:
        return "ring_label"
    if token == "(":
        return "branch_open"
    if token == ")":
        return "branch_close"
    if token == ".":
        return "dot"
    raise ValueError(f"unsupported South Star grammar token {token!r}")


def _rejected(
    code: str,
    detail: str,
    *,
    tokens: tuple[str, ...] = (),
) -> SouthStarGrammarConformance:
    return SouthStarGrammarConformance(
        passed=False,
        basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
        detail=detail,
        rejection_code=code,
        tokens=tokens,
    )
