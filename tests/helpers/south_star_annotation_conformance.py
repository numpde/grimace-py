from __future__ import annotations

from dataclasses import dataclass


ANNOTATION_CONFORMANCE_BASIS = (
    "south_star_current_subset_directional_marker_grammar"
)
_ATOM_TOKENS: frozenset[str] = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
)
_BOND_TOKENS: frozenset[str] = frozenset({"=", "/", "\\"})


@dataclass(frozen=True, slots=True)
class SouthStarAnnotationConformance:
    passed: bool
    detail: str


def south_star_annotation_conformance(
    smiles: str,
) -> SouthStarAnnotationConformance:
    tokens = _tokenize_current_subset(smiles)
    if tokens is None:
        return SouthStarAnnotationConformance(
            passed=False,
            detail="candidate contains tokens outside the current South Star subset",
        )
    if not _parentheses_balanced(tokens):
        return SouthStarAnnotationConformance(
            passed=False,
            detail="candidate branch parentheses are not balanced",
        )

    for index, token in enumerate(tokens):
        if token not in {"/", "\\"}:
            continue
        if index == 0 or index == len(tokens) - 1:
            return SouthStarAnnotationConformance(
                passed=False,
                detail="directional marker cannot appear at a string boundary",
            )
        if (
            tokens[index - 1] not in _ATOM_TOKENS
            and tokens[index - 1] not in {"(", ")"}
        ):
            return SouthStarAnnotationConformance(
                passed=False,
                detail=(
                    "directional marker must follow an atom, branch open, "
                    "or branch close"
                ),
            )
        if tokens[index + 1] not in _ATOM_TOKENS:
            return SouthStarAnnotationConformance(
                passed=False,
                detail="directional marker must precede an atom",
            )

    return SouthStarAnnotationConformance(
        passed=True,
        detail="directional markers satisfy the current South Star subset grammar",
    )


def _tokenize_current_subset(smiles: str) -> tuple[str, ...] | None:
    tokens = []
    index = 0
    while index < len(smiles):
        two_char = smiles[index : index + 2]
        if two_char in _ATOM_TOKENS:
            tokens.append(two_char)
            index += 2
            continue

        one_char = smiles[index]
        if one_char in _ATOM_TOKENS or one_char in _BOND_TOKENS or one_char in "()":
            tokens.append(one_char)
            index += 1
            continue

        return None
    return tuple(tokens)


def _parentheses_balanced(tokens: tuple[str, ...]) -> bool:
    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
