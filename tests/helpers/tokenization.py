from __future__ import annotations

from collections.abc import Iterable


STRUCTURAL_TOKENS = {"(", ")", ".", "-", "=", "#", ":", "/", "\\"}


def _sorted_atom_tokens(atom_tokens: Iterable[str]) -> tuple[str, ...]:
    unique_tokens = {token for token in atom_tokens if token and not token.startswith("[")}
    return tuple(sorted(unique_tokens, key=lambda token: (-len(token), token)))


def tokenize_smiles(
    smiles: str,
    *,
    atom_tokens: Iterable[str],
) -> tuple[str, ...]:
    known_atoms = _sorted_atom_tokens(atom_tokens)
    tokens: list[str] = []
    index = 0

    while index < len(smiles):
        char = smiles[index]

        if char in STRUCTURAL_TOKENS:
            tokens.append(char)
            index += 1
            continue

        if char == "[":
            end_index = smiles.find("]", index + 1)
            if end_index < 0:
                raise ValueError(f"Unterminated bracket atom in {smiles!r}")
            tokens.append(smiles[index : end_index + 1])
            index = end_index + 1
            continue

        if char == "%":
            if index + 1 < len(smiles) and smiles[index + 1] == "(":
                end_index = smiles.find(")", index + 2)
                if end_index < 0:
                    raise ValueError(f"Unterminated parenthesized ring label in {smiles!r}")
                label = smiles[index : end_index + 1]
                if not label[2:-1].isdigit():
                    raise ValueError(f"Invalid parenthesized ring label {label!r}")
                tokens.append(label)
                index = end_index + 1
                continue

            if index + 2 >= len(smiles):
                raise ValueError(f"Truncated percent ring label in {smiles!r}")
            label = smiles[index : index + 3]
            if not (label[1].isdigit() and label[2].isdigit()):
                raise ValueError(f"Invalid percent ring label {label!r}")
            tokens.append(label)
            index += 3
            continue

        if char.isdigit():
            tokens.append(char)
            index += 1
            continue

        matched_atom = next((token for token in known_atoms if smiles.startswith(token, index)), None)
        if matched_atom is None:
            raise ValueError(
                f"Could not tokenize {smiles!r} at offset {index}; remaining suffix {smiles[index:]!r}"
            )

        tokens.append(matched_atom)
        index += len(matched_atom)

    return tuple(tokens)


def expected_next_tokens_from_support(
    outputs: Iterable[str],
    prefix: str,
    *,
    atom_tokens: Iterable[str],
) -> tuple[str, ...]:
    prefix_tokens = tokenize_smiles(prefix, atom_tokens=atom_tokens)
    next_tokens = set()

    for output in outputs:
        output_tokens = tokenize_smiles(output, atom_tokens=atom_tokens)
        if len(output_tokens) <= len(prefix_tokens):
            continue
        if output_tokens[: len(prefix_tokens)] != prefix_tokens:
            continue
        next_tokens.add(output_tokens[len(prefix_tokens)])

    return tuple(sorted(next_tokens))
