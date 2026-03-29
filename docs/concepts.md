# Concepts

## Exact support

For a fixed molecule, root, and set of writer flags, the library defines a set
of complete SMILES strings. That set is the exact support.

`MolToSmilesEnum(...)` yields those complete strings.

## Decoder

`MolToSmilesDecoder(...)` exposes the same runtime as a stateful decoder.

At each step:

- `nextTokens()` returns the allowed next SMILES fragments
- `advance(token)` moves the decoder forward
- `prefix()` returns the current prefix

The tokens are literal SMILES fragments, not token ids. A token may be one
character or many characters, for example `"C"`, `"[C@H]"`, or `"%12"`.

## Why enum and decoder are separate

The decoder returns next tokens for one current state.

That is not enough to reconstruct the full support by itself, because one token
can lead to more than one successor state. The enum path and the decoder path
therefore share one backend, but they are different public operations.

## Rooted generation

The current public API is rooted. You must pass `rootedAtAtom >= 0`.

## Connected molecules

The current public API only supports singly-connected molecules. Disconnected
molecules fail fast.
