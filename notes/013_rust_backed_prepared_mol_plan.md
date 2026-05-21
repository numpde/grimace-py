Rust-backed PreparedMol plan
============================

Goal
----

Keep the Python public API stable:

```python
prepared = grimace.PrepareMol(mol)
payload = prepared.to_bytes()
restored = grimace.PreparedMol.from_bytes(payload)
```

Move ownership of the prepared object toward Rust so reading and runtime
consumption do not depend on Python dataclass records. RDKit remains allowed
only during `PrepareMol`.

Starting state
--------------

- Python `PrepareMol` uses RDKit to split fragments and prepare each connected
  fragment.
- Rust already owns and validates connected `PreparedSmilesGraphData`.
- Python `PreparedMol` currently stores private Python writer-flag and fragment
  records.
- Runtime accepts `PreparedMol` and consumes fragments without late RDKit
  preparation.
- `to_bytes()` / `from_bytes()` are public, but the current payload is JSON and
  not a compact long-term storage format.

Top alternatives
----------------

1. Keep Python `PreparedMol`, only replace JSON with binary.
   This is easy, but it keeps the real object in Python. It improves byte size
   but not the runtime boundary or future Rust fast-read path.

2. Add Rust `PreparedMol` and keep Python as a thin public wrapper.
   Python still performs RDKit preparation, but immediately hands primitive
   prepared fragments to Rust. Runtime consumes Rust-owned fragments. This
   preserves public API and moves the important object boundary now.

3. Move all preparation into Rust.
   This would be ideal only if Rust could call or replace RDKit preparation.
   Today Grimace depends on RDKit molecule objects and writer-surface behavior,
   so this is too large and not no-regret.

4. Make each fragment a standalone Rust `PreparedSmilesGraph` and keep only the
   fragment list in Python.
   This is halfway useful, but root mapping, writer flags, validation, and bytes
   still live in Python. It preserves the current split ownership smell.

5. Introduce a bulk dataset container first.
   This targets the final large-dataset use case, but it would bake storage
   decisions before the single-object contract is Rust-owned. It should follow,
   not precede, Rust-backed `PreparedMol`.

Principled approach
-------------------

Use alternative 2.

Implement a Rust `_core.PreparedMol` as the internal storage object. Python
`grimace.PreparedMol` remains the public class and wraps one Rust object. Python
`PrepareMol` still handles RDKit-specific preparation, then constructs the Rust
object from:

- schema version
- writer flag tuple
- ordered fragments
- original atom indices per fragment
- Rust `PreparedSmilesGraph` objects or dict-compatible graph payloads

The byte format should be owned by Rust. A compact binary encoding is preferable
once Rust owns the object, because it makes `from_bytes()` a direct Rust decode
rather than a Python JSON parse.

Implementation steps
--------------------

1. Add Rust `PreparedMol` data structs and Python binding methods:
   `new`, `to_bytes`, `from_bytes`, writer flag access, fragment count,
   fragment atom indices, and fragment graph access.
2. Make Python `PreparedMol` an opaque wrapper around `_core.PreparedMol`.
3. Change Python `PrepareMol` to construct the Rust object after RDKit
   preparation.
4. Change `_runtime.py` helper accessors to read from the wrapper/Rust object,
   not Python fragment dataclasses.
5. Keep all public runtime tests green after each step.
6. Add tests proving Python structural records are gone from `PreparedMol` and
   byte round trips still cover connected, disconnected, stereo, writer flags,
   and malformed payloads.

Non-goals for this pass
-----------------------

- No bulk dataset container yet.
- No RDKit-free preparation yet.
- No stable compact binary format promise yet.
- No extra public structural accessors.

Implemented result
------------------

- Python `PreparedMol` is now an opaque wrapper around `_core.PreparedMol`.
- Python no longer stores writer-flag or fragment dataclass records.
- Runtime helper accessors read writer flags, fragment atom indices, and
  prepared graphs from the Rust object.
- `to_bytes()` emits a versioned Rust binary payload.
- `from_bytes()` decodes and validates the binary payload in Rust.
- Tests cover connected, disconnected, stereo, writer flags, byte round trips,
  malformed binary payloads, malformed structural payloads, and no late RDKit
  preparation after `PrepareMol`.
