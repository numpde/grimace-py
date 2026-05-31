# PreparedMol zstd dictionary

This note defines the first no-regret shape for optional `PreparedMol`
compression with a shipped zstd dictionary.

## Decision

Keep raw `PreparedMol` bytes as the canonical storage format:

```text
GPM\0 ...
```

Compressed `PreparedMol` bytes should be ordinary zstd frames. Do not add a
Grimace compression envelope unless zstd frames prove insufficient.

Implicit built-in dictionary selection should use the zstd frame dictionary ID.
The compressed frame must carry a nonzero dictionary ID. The reader uses that
ID as a lookup key for shipped dictionaries, decompresses, then parses the
result as raw `GPM\0` bytes.

The dictionary ID is a lookup key, not a cryptographic identity. Artifact tests
must validate the shipped dictionary and manifest hashes separately.

The compressed form is a storage format. Once a built-in dictionary is shipped,
future releases should keep that dictionary readable unless there is an explicit
storage-breaking release. `compression="zstd"` may move to a newer default
dictionary, but `from_bytes(...)` must keep resolving older shipped dictionary
IDs.

`optional` should mean optional use by callers, not optional readability in the
distributed wheels. If public `PreparedMol.from_bytes(...)` accepts compressed
payloads, release wheels must include zstd support and the shipped built-in
dictionaries. A feature-gated build that cannot read compressed `PreparedMol`
bytes would be a different product surface and should not be released under the
same Python API without a very explicit compatibility decision.

## Artifact layout

Ship dictionary artifacts under:

```text
python/grimace/data/prepared_mol_zstd/
  YYYYMMDD_hhhhhhhh/
    default_v1.zstdict
    default_v1.json
```

The directory name is:

```text
{created_yyyymmdd}_{manifest_sha256[:8]}
```

The files inside use semantic names. The public identity is the semantic
dictionary ID declared by the manifest, not the path and not the zstd frame's
32-bit dictionary ID.

The date is creation metadata, not semantic identity. Do not create a second
artifact directory with the same manifest hash only because the generator was
rerun on a later date.

## Runtime behavior

Write:

```python
prepared.to_bytes()
```

returns raw `GPM\0` bytes.

```python
prepared.to_bytes(compression="zstd")
```

compresses raw `GPM\0` bytes with the shipped default dictionary and writes a
zstd frame containing the dictionary ID. Do not silently fall back to raw bytes
when compression is larger than raw; callers who ask for zstd should get a zstd
frame. Bulk storage layers may choose raw or compressed payloads themselves.

```python
prepared.to_bytes(compression="zstd", dictionary=dictionary)
```

compresses with an explicit dictionary.

Keep the public argument surface narrow:

- `to_bytes()` remains raw and argument-free in the common case
- compression options are keyword-only
- supported `compression` values are exactly `None`/omitted and `"zstd"`
- unknown compression names fail before doing work
- on write, `dictionary=...` is valid only with `compression="zstd"`

Compression output does not need to be byte-identical across supported zstd
library versions. The compatibility contract is that Grimace-produced frames
round-trip, carry the required metadata, and decompress to canonical raw
`GPM\0` bytes.

Do not duplicate the compression vocabulary across layers. Pick one owning
module for the public option names and keep Python/Rust tests that prove the
exposed signature, accepted values, and error behavior stay aligned.

Read:

```python
grimace.PreparedMol.from_bytes(payload)
```

should:

1. Parse raw `GPM\0` bytes directly.
2. Otherwise, detect a zstd frame.
3. Read the frame dictionary ID.
4. If the ID matches a shipped built-in dictionary, decompress with it.
5. Require decompressed bytes to start with `GPM\0`.
6. Parse with the existing raw `PreparedMol` parser.

For custom dictionaries:

```python
grimace.PreparedMol.from_bytes(payload, dictionary=dictionary)
```

uses the explicit dictionary only for zstd payloads. The dictionary argument is
keyword-only. Passing a dictionary while reading raw `GPM\0` bytes should fail,
because raw bytes are self-contained and silently ignoring the dictionary would
hide caller mistakes.

Detection is a prefix check:

```text
raw PreparedMol: b"GPM\0"
zstd frame:      b"\x28\xb5\x2f\xfd"
```

The zstd frame magic appears on disk as bytes `28 b5 2f fd`, which is the
standard zstd frame magic number `0xFD2FB528` in little-endian byte order. Do
not accept zstd skippable frames for v1; a compressed `PreparedMol` payload
must begin with the normal zstd frame magic.

For Grimace-produced frames, require:

- exactly one zstd frame
- no trailing compressed bytes
- a nonzero dictionary ID
- frame content size present
- content checksum present
- decompressed size below a named implementation limit

The reader should reject missing content size or missing checksum for v1
compressed `PreparedMol` payloads. Even with content size present,
decompression must be bounded so a hostile frame cannot allocate unbounded
memory before the raw `GPM\0` parser sees it. Use a named constant for the cap
and test it directly. The cap is part of the implementation contract: choose it
large enough for the largest supported raw `PreparedMol`, and document that
larger payloads are unsupported rather than trying to allocate and hope.

If the frame has no dictionary ID, has dictionary ID zero, or references an
unknown dictionary, `from_bytes(payload)` must fail with a clear error. An
explicit dictionary may be supplied for non-built-in payloads.

With an explicit dictionary:

- if the frame carries a nonzero dictionary ID, it must match the explicit
  dictionary
- if the frame omits a dictionary ID, the explicit dictionary may be used, but
  the frame must still satisfy the v1 frame rules other than the built-in
  dictionary-ID requirement, and the decompressed payload must still be a single
  valid raw `GPM\0` payload
- wrong explicit dictionaries must fail before returning a `PreparedMol`

On the write path, do not allow `dictionary=...` without
`compression="zstd"`.

An explicit dictionary should be a validated dictionary object, not an arbitrary
bytes blob passed through every call. The object should be constructed from
dictionary bytes, cache the zstd dictionary ID, reject invalid dictionaries, and
be immutable/thread-safe. It should expose only the identity needed for
debugging and storage metadata, not mutable codec internals. If this object
becomes public, document that custom compressed datasets are not self-contained:
users must preserve the dictionary bytes or semantic dictionary identity
alongside the data.

## Built-in corpus v1

Use the existing checked-in molecule fixture as the source:

```text
tests/fixtures/top_100000_CIDs.tsv.gz
```

Current fixture hashes:

```text
gzip sha256:         67d1d31c3eb27da5ae5a8b8c1a3369531061113464c0dbfaf46e274493acc1ea
uncompressed sha256: 605cbbd10e69225ccfb47e05594aa01b338df7238f2ba3da76d7f5060c08f1bf
```

The training corpus must be selected deterministically:

- pinned RDKit version: `2026.03.1`
- parseable molecules only
- `PrepareMol(...)` must succeed
- no random sampling
- stable row order
- raw samples are `PreparedMol.to_bytes()` before compression
- source provenance and redistribution terms must be recorded in the manifest
- skipped row counts must be recorded separately for parse failures and
  `PrepareMol(...)` failures

First candidate selection rule:

```text
head 10,000 parseable rows
+ hash-stratified 10,000 parseable rows by CID
```

This covers common simple molecules and a stable spread through the checked-in
fixture without making dictionary training depend on every row.

`default_v1` is a compression dictionary, not a semantic compatibility layer.
It may be used to compress any raw `PreparedMol` payload, even if the payload
was prepared with different writer options; the writer options below describe
the training samples, not a runtime validity check.

Make the hash-stratified part exact:

1. Build the parseable/successful candidate list in source row order.
2. Take the first 10,000 candidates as the head sample.
3. From the remaining candidates, sort by `sha256(CID)`, where `CID` is the
   ASCII decimal CID text from the fixture, then by source row number to break
   impossible hash ties.
4. Take the first 10,000 not already selected.
5. Preserve source row order when writing training samples, so training input is
   stable and human-auditable.

Selection is by source row/CID. Do not add hidden deduplication. If a later
dictionary deduplicates by SMILES or raw `PreparedMol` bytes, that policy must
be named in the manifest because it changes the training distribution.

Default writer options for `default_v1`:

```text
isomericSmiles=True
kekuleSmiles=False
allBondsExplicit=False
allHsExplicit=False
ignoreAtomMapNumbers=False
```

## Manifest and hashes

The manifest is for provenance and tests, not required for per-molecule read.
It should be canonicalized for hashing as UTF-8 JSON with:

```text
sort_keys=True
separators=(",", ":")
```

Use two hashes to avoid circular identity:

1. `training_identity_sha256`: recipe identity. It includes source fixture
   hashes, RDKit version, writer options, exact selection rule, selected
   CID/source-row digests, raw sample digest, generator version,
   `python-zstandard` version/backend, zstd library version, dictionary size,
   dictionary ID derivation rule, and training parameters. It excludes
   generated artifact bytes.
2. `manifest_sha256`: artifact identity. It includes the training identity,
   dictionary SHA-256, zstd dictionary ID, and shipped file names. This is the
   hash used in the artifact directory name.

The manifest hash excludes only self-referential or non-identity fields:

- `artifact_dir`
- `manifest_sha256`
- generated timestamps and dates that should not define identity

Every other field in the manifest is identity-bearing by default. Adding a new
non-identity field should be rare and should explain why it cannot affect
artifact meaning.

It includes:

- dictionary semantic ID, e.g. `prepared-mol-default-v1`
- training identity SHA-256
- zstd dictionary ID
- zstd dictionary SHA-256
- source fixture hashes
- RDKit version
- `PreparedMol` raw format magic and version
- writer options
- selection rule
- selected CID digest and selected source-row digest; the fixture and
  deterministic generator rule are the auditable source of the full list
- parse failure count
- preparation failure count
- sample count
- raw sample digest
- zstd training parameters

The zstd dictionary ID should be deterministic, derived from
`training_identity_sha256`, and forced during training. It must be nonzero and
must not collide with any shipped built-in dictionary. Do not derive it from
`manifest_sha256`; that would be circular because the manifest includes the
dictionary ID.

Derive the integer exactly as the generator does: scan
`training_identity_sha256` in four-byte chunks, interpret each chunk as a
little-endian `u32`, clear the top bit, and choose the first value in
`32768..2147483647` that does not collide with an existing shipped dictionary.
Fail generation if no candidate satisfies those rules.

Training reproducibility is not enough by itself. The shipped dictionary bytes
are the artifact of record. Use a Python generator with pinned
`python-zstandard` for `default_v1`; its training API accepts an explicit
`dict_id`, and its dictionary object exposes the resulting ID and bytes. Tests
should validate the checked-in dictionary hash rather than assuming every zstd
implementation will train byte-identical dictionaries.

The generator must fail unless the Python package version, Python package
backend, and underlying zstd library version match the pinned recipe. For
`python-zstandard==0.25.0`, the pinned CPython wheel reports backend `cext` and
zstd `1.5.7`; source-built or downstream wheels with a different backend or
library version should not generate the production `default_v1` artifact.

Dictionary lifecycle matters because each shipped built-in dictionary becomes
read debt. Before adding `default_v2`, measure wheel-size impact and update the
compatibility tests so both `default_v1` and `default_v2` payloads are readable.
Removing an older built-in dictionary is a storage-format break.

## Package and resource boundary

The dictionary must be available to `PreparedMol.from_bytes(...)` without RDKit
and without relying on the current working directory.

There are two implementation shapes:

1. Embed shipped built-in dictionaries in the Rust extension with
   `include_bytes!`. This makes implicit lookup purely Rust-owned, but
   duplicates bytes if the raw artifact is also shipped as package data.
2. Load dictionary package data from Python using `importlib.resources`, then
   pass dictionary bytes into Rust. This avoids embedding duplicates, but the
   Python wrapper becomes responsible for built-in resource lookup.

Do not decide this accidentally during implementation. Pick one boundary before
adding the public API. In either case, Rust should own frame parsing,
decompression, and raw `GPM\0` parsing.

If the Python-resource path is chosen, Python may perform only resource lookup:
for example, ask Rust for a validated frame dictionary ID, load the matching
resource bytes, then call back into Rust to decompress and parse. Rust must
still revalidate the frame with the selected dictionary. Python should not grow
its own partial zstd parser.

The zstd dependency is part of the binary compatibility story. Prefer the Rust
crate path that vendors or pins the zstd C implementation through `Cargo.lock`
for release wheels, rather than dynamically depending on whatever system zstd
is present at build time. If a system zstd path is ever allowed for downstream
builds, CI should still validate the released wheel path separately.

The release build should make the chosen zstd implementation visible in the
artifact manifest or build metadata. Otherwise size/timing changes caused by a
codec upgrade will be hard to diagnose.

Update third-party notices for the Rust zstd crate and bundled zstd C library
before shipping compressed support.

The wheel and sdist validation allowlists must be updated so the dictionary and
manifest are intentionally included and no generated training logs or raw sample
files are shipped by accident.

The checked-in source fixture can remain a test/training input; it does not
become runtime package data. Regenerating the dictionary requires a repo
checkout, not an installed wheel.

Built-in decompression should prepare and cache zstd decode dictionaries once,
then share immutable handles across calls. The cache must be thread-safe and
must not depend on Python object lifetime when parsing is owned by Rust.

Do not load or prepare the built-in dictionary at `import grimace` time. Load it
on first compressed write/read, then cache it. Compression and decompression
should release the GIL around CPU work if the selected PyO3/Rust API allows it.

This is a per-record random-access storage feature, not a complete dataset
container. For 100M-molecule stores, benchmark per-record zstd frames against
outer bulk compression and indexed container designs before claiming an
end-to-end storage recommendation.

## Generator boundary

Dictionary generation is an offline build/development step. For `default_v1`,
write it in Python using `python-zstandard`. Do not require the runtime to
train dictionaries, and do not make runtime reads depend on the generator's
implementation language.

The generator must:

- derive the deterministic dictionary ID from `training_identity_sha256`
- create a zstd dictionary that carries that dictionary ID
- write the dictionary bytes and canonical manifest
- fail if the selected generation library cannot force or verify the dictionary
  ID
- record the generator dependency/version and underlying zstd version

Rust generation remains viable later through lower-level zstd bindings, but the
safe high-level Rust training wrapper does not expose explicit dictionary ID
control. That makes Python the smaller first implementation while still keeping
runtime reading Rust-owned.

The Rust runtime must read and validate the shipped dictionary regardless of
how it was generated.

## Security and malformed inputs

Compressed reads add new failure modes that raw `GPM\0` parsing does not have:

- zstd frame with valid magic but no dictionary ID
- zstd frame with an unknown dictionary ID
- zstd frame with matching dictionary ID but corrupted payload
- zstd frame with missing content size
- zstd frame with huge declared content size
- zstd frame that decompresses beyond the configured output cap
- concatenated zstd frames
- zstd frame followed by trailing bytes
- zstd skippable frame before or after the data frame
- decompressed bytes that do not start with `GPM\0`
- decompressed bytes that start with `GPM\0` but fail raw parser validation

All of these should fail as `ValueError`/`PyValueError`-style malformed payload
errors, not as panics or partial reads.

Unknown non-raw, non-zstd prefixes should fail as malformed `PreparedMol`
payloads without invoking the zstd decoder.

If the chosen Rust zstd API cannot write frames with dictionary IDs, content
size, and checksum, and inspect dictionary ID, content size, checksum flag,
single-frame consumption, and trailing bytes on read, do not paper over that in
Python. Use lower-level zstd bindings or revisit a Grimace envelope.

Single-frame validation should be structural, not inferred from successful
decompression. The reader needs to determine the compressed size of the first
frame and require it to equal the input length before returning a molecule.

The content checksum is an accidental-corruption guard, not a cryptographic
authenticator. Treat dictionary and manifest SHA-256 hashes as artifact
integrity checks; do not imply that an arbitrary compressed payload is trusted
because its zstd checksum passes.

Dictionary compatibility and raw-format compatibility are separate. A payload
can reference a known dictionary and still decompress to a future raw
`PreparedMol` version that this release cannot parse. That should fail as an
unsupported raw payload after decompression, not as an unknown dictionary.

## Tests

Add tests before exposing compression as public API:

- raw `to_bytes()` remains byte-compatible and starts with `GPM\0`
- compressed `to_bytes(compression="zstd")` is a zstd frame, not a Grimace
  envelope
- compressed frames include a nonzero zstd dictionary ID
- compressed frames include content size and checksum
- compressed payloads are a single zstd frame with no trailing data
- built-in `from_bytes(compressed)` loads the matching built-in dictionary
  implicitly
- unknown, zero, or missing dictionary ID fails without an explicit dictionary
- missing content size fails
- missing checksum fails
- corrupted checksum fails
- oversized declared content size fails before decompression allocation
- decompression beyond the output cap fails
- concatenated frames and trailing compressed bytes fail
- zstd skippable frames fail
- explicit dictionary round-trips
- explicit dictionary object rejects invalid dictionary bytes
- explicit dictionary ID mismatch fails
- wrong explicit dictionary fails
- decompressed bytes must start with `GPM\0`
- decompressed `GPM\0` bytes must be consumed exactly by the raw parser
- compressed payloads with known dictionaries but unsupported raw payload
  versions fail at the raw parser layer
- `to_bytes(compression="zstd")` does not silently return raw bytes even when
  compression expands the payload
- `to_bytes(dictionary=...)` without zstd compression fails
- unknown compression names fail
- `from_bytes(raw_payload, dictionary=...)` fails instead of ignoring the
  dictionary
- non-`bytes` payloads remain rejected unless the public contract is
  deliberately expanded
- compression and decompression do not force dictionary loading at import time
- compressed read/write can run concurrently without Python-level mutable state
  races
- manifest hash recomputes
- training identity hash recomputes
- artifact directory hash prefix matches the manifest hash
- dictionary SHA-256 matches the manifest
- dictionary ID in the file matches the manifest
- source fixture hashes match the manifest
- skipped parse/preparation counts match the manifest
- no two shipped dictionaries share the same zstd dictionary ID
- old built-in dictionary artifacts remain readable when a newer default is
  added
- release/sdist validation proves only the dictionary and manifest are shipped,
  not generator scratch data or raw training samples
- concurrent compressed reads share cached dictionaries safely

## Implementation order

1. Add a Python `python-zstandard` generator that writes candidate artifacts to
   `~/tmp`.
2. Train head-only and head-plus-hash-stratified dictionaries.
3. Measure raw size, zstd without dictionary, zstd with dictionary, write time,
   read time, parse time, first-use dictionary load cost, and steady-state
   cached-read throughput. Include tiny molecules, typical molecules, larger
   molecules, disconnected molecules, and stereo-heavy examples.
4. Confirm the selected Rust zstd API can enforce the frame rules above.
5. Decide whether compressed readability is unconditional in release wheels or
   feature-gated for nonstandard builds only.
6. Decide the explicit dictionary object shape.
7. Decide the built-in resource boundary: Rust `include_bytes!` versus Python
   package resource loading.
8. Commit the artifact only after the manifest and validation tests are stable.
9. Add Rust-owned frame parsing and bounded decompression.
10. Add the Python API surface.
