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

Keep the public argument surface narrow:

- `to_bytes()` remains raw and argument-free in the common case
- compression options are keyword-only
- supported `compression` values are exactly `None`/omitted and `"zstd"`
- unknown compression names fail before doing work
- `dictionary_level` selects one shipped built-in dictionary by training level
- `level` selects the ordinary zstd compression level
- custom dictionary objects are not part of the v1 public API

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
- decompressed size at or below the 1 MiB raw `PreparedMol` limit

The reader should reject missing content size or missing checksum for v1
compressed `PreparedMol` payloads. Even with content size present,
decompression must be bounded so a hostile frame cannot allocate unbounded
memory before the raw `GPM\0` parser sees it. The cap is part of the
implementation contract: payloads above 1 MiB are unsupported rather than
allocated speculatively.

If the frame has no dictionary ID, has dictionary ID zero, or references an
unknown dictionary, `from_bytes(payload)` must fail with a clear error. v1 only
reads dictionaries shipped with the package; custom dictionaries can be designed
later if a real storage workflow needs them.

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

`default_v1` candidate selection rule:

```text
all parseable/preparable fixture rows, in source row order
```

This uses the whole checked-in fixture as the stable training source. The
manifest records the successful sample count plus skipped parse/preparation
counts, so a fixture or RDKit behavior change changes the artifact identity.

`default_v1` is a compression dictionary, not a semantic compatibility layer.
It may be used to compress any raw `PreparedMol` payload, even if the payload
was prepared with different writer options; the writer options below describe
the training samples, not a runtime validity check.

Make the selection exact:

1. Read the fixture in source row order.
2. Keep every row whose SMILES RDKit parses and whose molecule
   `grimace.PrepareMol(...)` prepares successfully.
3. Preserve source row order when writing training samples, so training input
   is stable and human-auditable.

Selection is by source row. Do not add hidden deduplication. If a later
dictionary deduplicates by CID, SMILES, or raw `PreparedMol` bytes, that policy
must be named in the manifest because it changes the training distribution.

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

Do not rely on zstd's zero-valued training defaults for the production recipe.
Record the intended values explicitly. For `default_v1`, `k=0` means optimize
the segment size, while `d=8`, `f=20`, `split_point=0.75`, `accel=1`,
`level=3`, `steps=4`, and `threads=0` name the actual fast-cover recipe used
by the generator. In `python-zstandard`, `threads=0` is the documented
single-thread mode for dictionary training.

Dictionary lifecycle matters because each shipped built-in dictionary becomes
read debt. Before adding `default_v2`, measure wheel-size impact and update the
compatibility tests so both `default_v1` and `default_v2` payloads are readable.
Removing an older built-in dictionary is a storage-format break.

## Package and resource boundary

The dictionary must be available to `PreparedMol.from_bytes(...)` without RDKit
and without relying on the current working directory.

The v1 implementation loads dictionary package data from Python using
`importlib.resources`, then uses `python-zstandard` to compress/decompress
ordinary zstd frames. The Rust core continues to own the raw `GPM\0` bytes and
their parser.

Keep this boundary explicit. Python may understand enough of the zstd frame
header to select a shipped dictionary by frame dictionary ID. It must still
reject malformed compressed payloads before returning a `PreparedMol`, and the
decompressed raw bytes must go through the Rust parser.

The zstd dependency is part of the compatibility story. Release wheels that
expose compressed `PreparedMol` reads must include the Python zstd dependency.
If a future Rust implementation replaces the Python codec, CI should validate
the release wheel path separately.

The release build should make the chosen zstd implementation visible in the
artifact manifest or build metadata. Otherwise size/timing changes caused by a
codec upgrade will be hard to diagnose.

Update third-party notices for the zstd package/runtime before shipping
compressed support.

The wheel and sdist validation allowlists must be updated so the dictionary and
manifest are intentionally included and no generated training logs or raw sample
files are shipped by accident.

The checked-in source fixture can remain a test/training input; it does not
become runtime package data. Regenerating the dictionary requires a repo
checkout, not an installed wheel.

Built-in decompression should prepare and cache zstd decode dictionaries once,
then share immutable handles across calls. The cache must be thread-safe and
must not affect raw `GPM\0` parsing.

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
- run a post-flight check on the exact written artifact: recompute hashes,
  verify the dictionary ID from the dictionary bytes, and prove the dictionary
  can round-trip a raw PreparedMol payload through zstd
- fail if the selected generation library cannot force or verify the dictionary
  ID
- record the generator dependency/version and underlying zstd version

Rust generation remains viable later through lower-level zstd bindings, but the
safe high-level Rust training wrapper does not expose explicit dictionary ID
control. That makes Python the smaller first implementation while still keeping
the raw `PreparedMol` format Rust-owned.

The runtime reader must read and validate the shipped dictionary regardless of
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

If the chosen zstd API cannot write frames with dictionary IDs, content
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
- unknown, zero, or missing dictionary ID fails
- missing content size fails
- missing checksum fails
- corrupted checksum fails
- oversized declared content size fails before decompression allocation
- decompression beyond the output cap fails
- concatenated frames and trailing compressed bytes fail
- zstd skippable frames fail
- decompressed bytes must start with `GPM\0`
- decompressed `GPM\0` bytes must be consumed exactly by the raw parser
- compressed payloads with known dictionaries but unsupported raw payload
  versions fail at the raw parser layer
- `to_bytes(compression="zstd")` does not silently return raw bytes even when
  compression expands the payload
- unexpected keyword arguments fail instead of being ignored
- unknown compression names fail
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

1. Generate shipped dictionaries from the pinned all-molecule fixture lane.
2. Validate each artifact before accepting it: manifest hashes, dictionary ID,
   source hashes, skipped counts, and a zstd round trip.
3. Keep compressed readability unconditional in release wheels.
4. Load built-in dictionaries as Python package resources and cache them by
   zstd dictionary ID.
5. Keep v1 custom dictionaries out of the public API.
6. Keep raw `GPM\0` parsing Rust-owned; handle zstd wrapping in Python until a
   Rust implementation is clearly needed.
7. Measure raw size, zstd without dictionary, zstd with dictionary, write time,
   read time, parse time, first-use dictionary load cost, and steady-state
   cached-read throughput.
