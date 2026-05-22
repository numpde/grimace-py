# Docs

Documentation is intentionally small and split by audience:

- [api/python.md](api/python.md): supported public Python API, semantics, and
  current limits
- [correctness-contracts.md](correctness-contracts.md): separation between
  principled SMILES/chemistry semantics and RDKit writer-parity behavior
- [enum-s.md](enum-s.md): provisional South Star semantic enumeration contract
  and package-readiness boundary
- [release-note-checklists/enum-s-export.md](release-note-checklists/enum-s-export.md):
  required release-note topics before exporting the provisional EnumS surface
- [testing-fixtures.md](testing-fixtures.md): fixture taxonomy and RDKit
  parity evidence boundaries
- [rdkit-serializer-coverage.md](rdkit-serializer-coverage.md): reviewed
  upstream RDKit serializer coverage status, known-gap policy, and maintenance
  workflow
- [timings.tsv](timings.tsv): raw benchmark summary data
- [timings.md](timings.md): generated benchmark table used by the README
- `architecture/`: internal design notes for runtime ownership and boundaries

Anything under `grimace._runtime`, `grimace._reference`, or `grimace._core`
remains internal unless documented explicitly in `api/python.md`.
