# Docs

Documentation is intentionally small and split by audience:

- [api/python.md](api/python.md): supported public Python API, semantics, and
  current limits
- [timings.md](timings.md): generated benchmark table used by the README
- `architecture/`: internal design notes for runtime ownership and boundaries

Anything under `grimace._runtime`, `grimace._reference`, or `grimace._core`
remains internal unless documented explicitly in `api/python.md`.
