# EnumS Export Release-Note Checklist

Use this checklist only when a release exports `MolToSmilesEnumS` from the
public `grimace` package. It is not itself a release note.

The release notes for that export must state:

- `MolToSmilesEnumS` has a semantic contract: it enumerates South Star semantic
  support under the documented default annotation policy.
- `MolToSmilesEnumS` is distinct from `MolToSmilesEnum` RDKit writer parity;
  users should not expect RDKit `canonical=False, doRandom=True` string-support
  equality from the semantic API.
- Unsupported boundaries still fail fast with `SouthStarUnsupportedFeatureError`
  and must not be described as partial support.
- Performance evidence boundary: performance evidence is an engineering
  guardrail unless a separate release-facing semantic-enumerator benchmark
  artifact exists.
- Avoid accidental speed claims: do not say or imply `MolToSmilesEnumS` is
  faster than RDKit without release-specific benchmark evidence for that
  exact semantic surface.
