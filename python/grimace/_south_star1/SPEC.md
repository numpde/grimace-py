# South Star 1 Ordinary Bounded Dialect v0

This package is a confined proof kernel for a small, explicit SMILES support
model. It is not the public Grimace runtime and it is not an RDKit writer model.

## Definition Boundary

The v0 support definition is:

```text
MoleculeFacts
  + ordinary_policy_for_facts(...)
  + OrdinarySmilesSemantics()
  -> enumerate_stereo_support(...)
```

RDKit may appear only at ingestion and audit boundaries. It is not part of
enumeration, candidate filtering, support definition, or witness repair.

## Input

The core input is a fixed finite `MoleculeFacts` graph:

- atom and bond facts are already perceived and fixed;
- components are fixed;
- stereo sites and ligand occurrences are explicit facts;
- policy choices are finite;
- ring-label domains are finite and least-free normalized.

RDKit ingestion is available only through `ordinary_molecule_facts_from_rdkit`.
For tetrahedral stereo, that adapter currently assumes RDKit atom ids preserve
the lexical order produced by `Chem.MolFromSmiles`. Arbitrarily renumbered RDKit
`Mol` objects are not supported stereo-ingestion provenance in v0.

## Supported v0 Surface

The current ordinary bounded dialect supports:

- plain neutral ordinary atoms accepted by `ordinary_policy_for_facts`;
- single, double, triple, and aromatic tree-bond spellings currently modeled by
  the ordinary policy;
- single and aromatic ring closures with bounded normalized labels;
- optional joint non-single ring closures where exactly one endpoint carries
  the double or triple bond-order marker;
- tetrahedral `@` / `@@` sites represented as explicit local relations;
- ordinary directional double-bond stereo represented as finite carrier-scope
  relations;
- disconnected inputs under the declared component traversal policy;
- RDKit audit as external falsification by parse-back and fact isomorphism.

## Explicitly Excluded v0 Surface

The current dialect rejects or omits:

- enhanced stereo groups;
- non-tetrahedral atom stereo;
- atropisomerism;
- `STEREOANY` and unknown/unspecified RDKit bond stereo classes;
- arbitrary-renumbered RDKit `Mol` stereo ingestion;
- non-single ring closures under the default policy;
- recursive ligand-equivalence refinement beyond the current immediate ligand
  color checks;
- RDKit writer-parity quirks unless they are explicitly modeled at the adapter
  or audit boundary.

## Conformance Corpus

The executable v0 conformance corpus lives in `tests/south_star1`:

- `tests.run_south_star_semantics` is the canonical unittest runner for the
  South Star 1 proof-kernel suite;
- `test_audit_rdkit.py` contains the RDKit external audit matrix and support
  stability checks;
- `test_boundary.py` enforces that only `rdkit_adapter.py` and
  `audit_rdkit.py` import RDKit;
- semantic tests exercise the RDKit-free policy, site-builder, CSP, renderer,
  support image, and fact-isomorphism layers.

The audit matrix is a falsifier for this spec, not the source of truth. A
passing RDKit audit means generated strings parse back to isomorphic facts under
the declared adapter, not that RDKit defines the support.

## Experimental Options

`OrdinaryStereoSiteOptions(ligand_equivalence="exact_graph_automorphism")`
broadens potential stereo-site recognition beyond the v0 immediate-color rule.
It uses an anchored label-preserving graph automorphism relation to decide
whether two ligand occurrences are equivalent.

This option is opt-in and is not part of the v0 conformance corpus. The v0
default remains `ligand_equivalence="immediate_color"`.

The graph exact mode is atom/bond graph exact. It does not make pre-existing
stereo labels part of the automorphism relation.

`OrdinaryStereoSiteOptions(
ligand_equivalence="exact_stereochemical_graph_automorphism"
)` is the stronger experimental relation. It uses the same atom/bond graph
automorphism search, then requires all non-ignored specified stereo facts to be
compatible under the shared `stereo_mapping.py` parity/reference-pair relation.
The candidate site's own existing site id is ignored during site construction
to avoid circularly using a site's stereo label as evidence for its own ligand
distinctness.

RDKit ingestion remains one-way and non-definitional for these experimental
options. The default one-pass adapter builds ordinary potential sites before it
overlays RDKit specified stereo, so a remote-stereo-dependent candidate that
requires `exact_stereochemical_graph_automorphism` is not discoverable in that
mode. `RdkitOrdinaryExtractionOptions(stereo_site_discovery_passes=2)` enables
an explicit operational second pass: overlay any discoverable specified stereo,
rebuild missing potential sites with those labels available, then require the
final overlay to resolve all supported RDKit stereo.

`RdkitOrdinaryExtractionOptions(stereo_site_discovery_mode="specified_closure")`
is the mathematical experimental mode for raw specified RDKit stereo. The
adapter extracts raw specified stereo records first, materializes them as a
temporary specified context, checks every raw site for ordinary eligibility with
its own site id ignored, and then rebuilds the full potential-site universe
under the accepted specified context.

### Specified Stereo Closure Certificate

The `specified_closure` mode accepts a raw specified stereo graph iff each raw
record is eligible under the context of all other raw specified records, with
its own candidate site ignored. The implementation exposes a certificate per
raw record recording the matched site, context records, ignored self-site ids,
and accept/reject reason.

### Known Specified-Closure RDKit Audit Boundary

The specified-closure audit has a known RDKit boundary mismatch for generated
strings whose unsanitized parse retains a raw tetrahedral tag that sanitized
`Chem.MolFromSmiles(...)` removes during RDKit cleanup. The diagnostic trace
classifies this as `SPECIFIED_TETRA_RECORD_LOSS`: South Star's source facts
and the unsanitized RDKit parse contain the raw tetrahedral record, while the
sanitized RDKit `Mol` used by ordinary adapter ingestion no longer exposes it.

This is not hidden by the conformance suite. The representative failure is
pinned by `trace_specified_closure_round_trip(...)`; the full specified-closure
RDKit audit remains a skipped diagnostic until the project chooses whether the
fix belongs in a SMILES-source raw-stereo ingestion path, a stricter closure
policy, or a different external audit contract.
