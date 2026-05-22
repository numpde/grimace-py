# South Star EnumS Export-Readiness Audit

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 242: Audit EnumS export readiness`

## Purpose

Revisit whether the provisional South Star semantic enumerator can move from
private `grimace._south_star` implementation toward public `MolToSmilesEnumS`
after the recent boundary work:

- compact derived-support fixtures and diagnostics;
- aromatic directional-surface boundary probe;
- query and dative fail-fast boundary hardening.

## Checked State

Current checked readiness snapshot:

- public promotion gates listed in code: `13`;
- exact first-domain cases: `6`;
- expanded-support cases: `74`;
- derived-support cases: `1`;
- unified-reference-backed readiness cases: `80`;
- shared-pipeline promotion candidates: `80`;
- public API blocker case ids: none;
- supported feature areas: `35`;
- support-gate blocker categories: `16`.

The private boundary remains:

`grimace._south_star.api.mol_to_smiles_enum_s_private(mol, *, policy_set=...)`

The inspectable contract still says:

- provisional name: `MolToSmilesEnumS`;
- accepted input: `rdkit.Chem.Mol`;
- exported from public package: `False`;
- RDKit parser is parse-back evidence, not the definition of support;
- RDKit writer parity surface remains `MolToSmilesEnum`.

## Readiness Evidence That Is Strong Now

The current branch has materially stronger pre-export evidence than the early
South Star seed:

- support and unsupported categories are manifested;
- graph/stereo semantic parse-back checks are separate from grammar membership;
- fixed-molecule query/dative/aromatic-directional boundaries are explicit;
- derived large-product fixtures avoid dumping enormous expected sets while
  preserving digest/runtime equality and full diagnostic parse-back;
- package-readiness and South Star semantic runners pass;
- public writer-parity API remains separate from South Star semantics.

This is enough to keep moving toward export deliberately.

## Why Not Export Immediately

Do not export `MolToSmilesEnumS` yet.

The blocker is not a failing fixture. The blocker is that the public contract is
not yet sufficiently specified and reviewed:

1. The public API shape is still implicit.

   The private function accepts an RDKit `Mol` and returns a prototype result
   with diagnostics. A public function needs a deliberate choice about whether
   it returns only strings, exposes diagnostics, accepts policies, uses
   RDKit-like flags, accepts SMILES strings, or remains `Mol`-only.

2. The docs still contain stale gap language.

   `docs/enum-s.md` still says broader polycyclic ring traversal,
   ring/tetrahedral interaction modeling, and several syntax areas are future
   gaps, even though representative versions of those areas are now covered.
   That section needs to be rewritten as current scope plus remaining boundary
   decisions before it can support a public export.

3. Explicit-review gates are not yet backed by artifacts.

   The package-readiness gate intentionally includes documentation,
   performance-evidence, release-note, and exported-surface review items. Those
   review items should be converted into concrete checked docs/tests before
   public export.

4. Performance evidence remains an engineering diagnostic.

   The branch has complexity guardrails and a benchmark manifest, but no
   release-grade performance artifact for a public semantic API. Export does
   not require speed claims, but docs and release notes must avoid implying
   performance guarantees.

## Serious Alternatives

### Alternative 1: Export Now With A Narrow Experimental Label

Pros:

- current automated readiness has no blocker case ids;
- the private API is already usable internally;
- early users could evaluate the semantic enumerator.

Cons:

- public return type and diagnostics contract would harden accidentally;
- stale docs would make the supported surface ambiguous;
- any later policy modularization could become a breaking public change.

Recommendation: do not choose.

### Alternative 2: Keep Private And Polish The Public Contract First

Pros:

- avoids accidental API regret;
- lets docs, tests, and release notes converge before export;
- keeps the South Star/RDKit parity distinction clean.

Cons:

- delays external use;
- may feel conservative despite strong internal coverage.

Recommendation: choose this.

### Alternative 3: Export A Separate Diagnostic API First

Pros:

- exposes the useful result object and diagnostics honestly;
- avoids pretending the public surface is just a drop-in string iterator.

Cons:

- likely too broad as the first public surface;
- diagnostics are still private engineering guardrails and may churn.

Recommendation: defer. Decide after the narrow string-enumeration contract is
specified.

### Alternative 4: Export Only Through `docs/enum-s.md` Examples

Pros:

- no package API commitment;
- lets reviewers run private examples.

Cons:

- encourages users to import private modules;
- creates an implicit unsupported API.

Recommendation: do not choose.

## Recommended Queue

1. Rewrite the EnumS package-readiness gap section.

   Replace stale future-gap language with current scope, remaining unsupported
   boundaries, and the exact reason export is still private.

2. Specify the public `MolToSmilesEnumS` surface.

   Write tests/contract text for input type, return type, ordering promise,
   diagnostics exposure, policy flags, unsupported error behavior, and naming.
   This should be a design slice before code export.

3. Convert explicit-review promotion gates into concrete checks where possible.

   Keep genuinely human review items explicit, but add checked evidence for
   docs/release/performance wording where a test can enforce the contract.

4. Re-run package-readiness after those slices and decide whether export is a
   code change or still premature.

