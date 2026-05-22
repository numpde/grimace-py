# South Star Post-Compositional Frontier

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 224: Reassess post-compositional semantic frontiers`

## Purpose

Reassess the South Star frontier after admitting the first mixed
ring/tetrahedral compositional fixture.

The goal is not to pick the smallest unsupported string. The goal is to choose
the next slice that most strengthens the principled, fixed-molecule,
one-truth reference model without importing RDKit writer parity or a separate
query/coordination product model accidentally.

## Current State

After `South Star 223`:

- the readiness matrix has 78 unified-reference-backed cases;
- expanded fixtures include two `compositional_stereo_independent_product`
  cases and two `compositional_stereo_coupled_component` cases;
- the small adversarial corpus still has 25 candidates, 23 supported, and 2
  unsupported;
- the two unsupported adversarial witnesses remain `C~C` and `N->[O]`.

The new ordinary fixed-molecule evidence is:

| Case | Source | Authority | Outputs |
| --- | --- | --- | ---: |
| `compositional_stereo_ring_tetra_plus_branch_tetra` | `F[C@H]1CCCC([C@H](Cl)Br)C1` | `unified_reference_compositional_stereo_coupled_component` | 576 |

This means the branch now has representative proof-backed coverage for:

- independent connected tetrahedral products;
- adjacent connected tetrahedral coupled components;
- disconnected tetrahedral products;
- mixed ring/tetrahedral coupled components.

## Serious Alternatives

### Alternative 1: Pin The Larger Polycyclic Ring/Tetra Branch Case

Representative witness: `F[C@H]1CC2CCC1C2[C@H](Cl)Br`.

Pros:

- fixed-molecule semantics;
- already supported;
- same `shared_ring_tetrahedral_system` proof shape as the smaller mixed
  ring/tetra fixture;
- would stress a 3160-output support set.

Cons:

- mostly scale evidence, not new proof-shape evidence;
- large fixture review cost;
- not the best next step if the component vocabulary still has known
  directional ambiguity.

Recommendation: defer until after the product/coupling vocabulary is tighter.

### Alternative 2: Reconcile Directional Product/Coupling Labels

Representative witnesses:

- `F/C=C/C/C=C/Cl`;
- `F/C=C/[C@H](Cl)Br`;
- `F[C@H]1CCCC(/C=C/Cl)C1`.

Pros:

- fixed-molecule semantics;
- directly protects the one-truth component partition model;
- blocks accidental overloading of "independent directional components";
- needed before using directional labels as general product/coupling authority
  for larger mixed stereo cases.

Cons:

- may require revising older feature labels or proof-helper coupling rules;
- less visually impressive than adding a large fixture.

Recommendation: choose this next. It is the highest-leverage cleanup before
larger directional/ring/stereo composition.

### Alternative 3: Aromatic Directional Overlays

Representative surface: aromatic bonds with directional markers or
directional-like input facts.

Pros:

- marker-placement relevant;
- natural follow-on after directional vocabulary cleanup.

Cons:

- still partly a policy question;
- aromatic support so far is mainly atom text and traversal support, not
  directional stereo semantics;
- a premature implementation risks treating parser artifacts as molecule
  semantics.

Recommendation: keep as a later policy/probe track.

### Alternative 4: Query Or Unspecified Bonds

Representative witness: `C~C`.

Pros:

- still one of the two tiny unsupported adversarial witnesses;
- useful fail-fast boundary.

Cons:

- not a fixed molecule;
- likely requires a query-product model, not just another bond token;
- would dilute the current South Star contract.

Recommendation: do not choose next.

### Alternative 5: Dative Or Coordination Bonds

Representative witness: `N->[O]`.

Pros:

- real chemistry-facing surface;
- still one of the tiny unsupported adversarial witnesses.

Cons:

- coordination semantics involve charge, valence, parser interpretation, and
  known RDKit writer quirks;
- easy to bolt on incorrectly as a text-token feature.

Recommendation: do not choose next.

## Recommendation

The next principled step is to reconcile directional product/coupling labels.

Reason: South Star is now past single-obligation fixtures. The main risk is no
longer missing a case; it is corrupting the component partition vocabulary.
Before adding larger mixed directional/ring/stereo fixtures, the project should
make clear whether directional obligations are independent, coupled, or
adjacent-but-factorable, and where that distinction is represented.

## Proposed Sequence

1. Reconcile existing directional fixture labels against the compositional
   proof classification.
2. Define the directional product/coupling proof standard.
3. Add or revise focused tests so directional product/coupling claims are
   explicit and not inferred from legacy feature-area names.
4. Only then decide whether to pin larger directional/ring/stereo fixtures,
   aromatic directional overlays, or the larger polycyclic mixed ring/tetra
   case.
