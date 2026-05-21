# South Star Post-Traversal Constraint Sequence

The connected-graph traversal spine is now good enough to stop choosing the next
work item by fixture pressure. The next sequence should widen only when a domain
can be expressed through the one-truth path:

1. molecule facts;
2. traversal events;
3. explicit constraints or obligations;
4. solver or deterministic local decision;
5. renderer;
6. grammar and semantic parse-back evidence;
7. package-readiness promotion diagnostics.

Temporary witnesses may remain as cross-checks, but they must not define
package-ready support.

## Sequencing Criteria

A family should move earlier if it removes duplicated witness logic, strengthens
the shared model, or blocks several later domains. A family should move later if
it needs a new semantic language, mixes unrelated concerns, or would only add
case-by-case support.

Every implementation slice should either promote evidence through the shared
pipeline or keep the unsupported boundary sharper. If neither is true, the slice
is probably fixture-driven and should be deferred.

## Proposed Order

1. **Retire nonstereo ring temporary witnesses.**
   Completed in `South Star 97`: nonstereo monocycle and saturated-monocycle
   fixture checks now use shared connected-graph traversal, ring-label policy,
   and renderer records rather than local traversal/rendering helpers.

2. **Lift tetrahedral atom stereo into traversal obligations.**
   Completed in `South Star 98`: acyclic tetrahedral fixture checks now use
   shared traversal output plus explicit ligand-order/token obligations instead
   of a local tetrahedral string renderer.

3. **Model disconnected composition as unified reference composition.**
   Runtime already composes per-fragment supports under an explicit
   all-fragment-orders policy. The remaining work is to remove witness-only
   composition evidence, pin fragment provenance, and make the readiness matrix
   distinguish fragment support from fragment-order policy.

4. **Extend atom-text facts before broad atom modifiers.**
   Explicit bracket hydrogens have regression evidence, but charge, isotope,
   atom map, radicals, and broader bracket rendering remain unsupported. These
   should be handled as atom-text facts and renderer obligations, not by adding
   isolated string cases.

5. **Decide aromatic semantics before aromatic support.**
   Aromatic rings need a separate semantic model: lowercase aromatic grammar,
   kekule/aromatic parse-back equivalence, and what "maximal annotation" means
   on aromatic surfaces. This should stay fail-fast until that model is named.

6. **Handle ring/tetrahedral interactions after tetrahedral obligations.**
   Ring-local tetrahedral ligand ordering should not be patched into ring
   traversal. It should reuse the tetrahedral obligation model once that model
   is traversal-native.

7. **Handle stereo on polycyclic ring systems after ring and tetrahedral
   obligations are unified.**
   Nonstereo polycyclic skeleton traversal is a graph problem. Polycyclic stereo
   adds coupled marker obligations across multiple closure choices and should
   wait until the simpler stereo-obligation families are unified.

## Follow-Up Backlog Shape

The next Backlog entries should be numbered after the current 96 row:

- `South Star 97: Retire nonstereo ring temporary witnesses`
- `South Star 98: Lift tetrahedral atom stereo into traversal obligations`
- `South Star 99: Promote disconnected composition evidence`
- `South Star 100: Define atom-text modifier obligation model`
- `South Star 101: Decide aromatic semantic boundary`
- `South Star 102: Model ring/tetrahedral interaction obligations`
- `South Star 103: Plan polycyclic stereo obligations`

Each row should require either a committed code/test slice or an explicit
Decision row. None should widen public API surface while any promoted behavior
is still temporary-witness-backed.
