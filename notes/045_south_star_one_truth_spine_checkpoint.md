# South Star One-Truth Spine Checkpoint

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 135: Run one-truth spine checkpoint`

## Scope

This checkpoint follows the spine-cleanup tranche:

- `South Star 131`: reference records have a narrow inert import boundary.
- `South Star 132`: the first-domain witness now uses shared carrier, slot,
  event, and fragment records.
- `South Star 133`: first-domain and expanded fixtures use the same
  `support_authority` vocabulary.
- `South Star 134`: every temporary witness authority must name a fold-in plan
  toward unified-reference backing.

This note does not widen support. It audits whether the Backlog is aligned
with the one-truth South Star model before more feature work.

## Current State

The direction is now mostly correct:

- shared record ownership is explicit in
  `python/grimace/_south_star/reference_model.py`;
- `reference_model.py` has no RDKit, `enum_s`, solver, support-gate, fixture,
  or renderer-vocabulary imports;
- shared traversal records remain inert; rendering is still outside the shared
  record module;
- first-domain witness search remains independent but no longer owns a local
  carrier/slot/event record language;
- fixture authority metadata is unified around `support_authority`;
- temporary witnesses are visibly temporary and have fold-in plans;
- the readiness matrix keeps unified-reference-backed, temporary-witness, and
  regression-backed cases separate.

This is enough to proceed, but not enough to call the South Star reference
model complete.

## Remaining Structural Blockers

1. `marker_equations.py` still imports
   `mol_to_smiles_enum_s_tree_traversals_for_case` from `enum_s.py`.

   That keeps a constraint module coupled to the prototype generator. The
   equation builder should consume traversal records, not import a traversal
   factory from the renderer/prototype module. This is not a behavior bug, but
   it is the wrong dependency direction for the one-truth model.

2. `SouthStarTreeTraversal` in `enum_s.py` is still a compatibility wrapper
   with `.render()`.

   The shared `SouthStarTraversal` record is inert, so the core rule holds, but
   the wrapper makes traversal objects look partly responsible for rendering.
   This should be retired once call sites can render through an explicit
   renderer function.

3. Expanded-domain witnesses are only partially folded into shared records.

   Ring, ring-stereo, disconnected-composition, and tetrahedral helpers consume
   more shared records than before, but the helper module still contains
   witness-specific result/obligation records. Some of those are legitimate
   constraint-family concepts; some are temporary witness scaffolding. The
   remaining distinction should be made explicit.

4. No fixture domain is yet `unified-reference-backed`.

   This is intentional. The metadata now prevents accidental promotion, but a
   later slice must define the criteria for moving a case from temporary witness
   or regression evidence to unified-reference authority.

5. Feature widening rows remain valid only after the structural rows.

   Radical atom text, broader syntax, adversarial corpora, and public API
   hardening should all consume the spine/authority model. They should not add
   new support authority names or helper-local record languages.

## Backlog Consequences

The next structural rows should be:

- remove the `marker_equations -> enum_s` traversal-factory dependency;
- retire `SouthStarTreeTraversal.render()` in favor of explicit rendering;
- classify and fold expanded-domain witness-local records;
- only then continue feature widening or package-export work.

The existing feature rows can remain, but their dependencies are real. If a
feature slice needs to happen before the structural rows, its Notes should say
why it strengthens the shared spine rather than bypassing it.

## Verification

The checkpoint is supported by the current guardrails:

- `tests.south_star.test_dependency_boundaries`;
- `tests.south_star.test_domain_manifest`;
- `tests.south_star.test_package_readiness`;
- `tests.run_south_star_package_readiness -q`;
- `tests.run_south_star_semantics -q`.

The tests verify boundaries and metadata. They do not prove full South Star
completeness.
