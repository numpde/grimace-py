# South Star Ring Directional/Tetrahedral Blockers

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 161d: Audit ring directional tetrahedral blockers`

## Finding

The current gate split is intentional enough to keep. It admits already-modeled
ring-local tetrahedral monocycle cases and ring-internal directional stereo
cases, but it still blocks directional stereo carried by an acyclic branch off a
ring.

Representative probes:

- `F[C@H]1CCCCC1`: supported as ring-local tetrahedral monocycle scope.
- `F[C@H]1C/C=C/CC1`: supported as a monocycle where the directional stereo is
  ring-local and the ring/tetrahedral obligation is already modeled.
- `C1CC(/C=C/Cl)CCC1`: blocked by `ring_molecule`.
- `F[C@H]1CCCC(/C=C/Cl)C1`: blocked by `ring_molecule` and
  `ring_tetrahedral_interaction`.

The narrower non-tetrahedral witness matters: the blocker is not only the
tetrahedral center. A directional branch attached to a ring is not yet covered
by the current ring traversal proof family. Admitting the tetrahedral witness
directly would silently combine three surfaces: ring traversal, exocyclic
directional marker equations, and ring-local tetrahedral ligand ordering.

## Decision

Keep these witnesses gated for now. Do not admit a special case until there is a
shared proof family for monocycles with directional acyclic branches.

The next principled slice is not "allow
`F[C@H]1CCCC(/C=C/Cl)C1`"; it is:

1. prove monocycle traversal with an exocyclic directional stereo branch;
2. then compose that proof with ring-local tetrahedral renderer obligations;
3. only then add a fixture for the full ring/tetrahedral/directional witness.

## Pinned Tests

`tests.south_star.test_support_gates` now pins:

- `C1CC(/C=C/Cl)CCC1` stays gated by `ring_molecule` and not `ring_stereo`;
- `F[C@H]1CCCC(/C=C/Cl)C1` stays gated by both `ring_molecule` and
  `ring_tetrahedral_interaction`.

These tests are boundary tests, not a claim that the witnesses are permanently
out of scope.

## Update After `South Star 163`

`South Star 163` admits the narrower exocyclic directional-branch monocycle
case, `C1CC(/C=C/Cl)CCC1`, after proving it through the existing monocycle
closure traversal plus branch-local marker equations.

The full `F[C@H]1CCCC(/C=C/Cl)C1` witness remains gated. Its remaining blocker
is the intended next composition step: ring-local tetrahedral renderer
obligations plus exocyclic directional marker equations on the same ring
traversal spine.

## Update After `South Star 164`

`South Star 164` admits the full `F[C@H]1CCCC(/C=C/Cl)C1` witness after the
exocyclic branch proof exists. The support gate now names the combined
ring-local tetrahedral plus exocyclic directional monocycle scope, and fixture
evidence checks that marker-equation and tetrahedral renderer-obligation proofs
agree on the same rendered support.
