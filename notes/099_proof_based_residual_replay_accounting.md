# Proof-based residual replay accounting

## Protected fact

An offline obligation is checked only when its exact semantic replay completed.
Serialized operation names, capability labels, lifecycle links, and transition
term presence describe a claimed proof; none of them is the proof itself.

## Ownership split

- Each residual-operation replay function owns the semantic decision and returns
  an explicit replay disposition.
- The branch classifier records evidence digests only for successful semantic
  replay.
- Lifecycle classification may credit a lifecycle manifest only when its linked
  residual digests and operation list exactly name replayed residual manifests.
- The producer emits a singular ring-opening term only for the singular v7
  surface it can represent honestly.

## Deliberate boundary

Single-site, single-bond, ordinary-policy ring openings and pairs are replayed.
Shared and non-single ring surfaces retain evidence and lifecycle provenance but
remain typed offline-incomplete. This slice does not generalize their chemistry.

## Alternative rejected

Annotating serialized manifests with a new `checked` field would make producer
claims authoritative and require a schema bump. Deriving credit inside the
verifier from successful replay keeps schema v7 and preserves producer-free
verification.

## Potential regret

The classifier carries small verifier-local digest sets. This is intentional:
it makes proof credit explicit without changing the artifact or coupling replay
functions to mutable manifest state.
