# Ordinary aromatic writer support

This slice makes neutral, unbracketed aromatic C, N, O, and S part of the
ordinary writer contract without adding an aromatic-specific proof object.
Facts, the serialized finite policy, and exact writer-state evolution remain
the authorities.

## Endpoint-aware single bonds

A fact-level aromatic bond joins two aromatic atoms and uses the configured
aromatic domain: elided `""`, explicit `":"`, or both. It never permits a
direction mark. A fact-level single bond whose two endpoints are aromatic has
different parser semantics: omission would denote an aromatic bond, so the
only legal spelling is explicit `"-"`, without direction. This package admits
that surface only when the bond is a graph bridge and therefore necessarily a
tree edge. A single bond with at most one aromatic endpoint retains the
ordinary single-bond policy, so toluene remains elided while biphenyl is
explicit.

## Atom and policy authority

The facts-only atom relation maps aromatic C/N/O/S to `c`/`n`/`o`/`s` and the
ordinary neutral organic subset to its existing token. The producer and the
producer-free verifier use that finite relation; neither derives aromaticity
from rendered text.

Prepared identity serializes the exact atom, tree-bond, and ring-endpoint
domains. Each branch replay verifies domain membership, endpoint-aware parser
semantics, direction legality, and the complete policy-state equation. Starting
from the source `WriterPolicyState`, atom, tree-bond, and ring-endpoint events
are applied in order; the resulting atom and bond histories must equal the
successor state exactly. Unrelated entries cannot change.

## Field to independent authority

| Field | Independent authority |
| --- | --- |
| Aromatic atom token | atom symbol and `is_aromatic` facts |
| Atom choice | serialized atom-text domain |
| Bond order | `BondFacts.order` |
| Endpoint aromaticity | endpoint `AtomFacts` |
| Tree-bond text | serialized tree domain and endpoint-aware semantics |
| Ring endpoint text | serialized ring domain and pair semantics |
| Direction mark | selected policy choice and fact bond order |
| Atom policy history | source state plus atom-emission events |
| Bond policy history | source state plus tree/ring-emission events |
| Ring lifecycle | exact open and closed closure records |
| Support and counts | continuation recurrence |
| Local semantics | branch and terminal facts verifiers |
| RDKit parity | version-pinned aromatic audit fixture |

## Preserved boundaries

Bracketed aromatic atoms such as `[nH]`, charged, isotopic, or mapped aromatic
surfaces, aromatic B and P, Kekule output, aromatic stereo, custom aromatic
domains, and aromatic/aromatic single bonds that can become ring closures are
not promoted. The last boundary is rejected during ordinary-policy preparation:
without a bridge proof the bond may occupy a ring-endpoint slot, whose omission
and pairing semantics differ from an explicit single tree bridge.
