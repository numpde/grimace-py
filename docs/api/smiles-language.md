# Principled SMILES language target

This is the target for a semantic prefix language over an admitted molecular
fact set, using [OpenSMILES v1.0 (2016-05-15)](https://opensmiles.org/opensmiles.html)
as its syntax and interpretation reference. It is a target contract, not a
claim that the current `GENERAL` implementation is complete. The separately
named RDKit language retains exact, version-qualified writer support as its
own contract.

## Input and identity

Preparation must either produce explicit molecular facts or reject an input
whose facts cannot be represented or preserved. Its admission boundary must
identify supported elements and isotopes, graph atoms, formal charge,
hydrogen counts and graph hydrogen atoms, atom class, bond orders,
connectivity, aromaticity, and configured or unspecified stereo. An unsupported
fact is an error; dropping it is not admission. Atom identifiers label the
graph-isomorphism witness; their values are not serialized identity. Source
atom-map property presence is a spelling constraint in `COMPACT`, distinct
from the generic chemical identity test.

An output represents the same admitted molecule when its graph atoms can be
matched bijectively, preserving element, isotope, formal charge, numeric atom
class, connectivity, bond order or equivalent aromatic bonding, and hydrogen
inventory. Graph hydrogen vertices remain graph atoms; non-graph hydrogen
counts remain counts. Converting either form requires an explicit, reviewed
admission and equivalence rule; this contract does not silently collapse them.
Configured tetrahedral
and double-bond stereo must describe the same assignment after traversal and
representation changes; unspecified stereo must remain unspecified. Aromatic
and Kekule spellings may be equivalent only when they reconstruct the same
admitted bonding and valence facts. If that equivalence cannot be established
for an input, it is not an excuse to accept extra strings. An omitted atom
class and explicit `:0` have the same numeric class value in the generic
semantic contract; `COMPACT` may separately preserve source map-property
presence, including zero. A parser's acceptance or RDKit object equality
alone does not establish this identity.

## Language and coverage

For each admitted input, the target language contains **every and only**
OpenSMILES v1.0 string whose standard interpretation preserves those facts.
It includes valid roots and traversal orders, branch placement, ring closure
events, and dot-separated serialization segments. A dot segment is not a
chemical connected component: `CC` and `C1.C1` can express the same graph;
so can `CCC` and `C1.C12.C2`. An open ring obligation can therefore cross a
dot, while a genuinely disconnected graph still requires disconnected output.

The coverage obligation includes standard atom and bond spellings: bare and
bracket atoms where valid, isotope and atom-class numeric spellings, admissible
hydrogen and charge spellings, aromatic and Kekule representations, explicit
and omitted bond symbols where interpretation agrees, and ring bond symbols
at either endpoint or consistently at both when allowed. The ring-label rule
is the OpenSMILES §3.4 syntax: one digit or `%` and exactly two digits, with
matching by numeric value and reuse after closure. Valid mixed lowercase aromatic and
uppercase spellings within one chemical component also belong when they
preserve the admitted facts. OpenSMILES §§3.1.4 and 3.1.7 define isotope and
atom-class fields using `NUMBER`, give zero-padded isotope examples, and
interpret atom classes numerically. The grammar sets minimum field widths but
no maximum digit count. This project interprets further leading zeros as the
same numeric value and includes arbitrarily padded forms in its target; the
standard does not expressly require that unbounded interpretation. Prefix
generation must reason incrementally about these forms; a bounded walk is
evidence about that walk, not a finite definition of the language.

Soundness requires every accepting path to be grammatical and fact-preserving.
Coverage requires a path for every qualifying string. The constructive proof
obligation is a mapping in both directions between standard serialization
events (atoms, bonds, branches, ring opens and closes, dots, and spelling
choices) and decoder transitions, with semantic constraints maintained through
the whole path. Bounded fixtures and parse-back checks test examples; they
cannot establish universal coverage, especially for unbounded lexical forms.

### Leading-dot branches and their semantic witness

The [OpenSMILES v1.0 grammar and §3.7.1](https://opensmiles.org/opensmiles.html)
allow `branch ::= '(' dot chain ')'`. Thus `C(.C)` is a standard spelling
for two disconnected carbon atoms, equivalent in graph identity to `C.C`.
RDKit 2026.03.4 was observed to reject `C(.C)` as SMILES input. That parser
result does not remove the spelling from the target language, and neither
parser acceptance nor graph equivalence establishes RDKit writer support.

The compact C/O fixture is checked by a temporary test-only interpreter in
[`tests/helpers/small_compact_graph.py`](../../tests/helpers/small_compact_graph.py).
It handles only C and O atoms, implicit single bonds, single-digit ring labels,
branches, and dots, including a leading dot inside a branch. It rejects other
tokens and malformed syntax. It reconstructs and sanitizes an RDKit graph
without asking RDKit to parse the emitted SMILES, then compares canonical
graph identity with the source. This is a scoped semantic witness, not a full
OpenSMILES validator, an exhaustive support oracle, or a proof that all
accepted prefixes remain live. The witness should fold into the shared
fact/event/constraint model; it is not a runtime fallback.

`COMPACT` is a restricted spelling policy applied to otherwise valid paths:
bare atoms where possible, unpadded numbers, omitted default fields, and the
smallest available ring label. It cannot repair missing traversal or dot-event
coverage. A future permissive lexical policy, whatever its public name, can
relax spelling only; it cannot stand in for the structural contract.

## Current production coverage

| Area | Current state | Remaining target obligation |
| --- | --- | --- |
| Atom and bond spellings | Implemented subset, including broad numeric atom spelling in `GENERAL` and compact spelling in `COMPACT` | Prove all admitted standard alternatives and fact-preserving equivalences |
| Ring labels | Labels `0`–`99` and compact smallest-free choice | Pair-consistent bond alternatives and complete ring-event coverage |
| Roots, branches, traversal | Connected traversal choices, terminal branches, and sibling branches within one residual component on admitted forest inputs, including supported aromatic graphs and tetrahedral witnesses | Map all standard serialization events both ways |
| Dot segments | Top-level and branch-local dots, including leading-dot branches, qualified for admitted stereo-free graphs in `GENERAL` and `COMPACT`; selected tetrahedral ring-segment witnesses also pass in both languages | Extend qualification to leading-dot stereo paths and other unqualified syntax |
| Aromaticity | Aromatic or whole-component Kekule choices for supported inputs | Valid mixed lowercase/uppercase spellings within a component |
| Stereo | Supported three-bond carbon with one hydrogen and four-bond carbon without hydrogen, plus selected E/Z cases, have semantic checks; selected tetrahedral Forest segments preserve chirality in both languages | Define and cover every admitted configuration and preserve unspecified stereo |

The native forest path now admits top-level and branch-local dot segments,
including leading-dot branches, with ring obligations crossing segment
boundaries on admitted stereo-free graphs, including supported aromatic graphs.
Selected tetrahedral Forest paths also admit ring-separated ligands, including
`F1.[C@@H]1(Cl)Br` in `GENERAL` and `F0.[C@@H]0(Cl)Br` in `COMPACT` for
`F[C@H](Cl)Br`. These are semantic parse-back witnesses, not RDKit writer
support claims. Directional E/Z Forest paths and leading-dot stereo paths
remain unqualified.
The aromatic source spelling and whole-component Kekule alternatives retain
one matching constraint across segments; mixed lowercase and uppercase
spellings within an aromatic component remain unqualified. The path also admits
terminal branches such as `C(C)`, `C(C(C))`, and `C(C)(C)` for those inputs in
`GENERAL` and `COMPACT`. It also admits branched ring paths such as
`C(CC1)CCC1` for cyclohexane. These examples establish specific supported
paths, not completeness of the Forest grammar or liveness of every prefix.
Public semantic witnesses include `C1.C1` for `CC` and `C1.C12.C2` for `CCC`;
`COMPACT` also admits corresponding smallest-label spellings. This is a
qualified subset of the target grammar: other stereo Forest paths, mixed aromatic spellings
within a component, and other
unqualified syntax remain open. A fixed-budget diagnostic may report that its
enumeration is incomplete; bounded results do not establish global exhaustive
support or prefix liveness. The current `GENERAL` name is provisional; naming
can follow a demonstrated contract.
