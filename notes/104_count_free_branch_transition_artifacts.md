# Count-free branch transition artifacts

Global support materialization and local transition proof have different owners.
The support artifact proves a complete continuation language; the branch
transition artifact proves one checked writer action and therefore contains no
support image, support string, count envelope, count DAG, or completion proof.

The existing branch-support payload remains the single serialization owned by
the support-artifact builder.  The new artifact reuses that builder and adds a
count-free projection identity containing only the selected live branch.

| Serialized field | Independent authority |
| --- | --- |
| Prepared identity | prepared molecule and runtime options |
| Source cursor | source snapshot |
| Source state | unique weighted state in source cursor |
| Emitted text | selected live branch |
| Events | selected branch certificate |
| Successor state | live branch transition |
| Successor cursor | selected text projection |
| Graph/ring delta | facts plus source/successor writer states |
| Local evidence | facts and emitted event |
| Residual term | typed event plus source residual state |
| Lifecycle links | exact branch-local lifecycle records |
| Coupling term | event, marker manifest, states, and residual work |

For a two-site shared directional ring carrier, the transition proof uses the
following narrower field authorities.  The opening term is deliberately a
distinct closed class from the singular one-site term.

| Serialized field | Independent authority |
| --- | --- |
| Carrier models | all fact-derived models for the bond |
| Compatible second choices | serialized policy plus the all-model relation |
| Opening intersections | union of independently derived restriction rows |
| Pair restrictions | selected independently derived restriction row |
| Source/successor snapshots | branch-local writer-state terms |
| Propagation result | producer-free residual-store execution |
| Affected component | replay propagation statistics |
| Discharged factors | facts plus source bond occurrences |
| Projected variables | exact source/successor domain difference |
| Bond occurrence | event endpoint roles and direction marks |
| Open/closed ring records | source and successor ring-state terms |
| Lifecycle capabilities | semantic operation and fact-derived model count |

Structural checking owns the closed three-object graph and its digests.  Live
checking reconstructs the immediate frontier with recursive counts disabled and
requires exact equality.  Facts-bound checking reuses the branch-local graph,
local-evidence, residual replay, lifecycle-accounting, and coupling checks.  An
unsupported semantic transition remains typed incomplete; flags, operation
names, capabilities, links, and term presence never confer proof credit.

The deliberate limitation is nonterminal snapshot branches only.  Terminal
artifacts, prefix reads, and global support completeness remain separate future
surfaces.
