# Directional ring coupling terms

Schema v8 closes each non-single directional ring coupling as an individual
proof object. The coupling term commits to the event, marker placement, writer
state and ring-state identities, residual snapshots, closure manifest, exact
stereo lifecycle, residual work, and optional closed-closure record.

Local coupled digests now name coupling terms rather than opaque live evidence.
Offline credit is registered per term only after its exact stereo lifecycle and
residual work have already been semantically replayed. Aggregate branch presence
or an unrelated replayed lifecycle cannot confer credit.

This changes proof representation only. The supported chemistry remains the
single-site ordinary DOUBLE ring carrier introduced in the preceding slice.
