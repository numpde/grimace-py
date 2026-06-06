---
title: Deviation diagnostics
---

`MolToSmilesDeviation(...)` reports the first place where a candidate leaves
the molecule's supported SMILES language under the requested writer flags.

Use it for diagnostics: model output validation, tokenizer boundary checks, or
explaining where a candidate leaves Grimace's supported writer language.

It returns `None` for an accepted candidate. Otherwise it returns a
`SmilesDeviation`.

## String candidates

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CCO")
kwargs = dict(
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)

assert grimace.MolToSmilesDeviation(mol, "CCO", **kwargs) is None

deviation = grimace.MolToSmilesDeviation(mol, "CCN", **kwargs)
assert deviation.accepted_text == "CC"
assert deviation.rejected_text == "N"
assert deviation.legal_next_tokens == ("O",)
```

## Token sequence candidates

String candidates are matched as text. Sequence candidates are atomic external
tokens, so boundaries matter:

```python
grimace.MolToSmilesDeviation(mol, "CCl", **kwargs).accepted_text
# 'CC'

grimace.MolToSmilesDeviation(mol, ("C", "Cl"), **kwargs).accepted_text
# 'C'
```

The sequence form is useful when a model or tokenizer already split the
candidate into external token strings.

For the meaning of Grimace tokens, see [Concepts](../concepts.html).
