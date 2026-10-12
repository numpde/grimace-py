"""Count native histories for one visible path through identical methane components."""

from rdkit import Chem

from grimace.decoding import SmilesLanguage, start


MAX_COMPONENTS = 7
MAX_FRONTIER = 10_000


for language in (SmilesLanguage.GENERAL, SmilesLanguage.COMPACT):
    for component_count in range(2, MAX_COMPONENTS + 1):
        mol = Chem.MolFromSmiles(".".join(["C"] * component_count))
        prefix = start(mol, language=language)
        path = ""
        for token in ["C"] + [part for _ in range(component_count - 1) for part in (".", "C")]:
            native_count = prefix._Prefix__native_prefix.frontier_size
            assert native_count <= MAX_FRONTIER, (language, component_count, path, native_count)
            choices = dict(prefix.choices())
            assert token in choices, (language, component_count, path, token)
            prefix = choices[token]
            path += token
            native_count = prefix._Prefix__native_prefix.frontier_size
            assert native_count <= MAX_FRONTIER, (language, component_count, path, native_count)
            next_choice_count = len(prefix.choices())
            print(language.value, component_count, path, native_count, next_choice_count)
