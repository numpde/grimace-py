from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.public_runtime import make_determinized_decoder
from tests.helpers.rdkit_known_stereo_gaps import (
    PinnedKnownStereoGapCase,
    load_pinned_known_stereo_gap_cases,
)
from tests.helpers.rdkit_writer_membership import (
    load_pinned_writer_membership_cases,
)
from tests.rdkit_serialization._support import (
    grimace_support,
    mol_from_pinned_source,
    rdkit_mol_to_smiles_kwargs_from_options,
    supported_public_kwargs_from_rdkit_options,
)


class KnownStereoGapTests(unittest.TestCase):
    """Diagnostic tests for coupled directional-stereo parity gaps.

    These cases are not part of the passing writer-membership corpus yet. They
    should move there once Grimace resolves coupled double-bond direction tokens
    with RDKit-equivalent traversal-order state. Run through
    tests.run_known_stereo_gaps; this module is outside default discovery.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_known_stereo_gap_cases(rdBase.rdkitVersion)
            writer_cases = load_pinned_writer_membership_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned known stereo-gap corpus for RDKit {rdBase.rdkitVersion}"
            )
        cls.writer_cases_by_id = {case.case_id: case for case in writer_cases}

    def _mol_from_case(self, case: PinnedKnownStereoGapCase) -> Chem.Mol:
        if case.writer_membership_case_id is not None:
            writer_case = self.writer_cases_by_id[case.writer_membership_case_id]
            return mol_from_pinned_source(writer_case)
        return mol_from_pinned_source(case)

    def _molecule_source_key(self, case: PinnedKnownStereoGapCase) -> str:
        if case.writer_membership_case_id is not None:
            return f"writer-membership:{case.writer_membership_case_id}"
        if case.smiles is not None:
            return f"smiles:{case.smiles}"
        return f"molblock:{case.case_id}"

    def _rdkit_output(self, mol: Chem.Mol, case: PinnedKnownStereoGapCase) -> str:
        if case.rdkit_random_vector_seed is not None:
            vector_index = case.rdkit_random_vector_index
            if vector_index is None:
                raise AssertionError(f"{case.case_id}: loader allowed unpaired seed")
            random_outputs = Chem.MolToRandomSmilesVect(
                Chem.Mol(mol),
                vector_index + 1,
                randomSeed=case.rdkit_random_vector_seed,
                isomericSmiles=case.isomeric_smiles,
            )
            return random_outputs[vector_index]
        return Chem.MolToSmiles(
            Chem.Mol(mol),
            **rdkit_mol_to_smiles_kwargs_from_options(
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
                canonical=case.rdkit_canonical,
                do_random=False,
            ),
        )

    def _assert_determinized_decoder_accepts(
        self,
        mol: Chem.Mol,
        case: PinnedKnownStereoGapCase,
    ) -> None:
        decoder = make_determinized_decoder(
            mol,
            **supported_public_kwargs_from_rdkit_options(
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ),
        )
        pos = 0
        while pos < len(case.expected):
            choice = next(
                (
                    choice
                    for choice in decoder.next_choices
                    if case.expected.startswith(choice.text, pos)
                ),
                None,
            )
            if choice is None:
                choices = tuple(choice.text for choice in decoder.next_choices)
                self.fail(
                    f"{case.case_id}: decoder rejected RDKit output at offset "
                    f"{pos}; prefix={case.expected[:pos]!r}; "
                    f"next={case.expected[pos:pos + 24]!r}; choices={choices!r}"
                )
            decoder = choice.next_state
            pos += len(choice.text)
        self.assertTrue(decoder.is_terminal, case.case_id)

    def test_pinned_rdkit_stereo_gap_outputs_are_in_grimace_support(self) -> None:
        support_cache: dict[tuple[str, int | None, bool], set[str]] = {}
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                rdkit_output = self._rdkit_output(mol, case)
                self.assertEqual(case.expected, rdkit_output)
                if case.check_grimace_decoder_path:
                    self._assert_determinized_decoder_accepts(mol, case)
                if not case.check_grimace_support:
                    continue

                support_key = (
                    self._molecule_source_key(case),
                    case.rooted_at_atom,
                    case.isomeric_smiles,
                )
                support = support_cache.get(support_key)
                if support is None:
                    try:
                        support = grimace_support(
                            mol,
                            rooted_at_atom=case.rooted_at_atom,
                            isomeric_smiles=case.isomeric_smiles,
                        )
                    except Exception as exc:
                        self.fail(
                            f"{case.case_id}: Grimace support enumeration failed "
                            f"with {type(exc).__name__}: {exc}"
                        )
                    support_cache[support_key] = support
                self.assertTrue(
                    case.expected in support,
                    f"{case.case_id}: missing RDKit output {case.expected!r} "
                    f"from Grimace support of size {len(support)}",
                )


if __name__ == "__main__":
    unittest.main()
