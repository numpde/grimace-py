from __future__ import annotations

from collections.abc import Callable
import unittest

import grimace
from grimace._mol_to_smiles_options import MOL_TO_SMILES_PREPARED_OPTIONS
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    choice_texts,
    public_enum_support,
    public_token_inventory,
    public_token_inventory_superset,
    reachable_outputs_from_decoder,
    supported_public_kwargs,
)


class SentinelPrepareError(Exception):
    pass


class PreparedMolRuntimeTests(unittest.TestCase):
    def _prepare_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        return {
            spec.public_name: kwargs[spec.public_name]
            for spec in MOL_TO_SMILES_PREPARED_OPTIONS
            if spec.public_name in kwargs
        }

    def _prepared_variants(
        self,
        mol: object,
        kwargs: dict[str, object],
    ) -> tuple[tuple[str, object], ...]:
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        return (
            ("prepared", prepared),
            ("round_tripped", grimace.PreparedMol.from_bytes(prepared.to_bytes())),
        )

    def _terminal_token_path(self, mol_or_prepared: object, kwargs: dict[str, object]) -> tuple[str, ...]:
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol_or_prepared, **kwargs)
        tokens: list[str] = []
        while not decoder.is_terminal:
            choices = decoder.next_choices
            self.assertTrue(choices)
            choice = choices[0]
            tokens.append(choice.text)
            decoder = choice.next_state
        return tuple(tokens)

    def _public_operation_calls(
        self,
        mol_or_prepared: object,
        kwargs: dict[str, object],
        token_candidate: tuple[str, ...],
    ) -> tuple[tuple[str, Callable[[], object]], ...]:
        string_candidate = "".join(token_candidate)
        return (
            (
                "enum",
                lambda: public_enum_support(mol_or_prepared, **kwargs),
            ),
            (
                "decoder",
                lambda: reachable_outputs_from_decoder(
                    grimace.MolToSmilesDecoder(mol_or_prepared, **kwargs)
                ),
            ),
            (
                "determinized_decoder",
                lambda: reachable_outputs_from_decoder(
                    grimace.MolToSmilesDeterminizedDecoder(mol_or_prepared, **kwargs)
                ),
            ),
            (
                "inventory",
                lambda: public_token_inventory(mol_or_prepared, **kwargs),
            ),
            (
                "inventory_superset",
                lambda: public_token_inventory_superset(mol_or_prepared, **kwargs),
            ),
            (
                "deviation_string",
                lambda: grimace.MolToSmilesDeviation(
                    mol_or_prepared,
                    string_candidate,
                    **kwargs,
                ),
            ),
            (
                "deviation_tokens",
                lambda: grimace.MolToSmilesDeviation(
                    mol_or_prepared,
                    token_candidate,
                    **kwargs,
                ),
            ),
        )

    def _with_prepare_mol_replaced(
        self,
        replacement: Callable[..., object],
        call: Callable[[], object],
    ) -> object:
        from grimace import _prepared_mol

        original_public_prepare_mol = grimace.PrepareMol
        original_module_prepare_mol = _prepared_mol.PrepareMol
        try:
            grimace.PrepareMol = replacement
            _prepared_mol.PrepareMol = replacement
            return call()
        finally:
            grimace.PrepareMol = original_public_prepare_mol
            _prepared_mol.PrepareMol = original_module_prepare_mol

    def _assert_runtime_equivalent(
        self,
        smiles: str,
        *,
        kwargs: dict[str, object],
    ) -> None:
        mol = parse_smiles(smiles)
        expected_support = public_enum_support(mol, **kwargs)
        expected_inventory = public_token_inventory(mol, **kwargs)
        expected_inventory_superset = public_token_inventory_superset(mol, **kwargs)

        expected_decoder = grimace.MolToSmilesDecoder(mol, **kwargs)
        expected_determinized = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)
        expected_decoder_outputs = reachable_outputs_from_decoder(expected_decoder)
        expected_determinized_outputs = reachable_outputs_from_decoder(expected_determinized)
        token_candidate = self._terminal_token_path(mol, kwargs)
        string_candidate = "".join(token_candidate)

        for variant_name, prepared in self._prepared_variants(mol, kwargs):
            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="enum"):
                self.assertEqual(expected_support, public_enum_support(prepared, **kwargs))

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="decoder"):
                decoder = grimace.MolToSmilesDecoder(prepared, **kwargs)
                self.assertEqual(choice_texts(expected_decoder), choice_texts(decoder))
                self.assertEqual(expected_decoder_outputs, reachable_outputs_from_decoder(decoder))

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="determinized_decoder"):
                determinized = grimace.MolToSmilesDeterminizedDecoder(prepared, **kwargs)
                self.assertEqual(choice_texts(expected_determinized), choice_texts(determinized))
                self.assertEqual(
                    expected_determinized_outputs,
                    reachable_outputs_from_decoder(determinized),
                )

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="inventory"):
                self.assertEqual(expected_inventory, public_token_inventory(prepared, **kwargs))

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="inventory_superset"):
                self.assertEqual(
                    expected_inventory_superset,
                    public_token_inventory_superset(prepared, **kwargs),
                )

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="deviation_string"):
                self.assertIsNone(grimace.MolToSmilesDeviation(prepared, string_candidate, **kwargs))
                self.assertEqual(
                    grimace.MolToSmilesDeviation(mol, "Z", **kwargs),
                    grimace.MolToSmilesDeviation(prepared, "Z", **kwargs),
                )

            with self.subTest(smiles=smiles, variant=variant_name, entrypoint="deviation_tokens"):
                self.assertIsNone(grimace.MolToSmilesDeviation(prepared, token_candidate, **kwargs))
                bad_candidate = token_candidate[:-1] + ("Z",)
                self.assertEqual(
                    grimace.MolToSmilesDeviation(mol, bad_candidate, **kwargs),
                    grimace.MolToSmilesDeviation(prepared, bad_candidate, **kwargs),
                )

    def test_connected_rooted_runtime_matches_rdkit_mol(self) -> None:
        self._assert_runtime_equivalent(
            "CCO",
            kwargs=supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
        )

    def test_empty_runtime_matches_rdkit_mol(self) -> None:
        for rooted_at_atom in (-1, 0):
            with self.subTest(rootedAtAtom=rooted_at_atom):
                self._assert_runtime_equivalent(
                    "",
                    kwargs=supported_public_kwargs(
                        isomericSmiles=False,
                        rootedAtAtom=rooted_at_atom,
                    ),
                )

    def test_connected_all_roots_runtime_matches_rdkit_mol(self) -> None:
        self._assert_runtime_equivalent(
            "CCO",
            kwargs=supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1),
        )

    def test_disconnected_all_roots_runtime_matches_rdkit_mol(self) -> None:
        self._assert_runtime_equivalent(
            "CCO.N.Cl",
            kwargs=supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1),
        )

    def test_disconnected_rooted_in_nonfirst_fragment_matches_rdkit_mol(self) -> None:
        self._assert_runtime_equivalent(
            "CCO.N.Cl",
            kwargs=supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3),
        )

    def test_stereo_runtime_matches_rdkit_mol(self) -> None:
        self._assert_runtime_equivalent(
            "F[C@H](Cl)Br",
            kwargs=supported_public_kwargs(isomericSmiles=True, rootedAtAtom=1),
        )

    def test_writer_surface_variants_runtime_match_rdkit_mol(self) -> None:
        cases = (
            (
                "kekule",
                "c1ccccc1",
                supported_public_kwargs(
                    isomericSmiles=False,
                    kekuleSmiles=True,
                    rootedAtAtom=0,
                ),
            ),
            (
                "explicit_bonds",
                "CC#N",
                supported_public_kwargs(
                    isomericSmiles=False,
                    allBondsExplicit=True,
                    rootedAtAtom=0,
                ),
            ),
            (
                "explicit_hs",
                "CO",
                supported_public_kwargs(
                    isomericSmiles=False,
                    allHsExplicit=True,
                    rootedAtAtom=0,
                ),
            ),
            (
                "ignored_atom_maps",
                "[CH3:7]C",
                supported_public_kwargs(
                    isomericSmiles=False,
                    ignoreAtomMapNumbers=True,
                    rootedAtAtom=0,
                ),
            ),
        )

        for name, smiles, kwargs in cases:
            with self.subTest(name=name):
                self._assert_runtime_equivalent(smiles, kwargs=kwargs)

    def test_prepared_runtime_does_not_call_late_rdkit_preparation(self) -> None:
        from rdkit import Chem
        from grimace import _runtime

        mol = parse_smiles("CCO.N.Cl")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3)
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        token_candidate = self._terminal_token_path(prepared, kwargs)
        expected_by_entrypoint = {
            entrypoint: call()
            for entrypoint, call in self._public_operation_calls(
                prepared,
                kwargs,
                token_candidate,
            )
        }
        original_get_mol_frags = Chem.GetMolFrags
        original_runtime_reference_prepare = (
            _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs
        )

        def fail_get_mol_frags(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("PreparedMol runtime must not call Chem.GetMolFrags")

        def fail_reference_preparation(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("PreparedMol runtime must not prepare from RDKit")

        try:
            Chem.GetMolFrags = fail_get_mol_frags
            _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs = (
                fail_reference_preparation
            )
            for entrypoint, call in self._public_operation_calls(
                prepared,
                kwargs,
                token_candidate,
            ):
                with self.subTest(entrypoint=entrypoint):
                    self.assertEqual(expected_by_entrypoint[entrypoint], call())
        finally:
            Chem.GetMolFrags = original_get_mol_frags
            _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs = (
                original_runtime_reference_prepare
            )

    def test_writer_flag_mismatches_are_rejected(self) -> None:
        mol = parse_smiles("CCO")
        prepared = grimace.PrepareMol(mol, isomericSmiles=False)
        kwargs = supported_public_kwargs(
            isomericSmiles=False,
            allHsExplicit=True,
            rootedAtAtom=0,
        )

        calls = (
            lambda: tuple(grimace.MolToSmilesEnum(prepared, **kwargs)),
            lambda: grimace.MolToSmilesDecoder(prepared, **kwargs).next_choices,
            lambda: grimace.MolToSmilesDeterminizedDecoder(prepared, **kwargs).next_choices,
            lambda: grimace.MolToSmilesTokenInventory(prepared, **kwargs),
            lambda: grimace.MolToSmilesTokenInventorySuperset(prepared, **kwargs),
            lambda: grimace.MolToSmilesDeviation(prepared, "CCO", **kwargs),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "writer flags"):
                    call()

    def test_empty_all_roots_enum_rejects_writer_flag_mismatch(self) -> None:
        prepared = grimace.PrepareMol(parse_smiles(""), isomericSmiles=False)
        kwargs = supported_public_kwargs(
            isomericSmiles=False,
            allHsExplicit=True,
            rootedAtAtom=-1,
        )

        with self.assertRaisesRegex(ValueError, "writer flags"):
            tuple(grimace.MolToSmilesEnum(prepared, **kwargs))

    def test_rdkit_input_public_operations_require_prepare_mol(self) -> None:
        mol = parse_smiles("CCO.N.Cl")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3)
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        token_candidate = self._terminal_token_path(prepared, kwargs)

        def fail_prepare_mol(*_args: object, **_kwargs: object) -> object:
            raise SentinelPrepareError("PrepareMol was required")

        for entrypoint, call in self._public_operation_calls(
            mol,
            kwargs,
            token_candidate,
        ):
            with self.subTest(entrypoint=entrypoint):
                with self.assertRaisesRegex(SentinelPrepareError, "required"):
                    self._with_prepare_mol_replaced(fail_prepare_mol, call)

    def test_prepared_input_public_operations_skip_prepare_mol(self) -> None:
        mol = parse_smiles("CCO.N.Cl")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3)
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        round_tripped = grimace.PreparedMol.from_bytes(prepared.to_bytes())
        token_candidate = self._terminal_token_path(prepared, kwargs)
        expected_by_entrypoint = {
            entrypoint: call()
            for entrypoint, call in self._public_operation_calls(
                prepared,
                kwargs,
                token_candidate,
            )
        }

        def fail_prepare_mol(*_args: object, **_kwargs: object) -> object:
            raise SentinelPrepareError("PreparedMol input must not call PrepareMol")

        for entrypoint, call in self._public_operation_calls(
            round_tripped,
            kwargs,
            token_candidate,
        ):
            with self.subTest(entrypoint=entrypoint):
                self.assertEqual(
                    expected_by_entrypoint[entrypoint],
                    self._with_prepare_mol_replaced(fail_prepare_mol, call),
                )

    def test_rdkit_input_public_operations_do_no_late_rdkit_preparation(self) -> None:
        from rdkit import Chem
        from grimace import _prepared_mol, _runtime

        mol = parse_smiles("CCO.N.Cl")
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=3)
        prepared = grimace.PrepareMol(mol, **self._prepare_kwargs(kwargs))
        token_candidate = self._terminal_token_path(prepared, kwargs)
        expected_by_entrypoint = {
            entrypoint: call()
            for entrypoint, call in self._public_operation_calls(
                prepared,
                kwargs,
                token_candidate,
            )
        }
        real_prepare_mol = _prepared_mol.PrepareMol

        def fail_late_preparation(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("runtime performed RDKit/reference preparation after PrepareMol")

        for entrypoint, operation_call in self._public_operation_calls(
            mol,
            kwargs,
            token_candidate,
        ):
            with self.subTest(entrypoint=entrypoint):
                original_get_mol_frags = Chem.GetMolFrags
                original_runtime_reference_prepare = (
                    _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs
                )

                def wrapped_prepare_mol(*args: object, **kwargs: object) -> object:
                    prepared_mol = real_prepare_mol(*args, **kwargs)
                    Chem.GetMolFrags = fail_late_preparation
                    _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs = (
                        fail_late_preparation
                    )
                    return prepared_mol

                try:
                    self.assertEqual(
                        expected_by_entrypoint[entrypoint],
                        self._with_prepare_mol_replaced(
                            wrapped_prepare_mol,
                            operation_call,
                        ),
                    )
                finally:
                    Chem.GetMolFrags = original_get_mol_frags
                    _runtime.prepare_smiles_graph_from_mol_to_smiles_kwargs = (
                        original_runtime_reference_prepare
                    )


if __name__ == "__main__":
    unittest.main()
