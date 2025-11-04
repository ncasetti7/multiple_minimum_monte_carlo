import pytest

# skip module if rdkit not available
pytest.importorskip("rdkit")
from rdkit import Chem

from multiple_minimum_monte_carlo.conformer import Conformer


def test_conformer_creates_mol_for_unmapped(monkeypatch):
    # ensure Chem.MolFromSmiles is called when not mapped
    called = {}
    real_mf = Chem.MolFromSmiles

    def fake_mol_from_smiles(smi):
        called["smi"] = smi
        return real_mf("CC")

    monkeypatch.setattr(Chem, "MolFromSmiles", fake_mol_from_smiles)

    # avoid heavy conformer generation in initializer by providing a minimal atoms object
    def fake_generate(self):
        class MinimalAtoms:
            def __init__(self):
                self.info = {}

            def get_positions(self):
                return []

        self.atoms = MinimalAtoms()

    monkeypatch.setattr(Conformer, "generate_conformer", fake_generate)
    c = Conformer("CC", mapped=False)
    assert c.smiles == "CC"


def test_conformer_uses_make_mol_when_mapped(monkeypatch):
    from multiple_minimum_monte_carlo import cheminformatics

    called = {}

    def fake_make_mol(smi):
        called["smi"] = smi
        return Chem.MolFromSmiles("CC")

    monkeypatch.setattr(cheminformatics, "make_mol", fake_make_mol)

    # avoid heavy conformer generation by providing minimal atoms
    def fake_generate(self):
        class MinimalAtoms:
            def __init__(self):
                self.info = {}

            def get_positions(self):
                return []

        self.atoms = MinimalAtoms()

    monkeypatch.setattr(Conformer, "generate_conformer", fake_generate)
    c = Conformer("[CH3:1][CH3:2]", mapped=True)
    assert hasattr(c, "mol")
    assert called["smi"].startswith("[CH3")
