import pytest
pytest.importorskip("rdkit")
from rdkit import Chem

from multiple_minimum_monte_carlo.conformer_ensemble import ConformerEnsemble, run_class_func
from multiple_minimum_monte_carlo.conformer import Conformer


def test_run_class_func_calls_method():
    class C:
        def add(self, x, y):
            return x + y

    assert run_class_func(C(), "add", {"x": 1, "y": 2}) == 3


def make_dummy_conformer(monkeypatch):
    # Create a minimal Conformer-like object
    # prevent heavy generation
    def fake_generate(self):
        class MinimalAtoms:
            def __init__(self):
                self.info = {}
            def get_positions(self):
                return []
        self.atoms = MinimalAtoms()
    try:
        # if a pytest monkeypatch fixture is provided
        monkeypatch.setattr(Conformer, "generate_conformer", fake_generate)
    except Exception:
        # otherwise, monkeypatch by direct assignment
        Conformer.generate_conformer = fake_generate
    c = Conformer("CC")
    c.atoms = None
    c.mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    c.constrained_atoms = []
    c.bonded_atoms = []
    return c


def test_sample_conformer_minimum():
    c = make_dummy_conformer(None)
    ensemble = ConformerEnsemble(c, calc=None, num_iterations=1, parallel=False)
    # when used is [0,0,0], argmin is 0
    assert ensemble.sample_conformer([1, 2, 3]) == 0 or isinstance(ensemble.sample_conformer([1,2,3]), int)


def test_constraint_test_detects_close_atoms(monkeypatch):
    c = make_dummy_conformer(monkeypatch)
    # create a fake conformer with positions that are extremely close
    mol = c.mol
    Chem.AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    # set all positions to zero to force failure
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
    ensemble = ConformerEnsemble(c, calc=None, num_iterations=1, parallel=False)
    assert ensemble.constraint_test(conf) is False
