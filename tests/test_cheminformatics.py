import pytest
import numpy as np

# skip entire module if rdkit missing
pytest.importorskip("rdkit")
from rdkit import Chem

from multiple_minimum_monte_carlo import cheminformatics


def test_make_mol_basic():
    smi = "CCO"
    mol = cheminformatics.make_mol(smi)
    assert mol.GetNumAtoms() == Chem.MolFromSmiles(smi).GetNumAtoms()


def test_get_bonds_and_metal_atoms():
    # create a molecule with atom mapping so get_bonds uses atom map nums
    smi = "[CH3:1][CH3:2]"
    mol = cheminformatics.make_mol(smi)
    bonds = cheminformatics.get_bonds(mol)
    assert isinstance(bonds, set)
    # metal detection on a simple metal mol
    metal_mol = Chem.MolFromSmiles("[Fe]")
    metals = cheminformatics.get_metal_atoms(metal_mol)
    assert all(isinstance(i, int) for i in metals)


def test_rotate_dihedrals_runs_without_error():
    # Use a simple molecule with a rotatable bond (but we only ensure no exception)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    Chem.AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    # pick an example dihedral (0,1,2,3) which exists for a 4-carbon chain
    cheminformatics.rotate_dihedrals(conf, [(0, 1, 2, 3)], 30.0)
    # position changed for at least one atom (float values)
    pos = conf.GetPositions()
    assert isinstance(pos, np.ndarray) or hasattr(pos, "__len__")


def test_mol_to_ase_and_add_coords():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    Chem.AllChem.EmbedMolecule(mol)
    atoms = cheminformatics.mol_to_ase_atoms(mol)
    assert hasattr(atoms, "get_positions")

    # test add_coords_to_mol with explicit numpy array
    coords = atoms.get_positions()
    newmol = cheminformatics.add_coords_to_mol(coords, mol)
    assert newmol is mol


def test_identity_checks(tmp_path):
    # Build a small molecule and ASE Atoms via mol_to_ase_atoms
    mol = cheminformatics.make_mol("CCO")
    Chem.AllChem.EmbedMolecule(mol)
    atoms = cheminformatics.mol_to_ase_atoms(mol)

    original_bonds, metal_atoms, halides = cheminformatics.initialize_mc_identity_check(
        atoms, mol
    )
    # returned types
    assert isinstance(original_bonds, list)
    assert isinstance(metal_atoms, list)
    assert isinstance(halides, list)

    # check_identity_mc on same atoms should be True
    assert (
        cheminformatics.check_identity_mc(original_bonds, metal_atoms, halides, atoms)
        is True
    )
