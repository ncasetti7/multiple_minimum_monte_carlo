""" "Module for handling cheminformatics tasks"""

from typing import List, Tuple
import math
import random
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from ase import Atoms
from ase.geometry.analysis import Analysis
import numpy as np


def make_mol(smi: str) -> Chem.Mol:
    """
    Initialize a rdkit molecule from a SMILES string while preserving atom mapping

    Args:
        smi: str, SMILES string

    Returns:
        mol: rdkit.Chem.Mol
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    og = Chem.MolFromSmiles(smi, ps)
    fake_map = []
    for atom in og.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            fake_map.append(og.GetNumAtoms() - 1)
        else:
            fake_map.append(atom.GetAtomMapNum() - 1)
    indices_order = sorted(range(len(fake_map)), key=lambda x: fake_map[x])
    mol = Chem.RenumberAtoms(og, indices_order)
    return mol


def mol_to_ase_atoms(mol: Chem.Mol) -> Atoms:
    """
    Converts an RDKit molecule object to an ASE Atoms object.

    Args:
        mol : Chem.Mol, An RDKit molecule object with 3D coordinates (conformer).

    Returns:
        atoms: ase.Atoms, An ase atoms object
    """

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert mol.GetNumConformers() > 0
    positions = mol.GetConformer().GetPositions()
    atoms = Atoms(symbols=symbols, positions=np.array(positions))
    return atoms


def add_ase_coords_to_mol(atoms: Atoms, mol: Chem.Mol) -> Chem.Mol:
    """
    Add a conformer to an RDKit molecule using coordinates from an ASE Atoms object.

    Args:
        atoms: ase.Atoms, An ASE atoms object with positions and chemical symbols.
        mol: Chem.Mol, An RDKit molecule to which the conformer will be added.

    Returns:
        mol: Chem.Mol, The same RDKit molecule with a new conformer added.
    """
    positions = atoms.get_positions()
    if len(positions) != mol.GetNumAtoms():
        raise ValueError("Number of atoms in ASE Atoms and RDKit Mol do not match.")

    conf = mol.GetConformer()
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
    return mol


def add_coords_to_mol(coords: np.array, mol: Chem.Mol) -> Chem.Mol:
    if len(coords) != mol.GetNumAtoms():
        raise ValueError("Number of atoms in ASE Atoms and RDKit Mol do not match.")
    conf = mol.GetConformer()
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos)
    return mol


def get_bonds(mol: Chem.Mol) -> List[Tuple]:
    """
    Get the bond strings of a molecule.

    Args:
        mol (Chem.Mol): Molecule.

    Returns:
        set: Set of bond strings.
    """
    bonds = set()
    for bond in mol.GetBonds():
        atom_1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        atom_2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()

        if atom_1 < atom_2:
            bonds.add((atom_1, atom_2))
        else:
            bonds.add((atom_2, atom_1))

    return bonds


def get_dihedral_matches(mol: Chem.Mol, heavy: bool) -> List[Tuple[int, int, int, int]]:
    """
    Identify unique dihedral (torsion) atom quartets in a molecule based on a strict SMARTS pattern.
    This function searches for all sets of four connected atoms (a, b, c, d) in the given molecule
    that match a predefined dihedral SMARTS pattern. The matches are filtered to ensure uniqueness
    based on the central bond (b, c). The selection of matches can be further refined based on the
    `heavy` parameter:
    - If `heavy` is True, only dihedrals where both terminal atoms (a and d) are not hydrogens are included.
    - If `heavy` is False, dihedrals where the third atom is carbon and the fourth atom is hydrogen are excluded.
    Args:
    mol : Chem.Mol
        An RDKit molecule object to search for dihedral matches.
    heavy : bool
        If True, only include dihedrals with heavy atom terminals (no hydrogens).
        If False, include all dihedrals except those ending with a C-H pair.
    Returns:
    uniqmatches : list of tuple of int
        A list of unique tuples (a, b, c, d), where each tuple contains the atom indices
        of a dihedral match in the molecule.
    """

    # this is rdkit's "strict" pattern
    pattern = r"*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*"
    qmol = Chem.MolFromSmarts(pattern)
    matches = mol.GetSubstructMatches(qmol)

    # these are all sets of 4 atoms, uniquify by middle two
    uniqmatches = []
    seen = set()
    for a, b, c, d in matches:
        if (b, c) not in seen and (c, b) not in seen:
            if heavy:
                if (
                    mol.GetAtomWithIdx(a).GetSymbol() != "H"
                    and mol.GetAtomWithIdx(d).GetSymbol() != "H"
                ):
                    seen.add((b, c))
                    uniqmatches.append((a, b, c, d))
            if not heavy:
                if (
                    mol.GetAtomWithIdx(c).GetSymbol() == "C"
                    and mol.GetAtomWithIdx(d).GetSymbol() == "H"
                ):
                    pass
                else:
                    seen.add((b, c))
                    uniqmatches.append((a, b, c, d))
    return uniqmatches


def rotate_dihedrals(
    conformer: Chem.rdchem.Conformer,
    dihedrals: List[Tuple[int, int, int, int]],
    stepsize: float,
) -> None:
    """
    Applies a random rotation to all the dihedrals

    Parameters
    ----------
    conformer : rdkit.Chem.rdchem.Conformer
        The conformer whose angles are going to be rotated (conformer = mol.GetConformer(cid))
    dihedrals : list
        A list of tuples of all the dihedrals that are going to be rotated.
    stepsize : float
        Angle in Degrees to do the steps between 0.0 and 360.0
    """

    rad_range = np.arange(stepsize, 360.0, stepsize)
    for dihedral in dihedrals:
        rad_ang = random.choice(rad_range)
        rad = math.pi * rad_ang / 180.0
        rdMolTransforms.SetDihedralRad(conformer, *dihedral, value=rad)


def get_bonded_atoms(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """
    Get the indices of bonded atoms in the molecule.

    Args:
    mol: Chem.Mol, An RDKit molecule object

    Returns:
    list: List of indices of bonded atoms.
    """
    bonded_atoms = []
    for bond in mol.GetBonds():
        atom_1 = bond.GetBeginAtomIdx()
        atom_2 = bond.GetEndAtomIdx()
        bonded_atoms.append((atom_1, atom_2))

    return bonded_atoms


def get_metal_atoms(mol: Chem.Mol) -> List[int]:
    """
    Get the indices of metal atoms in the reactant molecule.

    Returns:
    list: List of indices of metal atoms.
    """
    metal_list = [
        "Al",
        "Sb",
        "Ag",
        "As",
        "Ba",
        "Be",
        "Bi",
        "Cd",
        "Ca",
        "Cr",
        "Co",
        "Cu",
        "Au",
        "Fe",
        "Pb",
        "Mg",
        "Mn",
        "Hg",
        "Mo",
        "Ni",
        "Pd",
        "Pt",
        "K",
        "Rh",
        "Rb",
        "Ru",
        "Sc",
        "Ag",
        "Na",
        "Sr",
        "Ta",
        "Tl",
        "Th",
        "Ti",
        "U",
        "V",
        "Y",
        "Zn",
        "Zr",
    ]
    metal_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_list:
            metal_atoms.append(atom.GetIdx())

    return metal_atoms


def initialize_mc_identity_check(
    atoms: Atoms, original_mol: Chem.Mol
) -> Tuple[List, List, List]:
    """
    Initialize the identity check for the Monte Carlo conformer

    Args:
        atoms: ase.Atoms
        original_mol (rdkit.Chem.rdchem.Mol): Original molecule

    Returns:
        tuple: Set of bonds in the original molecule, indices of metal atoms, indices of halide atoms
    """
    metal_atoms = get_metal_atoms(original_mol)
    metal_atoms = [atom for atom in metal_atoms]
    halides = [
        atom.GetIdx()
        for atom in original_mol.GetAtoms()
        if atom.GetSymbol() in ["F", "Cl", "Br", "I"]
    ]
    original_bonds = []
    ana = Analysis(atoms)
    bonds = ana.all_bonds
    for b in bonds:
        for i, atoms in enumerate(b):
            for atom in atoms:
                if i not in metal_atoms and atom not in metal_atoms:
                    if i not in halides or atom not in halides:
                        if i > atom:
                            continue
                        original_bonds.append((i, atom))
    return original_bonds, metal_atoms, halides


def check_identity_mc(
    original_bonds: List, metal_atoms: List, halide_atoms: List, atoms: Atoms
) -> bool:
    """
    Check if the monte carlo conformer corresponds to the input molecule

    Args:
        original_bonds (list): Set of bonds in the original molecule
        metal_atoms (list): List of indices of metal atoms in the original molecule
        halide_atoms (list): List of indices of halide atoms in the original molecule
        atoms (ase.Atoms):

    Returns:
        bool: True if the conformer corresponds to the input molecule, False otherwise
    """
    ana = Analysis(atoms)
    bonds = ana.all_bonds
    mc_bonds = []
    for b in bonds:
        for i, atoms in enumerate(b):
            for atom in atoms:
                if i not in metal_atoms and atom not in metal_atoms:
                    if i not in halide_atoms or atom not in halide_atoms:
                        if i > atom:
                            continue
                        mc_bonds.append((i, atom))
    return set(original_bonds) == set(mc_bonds)
