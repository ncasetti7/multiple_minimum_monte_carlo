"""Module for managing conformer generation"""

from typing import Optional, List
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from multiple_minimum_monte_carlo import cheminformatics


class Conformer():
    def __init__(self, 
                 smiles: Optional[str] = None,
                 mapped: Optional[bool] = False,
                 input_xyz: Optional[str] = None,
                 charge: Optional[int] = None,
                 spin: Optional[int] = 1,
                 constrained_atoms: Optional[List[int]] = []) -> None:
        """
        Initializes a conformer object from a SMILES string, with optional atom mapping, input coordinates, and constrained atoms.
        Parameters:
            smiles (str): The SMILES string representing the molecule.
            mapped (Optional[bool], default=False): Whether the SMILES string is atom-mapped. If True, uses a custom molecule creation function; otherwise, hydrogens are added to the molecule.
            input_xyz (Optional[str], default=None): Path to an XYZ file containing input coordinates. If None, a conformer is generated.
            charge (Optional[int], default=None): 
            spin (Optional[int], default=1): 
            constrained_atoms (Optional[List[int]], default=[]): List of atom indices to be constrained during conformer generation or optimization.
        """
        self.smiles = smiles
        self.mapped = mapped
        self.input_xyz = input_xyz
        self.charge = charge
        self.spin = spin
        self.constrained_atoms = constrained_atoms
        
        # Check whether atom rearrangement is necessary 
        if self.smiles is not None:
            if self.mapped:
                self.mol = cheminformatics.make_mol(self.smiles)
            else:
                self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))

        # Check whether conformer generation/mapping is necessary
        if self.input_xyz is None and self.smiles is None:
            raise ValueError("Conformer needs either smiles or input_xyz!")
        elif self.input_xyz is None:
            self.generate_conformer()
        elif self.smiles is None:
            if self.charge is None:
                raise ValueError("Charge must be specified if no SMILES string provided!")
            self.mol_from_xyz()
            self.atoms = read(input_xyz)
        else:
            self.atoms = read(input_xyz)

        if self.charge is None:
            self.charge = Chem.GetFormalCharge(self.mol)
        self.bonded_atoms = cheminformatics.get_bonded_atoms(self.mol)
        self.atoms.info['charge'] = self.charge
        self.atoms.info['spin'] = self.spin

    def generate_conformer(self) -> None:
        AllChem.EmbedMolecule(self.mol)
        AllChem.UFFOptimizeMolecule(self.mol)
        self.atoms = cheminformatics.mol_to_ase_atoms(self.mol)

    def mol_from_xyz(self) -> Chem.Mol:
        raw_mol = Chem.MolFromXYZFile(self.input_xyz)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(mol,charge=self.charge)
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol)