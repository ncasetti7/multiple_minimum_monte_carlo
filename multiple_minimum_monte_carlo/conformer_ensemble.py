"""Module for running multiple minimum monte carlo"""

import os
import sys
from typing import Optional, List, Tuple
import random
from copy import copy
import logging
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import PeriodicTable, rdMolAlign
from multiple_minimum_monte_carlo.conformer import Conformer
from multiple_minimum_monte_carlo.calculation import Calculation
from multiple_minimum_monte_carlo import cheminformatics, multiproc

def run_class_func(cls, func_name, args):
    func = getattr(cls, func_name)
    return func(**args)

class ConformerEnsemble():
    def __init__(self,
                 conformer: Conformer,
                 calc: Calculation,
                 num_iterations: Optional[int] = 100,
                 energy_window: Optional[float] = 10.0,
                 max_bonds_rotate: Optional[int] = 3,
                 max_attempts: Optional[int] = 1000,
                 angle_step: Optional[float] = 30.0,
                 energy_threshold: Optional[float] = 1.0,
                 rmsd_threshold: Optional[float] = 0.3,
                 initial_optimization: Optional[bool] = True,
                 random_walk: Optional[bool] = False,
                 reduce_angle: Optional[bool] = False,
                 reduce_angle_every: Optional[int] = 50,
                 reduce_angle_by: Optional[int] = 2,
                 only_heavy: Optional[bool] = True,
                 parallel: Optional[bool] = True,
                 num_cpus: Optional[int] = 0,
                 verbose: Optional[bool] = False
                 ) -> None:
        """
        Initializes the conformer ensemble generator.
            Args:
                conformer (Conformer): The initial conformer structure to start the ensemble generation.
                calc (Calculation): A calculation object to perform energy minimizations
                num_iterations (int, optional): Number of Monte Carlo iterations to perform. Default is 100.
                energy_window (float, optional): Maximum energy window (in eV) above the minimum energy conformer to retain conformers. Default is 10.0.
                max_bonds_rotate (int, optional): Maximum number of rotatable bonds to rotate in each step. Default is 3.
                max_attempts (int, optional): 
                angle_step (float, optional): Step size (in degrees) for bond rotation. Default is 30.0.
                energy_threshold (float, optional): Energy threshold (in eV) for accepting new conformers. Default is 1.0.
                rmsd_threshold (float, optional): RMSD threshold (in Å) for distinguishing unique conformers. Default is 0.3.
                initial_optimziation (bool, optional):
                random_walk (bool, optional): If True, use random walk for bond rotations. Default is False.
                reduce_angle (bool, optional): If True, reduce angle step size during the search. Default is False.
                reduce_angle_every (int, optional): 
                reduce_angle_by (int, optional):
                only_heavy (bool, optional): If True, only rotate dihedrals associated with heavy atoms
                parallel (bool, optional): If True, perform calculations in parallel. Default is True.
                num_cpus (int, optional): Number of CPUs to use for parallel calculations. If 0, use all available CPUs. Default is 0.
                verbose (bool, optional)
        """
        self.conformer = conformer
        self.calc = calc
        self.num_iterations = num_iterations
        self.energy_window = energy_window
        self.max_bonds_rotate = max_bonds_rotate
        self.max_attempts = max_attempts
        self.angle_step = angle_step
        self.energy_threshold = energy_threshold
        self.rmsd_threshold = rmsd_threshold
        self.initial_optimization = initial_optimization
        self.random_walk = random_walk
        self.reduce_angle = reduce_angle
        self.reduce_angle_every = reduce_angle_every
        self.reduce_angle_by = reduce_angle_by
        self.only_heavy = only_heavy
        self.parallel = parallel
        self.num_cpus = num_cpus
        self.verbose = verbose
        if self.num_cpus == 0:
            self.num_cpus = os.cpu_count()
        if not self.parallel:
            self.num_cpus = 1
        if self.verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def log_info(self, message):
        if self.verbose:
            logging.info(message)

    def run_monte_carlo(self) -> List[np.ndarray]:
        """
        Runs a Monte Carlo search to generate a conformer ensemble by iteratively sampling, modifying, and optimizing molecular conformers.
        The method performs the following steps:
            1. Optionally runs an initial optimization on the starting conformer.
            2. Identifies rotatable dihedral angles, excluding those associated with constrained atoms.
            3. Iteratively samples conformers, applies random dihedral rotations, and optimizes the resulting structures.
            4. Filters out high-energy and duplicate conformers.
            5. Sorts the ensemble by energy and returns the final set of unique, low-energy conformers.
        
        Returns
        -------
        List[np.ndarray]
            A list of np.ndarray objects representing the coordinates of the final ensemble of unique, optimized conformers.
        """
        # Run the initial optimization
        if self.initial_optimization:
            self.log_info("Running intitial optimization")
            positions, energy = self.calc.run(self.conformer.atoms, self.conformer.constrained_atoms)
            self.conformer.atoms.positions = positions
        else:
            energy = self.calc.energy(self.conformer.atoms)
        # Get the dihedrals to rotate
        dihedrals = cheminformatics.get_dihedral_matches(self.conformer.mol, self.only_heavy)
        self.max_bonds_rotate = min(len(dihedrals), self.max_bonds_rotate)

        # Remove any dihedrals associated with constrained atoms
        final_dihedrals = []
        for dihedral in dihedrals:
            if dihedral[1] in self.conformer.constrained_atoms and dihedral[2] in self.conformer.constrained_atoms:
                # If the bond is constrained, we need to remove it from the list of rotatable bonds
                continue
            else:
                final_dihedrals.append(dihedral)
        dihedrals = final_dihedrals

        # Initialize information for identity checking
        # TODO: halides are weird with bond formation occasionally so they are currently gnore
        self.original_bonds, self.metal_atoms, self.halides = cheminformatics.initialize_mc_identity_check(self.conformer.atoms, self.conformer.mol)

        final_ensemble = [self.conformer.atoms.get_positions()]
        final_energies = [energy]
        used = [0]
        current_iter = 0
        while current_iter < self.num_iterations:
            self.log_info(f"Iteration: {current_iter} Current min energy: {min(final_energies)}")
            # Reduce angle if reduce_angle true
            if self.reduce_angle:
                if iter % self.reduce_angle_every == 0:
                    self.angle_step = self.angle_step / self.reduce_angle_by
            
            # Sample a conformer, rotate its dihedrals, and optimize it (if parallel, do this num_cpus times)
            calculation_input = []
            for _ in range(self.num_cpus):
                current_iter += 1
                success = False
                index = self.sample_conformer(used)
                success, positions = self.modify_conformer(final_ensemble[index], dihedrals)
                if success:
                    used[index] += 1
                    atoms_to_optimize = copy(self.conformer.atoms)
                    atoms_to_optimize.set_positions(positions)
                    calculation_input.append({"cls": self.calc, 
                                              "func_name": "run", 
                                              "args": {"atoms": atoms_to_optimize, 
                                                       "constrained_atoms": self.conformer.constrained_atoms}})
            if len(calculation_input) == 0:
                continue
            positions_and_energies = multiproc.parallel_run_proc(run_class_func, calculation_input, self.num_cpus)
            
            # Filter out high energy and duplicate conformers
            for positions, energy in positions_and_energies:
                if self.check_conformer(final_ensemble, final_energies, positions, energy):
                    final_ensemble.append(positions)
                    final_energies.append(energy)
                    used.append(0)

            # Sort all of the lists by energies
            final_ensemble, used, final_energies = zip(*sorted(zip(final_ensemble, used, final_energies), key=lambda x: x[2]))
            final_ensemble = list(final_ensemble)
            used = list(used)
            final_energies = list(final_energies)
        
        return final_ensemble

    def sample_conformer(self, used: List[int]) -> int:
        """
        Selects a conformer index based on the sampling strategy.
        If random_walk is enabled, randomly selects an index from the provided list of used indices.
        Otherwise, selects the index corresponding to the minimum value in the used list.
        Args:
            used (list or array-like): A list of indices or values representing conformer usage.
        Returns:
            int: The selected conformer index.
        """
        if self.random_walk:
            index = random.choices(used, k=1)[0]
        else:
            index = np.argmin(np.array(used))
        return index

    def modify_conformer(self, conformer: np.ndarray, all_dihedrals: List[Tuple[int, int, int, int]]) -> Tuple[bool, np.ndarray]:
        """
        Attempts to modify a given conformer by randomly rotating a subset of its dihedral angles.
        For a maximum number of attempts (`self.max_attempts`), this method:
            - Copies the input conformer and its associated molecule.
            - Randomly selects a subset of dihedral angles to rotate.
            - Applies a rotation to the selected dihedrals by a fixed angle step (`self.angle_step`).
            - Tests the modified conformer against a set of constraints (`self.constraint_test`).
        If a valid conformer is found that passes the constraint test, the process stops early.
        Args:
            conformer (np.ndarray): The input conformer coordinates to be modified.
            all_dihedrals (List[Tuple[int, int, int, int]]): List of all possible dihedral angles (as atom index tuples) in the molecule.
        Returns:
            Tuple[bool, np.ndarray]: 
                - A boolean indicating whether a valid conformer was found.
                - The resulting np.ndarray with the coordinates of the modified conformer.
        """
        success = False
        for _ in range(self.max_attempts):
            temp_mol = copy(self.conformer.mol)
            temp_mol = cheminformatics.add_coords_to_mol(conformer, temp_mol)
            num_dihedrals = random.randint(1, self.max_bonds_rotate)
            dihedrals = random.choices(all_dihedrals, k=num_dihedrals)
            cheminformatics.rotate_dihedrals(temp_mol.GetConformer(), dihedrals, self.angle_step)
            if self.constraint_test(temp_mol.GetConformer()):
                success = True
                break
        atoms = cheminformatics.mol_to_ase_atoms(temp_mol)
        return success, atoms.get_positions()

    def constraint_test(self, conf: Chem.rdchem.Conformer) -> bool:
        """
        Determine whether the conformer meets the constraint test 2 as defined in CITATION.
        This function checks whether the interatomic distances between non-bonded atoms is under 1/4 of the 
        van der Waals radius of the atoms. If the distance is under this threshold, the conformer is considered invalid.

        Parameters
        ----------
        conf : rdkit.Chem.rdchem.Conformer
            The conformer to be tested.
        Returns
        -------
        bool
            True if the conformer passes the constraint test, False otherwise.
        """
        # Get the 3D coordinates of the atoms in the conformer
        coords = conf.GetPositions()
        # Calculate the distance matrix between all pairs of atoms
        dist_matrix = distance_matrix(coords, coords)
        # Get the van der Waals radii for each atom in the conformer
        vdw_radii = [PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum()) for atom in self.conformer.mol.GetAtoms()]
        # Make a matrix of the van der Waals radii
        vdw_matrix = np.array([[vdw_radii[i] + vdw_radii[j] for j in range(len(vdw_radii))] for i in range(len(vdw_radii))])
        vdw_matrix = vdw_matrix / 4.0  # Scale the van der Waals radii by 1/4
        # Set the distances between bonded atoms to zero (to ignore them in the test)
        for bonded_atom_pair in self.conformer.bonded_atoms:
            i, j = bonded_atom_pair
            vdw_matrix[i][j] = 0.0
            vdw_matrix[j][i] = 0.0
        # Set the diagonal of the van der Waals matrix to zero (to ignore self-distances)
        np.fill_diagonal(vdw_matrix, 0.0)
        # Check if the distance between any two non-bonded atoms is less than 1/4 of the van der Waals radius
        difference_matrix = dist_matrix - vdw_matrix
        # If any distance is less than 0, the conformer fails the constraint test
        if np.any(difference_matrix < 0):
            return False
        # If all distances are greater than or equal to 0, the conformer passes the constraint test
        return True
    
    def check_conformer(self, ensemble: List[np.ndarray], 
                        energies: List[float], 
                        conf: np.ndarray, 
                        energy: float) -> bool:
        """
        Checks whether a given conformer should be added to the ensemble based on energy and structural similarity.
        This function evaluates if the provided conformer (`conf`) with its associated energy (`energy`) is sufficiently low in energy
        and structurally distinct from all conformers already present in the ensemble. The conformer is accepted if:
          - Its energy is within `self.energy_window` of the minimum energy in the current ensemble.
          - Its root-mean-square deviation (RMSD) from all conformers in the ensemble is greater than `self.rmsd_threshold`.
        Args:
            ensemble (list): List of conformers cooordinates currently in the ensemble, each represented as np.ndarray objects.
            energies (list): List of energies corresponding to the conformers in the ensemble.
            conf (np.ndarray): The candidate conformer to be evaluated.
            energy (float): The energy of the candidate conformer.
        Returns:
            bool: True if the conformer passes both the energy and RMSD criteria and should be added to the ensemble, False otherwise.
        """

        if energy > min(energies) + self.energy_window:
            return False
        temp_atoms = copy(self.conformer.atoms)
        temp_atoms.set_positions(conf)
        if not cheminformatics.check_identity_mc(self.original_bonds, self.metal_atoms, self.halides, temp_atoms):
            return False
        temp_mol = copy(self.conformer.mol)
        temp_mol = cheminformatics.add_coords_to_mol(conf, temp_mol)
        temp_reference_mol = copy(self.conformer.mol)
        atom_map = [(i,i) for i in range(len(temp_mol.GetAtoms()))]
        for reference_conf in ensemble:
            temp_reference_mol = cheminformatics.add_coords_to_mol(reference_conf, temp_reference_mol)
            rmsd = rdMolAlign.AlignMol(temp_mol, temp_reference_mol, atomMap=atom_map)
            if rmsd < self.rmsd_threshold:
                return False
        return True