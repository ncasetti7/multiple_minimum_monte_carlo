"""Module for running geometry optimizations"""

import os
import contextlib
from typing import Optional, List, Tuple
import numpy as np
import ase
import ase.calculators.calculator
from ase.optimize import BFGS
import ase.optimize.optimize
from ase.constraints import FixAtoms

EV_TO_KCAL = 23.0605

class Calculation():
    def __init__(self):
        pass

    def run(self, atoms : ase.Atoms, constrained_atoms : Optional[List[int]] = []) -> Tuple[np.ndarray, float]:
        pass

    def energy(self, atoms: ase.Atoms) -> float:
        pass

class ASEOptimization(Calculation):
    def __init__(self,
                 calc: ase.calculators.calculator.Calculator,
                 optimizer: Optional[ase.optimize.optimize.Optimizer] = BFGS, 
                 fmax: Optional[float] = 0.01, 
                 max_cycles: Optional[float] = 1000, 
                 verbose: Optional[float] = False) -> None:
        self.calc = calc
        self.optimizer = optimizer
        self.fmax = fmax
        self.max_cycles = max_cycles
        self.verbose = verbose

    def run(self, atoms : ase.Atoms, constrained_atoms : Optional[List[int]] = []) -> Tuple[ase.Atoms, float]:
        """
        Perform constrained optimization using ASE.

        Args:
            atoms (ase.Atoms): Molecule to optimize.
            constrained_atoms: Atomic indices to constrain

        Returns:
            atoms (ase.Atoms): Optimized ASE atoms object.
        """
        atoms.calc = self.calc
        if len(constrained_atoms) > 0:
            atoms.set_constraint(FixAtoms(constrained_atoms))
        # Perform optimization
        if self.verbose:
            opt = self.optimizer(atoms)
            opt.run(fmax=self.fmax, steps=self.max_cycles)
        else:
            with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f):
                opt = self.optimizer(atoms)
                opt.run(fmax=self.fmax, steps=self.max_cycles)
        return atoms.get_positions(), atoms.get_potential_energy() * EV_TO_KCAL
    
    def energy(self, atoms: ase.Atoms) -> float:
        """
        Return the energy of the input atoms object

        Args:
            atoms (ase.Atoms): Input atoms object

        Returns:
            energy (float): Energy of the atoms object
        """
        atoms.calc = self.calc
        return atoms.get_potential_energy() * EV_TO_KCAL