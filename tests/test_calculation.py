from multiple_minimum_monte_carlo.calculation import ASEOptimization
import numpy as np

EV_TO_KCAL = 23.0605

class DummyCalc:
    def __init__(self, energy=1.23):
        self._energy = energy

    def get_potential_energy(self, atoms=None):
        return self._energy


class DummyAtoms:
    def __init__(self):
        self.calc = None

    def set_constraint(self, constraints):
        self._constraints = constraints

    def get_potential_energy(self):
        return self.calc.get_potential_energy(self)

    def get_positions(self):
        # return a dummy positions array compatible with ASE
        return np.zeros((1, 3))

def test_energy_returns_calculator_energy():
    atoms = DummyAtoms()
    calc = DummyCalc(4.56)
    ao = ASEOptimization(calc=calc)
    atoms.calc = calc
    assert ao.energy(atoms) == 4.56 * EV_TO_KCAL


def test_run_applies_constraints_and_returns_tuple():
    atoms = DummyAtoms()
    calc = DummyCalc(7.89)
    # Provide a dummy optimizer that accepts atoms and has run()
    class DummyOpt:
        def __init__(self, atoms):
            self.atoms = atoms

        def run(self, fmax=None, steps=None):
            return None

    ao = ASEOptimization(calc=calc, optimizer=DummyOpt)
    atoms.calc = calc
    out_positions, energy = ao.run(atoms, constrained_atoms=[1, 2, 3])
    # run now returns positions array and energy scaled by EV_TO_KCAL
    assert hasattr(out_positions, "shape")
    assert energy == 7.89 * EV_TO_KCAL
