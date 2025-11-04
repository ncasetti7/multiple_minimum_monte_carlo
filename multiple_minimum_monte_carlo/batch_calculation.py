from typing import Optional, List, Tuple, Any
import numpy as np
import ase
from ase.constraints import FixAtoms
import torch

EV_TO_KCAL = 23.0605

class BatchCalculation:
    def __init__(self):
        pass

    def run(
        self, atoms_list: List[ase.Atoms], constrained_atoms_list: Optional[List[List[int]]] = None
    ) -> Tuple[np.ndarray, float]:
        pass

    def energy(self, atoms: ase.Atoms) -> float:
        pass

class TorchSimCalculation(BatchCalculation):
    def __init__(self, model: Any,
                 optimizer: Any,
                 device: Optional[str] = 'cpu',
                 dtype: Optional[Any] = torch.float32,
                 max_cycles: Optional[int] = 1000) -> None:
        try:
            from torch_sim.models.interface import ModelInterface
            from torch_sim.optimizers import Optimizer
        except ImportError:
            raise ImportError("TorchSim is not installed")
        self.model = model
        if not isinstance(self.model, ModelInterface):
            raise TypeError("model must be an instance of ModelInterface")
        self.optimizer = optimizer
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be an instance of Optimizer")
        self.device = device
        self.dtype = dtype
        self.max_cycles = max_cycles

    def run(self, atoms_list: List[ase.Atoms], constrained_atoms_list: Optional[List[List[int]]] = None):
        try:
            import torch_sim as ts
        except ImportError:
            raise ImportError("TorchSim is not installed")
        if constrained_atoms_list is not None:
            if len(atoms_list) != len(constrained_atoms_list):
                raise ValueError("Length of atoms_list and constrained_atoms_list must be the same")
            for atoms, constrained_atoms in zip(atoms_list, constrained_atoms_list):
                if len(constrained_atoms) > 0:
                    atoms.set_constraint(FixAtoms(constrained_atoms))
        final_state = ts.optimize(system=atoms_list,
                                  model=self.model,
                                  optimizer=self.optimizer)
        positions = final_state.positions.detach().numpy().astype(np.float64)
        positions = positions.reshape(len(atoms_list),-1, 3)
        energies = list(final_state.energy.detach().numpy().astype(np.float64) * EV_TO_KCAL)
        return positions, energies

    def energy(self, atoms_list: List[ase.Atoms]) -> float:
        try:
            import torch_sim as ts
        except ImportError:
            raise ImportError("TorchSim is not installed")
        state = ts.io.atoms_to_state(atoms_list, device=self.device, dtype=self.dtype)
        result = self.model(state)
        energies = list(result["energy"].detach().numpy().astype(np.float64) * EV_TO_KCAL)
        return energies