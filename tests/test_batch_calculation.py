import sys
import types
import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from ase import Atoms

from multiple_minimum_monte_carlo.batch_calculation import TorchSimCalculation


def _make_fake_torch_sim(monkeypatch, *, optimize_ret=None, energy_ret=None):
    """Install a fake torch_sim package into sys.modules with the pieces used by the code."""
    ts = types.ModuleType("torch_sim")
    # submodules
    models = types.ModuleType("torch_sim.models")
    interface = types.ModuleType("torch_sim.models.interface")
    optim_mod = types.ModuleType("torch_sim.optimizers")

    # Define ModelInterface and Optimizer base classes
    class ModelInterface:
        pass

    class Optimizer:
        pass

    interface.ModelInterface = ModelInterface
    optim_mod.Optimizer = Optimizer

    # Top-level ts.optimize function
    def optimize(system, model, optimizer):
        # Return a simple object with positions and energy tensors
        if optimize_ret is not None:
            return optimize_ret
        n = len(system)
        # assume each atoms object has 1 atom for simplicity
        positions = torch.zeros((n, 1, 3), dtype=torch.float32)
        energy = torch.tensor([1.0] * n, dtype=torch.float32)
        obj = types.SimpleNamespace(positions=positions, energy=energy)
        return obj

    ts.optimize = optimize

    # io submodule
    io = types.ModuleType("torch_sim.io")

    def atoms_to_state(atoms_list, device="cpu", dtype=torch.float32):
        # return a dummy state object accepted by model
        return {"atoms": atoms_list}

    io.atoms_to_state = atoms_to_state

    # pack modules into sys.modules
    # attach submodules as attributes and pack modules into sys.modules
    ts.models = models
    ts.models.interface = interface
    ts.optimizers = optim_mod
    ts.io = io
    sys.modules["torch_sim"] = ts
    sys.modules["torch_sim.models"] = models
    sys.modules["torch_sim.models.interface"] = interface
    sys.modules["torch_sim.optimizers"] = optim_mod
    sys.modules["torch_sim.io"] = io
    return ModelInterface, Optimizer


def test_init_raises_if_torchsim_missing(monkeypatch):
    # Ensure torch_sim is not present
    # Remove any existing torch_sim entries from sys.modules
    for k in list(sys.modules.keys()):
        if k.startswith("torch_sim"):
            monkeypatch.delitem(sys.modules, k, raising=False)
    # Importing class is fine, but instantiation should raise ImportError
    with pytest.raises(ImportError):
        TorchSimCalculation(model=None, optimizer=None)


def test_init_type_check(monkeypatch):
    ModelInterface, Optimizer = _make_fake_torch_sim(monkeypatch)
    # Pass objects that are not instances of the expected types
    with pytest.raises(TypeError):
        TorchSimCalculation(model=object(), optimizer=object())


def test_run_returns_positions_and_energies(monkeypatch):
    ModelInterface, Optimizer = _make_fake_torch_sim(monkeypatch)

    # create dummy model/optimizer instances of the required types
    class MyModel(ModelInterface):
        def __call__(self, state):
            return {"energy": torch.tensor([2.0])}

    my_model = MyModel()

    class MyOpt(Optimizer):
        pass

    my_opt = MyOpt()

    # instantiate calculation
    calc = TorchSimCalculation(model=my_model, optimizer=my_opt)

    # create two tiny ASE Atoms (1 atom each)
    atoms_list = [Atoms("H"), Atoms("H")]
    # matching constrained list
    positions, energies = calc.run(atoms_list, constrained_atoms_list=[[], []])
    assert isinstance(positions, np.ndarray)
    assert isinstance(energies, list)
    assert positions.shape[0] == len(atoms_list)
    assert len(energies) == len(atoms_list)


def test_run_mismatched_constrained_list_raises(monkeypatch):
    ModelInterface, Optimizer = _make_fake_torch_sim(monkeypatch)

    class MyModel(ModelInterface):
        pass

    class MyOpt(Optimizer):
        pass

    calc = TorchSimCalculation(model=MyModel(), optimizer=MyOpt())
    atoms_list = [Atoms("H")]
    with pytest.raises(ValueError):
        calc.run(atoms_list, constrained_atoms_list=[[], []])


def test_energy_returns_list(monkeypatch):
    ModelInterface, Optimizer = _make_fake_torch_sim(monkeypatch)

    class MyModel(ModelInterface):
        def __call__(self, state):
            return {"energy": torch.tensor([3.0, 4.0])}

    my_model = MyModel()
    class MyOpt(Optimizer):
        pass

    calc = TorchSimCalculation(model=my_model, optimizer=MyOpt())
    atoms_list = [Atoms("H"), Atoms("H")]
    energies = calc.energy(atoms_list)
    assert isinstance(energies, list)
    assert len(energies) == len(atoms_list)
