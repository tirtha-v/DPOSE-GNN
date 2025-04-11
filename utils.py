import torch
from torch import Tensor
import ase
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from torch_geometric.data import Batch
from ase import Atoms
from collections import Counter
from ase.data import chemical_symbols
from torch_geometric.data import Data


class AtomGraphConverter(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0, max_neighbors: int = 50):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def forward(self, atoms: ase.Atoms):
        r"""Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        if AseAtomsAdaptor is None:
            raise RuntimeError(
                "Unable to import pymatgen.io.ase.AseAtomsAdaptor. Make sure pymatgen is properly installed."
            )

        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.cutoff, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neighbors]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        # Create edge index using torch.stack
        edge_index = torch.stack([torch.tensor(_n_index), torch.tensor(_c_index)], dim=0)
        edge_distance = torch.tensor(n_distance)

        return edge_index, edge_distance, torch.tensor(_offsets)
    
def data_to_ase_atoms(data_batch: Batch):
    atoms_list = []
    num_molecules = data_batch.batch.max().item() + 1  # Number of molecules in the batch

    # Extract cell tensor if it exists and reshape it to (num_molecules, 3, 3)
    if hasattr(data_batch, 'cell'):
        cell_tensor = data_batch.cell.cpu().numpy().reshape(-1, 3, 3)  # Reshape to [num_molecules, 3, 3]

    for i in range(num_molecules):  # Iterate over each molecule in the batch
        # Extract atomic numbers and positions for the current molecule
        mask = data_batch.batch == i
        atomic_numbers = data_batch.z[mask].cpu().numpy()  # Convert to numpy array
        positions = data_batch.pos[mask].cpu().numpy()     # Convert to numpy array
        
        # Extract the cell matrix for the current molecule
        if hasattr(data_batch, 'cell'):
            molecule_cell = cell_tensor[i]  # Get the 3x3 cell matrix for the ith molecule
        else:
            molecule_cell = None

        # Create ASE Atoms object
        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=molecule_cell, pbc=True)
        atoms_list.append(atoms)
    
    return atoms_list

def get_chemical_formula_string(atoms):
    """
    Generate a chemical formula string for the whole system.

    Parameters:
        atoms (ase.Atoms): The Atoms object containing the chemical structure.

    Returns:
        str: A string representing the chemical formula.
    """
    from collections import Counter

    # Count the chemical symbols
    chemical_symbols = atoms.get_chemical_symbols()
    element_counts = Counter(chemical_symbols)

    # Create a sorted chemical formula string
    formula = ''.join(f"{symbol}{count}" for symbol, count in sorted(element_counts.items()))

    return formula


def qm9_to_xyz(data_entry):
    # Mapping for atomic number to element symbol (extend as needed)
    atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    num_atoms = data_entry.pos.shape[0]
    xyz_lines = [f"{num_atoms}", ""]  # First line is the number of atoms; second line is a blank/comment line.
    
    for atomic_number, pos in zip(data_entry.z.tolist(), data_entry.pos.tolist()):
        symbol = atomic_symbols.get(atomic_number, '?')
        xyz_lines.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    
    return "\n".join(xyz_lines)

def get_molecular_symbol_from_atoms(atoms):
    atom_counts = Counter(atoms.numbers)  # Count occurrences of each atomic number
    molecular_symbol = ''.join(
        f"{chemical_symbols[atom]}{(count if count > 1 else '')}" 
        for atom, count in sorted(atom_counts.items())
    )
    return molecular_symbol

def inference_qm9(model, test_loader, device):
    model.eval()
    total_var = []
    total_pred =[]
    mean_var =0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            val_pred = model(data.z, data.pos, data.batch)
            val_var = torch.var(val_pred, dim=1, unbiased=False)
            mean_var +=val_var.mean().item()
            pred_mean = val_pred.mean(dim=1).item()
            total_var.append(val_var.item())
            total_pred.append(pred_mean)
    return total_pred, total_var


def process_oc20_input(data_list, converter, device):
    input_data = []
    
    # Iterate over the data list (either metal or nonmetal)
    for slab in data_list:
        atomic_numbers = slab.atomic_numbers.clone().detach().long()  
        positions = slab.pos.clone().detach().float()  
        cell = slab.cell[0].clone().detach().float()  
        energy = torch.tensor(slab.y, dtype=torch.float32)  # Energy as a tensor

        # Step 2: Create ASE Atoms object
        atoms = Atoms(numbers=atomic_numbers.numpy(), positions=positions.numpy(), cell=cell.numpy(), pbc=True)
        
        # Step 3: Use the AtomGraphConverter to get edge_index and edge_weight (distances)
        edge_index, edge_weight, offsets = converter(atoms)
        
        # Step 4: Create a PyTorch Geometric Data object
        data = Data(
            z=atomic_numbers.to(device),  # Atomic numbers
            pos=positions.to(device),     # Atomic positions
            y=energy.to(device),          # Target property (energy)
            cell=cell.to(device),         # Cell tensor
            edge_index=edge_index.to(device),  # Add computed edge index
            edge_weight=edge_weight.to(device),  # Add computed edge weights (distances)
        )
        
        # Step 5: Append the data to the list
        input_data.append(data)
    
    return input_data

def inference_oc20(model, test_loader, device):
    
    model.eval()
    
    # Initialize lists to store atom-wise metrics
    pred_per_atom = []
    true_per_atom = []
    var_per_atom = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get predictions from the model
            val_pred = model(data)
            num_atoms = data.z.size(0)
            
            # Calculate mean prediction and variance per atom
            pred_mean = val_pred.detach().mean(dim=1).item()
            val_var = torch.var(val_pred, dim=1, unbiased=False)
            var_per_atom_value = torch.var(val_pred / num_atoms, dim=1, unbiased=False)
            
            # Calculate prediction and ground truth per atom
            p_per_atom = pred_mean / num_atoms
            t_per_atom = data.y.item() / num_atoms
            
            # Append results to lists
            pred_per_atom.append(p_per_atom)
            true_per_atom.append(t_per_atom)
            var_per_atom.append(var_per_atom_value.item())
    
    return pred_per_atom, true_per_atom, var_per_atom


def process_gold_input(results, converter):
   
    # Initialize the list to store SchNet input data
    data_list = []

    # Process all results in the results_bulk list
    for i, result in enumerate(results):
        # Extract the specifier for the structure
        if result.structure == 'surface':
            spec = result.surf
        else:
            spec = result[f'{result.structure}']

        # Extract the necessary data: atomic numbers, positions, cell, energy, and forces
        atomic_numbers = torch.tensor(result.numbers, dtype=torch.long)  # Atomic numbers as a tensor
        positions = torch.tensor(result.positions, dtype=torch.float32)  # Atomic positions as a tensor
        cell = torch.tensor(result.cell, dtype=torch.float32)  # Cell tensor
        energy = torch.tensor(result.energy, dtype=torch.float32)  # Energy as a tensor
        forces = torch.tensor(result.forces, dtype=torch.float32)
        
        # Compute the force magnitudes and max force per atom
        force_magnitudes = torch.norm(forces, dim=1)
        max_force_per_atom = torch.max(force_magnitudes) / atomic_numbers.size(0)
        energy_per_atom = energy / atomic_numbers.size(0)

        # Create ASE Atoms object for graph conversion
        atoms = Atoms(numbers=atomic_numbers.numpy(), positions=positions.numpy(), cell=cell.numpy(), pbc=True)

        # Get edge_index and edge_weight from the converter
        edge_index, edge_weight, offsets = converter(atoms)

        # Create a PyTorch Geometric Data object
        data = Data(
            z=atomic_numbers,  # Atomic numbers
            pos=positions,     # Atomic positions
            y=energy,          # Total energy
            y_atom=energy_per_atom,  # Energy per atom
            fmax_atom=max_force_per_atom,  # Max force per atom
            cell=cell,         # Cell tensor
            edge_index=edge_index,  # Edge index
            edge_weight=edge_weight,  # Edge weights (distances)
            structure=result.structure,
            spec=spec,         # Specifier for the structure
            idx=i              # Index for the structure
        )

        # Append the processed data to the indomain_data list
        data_list.append(data)

    # Return the list of processed in-domain data
    return data_list

def inference_gold(model, test_loader, device):
    model.eval()
    
    # Initialize lists to store results
    pred_per_atom = []
    true_per_atom = []
    var_atom = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get model predictions
            val_pred = model(data)
            num_atoms = data.z.size(0)
            energy_per_atom = val_pred / num_atoms
            
            # Compute the variance per atom
            var_per_atom = torch.var(energy_per_atom, dim=1, unbiased=False).item()
            
            # Get the predicted and true energy per atom
            pred_mean = val_pred.detach().mean(dim=1).item()
            p_per_atom = pred_mean / num_atoms
            t_per_atom = data.y.item() / num_atoms
            
            # Append the results to the lists
            pred_per_atom.append(p_per_atom)
            true_per_atom.append(t_per_atom)
            var_atom.append(var_per_atom)

    # Return the last three values
    return pred_per_atom, true_per_atom, var_atom
