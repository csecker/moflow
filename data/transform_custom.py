import json
import numpy as np
import pandas as pd
from rdkit import Chem
import random
import os


# Function to extract atomic numbers from a SMILES string
def extract_atomic_numbers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return set(atom.GetAtomicNum() for atom in mol.GetAtoms())
    else:
        return set()

# Function to calculate the number of atoms in a SMILES string
def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol.GetNumAtoms()
    else:
        return 0

# Function to count the number of bond types in a molecule
def count_bond_types(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Chem.Kekulize(mol)  # Ensure aromatic bonds are correctly perceived (deactivated)
        bond_types = set(bond.GetBondType() for bond in mol.GetBonds())
        return len(bond_types)
    else:
        return 0

# Getting custom dataset
df_custom = pd.read_csv('../data/custom.csv', index_col=0)

# Setting atomic num list
if os.getenv('VFGM_MOFLOW_ATOMIC_NUM_LIST', '') == '':
    # Extract unique atomic numbers from the entire SMILES list
    unique_atomic_numbers = set()
    for smi in df_custom['smiles']:
        unique_atomic_numbers.update(extract_atomic_numbers(smi))
    custom_atomic_num_list = sorted(list(unique_atomic_numbers))
    custom_atomic_num_list.append(0)
else:
    custom_atomic_num_list = os.getenv('VFGM_MOFLOW_ATOMIC_NUM_LIST', '')
    custom_atomic_num_list = custom_atomic_num_list.split(':')
    custom_atomic_num_list = [int(element) for element in custom_atomic_num_list]
print("Unique atomic numbers from all smiles:", custom_atomic_num_list)

# Setting max atoms
if os.getenv('VFGM_MOFLOW_MAX_ATOMS', '') == '':
    # Calculate the number of atoms for each SMILES
    atom_counts = [count_atoms(smi) for smi in df_custom['smiles']]
    max_atoms = max(atom_counts)
else:
    max_atoms = int(os.getenv('VFGM_MOFLOW_MAX_ATOMS', ''))
print("Max atoms:", max_atoms)

# Calculate the number of bond types for each SMILES
bond_types_counts = [count_bond_types(smi) for smi in df_custom['smiles']]
# Get the total number of unique bond types in the entire dataset
n_bonds = max(bond_types_counts)
print("Number of bonds for current dataset:", n_bonds)

# custom_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # 0 is for virtual node.
# max_atoms = 38
# n_bonds = 4


def one_hot_custom(data, out_size=max_atoms):
    num_max_id = len(custom_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = custom_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_fn_custom(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_custom(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    # Calculate the number of indices you want (10% of the list)
    num_indices = int(0.1 * len(df_custom))
    # Generate a list of random indices
    random_indices = sorted(random.sample(range(len(df_custom)), num_indices))

    # Write indices to file
    file_path = '../data/valid_idx_custom.json'
    with open(file_path, 'w') as json_out:
        json.dump(random_indices, json_out)

    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids
