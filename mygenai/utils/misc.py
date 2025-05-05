from torch_geometric.data import Batch

BOND_TYPES = { # human-readable bond types
    0: "single",
    1: "double",
    2: "triple",
    3: "aromatic",
    4: "no bond"
}

def create_single_molecule_loader(molecule):
    """DataLoader that only returns a single molecule, primarily for testing purposes"""
    water_batch = Batch.from_data_list([molecule] * 4)  # Batch size of 4
    class SingleMoleculeLoader:
        def __iter__(self):
            yield water_batch
        def __len__(self):
            return 1

    return SingleMoleculeLoader()

def get_ground_truth_bonds(molecule, natom, verbose=False):
    edges = molecule.edge_index.cpu().numpy()
    attrs = molecule.edge_attr.cpu().numpy()
    if verbose: print("Ground truth adjacency matrix:\nEdges")
    for i in range(min(10, edges.shape[1])):
        src, dst = edges[0, i], edges[1, i]
        if src < natom and dst < natom:  # only consider real atoms
            attr_idx = attrs[i].argmax()
            bond_type = BOND_TYPES[attr_idx]
            if verbose: print(f"  Atom {src} - Atom {dst}: {bond_type}")

    return edges, attrs
