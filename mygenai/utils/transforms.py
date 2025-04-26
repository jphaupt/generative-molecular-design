import torch
from torch_geometric.utils import remove_self_loops

class ExtractFeatures(object):
    """Extract only the first N features from node attributes"""
    def __init__(self, num_features=5):
        self.num_features = num_features

    def __call__(self, data):
        # Extract only the first num_features features
        data.x = data.x[:, :self.num_features]
        return data

class CompleteGraph(object):
    """
    This transform adds all pairwise edges into the edge index per data sample,
    then removes self loops, i.e. it builds a fully connected or complete graph
    """
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

class AddEdgeExistence(object):
    """
    This transform adds an edge_existence attribute to the data object.
    The edge_existence attribute is a tensor of ones, indicating that all
    edges in edge_index exist in sparse graphs.
    """
    def __call__(self, data):
        # All edges in edge_index exist in sparse graphs
        edge_existence = torch.ones(data.edge_index.shape[1], dtype=torch.float32)
        data.edge_existence = edge_existence
        return data

class SetTarget(object):
    def __call__(self, data):
        target = 4 # hardcoded HOMO-LUMO gap
        data.y = data.y[:, target]
        return data

class PadToFixedSize(object):
    """
    This transform pads graphs to a fixed number of nodes by adding dummy nodes with zero features.

    Args:
        num_nodes (int): The fixed number of nodes all graphs will have. Defaults to 29.
        allow_truncate (bool, optional): If True, will truncate graphs with more
                                         nodes than num_nodes. Default is False.
    """
    def __init__(self, num_nodes=29, allow_truncate=False):
        self.num_nodes = num_nodes
        self.allow_truncate = allow_truncate

    def __call__(self, data):
        # Current number of nodes
        current_nodes = data.num_nodes

        # If graph is already the right size, do nothing
        if current_nodes == self.num_nodes:
            return data

        # If graph is too large and truncation is not allowed, raise error
        if current_nodes > self.num_nodes and not self.allow_truncate:
            raise ValueError(f"Graph has {current_nodes} nodes, which exceeds fixed size {self.num_nodes}. "
                           f"Set allow_truncate=True to truncate graphs.")

        # Truncate if necessary
        if current_nodes > self.num_nodes:
            # Truncate node features
            data.x = data.x[:self.num_nodes]

            # Truncate positions if they exist
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos[:self.num_nodes]

            # Truncate edges connecting to removed nodes
            mask = (data.edge_index[0] < self.num_nodes) & (data.edge_index[1] < self.num_nodes)
            data.edge_index = data.edge_index[:, mask]

            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]

        # Pad if necessary
        elif current_nodes < self.num_nodes:
            # Pad node features
            padding_x = torch.zeros((self.num_nodes - current_nodes, data.x.size(1)),
                                  dtype=data.x.dtype, device=data.x.device)
            data.x = torch.cat([data.x, padding_x], dim=0)

            # Pad positions if they exist
            if hasattr(data, 'pos') and data.pos is not None:
                padding_pos = torch.zeros((self.num_nodes - current_nodes, data.pos.size(1)),
                                       dtype=data.pos.dtype, device=data.pos.device)
                data.pos = torch.cat([data.pos, padding_pos], dim=0)

        # Update number of nodes (important!)
        data.num_nodes = self.num_nodes

        return data
