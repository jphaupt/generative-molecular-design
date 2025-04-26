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
        # Determine which nodes are real by checking the padding feature
        # The last column in data.x indicates padding (1 = padding node, 0 = real node)
        is_padding = data.x[:, -1] > 0.5
        current_nodes = (~is_padding).sum().item()

        # Create complete graph
        num_nodes = data.num_nodes
        row = torch.arange(num_nodes, dtype=torch.long, device=data.x.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=data.x.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        # Handle edge attributes
        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = num_nodes * num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        # Remove self-loops
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        # Now add a "padding_edge" feature to edge attributes
        if data.edge_attr is not None:
            # Get indices of edges between real nodes
            real_node_mask = ~is_padding

            # Identify real edges (both endpoints are real nodes)
            src_real = real_node_mask[data.edge_index[0]]
            dst_real = real_node_mask[data.edge_index[1]]
            real_edges = src_real & dst_real

            # Add "is_padding_edge" feature (0 for real edges, 1 for padding)
            padding_feature = (~real_edges).float().unsqueeze(1)
            data.edge_attr = torch.cat([data.edge_attr, padding_feature], dim=1)

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
    """Pad graphs with explicit padding nodes"""
    def __init__(self, num_nodes=29, allow_truncate=False):
        self.num_nodes = num_nodes
        self.allow_truncate = allow_truncate

    def __call__(self, data):
        # Current number of nodes
        current_nodes = data.num_nodes

        # If graph is already the right size, do nothing
        if current_nodes == self.num_nodes:
            return data

        # Handle truncation cases as before...

        # Pad if necessary
        elif current_nodes < self.num_nodes:
            # First add a new "is_padding" feature to existing nodes (all 0s)
            padding_feature = torch.zeros((data.x.size(0), 1),
                                         dtype=data.x.dtype,
                                         device=data.x.device)
            data.x = torch.cat([data.x, padding_feature], dim=1)

            # Then create padding nodes with "is_padding" feature set to 1
            padding_x = torch.zeros((self.num_nodes - current_nodes, data.x.size(1)),
                                   dtype=data.x.dtype, device=data.x.device)
            padding_x[:, -1] = 1.0  # Set padding indicator to 1
            data.x = torch.cat([data.x, padding_x], dim=0)

        # Update number of nodes
        data.num_nodes = self.num_nodes

        return data
