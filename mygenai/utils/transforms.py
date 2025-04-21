import torch
from torch_geometric.utils import remove_self_loops

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
